import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
import os
import pandas as pd
import json
import math
import warnings
import torch.nn as nn
import torch.nn.functional as F

# --- Basic Configuration ---
warnings.filterwarnings("ignore", category=UserWarning)

# =============================================================================
#  1. Paths & Core Parameters for Phase 2B
# =============================================================================
# --- Data Paths (consistent with previous phases) ---
ALL_TRAIN_DATA_PATH = r"C:\Users\1\Documents\Solarpower\Data_processed\Train_val_test_split_new\train_selected_sites.csv"
ALL_VAL_DATA_PATH = r"C:\Users\1\Documents\Solarpower\Data_processed\Train_val_test_split_new\validation_enhanced.csv"
STATS_FILE_PATH = r"C:\Users\1\Documents\Solarpower\Data_processed\Train_val_test_split_new\scaler\campus_scale.txt"

# --- Model Paths ---
# !! Key: Load the completed Phase 2A model !!
PHASE2A_MODEL_PATH = r"C:\Python\Trained_models\Elite_Model_Transformer_Phase2B_Complete\transformer_elite_model_phase2b.pth"

# !! Key: Define the new save directory and filename for the Phase 2B model !!
PHASE2B_SAVE_DIR = r"C:\Python\Trained_models\Elite_Model_Transformer_Phase2B_Complete"
os.makedirs(PHASE2B_SAVE_DIR, exist_ok=True)
PHASE2B_MODEL_SAVE_PATH = os.path.join(PHASE2B_SAVE_DIR, 'transformer_elite_model_phase2b.pth')

# --- Feature Columns (consistent with previous phases) ---
TARGET_COLUMN = 'SolarGeneration'
WEATHER_FEATURES = ['AirTemperature', 'Ghi_hourly', 'CloudOpacity_hourly', "minutes_since_last_update",
                    "RelativeHumidity"]
TIME_FEATURES = ["zenith_sin", "azimuth_sin", "day_sin", "year_sin"]
ENCODER_FEATURES = WEATHER_FEATURES + TIME_FEATURES + [TARGET_COLUMN]
FUTURE_KNOWN_FEATURES = TIME_FEATURES
DECODER_INPUT_FEATURES = FUTURE_KNOWN_FEATURES + [TARGET_COLUMN]
TARGET_FEATURE_INDEX = DECODER_INPUT_FEATURES.index(TARGET_COLUMN)

# --- Training Hyperparameters ---
LOOKBACK_STEPS = 4 * 24
LOOKFORWARD_STEPS = 4 * 5
BATCH_SIZE = 32
# !! Key: Use a very small learning rate for this final fine-tuning phase !!
LEARNING_RATE_PHASE2B = 0.1e-6
EPOCHS_PHASE2B = 10  # Phase 2B can benefit from more epochs with a smaller learning rate
MAX_GRAD_NORM = 1.0
PATIENCE = 3
SENTINEL_VALUE = -999.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Transformer Model Hyperparameters (must match previous phases) ---
D_MODEL = 128
NHEAD = 8
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
DIM_FEEDFORWARD = 512
DROPOUT_RATE = 0.1


# =============================================================================
#  2. Reusable Modules (Model, Dataset, Scaling - Unchanged from Phase 2A)
# =============================================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TimeSeriesTransformer(nn.Module):
    def __init__(self, encoder_feature_dim: int, decoder_feature_dim: int, d_model: int, nhead: int,
                 num_encoder_layers: int, num_decoder_layers: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.encoder_embedding = nn.Linear(encoder_feature_dim, d_model)
        self.decoder_embedding = nn.Linear(decoder_feature_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.output_layer = nn.Linear(d_model, 1)

    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """
        FIXED: Generates a square subsequent mask.
        The incorrect `device=self.d_model.device` argument has been removed.
        The mask is created on the CPU by default and moved to the correct device in the forward pass.
        """
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                tgt_key_padding_mask: torch.Tensor,
                src_key_padding_mask: torch.Tensor) -> torch.Tensor:
        # This line correctly creates the mask and moves it to the device of the input tensor.
        tgt_mask = self._generate_square_subsequent_mask(tgt.size(1)).to(src.device)

        src_emb = self.encoder_embedding(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoder(src_emb)

        tgt_emb = self.decoder_embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoder(tgt_emb)

        memory = self.transformer.encoder(
            src=src_emb,
            src_key_padding_mask=src_key_padding_mask
        )

        decoder_output = self.transformer.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        return self.output_layer(decoder_output)


class SolarPowerDataset_Transformer(Dataset):
    def __init__(self, df, lookback, lookforward):
        self.df = df
        self.lookback = lookback
        self.lookforward = lookforward
        self.total_seq_len = lookback + lookforward
        self.campus_indices = []
        for campus_id in df['CampusKey'].unique():
            campus_df = df[df['CampusKey'] == campus_id]
            start_index = campus_df.index[0]
            num_samples = len(campus_df) - self.total_seq_len + 1
            if num_samples > 0:
                self.campus_indices.append({'start': start_index, 'count': num_samples})
        self.total_samples = sum(c['count'] for c in self.campus_indices)

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        campus_idx = 0
        while idx >= self.campus_indices[campus_idx]['count']:
            idx -= self.campus_indices[campus_idx]['count']
            campus_idx += 1
        start_pos = self.campus_indices[campus_idx]['start'] + idx
        end_pos = start_pos + self.total_seq_len
        sequence_slice = self.df.iloc[start_pos:end_pos]

        # Encoder data
        src_df = sequence_slice.iloc[:self.lookback].copy()
        src_key_padding_mask = torch.tensor(src_df[TARGET_COLUMN].isna().values, dtype=torch.bool)
        src_df.fillna(0, inplace=True)
        src = torch.tensor(src_df[ENCODER_FEATURES].values, dtype=torch.float32)

        # Decoder data
        future_slice = sequence_slice.iloc[self.lookback:]
        tgt_key_padding_mask = torch.tensor(future_slice[TARGET_COLUMN].isna().values, dtype=torch.bool)

        last_obs = sequence_slice.iloc[self.lookback - 1][TARGET_COLUMN]
        shifted = future_slice[TARGET_COLUMN].shift(1)
        shifted.iloc[0] = last_obs
        shifted = shifted.fillna(0)  # Use 0 for NaN in shifted target
        future_known_df = future_slice[FUTURE_KNOWN_FEATURES]
        tgt_input_df = pd.concat([future_known_df.reset_index(drop=True),
                                  shifted.reset_index(drop=True).rename(TARGET_COLUMN)], axis=1)
        tgt_input = torch.tensor(tgt_input_df[DECODER_INPUT_FEATURES].values, dtype=torch.float32)

        tgt_output_series = future_slice[TARGET_COLUMN].fillna(SENTINEL_VALUE)
        tgt_output = torch.tensor(tgt_output_series.values, dtype=torch.float32).unsqueeze(-1)

        return src, tgt_input, tgt_output, tgt_key_padding_mask, src_key_padding_mask


def apply_zscore_scaling(df_data, columns, stats):
    df_scaled = df_data.copy()
    for col in columns:
        if col in stats and stats[col]['std'] > 1e-6:
            df_scaled[col] = (df_scaled[col] - stats[col]['mean']) / stats[col]['std']
    return df_scaled


def scale_dataframe_by_campus(df, scaler_stats):
    if df.empty: return pd.DataFrame()
    scaled_dfs = []
    columns_to_scale = WEATHER_FEATURES + [TARGET_COLUMN]
    for campus_id_num, group in df.groupby('CampusKey'):
        group_copy = group.copy()
        campus_id = str(campus_id_num)
        campus_params = scaler_stats.get(campus_id)
        if campus_params:
            for col in WEATHER_FEATURES:
                if col in group_copy.columns:
                    group_copy[col] = group_copy[col].fillna(0)
            scaled_group = apply_zscore_scaling(group_copy, columns_to_scale, campus_params)
            scaled_dfs.append(scaled_group)
    if not scaled_dfs: return pd.DataFrame()
    return pd.concat(scaled_dfs).sort_index()


# =============================================================================
#  3. New Helper Functions for Phase 2B
# =============================================================================
def masked_rmse(pred, target, pad_mask):
    """Calculates RMSE only on valid (non-padded) positions."""
    valid_positions = ~pad_mask.squeeze(-1)
    if not valid_positions.any():
        return torch.tensor(0.0)
    error = (pred - target)[valid_positions]
    return torch.sqrt(torch.mean(error ** 2))


def find_continuous_segments(mask, min_length=3):
    """
    Identifies segments of continuous valid data.
    mask: [batch, seq_len] bool tensor (True = valid position)
    Returns: A boolean mask of the same shape, highlighting valid positions
             that are part of a continuous block of at least min_length.
    """
    continuity_mask = torch.zeros_like(mask)
    for b in range(mask.size(0)):
        # Using convolution to find start of sequences of 'min_length' Trues
        kernel = torch.ones(min_length, device=mask.device)
        # Pad sequence to handle edges correctly
        padded_seq = F.pad(mask[b].float(), (min_length - 1, 0))
        # Convolution finds where a full kernel fits
        conv_res = F.conv1d(padded_seq.view(1, 1, -1), kernel.view(1, 1, -1)).squeeze()
        # Mark all positions in the identified segments
        is_in_segment = conv_res >= min_length
        for i in torch.where(is_in_segment)[0]:
            continuity_mask[b, i:i + min_length] = True
    return continuity_mask & mask  # Ensure we only mark original valid positions


# =============================================================================
#  4. Core Logic: The SolarTransformerPhase2B Class
# =============================================================================
class SolarTransformerPhase2B:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.prev_preds_cache = {}  # {batch_idx: tensor}

    def process_batch(self, batch):
        """Helper to unpack and move a batch to the correct device."""
        src, tgt_in, tgt_out, tgt_mask, src_mask = batch
        return (
            src.to(DEVICE),
            tgt_in.to(DEVICE),
            tgt_out.to(DEVICE),
            tgt_mask.to(DEVICE),
            src_mask.to(DEVICE),
        )

    def train_epoch(self, loader, epoch):
        self.model.train()
        total_loss = 0
        # Epsilon for scheduled sampling: starts high (0.9) and decays
        epsilon = max(0.05, 0.9 - epoch * 0.1)
        print(f"Scheduled sampling rate (epsilon): {epsilon:.2f}")

        progress_bar = tqdm(enumerate(loader), total=len(loader), desc=f"Phase 2B Training", leave=True)
        for batch_idx, batch in progress_bar:
            # --- THIS IS THE CORRECTED LINE ---
            # Unpacks the batch with the correct variable names.
            # tgt_mask has shape [B, 20] and src_mask has shape [B, 96].
            src, tgt_in, tgt_out, tgt_mask, src_mask = self.process_batch(batch)

            # Phase 2A-style filling (non-training) to get a reasonable starting point
            # The correct src_mask and tgt_mask are now passed.
            filled_tgt = self.incremental_fill(src, src_mask, tgt_in, tgt_mask)

            # Apply scheduled sampling
            mixed_input = self.apply_scheduled_sampling(filled_tgt, tgt_in, batch_idx, epsilon, tgt_mask)

            # Core training step
            self.optimizer.zero_grad()
            output = self.model(src, mixed_input,
                                tgt_key_padding_mask=tgt_mask,
                                src_key_padding_mask=src_mask)

            # Continuous prediction-aware loss
            loss = self.continuous_aware_loss(output, tgt_out, tgt_mask, epoch)

            if not torch.isnan(loss) and loss.item() > 0:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), MAX_GRAD_NORM)
                self.optimizer.step()

            # Cache predictions for the next epoch's scheduled sampling
            self.cache_predictions(output.detach(), batch_idx)

            total_loss += loss.item()
            progress_bar.set_postfix(avg_loss=total_loss / (batch_idx + 1))

        return total_loss / len(loader)

    def incremental_fill(self, src, src_mask, tgt_in, tgt_mask):
        """Fills NaNs in the target input autoregressively without gradient."""
        with torch.no_grad():
            working_input = tgt_in.clone()
            # Only fill positions that are marked as padding (i.e., were originally NaN)
            nan_fill_mask = tgt_mask.clone()

            for step in range(LOOKFORWARD_STEPS):
                # If there are no NaNs left to fill, we can stop early
                if not nan_fill_mask.any():
                    break

                # Get model prediction
                pred = self.model(src, working_input, tgt_key_padding_mask=tgt_mask, src_key_padding_mask=src_mask)

                # Find the first NaN position for each item in the batch
                # and fill it with the prediction.
                # This is a simplified but effective parallel approach.
                if step < LOOKFORWARD_STEPS - 1:
                    is_current_step_nan = nan_fill_mask[:, step]
                    if is_current_step_nan.any():
                        working_input[is_current_step_nan, step + 1, TARGET_FEATURE_INDEX] = pred[
                            is_current_step_nan, step, 0]

        return working_input

    def apply_scheduled_sampling(self, filled_tgt, orig_tgt, batch_idx, epsilon, mask):
        """Mixes ground truth with predictions from the previous epoch."""
        if batch_idx not in self.prev_preds_cache or torch.rand(1).item() > epsilon:
            # Use ground-truth-filled target if no history or by chance
            return filled_tgt

        mixed_tgt = orig_tgt.clone()
        prev_preds = self.prev_preds_cache[batch_idx]

        # We can only replace values where we have a previous prediction
        valid_positions = ~mask.squeeze(-1)
        mixed_tgt[valid_positions, TARGET_FEATURE_INDEX] = prev_preds[valid_positions]

        return mixed_tgt

    def continuous_aware_loss(self, pred, target, mask, epoch):
        """Adaptive loss focusing more on continuous segments over time."""
        valid_mask = (target != SENTINEL_VALUE)
        if not valid_mask.any():
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        base_loss = F.mse_loss(pred[valid_mask], target[valid_mask])

        # Weight for continuity grows from 0 to 0.5 over 10 epochs
        alpha = min(0.5, epoch * 0.05)

        # Identify long continuous prediction areas (e.g., >= 5 steps)
        continuity_mask = find_continuous_segments(valid_mask.squeeze(-1), min_length=5)

        if continuity_mask.any():
            # Apply continuity mask to pred and target
            continuity_mask_expanded = continuity_mask.unsqueeze(-1)
            cont_loss = F.mse_loss(pred[continuity_mask_expanded], target[continuity_mask_expanded])
            return base_loss * (1 - alpha) + cont_loss * alpha
        else:
            return base_loss

    def cache_predictions(self, preds, batch_idx):
        """Caches predictions for use in the next epoch's scheduled sampling."""
        # Store only the single target variable prediction
        target_preds = preds[:, :, 0].clone()
        # NOTE: quantize_tensor was not provided. Storing full precision tensor.
        self.prev_preds_cache[batch_idx] = target_preds

    def validate(self, loader):
        """Runs the comprehensive three-mode validation."""
        print("Running comprehensive validation...")
        metrics = {
            'guided_rmse': self._run_validation(loader, mode='guided'),
            'continuous_5step_rmse': self._run_validation(loader, mode='continuous', steps=5),
            'full_autoregressive_rmse': self._run_validation(loader, mode='autoregressive')
        }
        return metrics

    def _run_validation(self, loader, mode='guided', steps=None):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            progress_bar = tqdm(loader, desc=f"Validation (Mode: {mode})", leave=False)
            for batch in progress_bar:
                src, tgt_in, tgt_out, tgt_mask, src_mask = self.process_batch(batch)

                pred = torch.zeros_like(tgt_out)
                if mode == 'guided':
                    pred = self.model(src, tgt_in, tgt_key_padding_mask=tgt_mask, src_key_padding_mask=src_mask)
                elif mode == 'continuous':
                    pred = self.rolling_forecast(src, src_mask, tgt_in, tgt_mask, steps)
                else:  # 'autoregressive'
                    pred = self.full_autoregressive(src, src_mask, tgt_out.size(1))

                loss = masked_rmse(pred, tgt_out, tgt_out == SENTINEL_VALUE)
                total_loss += loss.item()

        return total_loss / len(loader)

    def rolling_forecast(self, src, src_mask, tgt_in, tgt_mask, fixed_steps):
        """Predicts first `fixed_steps` with guidance, then autoregressively."""
        seq_len = tgt_in.size(1)
        # Start with the ground-truth guided input
        pred_input = tgt_in.clone()

        # The first prediction uses the full guided input
        preds = self.model(src, pred_input, tgt_key_padding_mask=tgt_mask, src_key_padding_mask=src_mask)

        # Autoregressively fill the rest
        for step in range(fixed_steps, seq_len - 1):
            # Update the next step's input with the current step's prediction
            pred_input[:, step + 1, TARGET_FEATURE_INDEX] = preds[:, step, 0]
            # Re-predict with the updated input
            preds = self.model(src, pred_input, tgt_key_padding_mask=tgt_mask, src_key_padding_mask=src_mask)

        return preds

    def full_autoregressive(self, src, src_mask, seq_len):
        """Pure autoregressive prediction from scratch."""
        # Start with a zero-tensor for the decoder input, except for known features
        # This requires a more careful construction if future known features vary.
        # For simplicity, we start with zeros and rely on the first prediction.
        pred_input = torch.zeros(src.size(0), seq_len, self.model.decoder_embedding.in_features).to(src.device)

        # Let's assume the first input can be bootstrapped from the encoder's last state,
        # but a simpler approach is to start with a zero/neutral input.

        for i in range(seq_len - 1):
            # No target padding mask needed in pure AR mode
            no_tgt_mask = torch.zeros(src.size(0), i + 1, dtype=torch.bool, device=src.device)
            output = self.model(src, pred_input[:, :i + 1, :], tgt_key_padding_mask=no_tgt_mask,
                                src_key_padding_mask=src_mask)

            # Update the *next* step's input in our sequence
            pred_input[:, i + 1, TARGET_FEATURE_INDEX] = output[:, i, 0]

        #  prediction over the full sequence
        final_preds = self.model(src, pred_input,
                                 tgt_key_padding_mask=torch.zeros_like(pred_input[..., 0], dtype=torch.bool),
                                 src_key_padding_mask=src_mask)
        return final_preds


# =============================================================================
#  5. Main Execution Logic for Phase 2B
# =============================================================================
if __name__ == "__main__":
    print(f"--- Transformer Elite Model Fine-Tuning (Phase 2B - Scheduled Sampling & Stability) ---")
    print(f"Using device: {DEVICE}")

    # 1. Load and prepare data
    print("Loading and preparing data...")
    df_train_all = pd.read_csv(ALL_TRAIN_DATA_PATH)
    df_val_all = pd.read_csv(ALL_VAL_DATA_PATH)
    with open(STATS_FILE_PATH, 'r') as f:
        scaling_stats = json.load(f)

    df_train_scaled = scale_dataframe_by_campus(df_train_all, scaling_stats)
    df_val_scaled = scale_dataframe_by_campus(df_val_all, scaling_stats)

    train_dataset = SolarPowerDataset_Transformer(df_train_scaled, LOOKBACK_STEPS, LOOKFORWARD_STEPS)
    val_dataset = SolarPowerDataset_Transformer(df_val_scaled, LOOKBACK_STEPS, LOOKFORWARD_STEPS)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    print(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    # 2. Initialize model
    model = TimeSeriesTransformer(
        encoder_feature_dim=len(ENCODER_FEATURES),
        decoder_feature_dim=len(DECODER_INPUT_FEATURES),
        d_model=D_MODEL, nhead=NHEAD, num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS, dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT_RATE
    ).to(DEVICE)

    # 3. !! Key Step: Load the Phase 2A trained model weights !!
    if os.path.exists(PHASE2A_MODEL_PATH):
        print(f"Successfully loaded Phase 2A pre-trained model: {PHASE2A_MODEL_PATH}")
        model.load_state_dict(torch.load(PHASE2A_MODEL_PATH, map_location=DEVICE))
    else:
        raise FileNotFoundError(f"CRITICAL: Phase 2A model not found at {PHASE2A_MODEL_PATH}. Cannot start Phase 2B.")

    # 4. Initialize optimizer and the new Trainer class
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE_PHASE2B)
    trainer = SolarTransformerPhase2B(model, optimizer)

    # 5. Start training
    print("\nStarting fine-tuning (Phase 2B)...")
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(EPOCHS_PHASE2B):
        print(f"\nEpoch {epoch + 1}/{EPOCHS_PHASE2B}")

        train_loss = trainer.train_epoch(train_loader, epoch)
        val_metrics = trainer.validate(val_loader)

        guided_loss = val_metrics['guided_rmse']
        continuous_loss = val_metrics['continuous_5step_rmse']
        ar_loss = val_metrics['full_autoregressive_rmse']

        print(f"  Epoch {epoch + 1} Summary:")
        print(f"    Train Loss (MSE): {train_loss:.6f}")
        print(f"    Validation Metrics (RMSE):")
        print(f"      - Guided: {guided_loss:.6f}")
        print(f"      - Continuous (5-step): {continuous_loss:.6f}")
        print(f"      - Full Autoregressive: {ar_loss:.6f}")

        # Early stopping based on the most stable metric: guided RMSE
        if guided_loss < best_val_loss:
            best_val_loss = guided_loss
            torch.save(model.state_dict(), PHASE2B_MODEL_SAVE_PATH)
            patience_counter = 0
            print(f"      -> New best validation loss. Model saved to {PHASE2B_MODEL_SAVE_PATH}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  > Early stopping triggered at Epoch {epoch + 1}.")
                break

    trainer.prev_preds_cache.clear()  # Clear cache after training
    print(f"\nElite Model Phase 2B fine-tuning complete.")
    print(f"Best guided validation RMSE: {best_val_loss:.6f}")
    print(f" model saved at: {PHASE2B_MODEL_SAVE_PATH}")
