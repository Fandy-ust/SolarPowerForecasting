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
PHASE2A_MODEL_PATH = r"C:\Python\Trained_models\Elite_Model_Transformer_Phase2A_Complete\transformer_elite_model_phase2a.pth"
PHASE2B_SAVE_DIR = r"C:\Python\Trained_models\Elite_Model_Transformer_Phase2B_Complete"
os.makedirs(PHASE2B_SAVE_DIR, exist_ok=True)
PHASE2B_MODEL_SAVE_PATH = os.path.join(PHASE2B_SAVE_DIR, 'transformer_elite_model_phase2b_V2.pth')

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
LEARNING_RATE_PHASE2B = 0.1e-6
EPOCHS_PHASE2B = 10
MAX_GRAD_NORM = 1.0
PATIENCE = 15
SENTINEL_VALUE = -10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Transformer Model Hyperparameters (must match previous phases) ---
D_MODEL = 128
NHEAD = 8
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
DIM_FEEDFORWARD = 512
DROPOUT_RATE = 0.1


# =============================================================================
#  2. Reusable Modules (Model, Dataset, Scaling - Unchanged)
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
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                tgt_key_padding_mask: torch.Tensor,
                src_key_padding_mask: torch.Tensor) -> torch.Tensor:
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

        src_df = sequence_slice.iloc[:self.lookback].copy()
        src_key_padding_mask = torch.tensor(src_df[TARGET_COLUMN].isna().values, dtype=torch.bool)
        src_df.fillna(0, inplace=True)
        src = torch.tensor(src_df[ENCODER_FEATURES].values, dtype=torch.float32)

        future_slice = sequence_slice.iloc[self.lookback:]
        tgt_key_padding_mask = torch.tensor(future_slice[TARGET_COLUMN].isna().values, dtype=torch.bool)

        last_obs = sequence_slice.iloc[self.lookback - 1][TARGET_COLUMN]
        shifted = future_slice[TARGET_COLUMN].shift(1)
        shifted.iloc[0] = last_obs
        shifted = shifted.fillna(0)
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
#  3. Helper Functions for Phase 2B (Unchanged)
# =============================================================================
def masked_rmse(pred, target, pad_mask):
    valid_positions = ~pad_mask.squeeze(-1)
    if not valid_positions.any():
        return torch.tensor(0.0)
    error = (pred - target)[valid_positions]
    return torch.sqrt(torch.mean(error ** 2))


def find_continuous_segments(mask, min_length=3):
    continuity_mask = torch.zeros_like(mask)
    for b in range(mask.size(0)):
        kernel = torch.ones(min_length, device=mask.device)
        padded_seq = F.pad(mask[b].float(), (min_length - 1, 0))
        conv_res = F.conv1d(padded_seq.view(1, 1, -1), kernel.view(1, 1, -1)).squeeze()
        is_in_segment = conv_res >= min_length
        for i in torch.where(is_in_segment)[0]:
            continuity_mask[b, i:i + min_length] = True
    return continuity_mask & mask


# =============================================================================
#  4. Core Logic: The SolarTransformerPhase2B Class (Refactored)
# =============================================================================
# =============================================================================
#  4. Core Logic: The SolarTransformerPhase2B Class (CORRECTED)
# =============================================================================
class SolarTransformerPhase2B:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    def process_batch(self, batch):
        src, tgt_in, tgt_out, tgt_mask, src_mask = batch
        return (
            src.to(DEVICE), tgt_in.to(DEVICE), tgt_out.to(DEVICE),
            tgt_mask.to(DEVICE), src_mask.to(DEVICE)
        )

    def train_epoch(self, loader, epoch, prediction_cache):
        self.model.train()
        total_loss = 0
        epsilon = max(0.05, 0.9 - epoch * 0.1)
        print(f"Scheduled sampling rate (epsilon): {epsilon:.2f}")

        progress_bar = tqdm(enumerate(loader), total=len(loader), desc=f"Phase 2B Training", leave=True)
        for batch_idx, batch in progress_bar:
            src, tgt_in, tgt_out, tgt_mask, src_mask = self.process_batch(batch)

            if batch_idx not in prediction_cache:
                print(f"Warning: Missing prediction for batch {batch_idx} in cache. Skipping.")
                continue

            cached_preds = prediction_cache[batch_idx].to(DEVICE)

            # --- The logic below requires cached_preds to have shape [B, 20] ---
            # --- The new generate_predictions_for_epoch ensures this. ---

            mixed_input = tgt_in.clone()

            # 1. Fill NaNs using the cached predictions
            nan_mask = tgt_mask.clone()
            # Use boolean indexing, which requires the mask and tensor to have the same shape.
            mixed_input[:, :, TARGET_FEATURE_INDEX][nan_mask] = cached_preds[nan_mask]

            # 2. Apply scheduled sampling to non-NaN values
            if torch.rand(1).item() < epsilon:
                ground_truth_mask = ~nan_mask
                mixed_input[:, :, TARGET_FEATURE_INDEX][ground_truth_mask] = cached_preds[ground_truth_mask]

            # Core training step
            self.optimizer.zero_grad()
            output = self.model(src, mixed_input, tgt_key_padding_mask=tgt_mask, src_key_padding_mask=src_mask)

            loss = self.continuous_aware_loss(output, tgt_out, tgt_mask, epoch)

            if not torch.isnan(loss) and loss.item() > 0:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), MAX_GRAD_NORM)
                self.optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(avg_loss=total_loss / (batch_idx + 1))

        return total_loss / len(loader)

    def generate_predictions_for_epoch(self, loader):
        """
        Generates a full set of predictions for the next epoch's training.
        This now uses the corrected, stable version of incremental_fill.
        """
        self.model.eval()
        epoch_predictions = {}
        with torch.no_grad():
            progress_bar = tqdm(enumerate(loader), total=len(loader), desc="Generating Predictions for Next Epoch",
                                leave=False)
            for batch_idx, batch in progress_bar:
                src, tgt_in, tgt_out, tgt_mask, src_mask = self.process_batch(batch)

                # This function now reliably returns a tensor of shape [B, 20, 1]
                filled_output = self.incremental_fill(src, src_mask, tgt_in, tgt_mask)

                # --- ADDED SAFETY CHECK ---
                # This will raise a clear error if the shape is wrong for any reason.
                expected_len = LOOKFORWARD_STEPS
                if filled_output.shape[1] != expected_len:
                    raise ValueError(
                        f"CRITICAL: `incremental_fill` produced tensor with incorrect sequence length "
                        f"in batch {batch_idx}. Expected {expected_len}, but got {filled_output.shape[1]}."
                    )

                # Store the predicted target variable on CPU, shape is [B, 20]
                epoch_predictions[batch_idx] = filled_output[:, :, 0].cpu()

        return epoch_predictions

    def incremental_fill(self, src, src_mask, tgt_in, tgt_mask):
        """
        CORRECTED AND ROBUST: Fills NaNs autoregressively and returns the full predicted sequence.
        This version explicitly builds the output tensor step-by-step to guarantee correct shape.
        """
        self.model.eval()
        with torch.no_grad():
            # The input used by the model in each loop, starts as the ground-truth
            working_input = tgt_in.clone()

            # The final output tensor we will return, initialized to zeros.
            # Its shape is guaranteed to be [B, 20, 1] from the start.
            final_predictions = torch.zeros_like(tgt_in[:, :, 0]).unsqueeze(-1)

            # Iterate through each of the 20 timesteps to generate the forecast
            for step in range(LOOKFORWARD_STEPS):
                # Get a prediction for the current state of the input sequence.
                # The model predicts all 20 steps, but we only trust the immediate next one.
                pred_all_steps = self.model(
                    src,
                    working_input,
                    tgt_key_padding_mask=tgt_mask,
                    src_key_padding_mask=src_mask
                )

                # Get the single prediction for the current step from the model's output
                pred_this_step = pred_all_steps[:, step, 0]

                # Store this reliable prediction in our final output tensor
                final_predictions[:, step, 0] = pred_this_step

                # For the next loop, update the *next* step in the input tensor
                # This is the autoregressive part.
                if step < LOOKFORWARD_STEPS - 1:
                    # Update the input for the next iteration using the prediction we just made.
                    # This only affects the input for the *next* loop, not the `final_predictions` tensor.
                    working_input[:, step + 1, TARGET_FEATURE_INDEX] = pred_this_step

        # Return the safely constructed predictions, which is guaranteed to have the correct shape.
        return final_predictions

    def continuous_aware_loss(self, pred, target, mask, epoch):
        """Adaptive loss focusing more on continuous segments over time."""
        valid_mask = (target != SENTINEL_VALUE)
        if not valid_mask.any():
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        base_loss = F.mse_loss(pred[valid_mask], target[valid_mask])
        alpha = min(0.5, epoch * 0.05)
        continuity_mask = find_continuous_segments(valid_mask.squeeze(-1), min_length=5)

        if continuity_mask.any():
            continuity_mask_expanded = continuity_mask.unsqueeze(-1)
            cont_loss = F.mse_loss(pred[continuity_mask_expanded], target[continuity_mask_expanded])
            return base_loss * (1 - alpha) + cont_loss * alpha
        else:
            return base_loss

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
                    # Use the corrected incremental_fill for the most accurate autoregressive test
                    pred = self.incremental_fill(src, src_mask, tgt_in, tgt_mask)

                loss = masked_rmse(pred, tgt_out, tgt_out == SENTINEL_VALUE)
                total_loss += loss.item()
        return total_loss / len(loader)

    def rolling_forecast(self, src, src_mask, tgt_in, tgt_mask, fixed_steps):
        """Predicts first `fixed_steps` with guidance, then autoregressively."""
        seq_len = tgt_in.size(1)
        pred_input = tgt_in.clone()
        # Use the robust incremental fill logic for the autoregressive part
        full_preds = self.incremental_fill(src, src_mask, tgt_in, tgt_mask)
        # Combine guided predictions and autoregressive predictions
        final_preds = self.model(src, pred_input, tgt_key_padding_mask=tgt_mask, src_key_padding_mask=src_mask)
        final_preds[:, fixed_steps:] = full_preds[:, fixed_steps:]
        return final_preds

    def full_autoregressive(self, src, src_mask, seq_len):
        # This function is now superseded by calling _run_validation with mode='autoregressive'
        # which uses the safer incremental_fill.
        tgt_in_dummy = torch.zeros(src.size(0), seq_len, self.model.decoder_embedding.in_features, device=src.device)
        tgt_mask_dummy = torch.ones(src.size(0), seq_len, dtype=torch.bool, device=src.device)
        return self.incremental_fill(src, src_mask, tgt_in_dummy, tgt_mask_dummy)


# =============================================================================
#  5. Main Execution Logic for Phase 2B (Refactored)
# =============================================================================
if __name__ == "__main__":
    print(f"--- Transformer Elite Model Fine-Tuning (Phase 2B - Staged Generation) ---")
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
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    print(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    # 2. Initialize model and load Phase 2A weights
    model = TimeSeriesTransformer(
        encoder_feature_dim=len(ENCODER_FEATURES), decoder_feature_dim=len(DECODER_INPUT_FEATURES),
        d_model=D_MODEL, nhead=NHEAD, num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS, dim_feedforward=DIM_FEEDFORWARD, dropout=DROPOUT_RATE
    ).to(DEVICE)
    if os.path.exists(PHASE2A_MODEL_PATH):
        print(f"Successfully loaded Phase 2A pre-trained model: {PHASE2A_MODEL_PATH}")
        model.load_state_dict(torch.load(PHASE2A_MODEL_PATH, map_location=DEVICE))
    else:
        raise FileNotFoundError(f"CRITICAL: Phase 2A model not found at {PHASE2A_MODEL_PATH}.")

    # 3. Initialize optimizer and trainer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE_PHASE2B)
    trainer = SolarTransformerPhase2B(model, optimizer)

    # 4. Prime the pump by generating initial predictions
    print("\nGenerating initial predictions for Epoch 1 using the loaded model...")
    prediction_cache = trainer.generate_predictions_for_epoch(train_loader)

    # 5. Start training
    print("\nStarting fine-tuning (Phase 2B)...")

    # --- CHANGE: We now track the best AUTOREGRESSIVE loss ---
    best_ar_loss = float('inf')
    patience_counter = 0

    for epoch in range(EPOCHS_PHASE2B):
        print(f"\nEpoch {epoch + 1}/{EPOCHS_PHASE2B}")

        train_loss = trainer.train_epoch(train_loader, epoch, prediction_cache)
        val_metrics = trainer.validate(val_loader)

        guided_loss = val_metrics['guided_rmse']
        continuous_loss = val_metrics['continuous_5step_rmse']
        ar_loss = val_metrics['full_autoregressive_rmse']  # This is the metric we care about

        print(f"  Epoch {epoch + 1} Summary:")
        print(f"    Train Loss (MSE): {train_loss:.6f}")
        print(f"    Validation Metrics (RMSE):")
        print(f"      - Guided: {guided_loss:.6f}")
        print(f"      - Continuous (5-step): {continuous_loss:.6f}")
        print(f"      - Full Autoregressive: {ar_loss:.6f}")

        # --- CHANGE: The saving condition now uses ar_loss ---
        if ar_loss < best_ar_loss:
            best_ar_loss = ar_loss
            torch.save(model.state_dict(), PHASE2B_MODEL_SAVE_PATH)
            patience_counter = 0
            print(f"      -> New best AUTOREGRESSIVE validation loss. Model saved to {PHASE2B_MODEL_SAVE_PATH}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  > Early stopping triggered at Epoch {epoch + 1}.")
                break

        # Generate predictions for the *next* epoch
        if epoch < EPOCHS_PHASE2B - 1:
            prediction_cache = trainer.generate_predictions_for_epoch(train_loader)

    print(f"\nElite Model Phase 2B fine-tuning complete.")
    print(f"Best full autoregressive validation RMSE: {best_ar_loss:.6f}")
    print(f"Model saved at: {PHASE2B_MODEL_SAVE_PATH}")
