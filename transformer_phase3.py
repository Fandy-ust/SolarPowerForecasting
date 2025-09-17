import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import json
import os
import time
import math
from tqdm import tqdm

# =============================================================================
#  1. Configuration and Constants
# =============================================================================
# --- Phase 3 Specific Parameters ---
print("--- INITIALIZING PHASE 3: FINAL AUTOREGRESSIVE POLISH ---")
PHASE3_EPOCHS = 5  # Short duration for fine-tuning
PHASE3_LEARNING_RATE = 1e-5  # Very low learning rate for delicate adjustments (0.1e-6)
GRADIENT_CLIP_VALUE = 0.5  # Helps prevent instability during fine-tuning

# --- File Paths ---
TRAIN_DATA_PATH = r"C:\Users\1\Documents\Solarpower\Data_processed\Train_val_test_split_new\train_selected_sites.csv"
VAL_DATA_PATH = r"C:\Users\1\Documents\Solarpower\Data_processed\Train_val_test_split_new\validation_enhanced.csv"
STATS_FILE_PATH = r"C:\Users\1\Documents\Solarpower\Data_processed\Train_val_test_split_new\scaler\campus_scale.txt"

PREV_BEST_MODEL_PATH = r"C:\Python\Trained_models\Elite_Model_Transformer_Phase3\Skip_phase_2_ver2"
FINAL_MODEL_SAVE_PATH = r"C:\Python\Trained_models\Elite_Model_Transformer_Phase3\Skip_phase_2_ver3"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOOKBACK_STEPS = 4 * 24
LOOKFORWARD_STEPS = 4 * 5
BATCH_SIZE = 32

D_MODEL = 128
NHEAD = 8
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
DIM_FEEDFORWARD = 512
DROPOUT_RATE = 0.1

TARGET_COLUMN = 'SolarGeneration'
WEATHER_FEATURES = ['AirTemperature', 'Ghi_hourly', 'CloudOpacity_hourly', "minutes_since_last_update",
                    "RelativeHumidity"]
TIME_FEATURES = ["zenith_sin", "azimuth_sin", "day_sin", "year_sin"]
ENCODER_FEATURES = WEATHER_FEATURES + TIME_FEATURES + [TARGET_COLUMN]
FUTURE_KNOWN_FEATURES = TIME_FEATURES
DECODER_INPUT_FEATURES = FUTURE_KNOWN_FEATURES + [TARGET_COLUMN]
TARGET_FEATURE_INDEX = DECODER_INPUT_FEATURES.index(TARGET_COLUMN)
SENTINEL_VALUE = -100


# =============================================================================
#  2. Model and Data Class Definitions (Unchanged)
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
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward,
                                          dropout=dropout, batch_first=True)
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
        memory = self.transformer.encoder(src=src_emb, src_key_padding_mask=src_key_padding_mask)
        decoder_output = self.transformer.decoder(tgt=tgt_emb, memory=memory, tgt_mask=tgt_mask,
                                                  tgt_key_padding_mask=tgt_key_padding_mask,
                                                  memory_key_padding_mask=src_key_padding_mask)
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
                self.campus_indices.append({'start': start_index, 'count': num_samples, 'id': campus_id})
        self.total_samples = sum(c['count'] for c in self.campus_indices)

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        campus_idx = 0
        while idx >= self.campus_indices[campus_idx]['count']:
            idx -= self.campus_indices[campus_idx]['count']
            campus_idx += 1
        campus_info = self.campus_indices[campus_idx]
        start_pos = campus_info['start'] + idx
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
        tgt_input_df = pd.concat(
            [future_known_df.reset_index(drop=True), shifted.reset_index(drop=True).rename(TARGET_COLUMN)], axis=1)
        tgt_input = torch.tensor(tgt_input_df[DECODER_INPUT_FEATURES].values, dtype=torch.float32)
        tgt_output_series = future_slice[TARGET_COLUMN].fillna(SENTINEL_VALUE)
        tgt_output = torch.tensor(tgt_output_series.values, dtype=torch.float32).unsqueeze(-1)
        return src, tgt_input, tgt_output, tgt_key_padding_mask, src_key_padding_mask


# =============================================================================
#  3. Scaling Functions (Unchanged)
# =============================================================================
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
                if col in group_copy.columns: group_copy[col] = group_copy[col].fillna(0)
            scaled_group = apply_zscore_scaling(group_copy, columns_to_scale, campus_params)
            scaled_dfs.append(scaled_group)
    return pd.concat(scaled_dfs).sort_index()


# =============================================================================
#  4. Phase 3 Training and Validation Loops (CORRECTED)
# =============================================================================
def train_epoch_fully_autoregressive(model, data_loader, optimizer, criterion, device):
    """Performs one training epoch in a fully autoregressive manner."""
    model.train()
    total_loss = 0
    progress_bar = tqdm(data_loader, desc="[Train] Fully Autoregressive")

    for src, tgt_in, tgt_out, tgt_key_padding_mask, src_key_padding_mask in progress_bar:
        src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
        src_key_padding_mask = src_key_padding_mask.to(device)

        optimizer.zero_grad()

        # --- Start of Autoregressive Generation with fix ---
        working_input = tgt_in.clone()
        final_preds = None  # To store the final predictions

        for step in range(LOOKFORWARD_STEPS):
            pred_all_steps = model(src, working_input, tgt_key_padding_mask=None,
                                   src_key_padding_mask=src_key_padding_mask)
            pred_this_step = pred_all_steps[:, step, 0]

            if step < LOOKFORWARD_STEPS - 1:
                # --- FIX APPLIED HERE ---
                # Detach the prediction to prevent the in-place modification error.
                # This severs the gradient history for this specific assignment, which is intended.
                working_input[:, step + 1, TARGET_FEATURE_INDEX] = pred_this_step.detach()

        # The predictions from the final step of the loop are our full forecast
        final_preds = pred_all_steps
        # --- End of Autoregressive Generation ---

        # Calculate loss on the entire generated sequence vs the ground truth
        valid_mask = tgt_out != SENTINEL_VALUE
        loss = criterion(final_preds[valid_mask], tgt_out[valid_mask])

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_VALUE)
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    return total_loss / len(data_loader)


@torch.no_grad()
def validate_fully_autoregressive(model, data_loader, criterion, device):
    """Performs validation in a fully autoregressive manner."""
    model.eval()
    total_loss = 0
    progress_bar = tqdm(data_loader, desc="[Validate] Fully Autoregressive")

    for src, tgt_in, tgt_out, tgt_key_padding_mask, src_key_padding_mask in progress_bar:
        src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
        src_key_padding_mask = src_key_padding_mask.to(device)

        working_input = tgt_in.clone()
        final_preds = None

        for step in range(LOOKFORWARD_STEPS):
            pred_all_steps = model(src, working_input, tgt_key_padding_mask=None,
                                   src_key_padding_mask=src_key_padding_mask)
            pred_this_step = pred_all_steps[:, step, 0]
            if step < LOOKFORWARD_STEPS - 1:
                # No .detach() needed here because of @torch.no_grad()
                working_input[:, step + 1, TARGET_FEATURE_INDEX] = pred_this_step

        final_preds = pred_all_steps

        valid_mask = tgt_out != SENTINEL_VALUE
        loss = criterion(final_preds[valid_mask], tgt_out[valid_mask])
        total_loss += loss.item()

    return total_loss / len(data_loader)


# =============================================================================
#  5. Main Execution Block (Unchanged)
# =============================================================================
if __name__ == "__main__":
    print(f"Using device: {DEVICE}")

    with open(STATS_FILE_PATH, 'r') as f:
        scaling_stats = json.load(f)

    df_train_all = pd.read_csv(TRAIN_DATA_PATH)
    df_train_scaled = scale_dataframe_by_campus(df_train_all, scaling_stats)
    train_dataset = SolarPowerDataset_Transformer(df_train_scaled, LOOKBACK_STEPS, LOOKFORWARD_STEPS)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    df_val_all = pd.read_csv(VAL_DATA_PATH)
    df_val_scaled = scale_dataframe_by_campus(df_val_all, scaling_stats)
    val_dataset = SolarPowerDataset_Transformer(df_val_scaled, LOOKBACK_STEPS, LOOKFORWARD_STEPS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = TimeSeriesTransformer(
        encoder_feature_dim=len(ENCODER_FEATURES), decoder_feature_dim=len(DECODER_INPUT_FEATURES),
        d_model=D_MODEL, nhead=NHEAD, num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS, dim_feedforward=DIM_FEEDFORWARD, dropout=DROPOUT_RATE
    ).to(DEVICE)

    if not os.path.exists(PREV_BEST_MODEL_PATH):
        raise FileNotFoundError(
            f"FATAL: Previous best model not found at {PREV_BEST_MODEL_PATH}. Cannot start Phase 3.")

    print(f"Loading weights from previous best model: {PREV_BEST_MODEL_PATH}")
    model.load_state_dict(torch.load(PREV_BEST_MODEL_PATH, map_location=DEVICE))

    optimizer = torch.optim.AdamW(model.parameters(), lr=PHASE3_LEARNING_RATE)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    print("\n--- Starting Phase 3 Fine-Tuning ---")

    for epoch in range(1, PHASE3_EPOCHS + 1):
        epoch_start_time = time.time()

        train_loss = train_epoch_fully_autoregressive(model, train_loader, optimizer, criterion, DEVICE)
        val_loss = validate_fully_autoregressive(model, val_loader, criterion, DEVICE)

        epoch_duration = time.time() - epoch_start_time

        print(
            f"Epoch {epoch}/{PHASE3_EPOCHS} | Time: {epoch_duration:.2f}s | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), FINAL_MODEL_SAVE_PATH)
            print(f"  -> Validation loss improved. Model saved to {FINAL_MODEL_SAVE_PATH}")

    print("\n--- Phase 3 Training Complete ---")
    print(f", polished model saved at: {FINAL_MODEL_SAVE_PATH}")

