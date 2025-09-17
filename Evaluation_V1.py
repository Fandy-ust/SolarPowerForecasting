import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from tqdm import tqdm
import random
import math

# =============================================================================
#  1. Configuration and Constants (Adapted from your training script)
# =============================================================================
# --- File Paths (Please verify these paths are correct) ---
VAL_DATA_PATH = r"C:\Users\1\Documents\Solarpower\Data_processed\Train_val_test_split_new\validation_enhanced.csv"
TEST_DATA_PATH = r"C:\Users\1\Documents\Solarpower\Data_processed\Train_val_test_split_new\test_enhanced.csv"  # <-- IMPORTANT: VERIFY THIS PATH
STATS_FILE_PATH = r"C:\Users\1\Documents\Solarpower\Data_processed\Train_val_test_split_new\scaler\campus_scale.txt"
MODEL_SAVE_PATH = r"C:\Python\Trained_models\Elite_Model_Transformer_Phase2B_Complete\transformer_final_autoregressive_SkipPhase2_V6_Jason.pth"

# --- Model & Data Parameters (Must match training script) ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOOKBACK_STEPS = 4 * 24
LOOKFORWARD_STEPS = 4 * 5
BATCH_SIZE = 32  # Can be larger for inference if GPU memory allows

# Model Architecture (Must match the saved model)
D_MODEL = 128
NHEAD = 8
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
DIM_FEEDFORWARD = 512
DROPOUT_RATE = 0.1

# Features (Must match training script)
TARGET_COLUMN = 'SolarGeneration'
WEATHER_FEATURES = ['AirTemperature', 'Ghi_hourly', 'CloudOpacity_hourly', "minutes_since_last_update",
                    "RelativeHumidity"]
TIME_FEATURES = ["zenith_sin", "azimuth_sin", "day_sin", "year_sin"]
ENCODER_FEATURES = WEATHER_FEATURES + TIME_FEATURES + [TARGET_COLUMN]
FUTURE_KNOWN_FEATURES = TIME_FEATURES
DECODER_INPUT_FEATURES = FUTURE_KNOWN_FEATURES + [TARGET_COLUMN]
TARGET_FEATURE_INDEX = DECODER_INPUT_FEATURES.index(TARGET_COLUMN)
SENTINEL_VALUE = -10.0

# Visualization
NUM_SAMPLES_TO_PLOT = 50 # Number of random examples to visualize


# =============================================================================
#  2. Model and Data Class Definitions (Copied *exactly* from your script)
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
        campus_id = campus_info['id']
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
        # Fix: Rename the Series to match TARGET_COLUMN
        shifted_renamed = pd.DataFrame({TARGET_COLUMN: shifted.reset_index(drop=True)})
        tgt_input_df = pd.concat([future_known_df.reset_index(drop=True), shifted_renamed], axis=1)
        tgt_input = torch.tensor(tgt_input_df[DECODER_INPUT_FEATURES].values, dtype=torch.float32)

        tgt_output_series = future_slice[TARGET_COLUMN].fillna(SENTINEL_VALUE)
        tgt_output = torch.tensor(tgt_output_series.values, dtype=torch.float32).unsqueeze(-1)

        return src, tgt_input, tgt_output, tgt_key_padding_mask, src_key_padding_mask, str(campus_id)


# =============================================================================
#  3. Scaling & Evaluation Functions
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
                if col in group_copy.columns:
                    group_copy[col] = group_copy[col].fillna(0)
            scaled_group = apply_zscore_scaling(group_copy, columns_to_scale, campus_params)
            scaled_dfs.append(scaled_group)
    if not scaled_dfs: return pd.DataFrame()
    return pd.concat(scaled_dfs).sort_index()


@torch.no_grad()
def autoregressive_predict(model, data_loader, device):
    """Generates full autoregressive predictions for an entire dataset."""
    model.eval()
    all_preds = []
    all_truths = []
    all_campuses = []

    progress_bar = tqdm(data_loader, desc="Generating Predictions")
    for src, tgt_in, tgt_out, tgt_key_padding_mask, src_key_padding_mask, campuses in progress_bar:
        src = src.to(device)
        tgt_in = tgt_in.to(device)
        src_key_padding_mask = src_key_padding_mask.to(device)

        working_input = tgt_in.clone()
        # Initialize preds tensor for accumulation
        preds = torch.zeros((src.size(0), LOOKFORWARD_STEPS), device=device)

        # Autoregressive generation loop
        for step in range(LOOKFORWARD_STEPS):
            pred_all_steps = model(src, working_input, tgt_key_padding_mask=None,
                                   src_key_padding_mask=src_key_padding_mask)
            pred_this_step = pred_all_steps[:, step, 0]
            preds[:, step] = pred_this_step  # Accumulate this step's prediction

            if step < LOOKFORWARD_STEPS - 1:
                working_input[:, step + 1, TARGET_FEATURE_INDEX] = pred_this_step

        # No need for final model call; use accumulated preds
        all_preds.append(preds.cpu().numpy())
        all_truths.append(tgt_out.squeeze(-1).cpu().numpy())
        all_campuses.extend(campuses)

    return np.vstack(all_preds), np.vstack(all_truths), all_campuses


def evaluate_and_visualize(model, data_loader, scaling_stats, dataset_name, device):
    """Runs the full evaluation and plotting pipeline for a given dataset."""
    print(f"\n--- Evaluating on {dataset_name} ---")

    # 1. Generate scaled predictions
    scaled_preds, scaled_truths, campuses = autoregressive_predict(model, data_loader, device)

    # 2. Inverse scale predictions and ground truths for interpretability
    unscaled_preds = np.zeros_like(scaled_preds)
    unscaled_truths = np.zeros_like(scaled_truths)

    for i in range(len(campuses)):
        campus = campuses[i]
        params = scaling_stats.get(campus, {}).get(TARGET_COLUMN, {'mean': 0, 'std': 1})
        unscaled_preds[i, :] = scaled_preds[i, :] * params['std'] + params['mean']
        unscaled_truths[i, :] = scaled_truths[i, :] * params['std'] + params['mean']

    # Clip predictions at 0, as negative solar generation is physically impossible
    unscaled_preds = np.maximum(0, unscaled_preds)

    # 3. Calculate metrics on valid (non-sentinel) data points
    valid_mask = scaled_truths != SENTINEL_VALUE
    if not np.any(valid_mask):
        print(f"Warning: No valid ground truth data found in {dataset_name} to evaluate against.")
        return

    # Flatten arrays for scikit-learn metrics, considering only valid points
    flat_truths = unscaled_truths[valid_mask]
    flat_preds = unscaled_preds[valid_mask]

    r2 = r2_score(flat_truths, flat_preds)
    rmse = np.sqrt(mean_squared_error(flat_truths, flat_preds))

    print(f"[{dataset_name}] Overall Autoregressive RMSE: {rmse:.4f}")
    print(f"[{dataset_name}] Overall Autoregressive R-squared (R²): {r2:.4f}")

    # Create a folder to save plots
    plot_dir = r"C:\Users\1\Documents\Solarpower\Plot"
    os.makedirs(plot_dir, exist_ok=True)

    # 4. Visualize a few random samples and save to folder
    print(f"Plotting and saving {NUM_SAMPLES_TO_PLOT} random samples from {dataset_name} to {plot_dir}...")
    sample_indices = random.sample(range(len(unscaled_preds)), k=min(NUM_SAMPLES_TO_PLOT, len(unscaled_preds)))

    time_steps = np.arange(1, LOOKFORWARD_STEPS + 1) * 15  # Time in minutes from forecast start

    for i, idx in enumerate(sample_indices):
        plt.figure(figsize=(16, 7))

        # Only plot ground truth points that are not sentinels
        truth_valid_mask = scaled_truths[idx] != SENTINEL_VALUE
        plt.plot(time_steps[truth_valid_mask], unscaled_truths[idx][truth_valid_mask], 'o-', label='Ground Truth',
                 color='royalblue', linewidth=2)
        plt.plot(time_steps, unscaled_preds[idx], 'x-', label='Autoregressive Forecast', color='orangered', alpha=0.9)

        plt.title(f'{dataset_name} Forecast - Sample {i + 1} (Campus: {campuses[idx]})', fontsize=16)
        plt.xlabel(f'Time into Future (minutes)', fontsize=12)
        plt.ylabel(f'{TARGET_COLUMN} (Unscaled)', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.ylim(bottom=-0.05 * np.max(unscaled_truths[idx][truth_valid_mask]))  # Give a little space below zero
        plt.xticks(np.arange(0, LOOKFORWARD_STEPS * 15 + 1, 60))  # Tick every hour
        plt.tight_layout()

        # Save the plot instead of showing
        plot_path = os.path.join(plot_dir, f'sample_{i + 1}_{campuses[idx]}.png')
        plt.savefig(plot_path)
        plt.close()  # Close the figure to free memory

    # 5. Additional visualization: RMSE and R² vs. forward steps
    print(f"Calculating and plotting metrics vs. forward steps for {dataset_name}...")
    steps = np.arange(1, LOOKFORWARD_STEPS + 1)
    rmse_per_step = []
    r2_per_step = []

    for step in range(LOOKFORWARD_STEPS):
        step_mask = valid_mask[:, step]  # Mask for this step across all samples
        if np.any(step_mask):
            step_truths = unscaled_truths[:, step][step_mask]
            step_preds = unscaled_preds[:, step][step_mask]
            step_rmse = np.sqrt(mean_squared_error(step_truths, step_preds))
            step_r2 = r2_score(step_truths, step_preds)
        else:
            step_rmse = np.nan
            step_r2 = np.nan
        rmse_per_step.append(step_rmse)
        r2_per_step.append(step_r2)

    # Plot RMSE vs. steps
    plt.figure(figsize=(10, 5))
    plt.plot(steps, rmse_per_step, 'o-', label='RMSE', color='darkred')
    plt.title(f'RMSE vs. Forward Steps - {dataset_name}', fontsize=16)
    plt.xlabel('Forward Step', fontsize=12)
    plt.ylabel('RMSE (kW)', fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    rmse_plot_path = os.path.join(plot_dir, 'rmse_vs_steps.png')
    plt.savefig(rmse_plot_path)
    plt.close()

    # Plot R² vs. steps
    plt.figure(figsize=(10, 5))
    plt.plot(steps, r2_per_step, 'o-', label='R²', color='darkgreen')
    plt.title(f'R² vs. Forward Steps - {dataset_name}', fontsize=16)
    plt.xlabel('Forward Step', fontsize=12)
    plt.ylabel('R²', fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    r2_plot_path = os.path.join(plot_dir, 'r2_vs_steps.png')
    plt.savefig(r2_plot_path)
    plt.close()

    print(f"Metrics plots saved to {plot_dir}")


# =============================================================================
#  4. Main Execution Block
# =============================================================================
if __name__ == "__main__":
    if not os.path.exists(MODEL_SAVE_PATH):
        raise FileNotFoundError(f"FATAL: Model file not found at {MODEL_SAVE_PATH}")
    if not os.path.exists(STATS_FILE_PATH):
        raise FileNotFoundError(f"FATAL: Scaling stats file not found at {STATS_FILE_PATH}")

    # Load scaling statistics from the text file
    with open(STATS_FILE_PATH, 'r') as f:
        scaling_stats = json.load(f)

    # Initialize the model architecture
    model = TimeSeriesTransformer(
        encoder_feature_dim=len(ENCODER_FEATURES), decoder_feature_dim=len(DECODER_INPUT_FEATURES),
        d_model=D_MODEL, nhead=NHEAD, num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS, dim_feedforward=DIM_FEEDFORWARD, dropout=DROPOUT_RATE
    ).to(DEVICE)

    # Load the trained weights of your best model
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    print(f"Successfully loaded the fine-tuned model from {MODEL_SAVE_PATH}\nDevice: {DEVICE}")

    # --- Process Validation Set ---
    if os.path.exists(VAL_DATA_PATH):
        df_val_all = pd.read_csv(VAL_DATA_PATH)
        df_val_scaled = scale_dataframe_by_campus(df_val_all, scaling_stats)
        val_dataset = SolarPowerDataset_Transformer(df_val_scaled, LOOKBACK_STEPS, LOOKFORWARD_STEPS)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        evaluate_and_visualize(model, val_loader, scaling_stats, "Validation Set", DEVICE)
    else:
        print(f"Warning: Validation data not found at {VAL_DATA_PATH}. Skipping validation set evaluation.")

    # --- Process Test Set ---
    if os.path.exists(TEST_DATA_PATH):
        df_test_all = pd.read_csv(TEST_DATA_PATH)
        df_test_scaled = scale_dataframe_by_campus(df_test_all, scaling_stats)
        test_dataset = SolarPowerDataset_Transformer(df_test_scaled, LOOKBACK_STEPS, LOOKFORWARD_STEPS)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        evaluate_and_visualize(model, test_loader, scaling_stats, "Test Set", DEVICE)
    else:
        print(f"\nWarning: Test data not found at {TEST_DATA_PATH}. Skipping test set evaluation.")
