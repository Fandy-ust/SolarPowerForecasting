import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import json
import os
import time
import math
from tqdm import tqdm

# Import shared utilities
from utils import (
    PositionalEncoding, TimeSeriesTransformer, SolarPowerDataset_Transformer,
    apply_zscore_scaling, scale_dataframe_by_campus, process_batch,
    autoregressive_generate, create_dataloaders,
    TARGET_COLUMN, WEATHER_FEATURES, TIME_FEATURES, ENCODER_FEATURES,
    FUTURE_KNOWN_FEATURES, DECODER_INPUT_FEATURES, get_target_feature_index,
    LOOKBACK_STEPS, LOOKFORWARD_STEPS, BATCH_SIZE, SENTINEL_VALUE,
    D_MODEL, NHEAD, NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, 
    DIM_FEEDFORWARD, DROPOUT_RATE
)

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

# Note: Other constants now imported from utils


# =============================================================================
#  2. Model and Data Class Definitions (Now imported from utils)
# =============================================================================


# Dataset class and scaling functions now imported from utils


# =============================================================================
#  4. Phase 3 Training and Validation Loops (CORRECTED)
# =============================================================================
def train_epoch_fully_autoregressive(model, data_loader, optimizer, criterion, device):
    """Performs one training epoch in a fully autoregressive manner."""
    model.train()
    total_loss = 0
    progress_bar = tqdm(data_loader, desc="[Train] Fully Autoregressive")

    for batch in progress_bar:
        src, tgt_in, tgt_out, tgt_key_padding_mask, src_key_padding_mask = process_batch(batch, device)

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
                target_idx = get_target_feature_index()
                working_input[:, step + 1, target_idx] = pred_this_step.detach()

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

    for batch in progress_bar:
        src, tgt_in, tgt_out, tgt_key_padding_mask, src_key_padding_mask = process_batch(batch, device)

        working_input = tgt_in.clone()
        final_preds = None

        for step in range(LOOKFORWARD_STEPS):
            pred_all_steps = model(src, working_input, tgt_key_padding_mask=None,
                                   src_key_padding_mask=src_key_padding_mask)
            pred_this_step = pred_all_steps[:, step, 0]
            if step < LOOKFORWARD_STEPS - 1:
                # No .detach() needed here because of @torch.no_grad()
                target_idx = get_target_feature_index()
                working_input[:, step + 1, target_idx] = pred_this_step

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
    df_val_all = pd.read_csv(VAL_DATA_PATH)
    
    # Scale dataframes
    df_train_scaled = scale_dataframe_by_campus(df_train_all, scaling_stats)
    df_val_scaled = scale_dataframe_by_campus(df_val_all, scaling_stats)
    
    # Create datasets
    train_dataset = SolarPowerDataset_Transformer(df_train_scaled, LOOKBACK_STEPS, LOOKFORWARD_STEPS)
    val_dataset = SolarPowerDataset_Transformer(df_val_scaled, LOOKBACK_STEPS, LOOKFORWARD_STEPS)
    
    # Create loaders with shuffle=True for training
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
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

