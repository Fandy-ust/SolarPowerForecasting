import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os
import pandas as pd
import json
import math
import warnings
import torch.nn as nn
import torch.nn.functional as F

# Import shared utilities
from utils import (
    PositionalEncoding, TimeSeriesTransformer, SolarPowerDataset_Transformer,
    masked_rmse, find_continuous_segments, apply_zscore_scaling, scale_dataframe_by_campus,
    process_batch, autoregressive_generate, create_dataloaders,
    TARGET_COLUMN, WEATHER_FEATURES, TIME_FEATURES, ENCODER_FEATURES,
    FUTURE_KNOWN_FEATURES, DECODER_INPUT_FEATURES, get_target_feature_index,
    LOOKBACK_STEPS, LOOKFORWARD_STEPS, BATCH_SIZE, MAX_GRAD_NORM, 
    PATIENCE, SENTINEL_VALUE, D_MODEL, NHEAD, NUM_ENCODER_LAYERS, 
    NUM_DECODER_LAYERS, DIM_FEEDFORWARD, DROPOUT_RATE
)

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
PHASE2B_MODEL_SAVE_PATH = os.path.join(PHASE2B_SAVE_DIR, 'transformer_elite_model_phase2b.pth')

# --- Feature Columns (consistent with previous phases) ---
TARGET_COLUMN = 'SolarGeneration'
WEATHER_FEATURES = ['AirTemperature', 'Ghi_hourly', 'CloudOpacity_hourly', "minutes_since_last_update",
                    "RelativeHumidity"]
TIME_FEATURES = ["zenith_sin", "azimuth_sin", "day_sin", "year_sin"]
# Feature definitions now imported from utils

# --- Training Hyperparameters ---
LOOKBACK_STEPS = 4 * 24
LOOKFORWARD_STEPS = 4 * 5
BATCH_SIZE = 32
LEARNING_RATE_PHASE2B = 0.1e-6
EPOCHS_PHASE2B = 10
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
#  4. Core Logic: The SolarTransformerPhase2B Class (CORRECTED)
# =============================================================================
class SolarTransformerPhase2B:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    def process_batch(self, batch):
        return process_batch(batch, DEVICE)

    def train_epoch(self, loader, epoch, prediction_cache):
        self.model.train()
        total_loss = 0

        # --- CORRECTED SCHEDULED SAMPLING LOGIC ---
        # We want the model to rely MORE on its own predictions over time.
        # This rate will START LOW and INCREASE with each epoch.
        # Epoch 1 (index 0): 10% | Epoch 2 (index 1): 20% | ... | Epoch 7 (index 6): 70%
        prediction_usage_rate = min(1.0, 0.1 + (epoch / (EPOCHS_PHASE2B - 1)) * 0.9)
        print(f"Scheduled sampling rate (prediction_usage_rate): {prediction_usage_rate:.2f}")

        progress_bar = tqdm(enumerate(loader), total=len(loader), desc=f"Phase 2B Training", leave=True)
        for batch_idx, batch in progress_bar:
            src, tgt_in, tgt_out, tgt_mask, src_mask = self.process_batch(batch)

            if batch_idx not in prediction_cache:
                print(f"Warning: Missing prediction for batch {batch_idx} in cache. Skipping.")
                continue

            cached_preds = prediction_cache[batch_idx].to(DEVICE)
            mixed_input = tgt_in.clone()

            # --- Create a mask for scheduled sampling ---
            # True where we will use the model's own prediction, False for ground truth
            # The probability of being True is our 'prediction_usage_rate'
            ss_mask = torch.rand_like(tgt_in[:, :, 0]) < prediction_usage_rate

            # We should only apply this to positions that originally had ground truth.
            # NaNs will always be filled from the cache.
            nan_mask = tgt_mask.clone()
            ground_truth_mask = ~nan_mask

            # Combine the masks: apply SS only where we have ground truth
            final_ss_mask = ss_mask & ground_truth_mask

            # Get target feature index from utils
            target_idx = get_target_feature_index()

            # 1. Fill NaNs using the cached predictions (always)
            mixed_input[:, :, target_idx][nan_mask] = cached_preds[nan_mask]

            # 2. Apply scheduled sampling to the ground-truth values
            mixed_input[:, :, target_idx][final_ss_mask] = cached_preds[final_ss_mask]

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
        Uses the shared autoregressive_generate function from utils.
        """
        return autoregressive_generate(self.model, src, src_mask, tgt_in, tgt_mask, DEVICE)

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
#  5. Main Execution Logic for Phase 2B (with new saving strategy)
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

    train_loader, val_loader = create_dataloaders(df_train_all, df_val_all, scaling_stats, BATCH_SIZE)
    print(f"Train samples: {len(train_loader.dataset)}, Validation samples: {len(val_loader.dataset)}")

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
