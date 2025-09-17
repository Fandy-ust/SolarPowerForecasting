"""
Utility functions and classes for Transformer-based solar power prediction models.

This module contains shared components used across different phases of the transformer training pipeline:
- Model architectures (PositionalEncoding, TimeSeriesTransformer)
- Dataset classes (SolarPowerDataset_Transformer)
- Loss functions (masked_mse_loss)
- Data preprocessing utilities (scaling functions)
- Common configurations and constants
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import math
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

# Feature definitions
TARGET_COLUMN = 'SolarGeneration'
WEATHER_FEATURES = ['AirTemperature', 'Ghi_hourly', 'CloudOpacity_hourly', "minutes_since_last_update", "RelativeHumidity"]
TIME_FEATURES = ["zenith_sin", "azimuth_sin", "day_sin", "year_sin"]
ENCODER_FEATURES = WEATHER_FEATURES + TIME_FEATURES + [TARGET_COLUMN]
FUTURE_KNOWN_FEATURES = TIME_FEATURES
DECODER_INPUT_FEATURES = FUTURE_KNOWN_FEATURES + [TARGET_COLUMN]

# Training hyperparameters
LOOKBACK_STEPS = 4 * 24
LOOKFORWARD_STEPS = 4 * 5
BATCH_SIZE = 32
MAX_GRAD_NORM = 1.0
PATIENCE = 5
SENTINEL_VALUE = -999.0

# Transformer model hyperparameters
D_MODEL = 128
NHEAD = 8
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
DIM_FEEDFORWARD = 512
DROPOUT_RATE = 0.1

# =============================================================================
# MODEL ARCHITECTURE COMPONENTS
# =============================================================================

class PositionalEncoding(nn.Module):
    """
    Positional encoding module for transformer models.
    Adds sinusoidal positional embeddings to input sequences.
    """
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
    """
    Transformer model for time series forecasting with encoder-decoder architecture.
    Supports both source and target key padding masks for handling missing data.
    """
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
        """Generate causal mask for decoder to prevent attention to future positions."""
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                tgt_key_padding_mask: torch.Tensor,
                src_key_padding_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the transformer model.
        
        Args:
            src: Source sequence (encoder input) [batch_size, seq_len, encoder_features]
            tgt: Target sequence (decoder input) [batch_size, seq_len, decoder_features]
            tgt_key_padding_mask: Mask for target padding positions [batch_size, seq_len]
            src_key_padding_mask: Mask for source padding positions [batch_size, seq_len]
        
        Returns:
            Model predictions [batch_size, seq_len, 1]
        """
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


# =============================================================================
# DATASET CLASS
# =============================================================================

class SolarPowerDataset_Transformer(Dataset):
    """
    Dataset class for transformer-based solar power prediction.
    Handles sequence generation with proper padding masks for missing data.
    """
    def __init__(self, df, lookback, lookforward):
        self.df = df
        self.lookback = lookback
        self.lookforward = lookforward
        self.total_seq_len = lookback + lookforward
        self.campus_indices = []
        
        # Build index for efficient sampling across different campuses
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
        """
        Generate a training sample with proper padding masks.
        
        Returns:
            src: Source sequence tensor [lookback, encoder_features]
            tgt_input: Decoder input tensor [lookforward, decoder_features]
            tgt_output: Target output tensor [lookforward, 1]
            tgt_key_padding_mask: Target padding mask [lookforward]
            src_key_padding_mask: Source padding mask [lookback]
        """
        # Find which campus this sample belongs to
        campus_idx = 0
        while idx >= self.campus_indices[campus_idx]['count']:
            idx -= self.campus_indices[campus_idx]['count']
            campus_idx += 1

        start_pos = self.campus_indices[campus_idx]['start'] + idx
        end_pos = start_pos + self.total_seq_len
        sequence_slice = self.df.iloc[start_pos:end_pos]

        # 1. Prepare encoder data
        src_df = sequence_slice.iloc[:self.lookback].copy()
        
        # Create source padding mask (must be done before filling NaN)
        src_key_padding_mask = torch.tensor(src_df[TARGET_COLUMN].isna().values, dtype=torch.bool)
        
        # Fill NaN values (safe because attention will be masked)
        src_df.fillna(0, inplace=True)
        src = torch.tensor(src_df[ENCODER_FEATURES].values, dtype=torch.float32)

        # 2. Prepare decoder data
        future_slice = sequence_slice.iloc[self.lookback:]
        
        # Create target padding mask
        tgt_key_padding_mask = torch.tensor(future_slice[TARGET_COLUMN].isna().values, dtype=torch.bool)
        
        # Create decoder input with shifted target values
        last_obs = sequence_slice.iloc[self.lookback - 1][TARGET_COLUMN]
        shifted = future_slice[TARGET_COLUMN].shift(1)
        shifted.iloc[0] = last_obs
        shifted = shifted.fillna(0)
        
        future_known_df = future_slice[FUTURE_KNOWN_FEATURES]
        tgt_input_df = pd.concat([future_known_df.reset_index(drop=True),
                                  shifted.reset_index(drop=True).rename(TARGET_COLUMN)], axis=1)
        tgt_input = torch.tensor(tgt_input_df[DECODER_INPUT_FEATURES].values, dtype=torch.float32)

        # Create decoder target output
        tgt_output_series = future_slice[TARGET_COLUMN].fillna(SENTINEL_VALUE)
        tgt_output = torch.tensor(tgt_output_series.values, dtype=torch.float32).unsqueeze(-1)

        return src, tgt_input, tgt_output, tgt_key_padding_mask, src_key_padding_mask


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

def masked_mse_loss(predictions: torch.Tensor, targets: torch.Tensor, sentinel_value: float = SENTINEL_VALUE):
    """
    Compute MSE loss while ignoring sentinel (missing) values.
    
    Args:
        predictions: Model predictions
        targets: Ground truth targets
        sentinel_value: Value used to mark missing/masked positions
    
    Returns:
        MSE loss computed only on non-sentinel values
    """
    mask = targets != sentinel_value
    if not mask.any():
        return torch.tensor(0.0, device=predictions.device, requires_grad=True)
    predictions_masked = torch.masked_select(predictions, mask)
    targets_masked = torch.masked_select(targets, mask)
    return F.mse_loss(predictions_masked, targets_masked)


# =============================================================================
# DATA PREPROCESSING UTILITIES
# =============================================================================

def apply_zscore_scaling(df_data, columns, stats):
    """
    Apply z-score normalization to specified columns using provided statistics.
    
    Args:
        df_data: DataFrame to scale
        columns: List of column names to scale
        stats: Dictionary containing mean and std for each column
    
    Returns:
        Scaled DataFrame
    """
    df_scaled = df_data.copy()
    for col in columns:
        if col in stats and stats[col]['std'] > 1e-6:
            df_scaled[col] = (df_scaled[col] - stats[col]['mean']) / stats[col]['std']
    return df_scaled


def scale_dataframe_by_campus(df, scaler_stats):
    """
    Scale DataFrame features by campus using campus-specific statistics.
    
    Args:
        df: Input DataFrame with CampusKey column
        scaler_stats: Dictionary mapping campus IDs to scaling statistics
    
    Returns:
        Scaled DataFrame with same structure as input
    """
    if df.empty:
        return pd.DataFrame()

    scaled_dfs = []
    columns_to_scale = WEATHER_FEATURES + [TARGET_COLUMN]

    for campus_id_num, group in df.groupby('CampusKey'):
        group_copy = group.copy()
        campus_id = str(campus_id_num)
        campus_params = scaler_stats.get(campus_id)

        if campus_params:
            # Fill NaN values in weather features with 0
            for col in WEATHER_FEATURES:
                if col in group_copy.columns:
                    group_copy[col] = group_copy[col].fillna(0)

            # Apply scaling
            scaled_group = apply_zscore_scaling(group_copy, columns_to_scale, campus_params)
            scaled_dfs.append(scaled_group)

    if not scaled_dfs:
        return pd.DataFrame()

    return pd.concat(scaled_dfs).sort_index()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_target_feature_index():
    """Get the index of the target column in decoder input features."""
    return DECODER_INPUT_FEATURES.index(TARGET_COLUMN)


def create_model(encoder_feature_dim=None, decoder_feature_dim=None, device=None):
    """
    Create a TimeSeriesTransformer model with default parameters.
    
    Args:
        encoder_feature_dim: Number of encoder input features (default: len(ENCODER_FEATURES))
        decoder_feature_dim: Number of decoder input features (default: len(DECODER_INPUT_FEATURES))
        device: Device to place the model on (default: cuda if available, else cpu)
    
    Returns:
        Initialized TimeSeriesTransformer model
    """
    if encoder_feature_dim is None:
        encoder_feature_dim = len(ENCODER_FEATURES)
    if decoder_feature_dim is None:
        decoder_feature_dim = len(DECODER_INPUT_FEATURES)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = TimeSeriesTransformer(
        encoder_feature_dim=encoder_feature_dim,
        decoder_feature_dim=decoder_feature_dim,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT_RATE
    ).to(device)
    
    return model