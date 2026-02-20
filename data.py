import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset

from sklearn.preprocessing import MinMaxScaler

import pickle
import os

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, seq_len):
        self.X = torch.tensor(X).float()
        self.y = torch.tensor(y).float()
        self.seq_len = seq_len
    
    def __len__(self):
        return len(self.X) - self.seq_len 
        
    def __getitem__(self, idx):
        # Transpose to [features, seq_len] for Conv1d
        # TCN expects input: (batch, features, seq_len)
        x_seq = self.X[idx:idx+self.seq_len].T  # Shape: (features, seq_len)
        y_target = self.y[idx+self.seq_len]
        return (x_seq, y_target)

def add_lag_features(df, target_cols, lags=[1, 3, 12]):
    """    
    Args:
        df: DataFrame with time series data
        target_cols: Columns to create lags for
        lags: [1, 3, 12] - month, quarter, year lags
    """
    df = df.copy()
    
    for col in target_cols:
        # Simple lags: 1, 3, 12
        for lag in lags:
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    
    # Drop rows with NaN from lag_12 (first 12 rows)
    df = df.dropna()
    
    return df

def add_seasonal_features(df):
    """
    Add cyclical seasonal encoding.
    
    Based on:
    - Hyndman & Athanasopoulos (2021): Seasonal decomposition
    - Makridakis et al. (2020): M4 competition best practices
    """
    df = df.copy()
    
    # Cyclical encoding (better than one-hot for neural networks)
    month = df.index.month
    df['month_sin'] = np.sin(2 * np.pi * month / 12)
    df['month_cos'] = np.cos(2 * np.pi * month / 12)
    
    quarter = df.index.quarter  
    df['quarter_sin'] = np.sin(2 * np.pi * quarter / 4)
    df['quarter_cos'] = np.cos(2 * np.pi * quarter / 4)
    
    # Philippine tax calendar indicators
    # April 15 - individual tax deadline
    # April 30 - corporate tax deadline
    df['is_tax_season'] = df.index.month.isin([3, 4, 5]).astype(int)
    
    # December year-end collections
    df['is_year_end'] = (df.index.month == 12).astype(int)
    
    return df

def transform_data(data, save_path="Transforms/default/scaler.pkl"):
    """Transform data and save scaler"""
    
    # Convert to absolute path if it's relative
    if not os.path.isabs(save_path):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        save_path = os.path.join(base_dir, save_path)
    
    # Create directory if it doesn't exist
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    with open(save_path, "wb") as f:
        pickle.dump(scaler, f)
    
    return data_scaled

def transform_with_scaler(data, scaler_path):
    """Transform data using an existing scaler (no fitting)"""
    
    # Convert to absolute path if it's relative
    if not os.path.isabs(scaler_path):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        scaler_path = os.path.join(base_dir, scaler_path)
    
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    
    return scaler.transform(data)

# Backwards-compatible wrapper used by notebooks/scripts that call
# `transform_data_with_scaler(...)`.
def transform_data_with_scaler(data, scaler_path):
    """Alias for `transform_with_scaler` kept for compatibility."""
    return transform_with_scaler(data, scaler_path)

def inverse_transform(data, load_path="Transforms/default/scaler.pkl"):
    """Inverse transform data using saved scaler"""
    
    # Convert to absolute path if it's relative
    if not os.path.isabs(load_path):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        load_path = os.path.join(base_dir, load_path)
    
    with open(load_path, "rb") as f:
        scaler = pickle.load(f)
    return scaler.inverse_transform(data)

def split_data(X, use_val=True):
    if use_val:
        train_size = int(0.6 * len(X))
        val_size = int(0.2 * len(X))

        train = X[:train_size]
        val = X[train_size:train_size + val_size]
        test = X[train_size + val_size:]

        return train, val, test