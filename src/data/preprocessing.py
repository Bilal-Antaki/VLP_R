"""
Improved preprocessing for trajectory prediction
Handles sequences properly without data leakage
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

# Get the absolute path of the script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Get the project root directory (2 levels up from script)
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))


def load_and_preprocess_data(feature_path):
    """
    Load and preprocess the selected features for model training
    
    Parameters:
    -----------
    feature_path : str
        Path to the selected features CSV file
        
    Returns:
    --------
    tuple : (X_train, Y_train, X_val, Y_val)
        Preprocessed training and validation data
    """
    # Load data
    df = pd.read_csv(feature_path)
    
    # Get feature columns (exclude targets and metadata)
    feature_cols = [col for col in df.columns 
                   if col not in ['X', 'Y', 'r', 'trajectory_id', 'step_id']]
    
    print(f"Using {len(feature_cols)} features: {feature_cols}")
    
    # Split into train and validation based on trajectory_id
    train_traj_ids = list(range(16))
    val_traj_ids = list(range(16, 20))
    
    # Prepare training data
    X_train, Y_train = [], []
    for traj_id in train_traj_ids:
        traj_data = df[df['trajectory_id'] == traj_id].sort_values('step_id')
        if len(traj_data) == 10:
            X_train.append(traj_data[feature_cols].values)
            Y_train.append(traj_data[['r']].values)
    
    # Prepare validation data
    X_val, Y_val = [], []
    for traj_id in val_traj_ids:
        traj_data = df[df['trajectory_id'] == traj_id].sort_values('step_id')
        if len(traj_data) == 10:
            X_val.append(traj_data[feature_cols].values)
            Y_val.append(traj_data[['r']].values)
    
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_val = np.array(X_val)
    Y_val = np.array(Y_val)
    
    print(f"\nTraining data: {X_train.shape}, {Y_train.shape}")
    print(f"Validation data: {X_val.shape}, {Y_val.shape}")
    
    return X_train, Y_train, X_val, Y_val


def create_lagged_features(df, feature_cols, n_lags=2):
    """
    Create lagged features for better temporal modeling
    """
    df_lagged = df.copy()
    
    # Create lagged features
    for lag in range(1, n_lags + 1):
        for col in ['X', 'Y']:
            df_lagged[f'{col}_lag{lag}'] = df_lagged.groupby('trajectory_id')[col].shift(lag)
    
    # Drop rows with NaN values from lagging
    df_lagged = df_lagged.groupby('trajectory_id').apply(
        lambda x: x.iloc[n_lags:] if len(x) > n_lags else x.iloc[0:0]
    ).reset_index(drop=True)
    
    return df_lagged


def normalize_by_trajectory(X_train, Y_train, X_val, Y_val):
    """
    Normalize features with trajectory-aware scaling
    """
    # Flatten for normalization but keep track of shapes
    train_shape = X_train.shape
    val_shape = X_val.shape
    
    # Normalize features
    feature_scaler = StandardScaler()
    X_train_flat = X_train.reshape(-1, X_train.shape[-1])
    X_val_flat = X_val.reshape(-1, X_val.shape[-1])
    
    X_train_scaled = feature_scaler.fit_transform(X_train_flat).reshape(train_shape)
    X_val_scaled = feature_scaler.transform(X_val_flat).reshape(val_shape)
    
    # Normalize targets
    target_scaler = StandardScaler()
    if Y_train.ndim == 3:  # Full sequences
        Y_train_flat = Y_train.reshape(-1, Y_train.shape[-1])
        Y_val_flat = Y_val.reshape(-1, Y_val.shape[-1])
        Y_train_scaled = target_scaler.fit_transform(Y_train_flat).reshape(Y_train.shape)
        Y_val_scaled = target_scaler.transform(Y_val_flat).reshape(Y_val.shape)
    else:  # Single targets
        Y_train_scaled = target_scaler.fit_transform(Y_train)
        Y_val_scaled = target_scaler.transform(Y_val)
    
    return (X_train_scaled, Y_train_scaled, X_val_scaled, Y_val_scaled), (feature_scaler, target_scaler)