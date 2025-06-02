"""
Training script for Multi-Layer Perceptron model
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path
from sklearn.metrics import mean_squared_error
import random
import os
import torch
import joblib

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.models.mlp import MLPModel
from src.config import TRAINING_CONFIG, MLP_CONFIG


def set_seed(seed=42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def train_model():
    """Train and evaluate MLP model"""
    # Set random seed
    set_seed(TRAINING_CONFIG['random_seed'])
    
    # Load data
    df = pd.read_csv('data/features/features_selected.csv')
    
    # Get feature columns
    feature_cols = [col for col in df.columns 
                   if col not in ['X', 'Y', 'r', 'trajectory_id', 'step_id']]
    
    # Split data
    train_df = df[df['trajectory_id'] < 16]
    val_df = df[df['trajectory_id'] >= 16]
    
    X_train = train_df[feature_cols].values
    Y_train = train_df[['r']].values
    
    X_val = val_df[feature_cols].values
    Y_val = val_df[['r']].values
    
    # Train model
    print(f"Training MLP model with architecture {MLP_CONFIG['hidden_sizes']}...")
    model = MLPModel(
        hidden_sizes=MLP_CONFIG['hidden_sizes'],
        dropout=MLP_CONFIG['dropout'],
        learning_rate=MLP_CONFIG['learning_rate'],
        epochs=MLP_CONFIG['epochs']
    )
    model.fit(X_train, Y_train)
    
    print("Training complete!")
    
    # Save model
    model_dir = Path('results/models')
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / 'mlp_model.pkl'
    
    # Save the entire model object
    torch.save({
        'model_state_dict': model.model.state_dict(),
        'model_config': {
            'hidden_sizes': model.hidden_sizes,
            'dropout': model.dropout,
            'input_size': X_train.shape[1]
        },
        'scaler_features': model.scaler_features,
        'scaler_targets': model.scaler_targets,
        'training_config': TRAINING_CONFIG,
        'mlp_config': MLP_CONFIG
    }, model_path)
    
    print(f"\nModel saved to: {model_path}")
    
    # Evaluate on validation set
    val_pred = model.predict(X_val)
    
    # Metrics
    rmse = np.sqrt(mean_squared_error(Y_val, val_pred))
    mae = np.mean(np.abs(Y_val - val_pred))
    
    print(f"\nValidation Metrics:")
    print(f"Radial Distance (r) RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    
    # Print sample predictions
    print(f"\nSample predictions (first 10):")
    print("True r | Pred r | Error")
    print("-" * 30)
    for i in range(min(10, len(Y_val))):
        error = abs(Y_val[i, 0] - val_pred[i, 0])
        print(f"{Y_val[i, 0]:6.2f} | {val_pred[i, 0]:6.2f} | {error:6.2f}")


if __name__ == "__main__":
    train_model() 