"""
Training script for Random Forest model
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path
from sklearn.metrics import mean_squared_error
import random
import os
import joblib

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.models.rf import RandomForestModel
from src.config import TRAINING_CONFIG, RF_CONFIG


def set_seed(seed=42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def train_model():
    """Train and evaluate Random Forest model"""
    # Set random seed
    set_seed(TRAINING_CONFIG['random_seed'])
    
    # Load data
    df = pd.read_csv('data/features/features_selected.csv')
    
    # Get feature columns
    feature_cols = [col for col in df.columns 
                   if col not in ['X', 'Y', 'r', 'trajectory_id', 'step_id']]
    
    # Split data
    train_traj_ids = list(range(16))
    val_traj_ids = list(range(16, 20))
    
    # Prepare training data
    train_df = df[df['trajectory_id'].isin(train_traj_ids)]
    X_train = train_df[feature_cols].values
    Y_train = train_df[['r']].values.ravel()
    
    # Prepare validation data
    val_df = df[df['trajectory_id'].isin(val_traj_ids)]
    X_val = val_df[feature_cols].values
    Y_val = val_df[['r']].values.ravel()
    
    # Train model
    print(f"Training Random Forest model with n_estimators={RF_CONFIG['n_estimators']}, max_depth={RF_CONFIG['max_depth']}...")
    model = RandomForestModel(
        n_estimators=RF_CONFIG['n_estimators'],
        max_depth=RF_CONFIG['max_depth'],
        random_state=RF_CONFIG['random_state']
    )
    model.fit(X_train, Y_train)
    
    print("Training complete!")
    
    # Save model
    model_dir = Path('results/models')
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / 'rf_model.pkl'
    
    joblib.dump({
        'model': model,
        'feature_count': X_train.shape[1],
        'training_config': TRAINING_CONFIG,
        'rf_config': RF_CONFIG
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
    
    # Feature importance
    print("\nTop 10 Feature Importances:")
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for idx, row in feature_importance.head(10).iterrows():
        print(f"{row['feature']}: {row['importance']:.4f}")


if __name__ == "__main__":
    train_model() 