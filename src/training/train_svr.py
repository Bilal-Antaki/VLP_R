"""
Training script for Support Vector Regression model
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
from src.models.svr import SVRModel
from src.config import TRAINING_CONFIG, SVR_CONFIG


def set_seed(seed=42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def prepare_data():
    """
    Load and prepare data for SVR model
    """
    # Load selected features
    df = pd.read_csv('data/features/features_selected.csv')
    
    # Get feature columns
    feature_cols = [col for col in df.columns 
                   if col not in ['X', 'Y', 'r', 'trajectory_id', 'step_id']]
    
    print(f"Using features: {feature_cols}")
    
    # Split by trajectory IDs
    train_traj_ids = list(range(16))
    val_traj_ids = list(range(16, 20))
    
    # Prepare training data
    train_df = df[df['trajectory_id'].isin(train_traj_ids)]
    X_train = train_df[feature_cols].values
    Y_train = train_df[['r']].values
    
    # Prepare validation data
    val_df = df[df['trajectory_id'].isin(val_traj_ids)]
    X_val = val_df[feature_cols].values
    Y_val = val_df[['r']].values
    
    # Get trajectory structure for validation
    val_trajectories = []
    for traj_id in val_traj_ids:
        traj_data = df[df['trajectory_id'] == traj_id].sort_values('step_id')
        if len(traj_data) == 10:
            val_trajectories.append({
                'X': traj_data[feature_cols].values,
                'Y': traj_data[['r']].values,
                'id': traj_id
            })
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Feature dimension: {X_train.shape[1]}")
    
    return (X_train, Y_train, X_val, Y_val), val_trajectories


def evaluate_model(model, X_val, Y_val, val_trajectories):
    """
    Evaluate model performance
    """
    # Predict on validation set
    val_pred = model.predict(X_val)
    
    # Point-wise metrics
    rmse = np.sqrt(mean_squared_error(Y_val, val_pred))
    mae = np.mean(np.abs(Y_val.ravel() - val_pred))
    
    print(f"\nPoint-wise Validation Metrics:")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    
    # Trajectory-level evaluation
    rmse_per_traj = []
    
    for traj in val_trajectories:
        # Predict
        predictions = model.predict(traj['X'])
        
        # Calculate RMSE for this trajectory
        mse = mean_squared_error(traj['Y'], predictions)
        rmse = np.sqrt(mse)
        rmse_per_traj.append(rmse)
    
    rmse_per_traj = np.array(rmse_per_traj)
    
    print(f"\nTrajectory-level Validation Metrics:")
    print(f"Radial Distance (r):")
    print(f"  RMSE: {rmse_per_traj.mean():.2f}")
    print(f"  Std: {rmse_per_traj.std():.2f}")
    
    # Print sample predictions
    if val_trajectories:
        sample_traj = val_trajectories[0]
        sample_pred = model.predict(sample_traj['X'])
        
        print(f"\nSample predictions for trajectory {sample_traj['id']}:")
        print("Step | True r | Pred r | Error")
        print("-" * 35)
        for i in range(min(5, len(sample_pred))):
            error = abs(sample_traj['Y'][i, 0] - sample_pred[i])
            print(f"{i+1:4d} | {sample_traj['Y'][i, 0]:6.2f} | {sample_pred[i]:6.2f} | {error:6.2f}")


def train_model():
    """Train SVR model"""
    # Set random seed for reproducibility
    set_seed(TRAINING_CONFIG['random_seed'])
    
    # Prepare data
    (X_train, Y_train, X_val, Y_val), val_trajectories = prepare_data()
    
    # Initialize model with config parameters
    model = SVRModel(
        kernel=SVR_CONFIG['kernel'],
        C=SVR_CONFIG['C'],
        epsilon=SVR_CONFIG['epsilon'],
        gamma=SVR_CONFIG['gamma']
    )
    
    print(f"\nTraining SVR model with kernel='{SVR_CONFIG['kernel']}', C={SVR_CONFIG['C']}, epsilon={SVR_CONFIG['epsilon']}, gamma={SVR_CONFIG['gamma']}...")
    
    # Fit the model
    model.fit(X_train, Y_train)
    
    print("Training complete!")
    
    # Save the model
    model_dir = Path('results/models')
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / 'svr_model.pkl'
    
    # Save using joblib (better for scikit-learn models)
    joblib.dump({
        'model': model,
        'feature_count': X_train.shape[1],
        'training_config': TRAINING_CONFIG,
        'svr_config': SVR_CONFIG
    }, model_path)
    
    print(f"\nModel saved to: {model_path}")
    
    # Evaluate on validation set (point-wise)
    evaluate_model(model, X_val, Y_val, val_trajectories)


def load_and_evaluate():
    """Load saved model and evaluate"""
    model_path = Path('results/models/svr_model.pkl')
    
    if not model_path.exists():
        print(f"Model not found at {model_path}. Train the model first.")
        return
    
    # Load model
    checkpoint = joblib.load(model_path)
    model = checkpoint['model']
    
    print(f"Model loaded from: {model_path}")
    
    # Load test data
    df = pd.read_csv('data/features/features_selected.csv')
    feature_cols = [col for col in df.columns 
                   if col not in ['X', 'Y', 'r', 'trajectory_id', 'step_id']]
    
    # Test on a specific trajectory
    test_traj_id = 19
    test_data = df[df['trajectory_id'] == test_traj_id].sort_values('step_id')
    
    if len(test_data) == 10:
        X_test = test_data[feature_cols].values
        Y_test = test_data[['r']].values
        
        # Predict
        predictions = model.predict(X_test)
        
        print(f"\nPredictions for trajectory {test_traj_id}:")
        print("Step | True r | Pred r | Error")
        print("-" * 35)
        for i in range(len(predictions)):
            error = abs(Y_test[i, 0] - predictions[i])
            print(f"{i+1:4d} | {Y_test[i, 0]:6.2f} | {predictions[i]:6.2f} | {error:6.2f}")
        
        # Overall metrics
        mae = np.mean(np.abs(Y_test[:, 0] - predictions))
        rmse = np.sqrt(mean_squared_error(Y_test[:, 0], predictions))
        
        print(f"\nMetrics for trajectory {test_traj_id}:")
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")


if __name__ == "__main__":
    train_model() 