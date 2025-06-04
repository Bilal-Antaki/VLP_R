"""
Training script for Linear Regression baseline model
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path
from sklearn.metrics import mean_squared_error
import random
import os
import joblib
from sklearn.linear_model import LinearRegression

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.models.linear import LinearBaseline
from src.config import TRAINING_CONFIG


def set_seed(seed=42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def prepare_data():
    """
    Load and prepare data for linear model
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
    X_train_list, Y_train_list = [], []
    for traj_id in train_traj_ids:
        traj_data = df[df['trajectory_id'] == traj_id].sort_values('step_id')
        if len(traj_data) == 10:
            X_train_list.append(traj_data[feature_cols].values)
            Y_train_list.append(traj_data[['r']].values)
    
    # Prepare validation data
    X_val_list, Y_val_list = [], []
    for traj_id in val_traj_ids:
        traj_data = df[df['trajectory_id'] == traj_id].sort_values('step_id')
        if len(traj_data) == 10:
            X_val_list.append(traj_data[feature_cols].values)
            Y_val_list.append(traj_data[['r']].values)
    
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
    
    # Convert to arrays
    X_train = np.array(X_train_list)
    Y_train = np.array(Y_train_list)
    X_val = np.array(X_val_list)
    Y_val = np.array(Y_val_list)
    
    # Flatten for sklearn (combine all trajectories)
    X_train_flat = X_train.reshape(-1, X_train.shape[-1])
    Y_train_flat = Y_train.reshape(-1, Y_train.shape[-1])
    X_val_flat = X_val.reshape(-1, X_val.shape[-1])
    Y_val_flat = Y_val.reshape(-1, Y_val.shape[-1])
    
    print(f"Training samples: {X_train_flat.shape[0]}")
    print(f"Validation samples: {X_val_flat.shape[0]}")
    print(f"Feature dimension: {X_train_flat.shape[1]}")
    
    return (X_train_flat, Y_train_flat, X_val_flat, Y_val_flat), (X_val, Y_val)


def evaluate_trajectories(model, X_val, Y_val):
    """
    Evaluate model on trajectory level
    """
    # Predict on flattened validation data
    X_val_flat = X_val.reshape(-1, X_val.shape[-1])
    val_pred_flat = model.predict(X_val_flat)
    
    # Reshape predictions back to trajectory format
    val_pred = val_pred_flat.reshape(X_val.shape[0], X_val.shape[1], -1)
    
    # Calculate RMSE for each trajectory
    rmse_per_traj = []
    
    for i in range(len(Y_val)):
        # Radial distance RMSE
        mse = mean_squared_error(Y_val[i], val_pred[i])
        rmse = np.sqrt(mse)
        rmse_per_traj.append(rmse)
    
    return np.array(rmse_per_traj), val_pred


def train_model():
    """Train and evaluate the linear baseline model"""
    print("Linear Baseline Model Training")
    print("=" * 50)
    
    # Prepare data
    (X_train_flat, Y_train_flat, X_val_flat, Y_val_flat), (X_val, Y_val) = prepare_data()
    
    # Initialize model
    model = LinearRegression()
    
    # Fit the model
    print("Training linear regression model...")
    model.fit(X_train_flat, Y_train_flat)
    
    print("Training complete!")
    
    # Save model
    model_dir = Path('results/models')
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / 'linear_baseline_model.pkl'
    
    joblib.dump({
        'model': model,
        'feature_count': X_train_flat.shape[1],
        'training_config': TRAINING_CONFIG
    }, model_path)
    
    print(f"Model saved to: {model_path}")
    
    # Evaluate on validation set
    print("\n" + "="*50)
    print("Validation Results")
    print("="*50)
    
    # Trajectory-level evaluation
    rmse_per_traj, val_pred = evaluate_trajectories(model, X_val, Y_val)
    
    # Print results
    print(f"\nRadial Distance (r) Metrics:")
    print(f"RMSE: {rmse_per_traj.mean():.2f} ± {rmse_per_traj.std():.2f}")
    print(f"MAE: {np.mean(np.abs(Y_val - val_pred)):.2f}")
    
    # Save model and results
    model_data = {
        'model': model,
        'feature_count': X_train_flat.shape[1],
        'training_config': TRAINING_CONFIG,
        'rmse_mean': rmse_per_traj.mean(),
        'rmse_std': rmse_per_traj.std()
    }
    
    # Save results
    results_dir = Path('results/results')
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / 'linear_baseline_results.pkl'
    
    joblib.dump(model_data, results_path)
    
    print(f"\nResults saved to: {results_path}")


def load_and_evaluate():
    """Load saved model and evaluate"""
    model_path = Path('results/models/linear_baseline_model.pkl')
    
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
        
        # Overall metrics
        mae_r = np.mean(np.abs(Y_test[:, 0] - predictions[:, 0]))
        rmse_r = np.sqrt(mean_squared_error(Y_test[:, 0], predictions[:, 0]))
        
        print(f"\nMetrics for trajectory {test_traj_id}:")
        print(f"MAE  - r: {mae_r:.2f}")
        print(f"RMSE - r: {rmse_r:.2f}")


if __name__ == "__main__":
    train_model()
