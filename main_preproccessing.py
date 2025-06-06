"""
Main script to run the complete feature engineering pipeline
This script should be placed in the project root directory
"""

import sys
from pathlib import Path
import os

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent / 'src'))

# Import the feature selection module which includes feature engineering
from src.data.feature_engineering import main as run_feature_engineering
from src.data.preprocessing import load_and_preprocess_data

# Create a wrapper to run the complete pipeline
def run_complete_pipeline():
    """
    Run the complete feature engineering and selection pipeline
    """
    print("="*60)
    print("TRAJECTORY PREDICTION - FEATURE ENGINEERING PIPELINE")
    print("="*60)
    
    # Step 1: Run feature engineering
    print("\nStep 1: Engineering features...")
    print("-"*40)
    features_df, feature_names = run_feature_engineering()
    
    # Step 2: Run feature selection
    print("\n\nStep 2: Selecting best features...")
    print("Method: LASSO")
    print("-"*40)
    
    # Import and run feature selection
    from src.data.feature_selection import main as run_feature_selection
    selected_features = run_feature_selection()
    
    # Step 3: Load and preprocess the selected features
    print("\n\nStep 3: Loading and preprocessing selected features...")
    print("-"*40)
    feature_path = os.path.join("data", "features", "features_selected.csv")
    X_train, Y_train, X_val, Y_val = load_and_preprocess_data(feature_path)
    print(f"Training set shape: X={X_train.shape}, Y={Y_train.shape}")
    print(f"Validation set shape: X={X_val.shape}, Y={Y_val.shape}")
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("="*60)
    print(f"\n✓ All features saved to: data/features/features_all.csv")
    print(f"✓ Selected features (7 total) saved to: data/features/features_selected.csv")
    print(f"✓ Feature importance plot saved to: data/features/feature_importance.png")
    print(f"✓ Data preprocessed and split into training/validation sets")
    print(f"\nSelected features include mandatory PL and RMS plus top 5 from Lasso")
    
    return features_df, selected_features, (X_train, Y_train, X_val, Y_val)


if __name__ == "__main__":
    # Run the complete pipeline
    features_df, selected_features, (X_train, Y_train, X_val, Y_val) = run_complete_pipeline()
    print(X_train.shape)
    print(Y_train.shape)
