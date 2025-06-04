"""
Main entry point for the Position Estimation project
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.training.train_lstm import train_model as train_lstm
from src.training.train_linear import train_model as train_linear
from src.training.train_svr import train_model as train_svr
from src.training.train_rf import train_model as train_rf
from src.training.train_mlp import train_model as train_mlp


def main():
    

    print("Training LSTM Model")
    print("=" * 60)
    train_lstm()
    print("\n")
    
    print("=" * 60)
    print("Training Linear Baseline Model")
    print("=" * 60)
    train_linear()
    print("\n")
    
    print("=" * 60)
    print("Training SVR Model")
    print("=" * 60)
    train_svr()
    print("\n")
    
    print("=" * 60)
    print("Training Random Forest Model")
    print("=" * 60)
    train_rf()
    print("\n")
    
    print("=" * 60)
    print("Training MLP (Multi-Layer Perceptron) Model")
    print("=" * 60)
    train_mlp()
    print("\n")

    compare_models()


def compare_models():
    # Check if all models exist
    lstm_path = Path('results/models/lstm_best_model.pth')
    linear_path = Path('results/models/linear_baseline_model.pkl')
    svr_path = Path('results/models/svr_model.pkl')
    rf_path = Path('results/models/rf_model.pkl')
    mlp_path = Path('results/models/mlp_model.pkl')
    
    if not all([lstm_path.exists(), linear_path.exists(), svr_path.exists(), 
                rf_path.exists(), mlp_path.exists()]):
        print("All models need to be trained first for comparison.")
        return


if __name__ == "__main__":
    main()