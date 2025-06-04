"""
Feature selection for trajectory prediction
Selects the most informative features for predicting radial distance
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')
from sklearn.feature_selection import mutual_info_regression, SelectKBest
from sklearn.model_selection import cross_val_score


class FeatureSelector:
    def __init__(self, target_col='r', n_features=7):
        """
        Initialize the feature selector
        
        Parameters:
        -----------
        target_col : str
            Target column name (default: 'r' for radial distance)
        n_features : int
            Total number of features to select (including PL and RMS)
        """
        self.target_col = target_col
        self.n_features = n_features
        self.feature_scores = {}
        self.selected_features = []
        self.scaler = StandardScaler()
        # Always include these base features
        self.mandatory_features = ['PL', 'RMS']
        
    def load_features(self, features_path='data/features/features_all.csv'):
        """
        Load the engineered features
        
        Parameters:
        -----------
        features_path : str
            Path to the features CSV file
            
        Returns:
        --------
        pd.DataFrame : DataFrame with all features
        """
        df = pd.read_csv(features_path)
        print(f"Loaded features from: {features_path}")
        print(f"Shape: {df.shape}")
        return df
    
    def prepare_data(self, df):
        """
        Prepare data for feature selection
        FIXED: Ensures we work with properly structured trajectory data
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with all features
            
        Returns:
        --------
        tuple : (X, y, feature_names)
        """
        # Verify trajectory structure exists
        if 'trajectory_id' not in df.columns or 'step_id' not in df.columns:
            raise ValueError("Data must contain trajectory_id and step_id columns")
        
        # Identify feature columns (exclude targets and metadata)
        exclude_cols = ['X', 'Y', 'r', 'trajectory_id', 'step_id']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Remove any remaining NaN or infinite values
        df_clean = df.copy()
        df_clean[feature_cols] = df_clean[feature_cols].replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN values within each trajectory independently
        for traj_id in df_clean['trajectory_id'].unique():
            traj_mask = df_clean['trajectory_id'] == traj_id
            traj_data = df_clean.loc[traj_mask, feature_cols]
            # Fill with trajectory-specific means, then 0 for any remaining NaN
            df_clean.loc[traj_mask, feature_cols] = traj_data.fillna(traj_data.mean()).fillna(0)
        
        # Extract features and targets
        X = df_clean[feature_cols].values
        y = df_clean[self.target_col].values.reshape(-1, 1)  # Ensure 2D for consistency
        
        print(f"\nFeature matrix shape: {X.shape}")
        print(f"Target matrix shape: {y.shape}")
        print(f"Number of trajectories: {df_clean['trajectory_id'].nunique()}")
        
        return X, y, feature_cols
    
    def select_with_lasso(self, X, y, feature_names):
        """
        Select features using Lasso regularization
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray  
            Target values (radial distance)
        feature_names : list
            Names of features
            
        Returns:
        --------
        list : Selected feature names
        """
        print(f"\n--- Lasso-based Feature Selection (Top {self.n_features} features) ---")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Lasso with cross-validation
        lasso = LassoCV(cv=5, random_state=42, max_iter=5000, n_alphas=100)
        lasso.fit(X_scaled, y.ravel())  # Use ravel() for 1D target
        
        # Get feature importances (absolute coefficients)
        importance = np.abs(lasso.coef_)
        
        # Create a dict of feature scores
        feature_score_dict = dict(zip(feature_names, importance))
        
        # Get top features (excluding mandatory ones first)
        non_mandatory_features = [f for f in feature_names if f not in self.mandatory_features]
        non_mandatory_scores = [(f, feature_score_dict[f]) for f in non_mandatory_features]
        non_mandatory_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select top features after mandatory ones
        n_additional = self.n_features - len(self.mandatory_features)
        selected_additional = [f for f, score in non_mandatory_scores[:n_additional]]
        
        # Combine mandatory and selected features
        selected_features = self.mandatory_features + selected_additional
        
        # Print all selected features with scores
        print(f"\nSelected {len(selected_features)} features:")
        for feat in selected_features:
            print(f"  {feat}: {feature_score_dict[feat]:.4f}")
        
        # Store scores
        self.feature_scores['lasso'] = feature_score_dict
        
        return selected_features
    
    def select_with_mutual_info(self, X, y, feature_names):
        """
        Select features using mutual information
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix  
        y : np.ndarray
            Target values (radial distance)
        feature_names : list
            Names of features
            
        Returns:
        --------
        list : Selected feature names
        """
        print(f"\n--- Mutual Information Feature Selection (Top {self.n_features} features) ---")
        
        # Calculate mutual information scores
        mi_scores = mutual_info_regression(X, y.ravel(), random_state=42)
        
        # Create a dict of feature scores
        feature_score_dict = dict(zip(feature_names, mi_scores))
        
        # Print scores for mandatory features
        print(f"\nMandatory feature scores:")
        for feat in self.mandatory_features:
            if feat in feature_score_dict:
                print(f"  {feat}: {feature_score_dict[feat]:.4f}")
        
        # Get top features (excluding mandatory ones first)
        non_mandatory_features = [f for f in feature_names if f not in self.mandatory_features]
        non_mandatory_scores = [(f, feature_score_dict[f]) for f in non_mandatory_features]
        non_mandatory_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select top features after mandatory ones
        n_additional = self.n_features - len(self.mandatory_features)
        selected_additional = [f for f, score in non_mandatory_scores[:n_additional]]
        
        # Combine mandatory and selected features
        selected_features = self.mandatory_features + selected_additional
        
        # Print all selected features with scores
        print(f"\nSelected {len(selected_features)} features:")
        for feat in selected_features:
            print(f"  {feat}: {feature_score_dict[feat]:.4f}")
        
        # Store scores
        self.feature_scores['mutual_info'] = feature_score_dict
        
        return selected_features
    
    def visualize_feature_importance(self, selected_features):
        """
        Visualize feature importance scores
        
        Parameters:
        -----------
        selected_features : list
            List of selected feature names
        """
        if not self.feature_scores:
            print("No feature scores to visualize")
            return
        
        # Get the scores for the method used
        method_scores = self.feature_scores['lasso']
        
        # Get scores for selected features
        selected_scores = [(feat, method_scores.get(feat, 0)) for feat in selected_features]
        selected_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        features, scores = zip(*selected_scores)
        positions = range(len(features))
        
        # Create bar plot
        bars = plt.bar(positions, scores)
        
        # Color mandatory features differently
        for i, feat in enumerate(features):
            if feat in self.mandatory_features:
                bars[i].set_color('darkred')
            else:
                bars[i].set_color('steelblue')
        
        plt.xlabel('Features', fontsize=12)
        plt.ylabel('Importance Score', fontsize=12)
        plt.title('Lasso Feature Importance - Top 7 Features', fontsize=14)
        plt.xticks(positions, features, rotation=45, ha='right')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='darkred', label='Mandatory (PL, RMS)'),
            Patch(facecolor='steelblue', label='Selected')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.savefig('data/features/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("\nFeature importance plot saved to: data/features/feature_importance.png")
    
    def select_features(self):
        """
        Main method to perform feature selection using Lasso
        
        Returns:
        --------
        list : Selected feature names
        """
        # Load features
        df = self.load_features()
        
        # Prepare data
        X, y, feature_names = self.prepare_data(df)
        
        # Ensure mandatory features exist in the data
        missing_mandatory = [f for f in self.mandatory_features if f not in feature_names]
        if missing_mandatory:
            print(f"Warning: Mandatory features {missing_mandatory} not found in data!")
            self.mandatory_features = [f for f in self.mandatory_features if f in feature_names]
        
        # Apply Lasso feature selection
        selected_features = self.select_with_lasso(X, y, feature_names)
        self.selected_features = selected_features
        
        # Visualize feature importance
        self.visualize_feature_importance(selected_features)
        
        return selected_features
    
    def save_selected_features(self, df, selected_features):
        """
        Save dataset with only selected features
        
        Parameters:
        -----------
        df : pd.DataFrame
            Original DataFrame with all features
        selected_features : list
            List of selected feature names
        """
        # Include targets and metadata in the output
        output_cols = ['r', 'trajectory_id', 'step_id'] + selected_features
        
        # Ensure all columns exist
        output_cols = [col for col in output_cols if col in df.columns]
        
        # Create output DataFrame
        output_df = df[output_cols].copy()
        for col in selected_features:
            if col in output_df.columns:
                output_df[col] = output_df[col].round(2)
        
        # Save to CSV
        output_path = Path('data/features/features_selected.csv')
        output_df.to_csv(output_path, index=False)
        
        print(f"\n--- Feature Selection Complete ---")
        print(f"Selected {len(selected_features)} features")
        print(f"Saved to: {output_path}")
        print(f"Output shape: {output_df.shape}")
        
        # Print selected features
        print(f"\nSelected features:")
        for i, feature in enumerate(selected_features, 1):
            print(f"{i:2d}. {feature}")


def main():
    """
    Main function to perform feature selection using Lasso
    """
    # Initialize selector with 7 features total (including PL and RMS) for radial distance
    selector = FeatureSelector(target_col='r', n_features=7)
    
    # Perform feature selection
    selected_features = selector.select_features()
    
    return selected_features