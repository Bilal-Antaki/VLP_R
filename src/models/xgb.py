"""
XGBoost model for trajectory prediction
Predicts radial distance using gradient boosting
"""

import xgboost as xgb
import numpy as np
from sklearn.preprocessing import StandardScaler


class XGBoostModel:
    """
    XGBoost for radial distance prediction
    """
    
    def __init__(self, n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42):
        """
        Initialize XGBoost model
        
        Parameters:
        -----------
        n_estimators : int
            Number of boosting rounds
        max_depth : int
            Maximum tree depth
        learning_rate : float
            Learning rate (eta)
        random_state : int
            Random state for reproducibility
        """
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            objective='reg:squarederror'
        )
        self.scaler_features = StandardScaler()
        self.is_fitted = False
        
    def fit(self, X, y):
        """Fit the XGBoost model"""
        # Scale features
        X_scaled = self.scaler_features.fit_transform(X)
        
        # Fit model
        self.model.fit(X_scaled, y)
        
        self.is_fitted = True
        
    def predict(self, X):
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Scale features
        X_scaled = self.scaler_features.transform(X)
        
        # Predict
        pred = self.model.predict(X_scaled)
        
        return pred.astype(int) 