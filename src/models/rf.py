"""
Random Forest model for trajectory prediction
Predicts radial distance using ensemble of decision trees
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler


class RandomForestModel:
    """
    Random Forest for radial distance prediction
    """
    
    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        """
        Initialize Random Forest model
        
        Parameters:
        -----------
        n_estimators : int
            Number of trees in the forest
        max_depth : int or None
            Maximum depth of the trees
        random_state : int
            Random state for reproducibility
        """
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )
        self.is_fitted = False
        
    def fit(self, X, y):
        """Fit the Random Forest model"""
        self.model.fit(X, y)
        self.is_fitted = True
        
    def predict(self, X):
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        return self.model.predict(X) 