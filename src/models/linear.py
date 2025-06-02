"""
Linear baseline model for trajectory prediction
Uses simple linear regression to predict radial distance from features
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


class LinearBaseline:
    def __init__(self):
        """
        Initialize the linear baseline model
        """
        self.model = LinearRegression()
        self.scaler_features = StandardScaler()
        self.scaler_target = StandardScaler()
        self.is_fitted = False
        
    def fit(self, X, y):
        """
        Fit the linear model
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training features
        y : array-like of shape (n_samples, 1)
            Target values (radial distance)
        """
        # Scale features
        X_scaled = self.scaler_features.fit_transform(X)
        
        # Scale targets
        y_scaled = self.scaler_target.fit_transform(y.reshape(-1, 1)).ravel()
        
        # Fit model
        self.model.fit(X_scaled, y_scaled)
        
        self.is_fitted = True
        
    def predict(self, X):
        """
        Make predictions
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Features to predict on
            
        Returns:
        --------
        array-like of shape (n_samples, 1) : Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Scale features
        X_scaled = self.scaler_features.transform(X)
        
        # Predict
        pred_scaled = self.model.predict(X_scaled)
        
        # Inverse transform predictions
        pred = self.scaler_target.inverse_transform(pred_scaled.reshape(-1, 1))
        
        return pred
