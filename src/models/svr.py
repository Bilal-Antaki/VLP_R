"""
Support Vector Regression model for trajectory prediction
Predicts radial distance using SVR with RBF kernel
"""

import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler


class SVRModel:
    """
    Support Vector Regression for radial distance prediction
    """
    
    def __init__(self, kernel='rbf', C=1.0, epsilon=0.1, gamma='scale'):
        """
        Initialize SVR model
        
        Parameters:
        -----------
        kernel : str
            Kernel type for SVR
        C : float
            Regularization parameter
        epsilon : float
            Epsilon-tube within which no penalty is associated
        gamma : str or float
            Kernel coefficient
        """
        self.model = SVR(kernel=kernel, C=C, epsilon=epsilon, gamma=gamma)
        self.scaler_features = StandardScaler()
        self.scaler_target = StandardScaler()
        self.is_fitted = False
        
    def fit(self, X, y):
        """
        Fit the SVR model
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training features
        y : array-like of shape (n_samples, 1)
            Target values
        """
        # Scale features
        X_scaled = self.scaler_features.fit_transform(X)
        
        # Scale targets separately
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
        pred = self.scaler_target.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
        
        return pred 