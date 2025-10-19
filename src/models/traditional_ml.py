"""
Traditional ML models for concrete strength prediction.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import pickle


class TraditionalMLModels:
    """
    Traditional ML models for concrete strength prediction.
    """
    
    def __init__(self):
        """Initialize ML models."""
        self.models = {
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=20,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            ),
            'xgboost': xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            ),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=1.0),
            'mlp': MLPRegressor(
                hidden_layer_sizes=(100, 50),
                max_iter=500,
                random_state=42
            )
        }
        self.scaler = StandardScaler()
        self.trained_models = {}
        
    def train(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        models_to_train: list = None
    ) -> Dict[str, Any]:
        """
        Train ML models.
        
        Args:
            X_train: Training features
            y_train: Training targets
            models_to_train: List of model names to train. If None, trains all.
            
        Returns:
            Dictionary with training results
        """
        if models_to_train is None:
            models_to_train = list(self.models.keys())
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        results = {}
        
        for model_name in models_to_train:
            if model_name not in self.models:
                print(f"Warning: Model {model_name} not found. Skipping.")
                continue
                
            print(f"Training {model_name}...")
            model = self.models[model_name]
            model.fit(X_train_scaled, y_train)
            self.trained_models[model_name] = model
            
            # Training predictions
            train_pred = model.predict(X_train_scaled)
            train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
            train_mae = mean_absolute_error(y_train, train_pred)
            train_r2 = r2_score(y_train, train_pred)
            
            results[model_name] = {
                'train_rmse': train_rmse,
                'train_mae': train_mae,
                'train_r2': train_r2
            }
            
            print(f"  Train RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}, R2: {train_r2:.4f}")
        
        return results
    
    def evaluate(
        self, 
        X_test: pd.DataFrame, 
        y_test: pd.Series
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate trained models.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary with evaluation metrics for each model
        """
        X_test_scaled = self.scaler.transform(X_test)
        
        results = {}
        
        for model_name, model in self.trained_models.items():
            predictions = model.predict(X_test_scaled)
            
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            
            results[model_name] = {
                'test_rmse': rmse,
                'test_mae': mae,
                'test_r2': r2,
                'predictions': predictions
            }
            
            print(f"{model_name}:")
            print(f"  Test RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
        
        return results
    
    def predict(
        self, 
        X: pd.DataFrame, 
        model_name: str
    ) -> np.ndarray:
        """
        Make predictions with a specific model.
        
        Args:
            X: Input features
            model_name: Name of the model to use
            
        Returns:
            Array of predictions
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} has not been trained yet.")
        
        X_scaled = self.scaler.transform(X)
        return self.trained_models[model_name].predict(X_scaled)
    
    def save_models(self, path: str):
        """
        Save trained models to disk.
        
        Args:
            path: Directory path to save models
        """
        import os
        os.makedirs(path, exist_ok=True)
        
        # Save scaler
        with open(f"{path}/scaler.pkl", 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save each model
        for model_name, model in self.trained_models.items():
            with open(f"{path}/{model_name}.pkl", 'wb') as f:
                pickle.dump(model, f)
        
        print(f"Models saved to {path}")
    
    def load_models(self, path: str):
        """
        Load trained models from disk.
        
        Args:
            path: Directory path to load models from
        """
        import os
        
        # Load scaler
        with open(f"{path}/scaler.pkl", 'rb') as f:
            self.scaler = pickle.load(f)
        
        # Load models
        for model_file in os.listdir(path):
            if model_file.endswith('.pkl') and model_file != 'scaler.pkl':
                model_name = model_file.replace('.pkl', '')
                with open(f"{path}/{model_file}", 'rb') as f:
                    self.trained_models[model_name] = pickle.load(f)
        
        print(f"Models loaded from {path}")


def train_traditional_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Tuple[TraditionalMLModels, Dict[str, Any]]:
    """
    Convenience function to train and evaluate traditional ML models.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        y_test: Test targets
        
    Returns:
        Tuple of (TraditionalMLModels instance, evaluation results)
    """
    ml_models = TraditionalMLModels()
    
    print("Training traditional ML models...")
    print("=" * 50)
    ml_models.train(X_train, y_train)
    
    print("\nEvaluating models...")
    print("=" * 50)
    results = ml_models.evaluate(X_test, y_test)
    
    return ml_models, results
