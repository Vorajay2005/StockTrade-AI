"""
Simplified ML Model using XGBoost instead of LSTM for better compatibility
Works with Python 3.13 and doesn't require TensorFlow
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import logging
import os
from typing import Tuple, Dict, List, Optional
from config import FEATURES, MODEL_DIR, SCALER_DIR
import warnings
warnings.filterwarnings('ignore')

class SimpleMLPredictor:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.model = None
        self.scaler = None
        self.feature_scaler = None
        self.sequence_length = 30  # Reduced for XGBoost
        self.logger = logging.getLogger(__name__)
        
        # Create directories
        os.makedirs(MODEL_DIR, exist_ok=True)
        os.makedirs(SCALER_DIR, exist_ok=True)
        
        self.model_path = os.path.join(MODEL_DIR, f"{symbol}_xgb_model.pkl")
        self.scaler_path = os.path.join(SCALER_DIR, f"{symbol}_price_scaler.pkl")
        self.feature_scaler_path = os.path.join(SCALER_DIR, f"{symbol}_feature_scaler.pkl")
        
    def prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for XGBoost training
        
        Args:
            data: DataFrame with features and target
            
        Returns:
            Tuple of (X, y)
        """
        # Select features
        available_features = [col for col in FEATURES if col in data.columns]
        if not available_features:
            self.logger.error("No features found in data")
            return None, None
        
        feature_data = data[available_features].copy()
        target_data = data['close'].copy()
        
        # Handle missing values
        feature_data = feature_data.fillna(method='ffill').fillna(method='bfill')
        target_data = target_data.fillna(method='ffill').fillna(method='bfill')
        
        # Create lag features for time series
        for lag in [1, 2, 3, 5, 10]:
            for col in ['close', 'volume']:
                if col in feature_data.columns:
                    feature_data[f'{col}_lag_{lag}'] = feature_data[col].shift(lag)
        
        # Create rolling statistics
        for window in [5, 10, 20]:
            if 'close' in feature_data.columns:
                feature_data[f'close_rolling_mean_{window}'] = feature_data['close'].rolling(window).mean()
                feature_data[f'close_rolling_std_{window}'] = feature_data['close'].rolling(window).std()
        
        # Drop rows with NaN values
        feature_data = feature_data.dropna()
        target_data = target_data.loc[feature_data.index]
        
        # Scale features
        if self.feature_scaler is None:
            self.feature_scaler = StandardScaler()
            feature_scaled = self.feature_scaler.fit_transform(feature_data)
        else:
            feature_scaled = self.feature_scaler.transform(feature_data)
        
        # Create target (next day return)
        target_return = target_data.pct_change().shift(-1).dropna()
        feature_scaled = feature_scaled[:-1]  # Align with target
        
        return feature_scaled, target_return.values
    
    def train_model(self, data: pd.DataFrame, save_model: bool = True) -> Dict:
        """
        Train XGBoost model
        
        Args:
            data: DataFrame with features and target
            save_model: Whether to save the trained model
            
        Returns:
            Dictionary with training metrics
        """
        self.logger.info(f"Training XGBoost model for {self.symbol}")
        
        # Prepare data
        X, y = self.prepare_data(data)
        if X is None:
            return {}
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, shuffle=False, random_state=42
        )
        
        # Create and train model
        self.model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Make predictions
        train_predictions = self.model.predict(X_train)
        val_predictions = self.model.predict(X_val)
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
        val_rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
        train_mae = mean_absolute_error(y_train, train_predictions)
        val_mae = mean_absolute_error(y_val, val_predictions)
        train_r2 = r2_score(y_train, train_predictions)
        val_r2 = r2_score(y_val, val_predictions)
        
        # Save model and scaler
        if save_model:
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.feature_scaler, self.feature_scaler_path)
        
        metrics = {
            'train_rmse': train_rmse,
            'val_rmse': val_rmse,
            'train_mae': train_mae,
            'val_mae': val_mae,
            'train_r2': train_r2,
            'val_r2': val_r2,
            'model_saved': save_model
        }
        
        self.logger.info(f"Model training completed. Val RMSE: {val_rmse:.4f}, Val R2: {val_r2:.4f}")
        
        return metrics
    
    def load_model(self) -> bool:
        """
        Load saved model and scalers
        
        Returns:
            True if successfully loaded, False otherwise
        """
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                self.feature_scaler = joblib.load(self.feature_scaler_path)
                self.logger.info(f"Model loaded successfully for {self.symbol}")
                return True
            else:
                self.logger.warning(f"No saved model found for {self.symbol}")
                return False
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            return False
    
    def predict(self, data: pd.DataFrame, steps: int = 1) -> Dict:
        """
        Make predictions using the trained model
        
        Args:
            data: DataFrame with recent data
            steps: Number of future steps to predict
            
        Returns:
            Dictionary with predictions and confidence metrics
        """
        if self.model is None:
            if not self.load_model():
                return {}
        
        try:
            # Prepare data
            X, _ = self.prepare_data(data)
            if X is None:
                return {}
            
            # Make prediction (predict return)
            predicted_return = self.model.predict(X[-1:].reshape(1, -1))[0]
            
            # Convert return to price
            current_price = data['close'].iloc[-1]
            predicted_price = current_price * (1 + predicted_return)
            
            # Calculate confidence based on model's feature importance and prediction magnitude
            feature_importance = self.model.feature_importances_
            confidence = min(0.9, max(0.1, np.mean(feature_importance) / (1 + abs(predicted_return))))
            
            # Determine signal
            if predicted_return > 0.02:  # 2% threshold
                signal = 'BUY'
            elif predicted_return < -0.02:  # -2% threshold
                signal = 'SELL'
            else:
                signal = 'HOLD'
            
            return {
                'predictions': [predicted_price],
                'confidence': [confidence],
                'current_price': current_price,
                'predicted_price': predicted_price,
                'price_change': predicted_return,
                'price_change_percent': predicted_return * 100,
                'signal': signal,
                'signal_strength': confidence,
                'timestamp': data.index[-1]
            }
            
        except Exception as e:
            self.logger.error(f"Error making prediction: {str(e)}")
            return {}
    
    def retrain_model(self, new_data: pd.DataFrame) -> Dict:
        """
        Retrain model with new data
        
        Args:
            new_data: New data to retrain with
            
        Returns:
            Dictionary with retraining metrics
        """
        self.logger.info(f"Retraining model for {self.symbol}")
        
        # Load existing model if available
        self.load_model()
        
        # Train with new data
        metrics = self.train_model(new_data, save_model=True)
        
        self.logger.info(f"Model retrained successfully for {self.symbol}")
        return metrics
    
    def evaluate_model(self, test_data: pd.DataFrame) -> Dict:
        """
        Evaluate model performance on test data
        
        Args:
            test_data: Test dataset
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            if not self.load_model():
                return {}
        
        # Prepare test data
        X_test, y_test = self.prepare_data(test_data)
        if X_test is None:
            return {}
        
        # Make predictions
        predictions = self.model.predict(X_test)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        # Calculate directional accuracy
        actual_direction = np.sign(y_test)
        predicted_direction = np.sign(predictions)
        directional_accuracy = np.mean(actual_direction == predicted_direction)
        
        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'directional_accuracy': directional_accuracy,
            'num_predictions': len(predictions)
        }
    
    def get_feature_importance(self) -> Dict:
        """
        Get feature importance from XGBoost model
        
        Returns:
            Dictionary with feature importance scores
        """
        if self.model is None:
            return {}
        
        feature_names = [f'feature_{i}' for i in range(len(self.model.feature_importances_))]
        importance_dict = dict(zip(feature_names, self.model.feature_importances_))
        
        return importance_dict
    
    def get_model_info(self) -> Dict:
        """
        Get model information
        
        Returns:
            Dictionary with model information
        """
        if self.model is None:
            return {}
        
        return {
            'symbol': self.symbol,
            'model_type': 'XGBoost',
            'n_estimators': self.model.n_estimators,
            'max_depth': self.model.max_depth,
            'learning_rate': self.model.learning_rate,
            'model_path': self.model_path,
            'feature_scaler_path': self.feature_scaler_path
        }

# Alias for compatibility
LSTMPredictor = SimpleMLPredictor

# Example usage
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data
    dates = pd.date_range('2020-01-01', periods=500, freq='D')
    np.random.seed(42)
    
    # Generate sample data with features
    close = 100 + np.cumsum(np.random.randn(500) * 0.02)
    high = close + np.abs(np.random.randn(500) * 0.5)
    low = close - np.abs(np.random.randn(500) * 0.5)
    open_price = close + np.random.randn(500) * 0.1
    volume = np.random.randint(1000, 10000, 500)
    
    # Create technical indicators (simplified)
    sample_data = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume,
        'sma_5': close.rolling(5).mean(),
        'sma_10': close.rolling(10).mean(),
        'sma_20': close.rolling(20).mean(),
        'sma_50': close.rolling(50).mean(),
        'ema_12': close.ewm(span=12).mean(),
        'ema_26': close.ewm(span=26).mean(),
        'rsi': 50 + np.random.randn(500) * 10,
        'macd': np.random.randn(500) * 0.1,
        'macd_signal': np.random.randn(500) * 0.1,
        'price_change': close.diff(),
        'volume_change': volume - np.mean(volume),
        'volatility': close.rolling(20).std()
    }, index=dates)
    
    # Fill NaN values
    sample_data = sample_data.fillna(method='ffill')
    
    # Test ML model
    predictor = SimpleMLPredictor('TEST.NS')
    
    # Train model
    print("Training XGBoost model...")
    metrics = predictor.train_model(sample_data)
    print(f"Training metrics: {metrics}")
    
    # Make predictions
    print("\nMaking predictions...")
    predictions = predictor.predict(sample_data.tail(100))
    print(f"Predictions: {predictions}")
    
    # Evaluate model
    print("\nEvaluating model...")
    evaluation = predictor.evaluate_model(sample_data.tail(100))
    print(f"Evaluation: {evaluation}")
    
    # Get model info
    print("\nModel info:")
    info = predictor.get_model_info()
    print(info)