"""
Simple ML Model using only scikit-learn for maximum compatibility
No external ML libraries required beyond standard scikit-learn
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
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
        self.sequence_length = 30
        self.logger = logging.getLogger(__name__)
        
        # Create directories
        os.makedirs(MODEL_DIR, exist_ok=True)
        os.makedirs(SCALER_DIR, exist_ok=True)
        
        self.model_path = os.path.join(MODEL_DIR, f"{symbol}_ml_model.pkl")
        self.scaler_path = os.path.join(SCALER_DIR, f"{symbol}_price_scaler.pkl")
        self.feature_scaler_path = os.path.join(SCALER_DIR, f"{symbol}_feature_scaler.pkl")
        
    def prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for ML training
        
        Args:
            data: DataFrame with features and target
            
        Returns:
            Tuple of (X, y)
        """
        # Select available features
        available_features = [col for col in FEATURES if col in data.columns]
        if not available_features:
            # Use basic OHLCV if no technical indicators
            available_features = ['open', 'high', 'low', 'close', 'volume']
            available_features = [col for col in available_features if col in data.columns]
        
        if not available_features:
            self.logger.error("No features found in data")
            return None, None
        
        feature_data = data[available_features].copy()
        target_data = data['close'].copy()
        
        # Handle missing values
        feature_data = feature_data.fillna(method='ffill').fillna(method='bfill')
        target_data = target_data.fillna(method='ffill').fillna(method='bfill')
        
        # Create additional features
        self._add_technical_features(feature_data)
        
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
    
    def _add_technical_features(self, data: pd.DataFrame):
        """Add basic technical features if not present"""
        if 'close' in data.columns:
            # Simple moving averages
            for window in [5, 10, 20]:
                col_name = f'sma_{window}'
                if col_name not in data.columns:
                    data[col_name] = data['close'].rolling(window).mean()
            
            # Price changes
            if 'price_change' not in data.columns:
                data['price_change'] = data['close'].diff()
            
            # Volatility
            if 'volatility' not in data.columns:
                data['volatility'] = data['close'].rolling(20).std()
            
            # RSI-like indicator
            if 'rsi' not in data.columns:
                delta = data['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss
                data['rsi'] = 100 - (100 / (1 + rs))
        
        if 'volume' in data.columns:
            # Volume changes
            if 'volume_change' not in data.columns:
                data['volume_change'] = data['volume'].diff()
    
    def train_model(self, data: pd.DataFrame, save_model: bool = True) -> Dict:
        """
        Train Random Forest model
        
        Args:
            data: DataFrame with features and target
            save_model: Whether to save the trained model
            
        Returns:
            Dictionary with training metrics
        """
        self.logger.info(f"Training Random Forest model for {self.symbol}")
        
        # Prepare data
        X, y = self.prepare_data(data)
        if X is None:
            return {}
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, shuffle=False, random_state=42
        )
        
        # Create and train model
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        
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
            
            # Calculate confidence based on model's feature importance and prediction variance
            if hasattr(self.model, 'feature_importances_'):
                feature_importance = np.mean(self.model.feature_importances_)
                confidence = min(0.9, max(0.1, feature_importance / (1 + abs(predicted_return))))
            else:
                confidence = 0.5  # Default confidence
            
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
        Get feature importance from Random Forest model
        
        Returns:
            Dictionary with feature importance scores
        """
        if self.model is None or not hasattr(self.model, 'feature_importances_'):
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
            'model_type': 'Random Forest',
            'n_estimators': getattr(self.model, 'n_estimators', 'N/A'),
            'max_depth': getattr(self.model, 'max_depth', 'N/A'),
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
    
    # Generate sample data
    close = 100 + np.cumsum(np.random.randn(500) * 0.02)
    high = close + np.abs(np.random.randn(500) * 0.5)
    low = close - np.abs(np.random.randn(500) * 0.5)
    open_price = close + np.random.randn(500) * 0.1
    volume = np.random.randint(1000, 10000, 500)
    
    sample_data = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)
    
    # Test ML model
    predictor = SimpleMLPredictor('TEST.NS')
    
    # Train model
    print("Training Random Forest model...")
    metrics = predictor.train_model(sample_data)
    print(f"Training metrics: {metrics}")
    
    # Make predictions
    print("\nMaking predictions...")
    predictions = predictor.predict(sample_data.tail(100))
    print(f"Predictions: {predictions}")
    
    # Get model info
    print("\nModel info:")
    info = predictor.get_model_info()
    print(info)