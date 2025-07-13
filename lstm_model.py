"""
LSTM Neural Network Model for Stock Price Prediction
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import logging
import os
from typing import Tuple, Dict, List, Optional
from config import LSTM_CONFIG, FEATURES, MODEL_DIR, SCALER_DIR
import warnings
warnings.filterwarnings('ignore')

class LSTMPredictor:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.model = None
        self.scaler = None
        self.feature_scaler = None
        self.sequence_length = LSTM_CONFIG["sequence_length"]
        self.logger = logging.getLogger(__name__)
        
        # Create directories
        os.makedirs(MODEL_DIR, exist_ok=True)
        os.makedirs(SCALER_DIR, exist_ok=True)
        
        self.model_path = os.path.join(MODEL_DIR, f"{symbol}_lstm_model.h5")
        self.scaler_path = os.path.join(SCALER_DIR, f"{symbol}_price_scaler.pkl")
        self.feature_scaler_path = os.path.join(SCALER_DIR, f"{symbol}_feature_scaler.pkl")
        
    def prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for LSTM training
        
        Args:
            data: DataFrame with features and target
            
        Returns:
            Tuple of (X, y, feature_data)
        """
        # Select features
        available_features = [col for col in FEATURES if col in data.columns]
        if not available_features:
            self.logger.error("No features found in data")
            return None, None, None
        
        feature_data = data[available_features].copy()
        target_data = data['close'].copy()
        
        # Handle missing values
        feature_data = feature_data.fillna(method='ffill').fillna(method='bfill')
        target_data = target_data.fillna(method='ffill').fillna(method='bfill')
        
        # Scale features
        if self.feature_scaler is None:
            self.feature_scaler = StandardScaler()
            feature_scaled = self.feature_scaler.fit_transform(feature_data)
        else:
            feature_scaled = self.feature_scaler.transform(feature_data)
        
        # Scale target
        if self.scaler is None:
            self.scaler = MinMaxScaler(feature_range=(0, 1))
            target_scaled = self.scaler.fit_transform(target_data.values.reshape(-1, 1))
        else:
            target_scaled = self.scaler.transform(target_data.values.reshape(-1, 1))
        
        # Create sequences
        X, y = [], []
        for i in range(self.sequence_length, len(feature_scaled)):
            X.append(feature_scaled[i-self.sequence_length:i])
            y.append(target_scaled[i])
        
        return np.array(X), np.array(y), feature_scaled
    
    def build_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """
        Build LSTM model architecture
        
        Args:
            input_shape: Shape of input data (sequence_length, features)
            
        Returns:
            Compiled LSTM model
        """
        model = Sequential()
        
        # First LSTM layer
        model.add(LSTM(
            units=LSTM_CONFIG["lstm_units"][0],
            return_sequences=True,
            input_shape=input_shape
        ))
        model.add(Dropout(LSTM_CONFIG["dropout_rate"]))
        model.add(BatchNormalization())
        
        # Second LSTM layer
        model.add(LSTM(
            units=LSTM_CONFIG["lstm_units"][1],
            return_sequences=False
        ))
        model.add(Dropout(LSTM_CONFIG["dropout_rate"]))
        model.add(BatchNormalization())
        
        # Dense layers
        model.add(Dense(units=50, activation='relu'))
        model.add(Dropout(LSTM_CONFIG["dropout_rate"]))
        
        model.add(Dense(units=25, activation='relu'))
        model.add(Dropout(LSTM_CONFIG["dropout_rate"]))
        
        # Output layer
        model.add(Dense(units=1, activation='linear'))
        
        # Compile model
        optimizer = Adam(learning_rate=LSTM_CONFIG["learning_rate"])
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train_model(self, data: pd.DataFrame, save_model: bool = True) -> Dict:
        """
        Train LSTM model
        
        Args:
            data: DataFrame with features and target
            save_model: Whether to save the trained model
            
        Returns:
            Dictionary with training history and metrics
        """
        self.logger.info(f"Training LSTM model for {self.symbol}")
        
        # Prepare data
        X, y, feature_data = self.prepare_data(data)
        if X is None:
            return {}
        
        # Split data
        train_size = int(len(X) * (1 - LSTM_CONFIG["validation_split"]))
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]
        
        # Build model
        self.model = self.build_model((X.shape[1], X.shape[2]))
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7
            )
        ]
        
        if save_model:
            callbacks.append(
                ModelCheckpoint(
                    self.model_path,
                    monitor='val_loss',
                    save_best_only=True,
                    save_weights_only=False
                )
            )
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            epochs=LSTM_CONFIG["epochs"],
            batch_size=LSTM_CONFIG["batch_size"],
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate model
        train_predictions = self.model.predict(X_train)
        val_predictions = self.model.predict(X_val)
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
        val_rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
        train_mae = mean_absolute_error(y_train, train_predictions)
        val_mae = mean_absolute_error(y_val, val_predictions)
        train_r2 = r2_score(y_train, train_predictions)
        val_r2 = r2_score(y_val, val_predictions)
        
        # Save scalers
        if save_model:
            joblib.dump(self.scaler, self.scaler_path)
            joblib.dump(self.feature_scaler, self.feature_scaler_path)
        
        metrics = {
            'train_rmse': train_rmse,
            'val_rmse': val_rmse,
            'train_mae': train_mae,
            'val_mae': val_mae,
            'train_r2': train_r2,
            'val_r2': val_r2,
            'history': history.history,
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
                self.model = load_model(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
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
            available_features = [col for col in FEATURES if col in data.columns]
            if not available_features:
                self.logger.error("No features found in data for prediction")
                return {}
            
            feature_data = data[available_features].copy()
            feature_data = feature_data.fillna(method='ffill').fillna(method='bfill')
            
            # Scale features
            feature_scaled = self.feature_scaler.transform(feature_data)
            
            # Get last sequence
            if len(feature_scaled) < self.sequence_length:
                self.logger.error(f"Not enough data for prediction. Need {self.sequence_length}, got {len(feature_scaled)}")
                return {}
            
            predictions = []
            confidence_scores = []
            
            # Make predictions
            for step in range(steps):
                # Get input sequence
                input_sequence = feature_scaled[-self.sequence_length:]
                input_sequence = input_sequence.reshape(1, self.sequence_length, -1)
                
                # Make prediction
                prediction_scaled = self.model.predict(input_sequence, verbose=0)
                prediction = self.scaler.inverse_transform(prediction_scaled)[0][0]
                
                predictions.append(prediction)
                
                # Calculate confidence (simplified approach)
                # In production, you might want to use ensemble methods or uncertainty quantification
                confidence = min(0.9, max(0.1, 1.0 / (1.0 + abs(prediction - data['close'].iloc[-1]) / data['close'].iloc[-1])))
                confidence_scores.append(confidence)
                
                # For multi-step prediction, update the feature data
                if step < steps - 1:
                    # This is a simplified approach - in practice, you'd need to update all features
                    # For now, we'll just use the last sequence
                    pass
            
            # Calculate trading signal
            current_price = data['close'].iloc[-1]
            predicted_price = predictions[0]
            price_change = (predicted_price - current_price) / current_price
            
            # Determine signal
            if price_change > 0.02:  # 2% threshold
                signal = 'BUY'
            elif price_change < -0.02:  # -2% threshold
                signal = 'SELL'
            else:
                signal = 'HOLD'
            
            return {
                'predictions': predictions,
                'confidence': confidence_scores,
                'current_price': current_price,
                'predicted_price': predicted_price,
                'price_change': price_change,
                'price_change_percent': price_change * 100,
                'signal': signal,
                'signal_strength': confidence_scores[0] if confidence_scores else 0,
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
        X_test, y_test, _ = self.prepare_data(test_data)
        if X_test is None:
            return {}
        
        # Make predictions
        predictions = self.model.predict(X_test)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        # Calculate directional accuracy
        actual_direction = np.sign(np.diff(self.scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()))
        predicted_direction = np.sign(np.diff(self.scaler.inverse_transform(predictions).flatten()))
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
        Get feature importance (simplified approach)
        
        Returns:
            Dictionary with feature importance scores
        """
        if self.model is None:
            return {}
        
        # This is a simplified approach
        # In practice, you might want to use techniques like SHAP or LIME
        feature_names = [col for col in FEATURES if col in FEATURES]
        
        # For now, return equal importance
        importance = {feature: 1.0 / len(feature_names) for feature in feature_names}
        
        return importance
    
    def save_model_summary(self, filepath: str):
        """
        Save model summary to file
        
        Args:
            filepath: Path to save the summary
        """
        if self.model is None:
            return
        
        with open(filepath, 'w') as f:
            self.model.summary(print_fn=lambda x: f.write(x + '\n'))
    
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
            'model_type': 'LSTM',
            'sequence_length': self.sequence_length,
            'total_params': self.model.count_params(),
            'layers': len(self.model.layers),
            'model_path': self.model_path,
            'scaler_path': self.scaler_path,
            'feature_scaler_path': self.feature_scaler_path
        }

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
    
    # Test LSTM model
    predictor = LSTMPredictor('TEST.NS')
    
    # Train model
    print("Training LSTM model...")
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