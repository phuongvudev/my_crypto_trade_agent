"""
Machine Learning based trading strategy.
Uses XGBoost for trade signal classification and LSTM for price prediction.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import joblib
import json
import os
from datetime import datetime
from loguru import logger

# ML model imports
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    import xgboost as xgb
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score
except ImportError as e:
    logger.warning(f"ML libraries not available: {e}")

# Local imports
from features.feature_engineering import FeatureEngineer


class MLTradingStrategy:
    """
    Machine Learning based trading strategy combining XGBoost and LSTM models.
    XGBoost predicts trade signals (buy/hold/sell) while LSTM predicts price direction.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize ML trading strategy.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.feature_engineer = FeatureEngineer(config)
        
        # Model paths
        self.lstm_path = config['models']['lstm_path']
        self.xgb_path = config['models']['xgb_path']
        
        # Strategy parameters
        self.confidence_threshold = config['strategy']['ml_model']['confidence_threshold']
        self.lookback_period = config['strategy']['ml_model']['lstm_lookback']
        self.feature_count = config['strategy']['ml_model']['xgb_features']
        
        # Models and scalers
        self.lstm_model = None
        self.xgb_model = None
        self.feature_scaler = None
        self.price_scaler = None
        
        # Model loaded flags
        self.models_loaded = False
        
        # Load models if they exist
        self._load_models()
        
    def _load_models(self) -> None:
        """Load pre-trained models if available."""
        try:
            # Load LSTM model
            if os.path.exists(self.lstm_path):
                self.lstm_model = load_model(self.lstm_path)
                logger.info(f"Loaded LSTM model from {self.lstm_path}")
            else:
                logger.warning(f"LSTM model not found at {self.lstm_path}")
                
            # Load XGBoost model
            if os.path.exists(self.xgb_path):
                self.xgb_model = joblib.load(self.xgb_path)
                logger.info(f"Loaded XGBoost model from {self.xgb_path}")
            else:
                logger.warning(f"XGBoost model not found at {self.xgb_path}")
                
            # Load scalers
            scaler_dir = os.path.dirname(self.xgb_path)
            feature_scaler_path = os.path.join(scaler_dir, 'feature_scaler.pkl')
            price_scaler_path = os.path.join(scaler_dir, 'price_scaler.pkl')
            
            if os.path.exists(feature_scaler_path):
                self.feature_scaler = joblib.load(feature_scaler_path)
                logger.info("Loaded feature scaler")
                
            if os.path.exists(price_scaler_path):
                self.price_scaler = joblib.load(price_scaler_path)
                logger.info("Loaded price scaler")
                
            self.models_loaded = (self.lstm_model is not None and 
                                self.xgb_model is not None and
                                self.feature_scaler is not None and
                                self.price_scaler is not None)
                                
            if self.models_loaded:
                logger.info("All ML models loaded successfully")
            else:
                logger.warning("Some ML models are missing - will use fallback strategy")
                
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            self.models_loaded = False
            
    def generate_signal(self, 
                       df: pd.DataFrame, 
                       current_position: float = 0.0) -> Dict[str, Any]:
        """
        Generate trading signal using ML models.
        
        Args:
            df: OHLCV DataFrame with recent market data
            current_position: Current position size (-1 to 1)
            
        Returns:
            Dictionary containing signal, confidence, and additional info
        """
        try:
            if not self.models_loaded:
                logger.warning("Models not loaded - using fallback signal")
                return self._fallback_signal(df, current_position)
                
            if len(df) < self.lookback_period:
                logger.warning(f"Insufficient data for ML prediction: {len(df)} < {self.lookback_period}")
                return self._fallback_signal(df, current_position)
                
            # Compute features
            features = self.feature_engineer.compute_features(df)
            
            if len(features) == 0:
                logger.error("No features computed")
                return self._fallback_signal(df, current_position)
                
            # XGBoost signal prediction
            xgb_signal, xgb_confidence = self._predict_xgb_signal(features)
            
            # LSTM price prediction
            lstm_prediction = self._predict_lstm_price(df)
            
            # Combine signals
            combined_signal = self._combine_signals(xgb_signal, xgb_confidence, 
                                                  lstm_prediction, current_position)
            
            return combined_signal
            
        except Exception as e:
            logger.error(f"Error generating ML signal: {e}")
            return self._fallback_signal(df, current_position)
            
    def _predict_xgb_signal(self, features: np.ndarray) -> Tuple[int, float]:
        """
        Predict trading signal using XGBoost classifier.
        
        Args:
            features: Feature array
            
        Returns:
            Tuple of (signal, confidence)
        """
        try:
            # Use latest features
            latest_features = features[-1:].reshape(1, -1)
            
            # Scale features
            if self.feature_scaler is not None:
                latest_features = self.feature_scaler.transform(latest_features)
                
            # Predict signal and probability
            signal = self.xgb_model.predict(latest_features)[0]
            probabilities = self.xgb_model.predict_proba(latest_features)[0]
            
            # Get confidence as maximum probability
            confidence = np.max(probabilities)
            
            logger.debug(f"XGBoost prediction: signal={signal}, confidence={confidence:.3f}")
            
            return int(signal), float(confidence)
            
        except Exception as e:
            logger.error(f"Error in XGBoost prediction: {e}")
            return 0, 0.0  # Hold signal with zero confidence
            
    def _predict_lstm_price(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Predict next price using LSTM model.
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            Dictionary with price prediction info
        """
        try:
            # Prepare price sequence
            prices = df['close'].values[-self.lookback_period:]
            
            if self.price_scaler is not None:
                prices_scaled = self.price_scaler.transform(prices.reshape(-1, 1)).flatten()
            else:
                prices_scaled = prices
                
            # Reshape for LSTM input
            X = prices_scaled.reshape(1, self.lookback_period, 1)
            
            # Predict next price
            prediction_scaled = self.lstm_model.predict(X, verbose=0)[0, 0]
            
            # Inverse transform if scaler is available
            if self.price_scaler is not None:
                prediction = self.price_scaler.inverse_transform([[prediction_scaled]])[0, 0]
            else:
                prediction = prediction_scaled
                
            current_price = df['close'].iloc[-1]
            price_change_pct = (prediction - current_price) / current_price
            
            result = {
                'predicted_price': float(prediction),
                'current_price': float(current_price),
                'price_change_pct': float(price_change_pct),
                'direction': 1 if price_change_pct > 0 else -1
            }
            
            logger.debug(f"LSTM prediction: {result}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in LSTM prediction: {e}")
            return {
                'predicted_price': df['close'].iloc[-1],
                'current_price': df['close'].iloc[-1],
                'price_change_pct': 0.0,
                'direction': 0
            }
            
    def _combine_signals(self, 
                        xgb_signal: int, 
                        xgb_confidence: float,
                        lstm_prediction: Dict[str, float],
                        current_position: float) -> Dict[str, Any]:
        """
        Combine XGBoost and LSTM signals to generate final trading decision.
        
        Args:
            xgb_signal: XGBoost signal (-1, 0, 1)
            xgb_confidence: XGBoost confidence (0-1)
            lstm_prediction: LSTM price prediction dictionary
            current_position: Current position
            
        Returns:
            Combined signal dictionary
        """
        # Base signal from XGBoost
        final_signal = xgb_signal
        
        # Adjust confidence based on LSTM agreement
        lstm_direction = lstm_prediction['direction']
        price_change_magnitude = abs(lstm_prediction['price_change_pct'])
        
        # Check if LSTM agrees with XGBoost
        signals_agree = (xgb_signal * lstm_direction > 0) or (xgb_signal == 0)
        
        # Adjust confidence
        if signals_agree and price_change_magnitude > 0.001:  # 0.1% minimum change
            # Boost confidence when signals agree and price change is significant
            confidence_boost = min(0.2, price_change_magnitude * 10)
            final_confidence = min(1.0, xgb_confidence + confidence_boost)
        elif not signals_agree:
            # Reduce confidence when signals disagree
            final_confidence = xgb_confidence * 0.5
        else:
            final_confidence = xgb_confidence
            
        # Apply confidence threshold
        if final_confidence < self.confidence_threshold:
            final_signal = 0  # Hold if confidence is too low
            
        # Position sizing based on confidence
        if final_signal != 0:
            position_size = final_confidence * 0.5  # Max 50% of portfolio
        else:
            position_size = 0.0
            
        # Risk management: don't increase position in same direction if already positioned
        if current_position > 0.1 and final_signal > 0:
            final_signal = 0  # Don't add to long position
        elif current_position < -0.1 and final_signal < 0:
            final_signal = 0  # Don't add to short position
            
        result = {
            'signal': final_signal,
            'confidence': final_confidence,
            'position_size': position_size,
            'xgb_signal': xgb_signal,
            'xgb_confidence': xgb_confidence,
            'lstm_prediction': lstm_prediction,
            'signals_agree': signals_agree,
            'timestamp': datetime.now().isoformat(),
            'strategy': 'ml_based'
        }
        
        logger.info(f"ML Signal: {final_signal}, Confidence: {final_confidence:.3f}, "
                   f"Size: {position_size:.3f}")
        
        return result
        
    def _fallback_signal(self, df: pd.DataFrame, current_position: float) -> Dict[str, Any]:
        """
        Fallback signal generation when ML models are not available.
        Uses simple moving average crossover strategy.
        
        Args:
            df: OHLCV DataFrame
            current_position: Current position
            
        Returns:
            Fallback signal dictionary
        """
        try:
            if len(df) < 20:
                return {
                    'signal': 0,
                    'confidence': 0.0,
                    'position_size': 0.0,
                    'strategy': 'fallback_hold',
                    'timestamp': datetime.now().isoformat(),
                    'reason': 'insufficient_data'
                }
                
            # Simple moving average crossover
            ma_short = df['close'].rolling(window=10).mean()
            ma_long = df['close'].rolling(window=20).mean()
            
            # Current values
            current_short = ma_short.iloc[-1]
            current_long = ma_long.iloc[-1]
            prev_short = ma_short.iloc[-2]
            prev_long = ma_long.iloc[-2]
            
            # Crossover detection
            signal = 0
            confidence = 0.3  # Low confidence for fallback strategy
            
            if current_short > current_long and prev_short <= prev_long:
                signal = 1  # Golden cross - buy signal
            elif current_short < current_long and prev_short >= prev_long:
                signal = -1  # Death cross - sell signal
                
            # Simple position sizing
            position_size = 0.2 if signal != 0 else 0.0  # 20% position
            
            return {
                'signal': signal,
                'confidence': confidence,
                'position_size': position_size,
                'ma_short': float(current_short),
                'ma_long': float(current_long),
                'strategy': 'fallback_ma_crossover',
                'timestamp': datetime.now().isoformat(),
                'reason': 'ml_models_unavailable'
            }
            
        except Exception as e:
            logger.error(f"Error in fallback signal: {e}")
            return {
                'signal': 0,
                'confidence': 0.0,
                'position_size': 0.0,
                'strategy': 'fallback_error',
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
            
    def train_models(self, 
                    df: pd.DataFrame, 
                    signals: np.ndarray,
                    test_size: float = 0.2) -> Dict[str, Any]:
        """
        Train ML models on historical data.
        
        Args:
            df: Historical OHLCV data
            signals: Target signals (-1, 0, 1)
            test_size: Fraction of data for testing
            
        Returns:
            Training results dictionary
        """
        logger.info("Starting ML model training...")
        
        try:
            # Compute features
            features = self.feature_engineer.compute_features(df)
            
            if len(features) == 0:
                raise ValueError("No features computed")
                
            # Prepare data
            X = features[self.lookback_period:]  # Skip initial rows due to lookback
            y = signals[self.lookback_period:]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            
            # Train XGBoost model
            xgb_results = self._train_xgb_model(X_train, X_test, y_train, y_test)
            
            # Train LSTM model
            lstm_results = self._train_lstm_model(df)
            
            # Save models
            self._save_models()
            
            return {
                'xgb_results': xgb_results,
                'lstm_results': lstm_results,
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'feature_count': X.shape[1]
            }
            
        except Exception as e:
            logger.error(f"Error training models: {e}")
            return {'error': str(e)}
            
    def _train_xgb_model(self, X_train, X_test, y_train, y_test) -> Dict[str, Any]:
        """Train XGBoost classifier."""
        logger.info("Training XGBoost model...")
        
        # Scale features
        self.feature_scaler = RobustScaler()
        X_train_scaled = self.feature_scaler.fit_transform(X_train)
        X_test_scaled = self.feature_scaler.transform(X_test)
        
        # Train XGBoost
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric='mlogloss'
        )
        
        self.xgb_model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.xgb_model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"XGBoost training completed. Accuracy: {accuracy:.3f}")
        
        return {
            'accuracy': accuracy,
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
    def _train_lstm_model(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Train LSTM price prediction model."""
        logger.info("Training LSTM model...")
        
        # Prepare price sequences
        prices = df['close'].values
        
        # Scale prices
        self.price_scaler = StandardScaler()
        prices_scaled = self.price_scaler.fit_transform(prices.reshape(-1, 1)).flatten()
        
        # Create sequences
        X_sequences, y_sequences = self._create_lstm_sequences(prices_scaled)
        
        # Split data
        split_idx = int(len(X_sequences) * 0.8)
        X_train, X_test = X_sequences[:split_idx], X_sequences[split_idx:]
        y_train, y_test = y_sequences[:split_idx], y_sequences[split_idx:]
        
        # Build LSTM model
        self.lstm_model = tf.keras.Sequential([
            tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(self.lookback_period, 1)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(50, return_sequences=False),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(25),
            tf.keras.layers.Dense(1)
        ])
        
        self.lstm_model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Train model
        history = self.lstm_model.fit(
            X_train, y_train,
            batch_size=32,
            epochs=50,
            validation_data=(X_test, y_test),
            verbose=0
        )
        
        # Evaluate
        train_loss = history.history['loss'][-1]
        val_loss = history.history['val_loss'][-1]
        
        logger.info(f"LSTM training completed. Val Loss: {val_loss:.6f}")
        
        return {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'epochs': len(history.history['loss'])
        }
        
    def _create_lstm_sequences(self, prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training."""
        X, y = [], []
        
        for i in range(self.lookback_period, len(prices)):
            X.append(prices[i-self.lookback_period:i])
            y.append(prices[i])
            
        return np.array(X), np.array(y)
        
    def _save_models(self) -> None:
        """Save trained models and scalers."""
        try:
            os.makedirs(os.path.dirname(self.lstm_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.xgb_path), exist_ok=True)
            
            # Save LSTM model
            if self.lstm_model is not None:
                self.lstm_model.save(self.lstm_path)
                logger.info(f"Saved LSTM model to {self.lstm_path}")
                
            # Save XGBoost model
            if self.xgb_model is not None:
                joblib.dump(self.xgb_model, self.xgb_path)
                logger.info(f"Saved XGBoost model to {self.xgb_path}")
                
            # Save scalers
            scaler_dir = os.path.dirname(self.xgb_path)
            if self.feature_scaler is not None:
                joblib.dump(self.feature_scaler, os.path.join(scaler_dir, 'feature_scaler.pkl'))
                
            if self.price_scaler is not None:
                joblib.dump(self.price_scaler, os.path.join(scaler_dir, 'price_scaler.pkl'))
                
            logger.info("All models and scalers saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")


# Helper functions for model management
def load_lstm_model(model_path: str):
    """Load LSTM model from file."""
    try:
        if os.path.exists(model_path):
            return load_model(model_path)
        else:
            logger.warning(f"LSTM model not found at {model_path}")
            return None
    except Exception as e:
        logger.error(f"Error loading LSTM model: {e}")
        return None


def load_xgb_model(model_path: str):
    """Load XGBoost model from file."""
    try:
        if os.path.exists(model_path):
            return joblib.load(model_path)
        else:
            logger.warning(f"XGBoost model not found at {model_path}")
            return None
    except Exception as e:
        logger.error(f"Error loading XGBoost model: {e}")
        return None


# Example usage and testing
if __name__ == "__main__":
    """
    Example usage of MLTradingStrategy for testing.
    """
    # Mock configuration
    config = {
        'models': {
            'lstm_path': 'models/lstm_price_forecast.keras',
            'xgb_path': 'models/xgb_trade_signal.pkl'
        },
        'strategy': {
            'ml_model': {
                'confidence_threshold': 0.6,
                'lstm_lookback': 60,
                'xgb_features': 20
            }
        }
    }
    
    # Create sample data
    dates = pd.date_range(start='2024-01-01', periods=200, freq='1min')
    np.random.seed(42)
    
    prices = np.random.random(200).cumsum() + 50000
    df = pd.DataFrame({
        'open': prices,
        'high': prices * 1.001,
        'low': prices * 0.999,
        'close': prices,
        'volume': np.random.lognormal(10, 0.5, 200)
    }, index=dates)
    
    print("📊 Testing ML Trading Strategy")
    print(f"Sample data shape: {df.shape}")
    
    # Initialize strategy
    strategy = MLTradingStrategy(config)
    
    # Generate signal
    print("\n🔮 Generating ML signal...")
    signal = strategy.generate_signal(df)
    
    print("Signal result:")
    for key, value in signal.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
            
    # Test fallback signal
    print("\n🔄 Testing fallback signal...")
    strategy.models_loaded = False
    fallback_signal = strategy.generate_signal(df)
    
    print("Fallback signal result:")
    for key, value in fallback_signal.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
            
    print("\n✅ ML Trading Strategy test completed")
