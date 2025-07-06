#!/usr/bin/env python3
"""
Training script for ML models used in the crypto trading AI agent.
Creates initial LSTM and XGBoost models for price prediction and signal generation.
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta
import pickle
import json
from typing import Tuple, Dict, Any
from loguru import logger

# ML libraries
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split
    import xgboost as xgb
except ImportError as e:
    logger.error(f"Missing ML dependencies: {e}")
    sys.exit(1)

# Local imports
from trading.exchange import ExchangeManager
from features.feature_engineering import FeatureEngineer


class ModelTrainer:
    """
    Trains and saves ML models for the trading system.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the model trainer."""
        self.exchange_manager = ExchangeManager(config_path)
        self.feature_engineer = FeatureEngineer()
        
        # Create models directory
        os.makedirs("models", exist_ok=True)
        
        # Model configurations
        self.lstm_config = {
            'sequence_length': 60,
            'features': ['close', 'volume', 'rsi', 'macd'],
            'epochs': 50,
            'batch_size': 32,
            'validation_split': 0.2
        }
        
        self.xgb_config = {
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        
    def fetch_training_data(self, days: int = 90) -> pd.DataFrame:
        """
        Fetch historical data for training.
        
        Args:
            days: Number of days of historical data to fetch
            
        Returns:
            DataFrame with historical OHLCV data
        """
        logger.info(f"Fetching {days} days of training data...")
        
        try:
            # Fetch historical data
            df = self.exchange_manager.fetch_historical_data(
                timeframe='1h',  # Use hourly data for training
                days=days
            )
            
            if df.empty:
                logger.error("No training data available")
                return pd.DataFrame()
                
            logger.info(f"Fetched {len(df)} candles for training")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching training data: {e}")
            # Create synthetic data as fallback
            logger.info("Creating synthetic training data as fallback...")
            return self._create_synthetic_data(days)
            
    def _create_synthetic_data(self, days: int) -> pd.DataFrame:
        """
        Create synthetic OHLCV data for training when real data is unavailable.
        
        Args:
            days: Number of days to simulate
            
        Returns:
            DataFrame with synthetic OHLCV data
        """
        logger.info("Generating synthetic training data...")
        
        # Create datetime index
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        date_range = pd.date_range(start=start_time, end=end_time, freq='1H')
        
        # Generate realistic price movement
        np.random.seed(42)
        n_periods = len(date_range)
        
        # Start with base price
        base_price = 45000.0  # BTC-like price
        
        # Generate random walk with trend and volatility
        returns = np.random.normal(0.0001, 0.02, n_periods)  # Small positive drift with 2% volatility
        price_multipliers = np.exp(np.cumsum(returns))
        
        # Calculate OHLC from close prices
        close_prices = base_price * price_multipliers
        
        # Generate OHLC with realistic relationships
        high_noise = np.random.uniform(1.0, 1.02, n_periods)
        low_noise = np.random.uniform(0.98, 1.0, n_periods)
        open_noise = np.random.uniform(0.995, 1.005, n_periods)
        
        high_prices = close_prices * high_noise
        low_prices = close_prices * low_noise
        open_prices = np.roll(close_prices, 1) * open_noise
        open_prices[0] = close_prices[0]  # First open = first close
        
        # Ensure OHLC relationships are maintained
        for i in range(n_periods):
            high_prices[i] = max(high_prices[i], open_prices[i], close_prices[i])
            low_prices[i] = min(low_prices[i], open_prices[i], close_prices[i])
        
        # Generate volume data
        volume = np.random.lognormal(mean=2.0, sigma=0.5, size=n_periods) * 100
        
        # Create DataFrame
        df = pd.DataFrame({
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volume,
            'timestamp': [int(dt.timestamp() * 1000) for dt in date_range]
        }, index=date_range)
        
        logger.info(f"Generated {len(df)} synthetic candles")
        return df
        
    def prepare_lstm_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
        """
        Prepare data for LSTM training.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Tuple of (X, y, scaler) for training
        """
        logger.info("Preparing LSTM training data...")
        
        # Compute features
        df_features = self.feature_engineer.compute_features(df)
        
        # If features are empty, use original OHLCV data
        if df_features.empty:
            logger.warning("No features computed, using original OHLCV data")
            df_features = df[['close', 'volume']].copy()
        
        # First, add original OHLCV data to features if not present
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col not in df_features.columns and col in df.columns:
                df_features[col] = df[col]
        
        # Select features for LSTM - start with basic price/volume
        feature_columns = []
        
        # Add basic OHLCV features if available
        for col in ['close', 'volume']:
            if col in df_features.columns:
                feature_columns.append(col)
        
        # Add technical indicators if available
        available_features = []
        for col in ['rsi', 'macd', 'bb_middle', 'sma_20', 'ema_12']:
            if col in df_features.columns:
                available_features.append(col)
        
        feature_columns.extend(available_features[:3])  # Limit to prevent overfitting
        
        # Ensure we have at least one feature
        if not feature_columns:
            logger.error("No suitable features found for LSTM training")
            raise ValueError("No features available for LSTM training")
        
        logger.info(f"Using features for LSTM: {feature_columns}")
        
        # Prepare data
        data = df_features[feature_columns].dropna().values
        
        # Scale the data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)
        
        # Create sequences
        sequence_length = self.lstm_config['sequence_length']
        X, y = [], []
        
        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i-sequence_length:i])
            y.append(scaled_data[i, 0])  # Predict close price (first feature)
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"Prepared LSTM data: X shape {X.shape}, y shape {y.shape}")
        return X, y, scaler
        
    def train_lstm_model(self, X: np.ndarray, y: np.ndarray) -> tf.keras.Model:
        """
        Train LSTM model for price prediction.
        
        Args:
            X: Input sequences
            y: Target values
            
        Returns:
            Trained LSTM model
        """
        logger.info("Training LSTM model...")
        
        # Build model
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        # Train model
        history = model.fit(
            X, y,
            epochs=self.lstm_config['epochs'],
            batch_size=self.lstm_config['batch_size'],
            validation_split=self.lstm_config['validation_split'],
            verbose=1,
            shuffle=False  # Important for time series
        )
        
        logger.info("LSTM training completed")
        return model
        
    def prepare_xgboost_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for XGBoost training.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Tuple of (X, y) for training
        """
        logger.info("Preparing XGBoost training data...")
        
        # Compute features
        df_features = self.feature_engineer.compute_features(df)
        
        # If features are empty, use original OHLCV data
        if df_features.empty:
            logger.warning("No features computed, using original OHLCV data")
            df_features = df.copy()
        
        # Add original OHLCV data to features if not present
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col not in df_features.columns and col in df.columns:
                df_features[col] = df[col]
        
        # Ensure we have close price for target calculation
        if 'close' not in df_features.columns:
            logger.error("Close price not available for target calculation")
            raise ValueError("Close price required for XGBoost training")
        
        # Create target variable (future price direction)
        df_features['future_return'] = df_features['close'].pct_change().shift(-1)
        # Convert signals to 0, 1, 2 for XGBoost compatibility
        df_features['signal'] = np.where(df_features['future_return'] > 0.002, 2,  # Buy signal (class 2)
                                       np.where(df_features['future_return'] < -0.002, 0, 1))  # Sell/Hold signals (class 0/1)
        
        # Select features for XGBoost (exclude price-based features to avoid data leakage)
        feature_columns = []
        exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'timestamp', 'future_return', 'signal']
        
        for col in df_features.columns:
            if col not in exclude_cols and not df_features[col].isna().all():
                feature_columns.append(col)
        
        # Ensure we have at least some features
        if len(feature_columns) < 3:
            # Add some basic derived features if we don't have enough
            logger.info("Not enough technical features, adding basic derived features")
            if 'close' in df_features.columns:
                df_features['price_change'] = df_features['close'].pct_change()
                df_features['price_volatility'] = df_features['close'].rolling(10).std()
                feature_columns.extend(['price_change', 'price_volatility'])
        
        # If still no features, use some basic ones
        if not feature_columns:
            logger.warning("No technical features available, using basic price ratios")
            if all(col in df_features.columns for col in ['high', 'low', 'close']):
                df_features['hl_ratio'] = (df_features['high'] - df_features['low']) / df_features['close']
                feature_columns.append('hl_ratio')
        
        if not feature_columns:
            logger.error("No suitable features found for XGBoost training")
            raise ValueError("No features available for XGBoost training")
        
        logger.info(f"Using {len(feature_columns)} features for XGBoost: {feature_columns[:5]}...")
        
        # Prepare data
        data = df_features[feature_columns + ['signal']].dropna()
        
        X = data[feature_columns].values
        y = data['signal'].values
        
        logger.info(f"Prepared XGBoost data: X shape {X.shape}, y shape {y.shape}")
        logger.info(f"Signal distribution: {np.bincount(y)}")  # Now handles 0, 1, 2
        
        return X, y
        
    def train_xgboost_model(self, X: np.ndarray, y: np.ndarray) -> xgb.XGBClassifier:
        """
        Train XGBoost model for signal generation.
        
        Args:
            X: Feature matrix
            y: Target signals
            
        Returns:
            Trained XGBoost model
        """
        logger.info("Training XGBoost model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        model = xgb.XGBClassifier(**self.xgb_config)
        model.fit(X_train, y_train)
        
        # Evaluate
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        logger.info(f"XGBoost training completed")
        logger.info(f"Train accuracy: {train_score:.3f}")
        logger.info(f"Test accuracy: {test_score:.3f}")
        
        return model
        
    def save_models(self, lstm_model: tf.keras.Model, xgb_model: xgb.XGBClassifier, 
                   lstm_scaler: MinMaxScaler) -> None:
        """
        Save trained models to disk.
        
        Args:
            lstm_model: Trained LSTM model
            xgb_model: Trained XGBoost model
            lstm_scaler: Scaler for LSTM data preprocessing
        """
        logger.info("Saving models...")
        
        try:
            # Save LSTM model
            lstm_model.save("models/lstm_price_forecast.keras")
            logger.info("✅ LSTM model saved to models/lstm_price_forecast.keras")
            
            # Save XGBoost model
            with open("models/xgb_trade_signal.pkl", "wb") as f:
                pickle.dump(xgb_model, f)
            logger.info("✅ XGBoost model saved to models/xgb_trade_signal.pkl")
            
            # Save LSTM scaler
            with open("models/lstm_scaler.pkl", "wb") as f:
                pickle.dump(lstm_scaler, f)
            logger.info("✅ LSTM scaler saved to models/lstm_scaler.pkl")
            
            # Save model metadata
            metadata = {
                "lstm": {
                    "sequence_length": self.lstm_config['sequence_length'],
                    "features": self.lstm_config['features'],
                    "trained_date": datetime.now().isoformat()
                },
                "xgboost": {
                    "config": self.xgb_config,
                    "trained_date": datetime.now().isoformat()
                }
            }
            
            with open("models/model_metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
            logger.info("✅ Model metadata saved to models/model_metadata.json")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
            raise
            
    def train_all_models(self) -> None:
        """Train and save all ML models."""
        logger.info("🚀 Starting ML model training...")
        
        try:
            # Fetch training data
            df = self.fetch_training_data(days=90)
            
            if df.empty:
                logger.error("No training data available")
                return
                
            # Train LSTM model
            logger.info("📈 Training LSTM model for price prediction...")
            X_lstm, y_lstm, lstm_scaler = self.prepare_lstm_data(df)
            lstm_model = self.train_lstm_model(X_lstm, y_lstm)
            
            # Train XGBoost model
            logger.info("🎯 Training XGBoost model for signal generation...")
            X_xgb, y_xgb = self.prepare_xgboost_data(df)
            xgb_model = self.train_xgboost_model(X_xgb, y_xgb)
            
            # Save models
            self.save_models(lstm_model, xgb_model, lstm_scaler)
            
            logger.info("🎉 All models trained and saved successfully!")
            
        except Exception as e:
            logger.error(f"Error in model training: {e}")
            raise


def main():
    """Main function to run model training."""
    logger.info("🤖 Crypto Trading AI - Model Training")
    logger.info("=" * 50)
    
    try:
        trainer = ModelTrainer()
        trainer.train_all_models()
        
        print("\n✅ Model training completed successfully!")
        print("\nModels saved:")
        print("- models/lstm_price_forecast.keras (LSTM price prediction)")
        print("- models/xgb_trade_signal.pkl (XGBoost signal generation)")
        print("- models/lstm_scaler.pkl (LSTM data scaler)")
        print("- models/model_metadata.json (Model metadata)")
        
        print("\nYou can now run the trading system with ML models:")
        print("python main.py --mode paper --strategy ml_based")
        
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
