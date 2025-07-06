"""
Feature engineering module for technical indicators and market features.
Computes various technical indicators used by ML models and trading strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import talib
from loguru import logger
import warnings

warnings.filterwarnings('ignore')


class FeatureEngineer:
    """
    Handles feature engineering for trading strategies and ML models.
    Computes technical indicators, price patterns, and market microstructure features.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize feature engineer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
    def compute_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Compute comprehensive feature set from OHLCV data.
        
        Args:
            df: DataFrame with OHLCV data (columns: open, high, low, close, volume)
            
        Returns:
            Feature array ready for ML models
        """
        logger.debug(f"Computing features for {len(df)} candles")
        
        if len(df) < 50:
            logger.warning(f"Insufficient data for feature computation: {len(df)} candles")
            return np.array([])
            
        try:
            # Create feature DataFrame
            features_df = pd.DataFrame(index=df.index)
            
            # Price-based features
            features_df = self._add_price_features(features_df, df)
            
            # Moving averages
            features_df = self._add_moving_average_features(features_df, df)
            
            # Momentum indicators
            features_df = self._add_momentum_features(features_df, df)
            
            # Volatility indicators
            features_df = self._add_volatility_features(features_df, df)
            
            # Volume indicators
            features_df = self._add_volume_features(features_df, df)
            
            # Pattern recognition
            features_df = self._add_pattern_features(features_df, df)
            
            # Market microstructure
            features_df = self._add_microstructure_features(features_df, df)
            
            # Time-based features
            features_df = self._add_time_features(features_df, df)
            
            # Clean features
            features_df = self._clean_features(features_df)
            
            logger.debug(f"Generated {features_df.shape[1]} features")
            return features_df
            
        except Exception as e:
            logger.error(f"Error computing features: {e}")
            return pd.DataFrame()
            
    def _add_price_features(self, features_df: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic price-based features."""
        # Price changes
        features_df['price_change'] = df['close'].pct_change()
        features_df['price_change_abs'] = np.abs(features_df['price_change'])
        
        # Log returns
        features_df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        
        # High-Low spread
        features_df['hl_spread'] = (df['high'] - df['low']) / df['close']
        
        # Open-Close spread
        features_df['oc_spread'] = (df['close'] - df['open']) / df['open']
        
        # Price position within candle
        features_df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        # Price momentum
        for period in [3, 5, 10]:
            features_df[f'momentum_{period}'] = df['close'] / df['close'].shift(period) - 1
            
        return features_df
        
    def _add_moving_average_features(self, features_df: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Add moving average based features."""
        periods = [5, 10, 20, 50]
        
        for period in periods:
            if len(df) >= period:
                # Simple Moving Average
                ma = df['close'].rolling(window=period).mean()
                features_df[f'sma_{period}'] = df['close'] / ma - 1
                
                # Exponential Moving Average
                ema = df['close'].ewm(span=period).mean()
                features_df[f'ema_{period}'] = df['close'] / ema - 1
                
                # Moving average slope
                features_df[f'ma_slope_{period}'] = ma.pct_change(periods=3)
                
        # Moving average crossovers
        if len(df) >= 50:
            ma_fast = df['close'].rolling(window=10).mean()
            ma_slow = df['close'].rolling(window=50).mean()
            features_df['ma_crossover'] = (ma_fast > ma_slow).astype(int)
            features_df['ma_distance'] = (ma_fast - ma_slow) / ma_slow
            
        return features_df
        
    def _add_momentum_features(self, features_df: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum-based technical indicators."""
        # RSI
        if len(df) >= 14:
            features_df['rsi'] = talib.RSI(df['close'].values, timeperiod=14)
            features_df['rsi_oversold'] = (features_df['rsi'] < 30).astype(int)
            features_df['rsi_overbought'] = (features_df['rsi'] > 70).astype(int)
            
        # MACD
        if len(df) >= 26:
            macd, macd_signal, macd_hist = talib.MACD(df['close'].values, 
                                                     fastperiod=12, 
                                                     slowperiod=26, 
                                                     signalperiod=9)
            features_df['macd'] = macd
            features_df['macd_signal'] = macd_signal
            features_df['macd_histogram'] = macd_hist
            features_df['macd_crossover'] = (macd > macd_signal).astype(int)
            
        # Stochastic
        if len(df) >= 14:
            slowk, slowd = talib.STOCH(df['high'].values, 
                                      df['low'].values, 
                                      df['close'].values,
                                      fastk_period=14, 
                                      slowk_period=3, 
                                      slowd_period=3)
            features_df['stoch_k'] = slowk
            features_df['stoch_d'] = slowd
            features_df['stoch_oversold'] = (slowk < 20).astype(int)
            features_df['stoch_overbought'] = (slowk > 80).astype(int)
            
        # Williams %R
        if len(df) >= 14:
            features_df['williams_r'] = talib.WILLR(df['high'].values, 
                                                   df['low'].values, 
                                                   df['close'].values, 
                                                   timeperiod=14)
            
        # Commodity Channel Index
        if len(df) >= 20:
            features_df['cci'] = talib.CCI(df['high'].values, 
                                          df['low'].values, 
                                          df['close'].values, 
                                          timeperiod=20)
            
        return features_df
        
    def _add_volatility_features(self, features_df: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility-based indicators."""
        # Bollinger Bands
        if len(df) >= 20:
            bb_upper, bb_middle, bb_lower = talib.BBANDS(df['close'].values, 
                                                        timeperiod=20, 
                                                        nbdevup=2, 
                                                        nbdevdn=2)
            features_df['bb_upper'] = bb_upper
            features_df['bb_lower'] = bb_lower
            features_df['bb_width'] = (bb_upper - bb_lower) / bb_middle
            features_df['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)
            
        # Average True Range
        if len(df) >= 14:
            features_df['atr'] = talib.ATR(df['high'].values, 
                                          df['low'].values, 
                                          df['close'].values, 
                                          timeperiod=14)
            features_df['atr_ratio'] = features_df['atr'] / df['close']
            
        # Historical volatility
        for period in [10, 20]:
            if len(df) >= period:
                returns = df['close'].pct_change()
                features_df[f'volatility_{period}'] = returns.rolling(window=period).std()
                
        return features_df
        
    def _add_volume_features(self, features_df: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based indicators."""
        # Volume changes
        features_df['volume_change'] = df['volume'].pct_change()
        features_df['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()
        
        # On-Balance Volume
        features_df['obv'] = talib.OBV(df['close'].values, df['volume'].values)
        features_df['obv_ma'] = features_df['obv'].rolling(window=10).mean()
        
        # Volume-Price Trend
        features_df['vpt'] = talib.TRIX(df['close'].values, timeperiod=14)
        
        # Accumulation/Distribution Line
        features_df['ad_line'] = talib.AD(df['high'].values, 
                                         df['low'].values, 
                                         df['close'].values, 
                                         df['volume'].values)
        
        # Money Flow Index
        if len(df) >= 14:
            features_df['mfi'] = talib.MFI(df['high'].values, 
                                          df['low'].values, 
                                          df['close'].values, 
                                          df['volume'].values, 
                                          timeperiod=14)
            
        # Volume Weighted Average Price approximation
        if len(df) >= 20:
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            vwap = (typical_price * df['volume']).rolling(window=20).sum() / df['volume'].rolling(window=20).sum()
            features_df['vwap_ratio'] = df['close'] / vwap - 1
            
        return features_df
        
    def _add_pattern_features(self, features_df: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Add candlestick pattern recognition features."""
        if len(df) >= 10:
            # Doji patterns
            features_df['doji'] = talib.CDLDOJI(df['open'].values, 
                                               df['high'].values, 
                                               df['low'].values, 
                                               df['close'].values)
            
            # Hammer patterns
            features_df['hammer'] = talib.CDLHAMMER(df['open'].values, 
                                                   df['high'].values, 
                                                   df['low'].values, 
                                                   df['close'].values)
            
            # Engulfing patterns
            features_df['engulfing'] = talib.CDLENGULFING(df['open'].values, 
                                                         df['high'].values, 
                                                         df['low'].values, 
                                                         df['close'].values)
            
            # Morning/Evening Star
            features_df['morning_star'] = talib.CDLMORNINGSTAR(df['open'].values, 
                                                              df['high'].values, 
                                                              df['low'].values, 
                                                              df['close'].values)
            
            features_df['evening_star'] = talib.CDLEVENINGSTAR(df['open'].values, 
                                                              df['high'].values, 
                                                              df['low'].values, 
                                                              df['close'].values)
            
        return features_df
        
    def _add_microstructure_features(self, features_df: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Add market microstructure features."""
        # Price gaps
        features_df['gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        features_df['gap_up'] = (features_df['gap'] > 0.001).astype(int)
        features_df['gap_down'] = (features_df['gap'] < -0.001).astype(int)
        
        # Intraday returns
        features_df['intraday_return'] = (df['close'] - df['open']) / df['open']
        features_df['overnight_return'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        
        # High-Low volatility
        features_df['hl_volatility'] = (df['high'] - df['low']) / df['open']
        
        # Volume-weighted returns
        features_df['vw_return'] = features_df['price_change'] * np.log(df['volume'] + 1)
        
        return features_df
        
    def _add_time_features(self, features_df: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features."""
        # Hour of day (for intraday patterns)
        features_df['hour'] = df.index.hour
        features_df['minute'] = df.index.minute
        
        # Day of week
        features_df['day_of_week'] = df.index.dayofweek
        
        # Market session (assuming UTC timezone)
        features_df['asian_session'] = ((df.index.hour >= 0) & (df.index.hour < 8)).astype(int)
        features_df['european_session'] = ((df.index.hour >= 8) & (df.index.hour < 16)).astype(int)
        features_df['us_session'] = ((df.index.hour >= 16) & (df.index.hour < 24)).astype(int)
        
        return features_df
        
    def _clean_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalize features."""
        # Fill NaN values
        features_df = features_df.fillna(method='ffill').fillna(0)
        
        # Remove infinite values
        features_df = features_df.replace([np.inf, -np.inf], 0)
        
        # Clip extreme values (beyond 3 standard deviations)
        for column in features_df.columns:
            if features_df[column].dtype in ['float64', 'float32']:
                mean = features_df[column].mean()
                std = features_df[column].std()
                if std > 0:
                    features_df[column] = features_df[column].clip(
                        lower=mean - 3 * std,
                        upper=mean + 3 * std
                    )
        
        return features_df
        
    def get_feature_names(self) -> List[str]:
        """
        Get list of feature names.
        
        Returns:
            List of feature names
        """
        # This would return the actual feature names from the last computation
        # For now, return a representative list
        feature_names = [
            'price_change', 'price_change_abs', 'log_return', 'hl_spread', 'oc_spread',
            'price_position', 'momentum_3', 'momentum_5', 'momentum_10',
            'sma_5', 'sma_10', 'sma_20', 'sma_50', 'ema_5', 'ema_10', 'ema_20', 'ema_50',
            'ma_slope_5', 'ma_slope_10', 'ma_slope_20', 'ma_slope_50',
            'ma_crossover', 'ma_distance', 'rsi', 'rsi_oversold', 'rsi_overbought',
            'macd', 'macd_signal', 'macd_histogram', 'macd_crossover',
            'stoch_k', 'stoch_d', 'stoch_oversold', 'stoch_overbought',
            'williams_r', 'cci', 'bb_upper', 'bb_lower', 'bb_width', 'bb_position',
            'atr', 'atr_ratio', 'volatility_10', 'volatility_20',
            'volume_change', 'volume_ma_ratio', 'obv', 'obv_ma', 'vpt', 'ad_line',
            'mfi', 'vwap_ratio', 'doji', 'hammer', 'engulfing', 'morning_star', 'evening_star',
            'gap', 'gap_up', 'gap_down', 'intraday_return', 'overnight_return',
            'hl_volatility', 'vw_return', 'hour', 'minute', 'day_of_week',
            'asian_session', 'european_session', 'us_session'
        ]
        return feature_names
        
    def compute_single_features(self, df: pd.DataFrame, feature_list: List[str]) -> pd.DataFrame:
        """
        Compute specific features only.
        
        Args:
            df: OHLCV DataFrame
            feature_list: List of specific features to compute
            
        Returns:
            DataFrame with requested features
        """
        features_df = pd.DataFrame(index=df.index)
        
        for feature in feature_list:
            if 'rsi' in feature and len(df) >= 14:
                features_df['rsi'] = talib.RSI(df['close'].values, timeperiod=14)
            elif 'macd' in feature and len(df) >= 26:
                macd, macd_signal, macd_hist = talib.MACD(df['close'].values)
                features_df['macd'] = macd
                features_df['macd_signal'] = macd_signal
                features_df['macd_histogram'] = macd_hist
            elif 'sma' in feature:
                period = int(feature.split('_')[1])
                if len(df) >= period:
                    features_df[feature] = df['close'].rolling(window=period).mean()
            # Add more specific feature computations as needed
            
        return features_df.fillna(method='ffill').fillna(0)


# Example usage and testing
if __name__ == "__main__":
    """
    Example usage of FeatureEngineer for testing.
    """
    # Create sample OHLCV data
    import datetime
    
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1min')
    np.random.seed(42)
    
    # Generate realistic price data
    initial_price = 50000
    price_changes = np.random.normal(0, 0.001, 100)
    prices = [initial_price]
    
    for change in price_changes[1:]:
        prices.append(prices[-1] * (1 + change))
    
    # Create OHLCV DataFrame
    df = pd.DataFrame({
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.0005))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.0005))) for p in prices],
        'close': prices,
        'volume': np.random.lognormal(10, 0.5, 100)
    }, index=dates)
    
    # Make sure high >= low and open, close are within high-low range
    df['high'] = np.maximum(df['high'], np.maximum(df['open'], df['close']))
    df['low'] = np.minimum(df['low'], np.minimum(df['open'], df['close']))
    
    print("📊 Sample OHLCV data:")
    print(df.head())
    print(f"\nData shape: {df.shape}")
    
    # Initialize feature engineer
    feature_engineer = FeatureEngineer()
    
    # Compute features
    print("\n🔧 Computing features...")
    features = feature_engineer.compute_features(df)
    
    print(f"Feature matrix shape: {features.shape}")
    print(f"Number of features: {features.shape[1] if len(features.shape) > 1 else 0}")
    
    # Get feature names
    feature_names = feature_engineer.get_feature_names()
    print(f"Available features: {len(feature_names)}")
    print("First 10 features:", feature_names[:10])
    
    if len(features) > 0:
        print(f"\nFeature statistics:")
        features_df = pd.DataFrame(features, columns=feature_names[:features.shape[1]])
        print(features_df.describe())
        
        # Check for NaN or infinite values
        nan_count = np.isnan(features).sum()
        inf_count = np.isinf(features).sum()
        print(f"\nData quality check:")
        print(f"NaN values: {nan_count}")
        print(f"Infinite values: {inf_count}")
        
        if nan_count == 0 and inf_count == 0:
            print("✅ Features are clean and ready for ML models")
        else:
            print("⚠️ Features contain NaN or infinite values")
    else:
        print("❌ No features computed - check input data")
