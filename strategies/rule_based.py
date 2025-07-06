"""
Rule-based trading strategy using traditional technical indicators.
Implements moving average crossover, RSI, and MACD strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from loguru import logger
import talib

# Local imports
from features.feature_engineering import FeatureEngineer


class RuleBasedStrategy:
    """
    Rule-based trading strategy using technical analysis indicators.
    Combines multiple indicators for signal generation with risk management.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize rule-based strategy.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.feature_engineer = FeatureEngineer(config)
        
        # Strategy parameters from config
        strategy_config = config['strategy']['rule_based']
        self.ma_fast = strategy_config['ma_fast']
        self.ma_slow = strategy_config['ma_slow']
        self.rsi_period = strategy_config['rsi_period']
        self.rsi_oversold = strategy_config['rsi_oversold']
        self.rsi_overbought = strategy_config['rsi_overbought']
        
        # Risk management
        risk_config = config['trading']['risk_management']
        self.max_position_pct = risk_config['max_position_size_pct']
        self.stop_loss_pct = risk_config['stop_loss_pct']
        self.take_profit_pct = risk_config['take_profit_pct']
        
        # Internal state
        self.last_signal = 0
        self.signal_history = []
        
    def generate_signal(self, 
                       df: pd.DataFrame, 
                       current_position: float = 0.0,
                       entry_price: Optional[float] = None) -> Dict[str, Any]:
        """
        Generate trading signal using rule-based strategy.
        
        Args:
            df: OHLCV DataFrame with recent market data
            current_position: Current position size (-1 to 1)
            entry_price: Entry price for current position (if any)
            
        Returns:
            Dictionary containing signal, confidence, and additional info
        """
        try:
            if len(df) < max(self.ma_slow, self.rsi_period, 26):  # 26 for MACD
                logger.warning(f"Insufficient data for rule-based strategy: {len(df)}")
                return self._no_signal_response("insufficient_data")
                
            current_price = df['close'].iloc[-1]
            
            # Check for stop-loss or take-profit if we have a position
            if current_position != 0 and entry_price is not None:
                exit_signal = self._check_exit_conditions(current_price, entry_price, current_position)
                if exit_signal['signal'] != 0:
                    return exit_signal
                    
            # Compute technical indicators
            indicators = self._compute_indicators(df)
            
            # Generate signals from each strategy component
            ma_signal = self._moving_average_signal(indicators)
            rsi_signal = self._rsi_signal(indicators)
            macd_signal = self._macd_signal(indicators)
            volume_signal = self._volume_signal(df, indicators)
            
            # Combine signals
            combined_signal = self._combine_signals(ma_signal, rsi_signal, macd_signal, volume_signal)
            
            # Apply position management rules
            final_signal = self._apply_position_rules(combined_signal, current_position)
            
            # Calculate confidence and position size
            confidence = self._calculate_confidence(ma_signal, rsi_signal, macd_signal, volume_signal)
            position_size = self._calculate_position_size(final_signal['signal'], confidence)
            
            result = {
                'signal': final_signal['signal'],
                'confidence': confidence,
                'position_size': position_size,
                'entry_price': current_price if final_signal['signal'] != 0 else None,
                'stop_loss': self._calculate_stop_loss(current_price, final_signal['signal']),
                'take_profit': self._calculate_take_profit(current_price, final_signal['signal']),
                'indicators': indicators,
                'component_signals': {
                    'ma_signal': ma_signal,
                    'rsi_signal': rsi_signal,
                    'macd_signal': macd_signal,
                    'volume_signal': volume_signal
                },
                'strategy': 'rule_based',
                'timestamp': datetime.now().isoformat(),
                'reason': final_signal['reason']
            }
            
            # Update internal state
            self.last_signal = final_signal['signal']
            self.signal_history.append(result)
            
            # Keep only last 100 signals in history
            if len(self.signal_history) > 100:
                self.signal_history.pop(0)
                
            logger.info(f"Rule-based Signal: {final_signal['signal']}, "
                       f"Confidence: {confidence:.3f}, Size: {position_size:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating rule-based signal: {e}")
            return self._no_signal_response(f"error: {e}")
            
    def _compute_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute technical indicators."""
        indicators = {}
        
        # Moving Averages
        indicators['ma_fast'] = df['close'].rolling(window=self.ma_fast).mean()
        indicators['ma_slow'] = df['close'].rolling(window=self.ma_slow).mean()
        
        # RSI
        indicators['rsi'] = talib.RSI(df['close'].values, timeperiod=self.rsi_period)
        
        # MACD
        macd, macd_signal, macd_hist = talib.MACD(df['close'].values, 
                                                 fastperiod=12, 
                                                 slowperiod=26, 
                                                 signalperiod=9)
        indicators['macd'] = macd
        indicators['macd_signal'] = macd_signal
        indicators['macd_histogram'] = macd_hist
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = talib.BBANDS(df['close'].values, 
                                                    timeperiod=20, 
                                                    nbdevup=2, 
                                                    nbdevdn=2)
        indicators['bb_upper'] = bb_upper
        indicators['bb_middle'] = bb_middle
        indicators['bb_lower'] = bb_lower
        
        # Volume indicators
        indicators['volume_sma'] = df['volume'].rolling(window=20).mean()
        indicators['obv'] = talib.OBV(df['close'].values, df['volume'].values)
        
        # ATR for volatility
        indicators['atr'] = talib.ATR(df['high'].values, 
                                     df['low'].values, 
                                     df['close'].values, 
                                     timeperiod=14)
        
        return indicators
        
    def _moving_average_signal(self, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Generate signal from moving average crossover."""
        ma_fast_current = indicators['ma_fast'].iloc[-1]
        ma_slow_current = indicators['ma_slow'].iloc[-1]
        ma_fast_prev = indicators['ma_fast'].iloc[-2]
        ma_slow_prev = indicators['ma_slow'].iloc[-2]
        
        signal = 0
        strength = 0.0
        reason = "ma_neutral"
        
        # Golden Cross (bullish)
        if ma_fast_current > ma_slow_current and ma_fast_prev <= ma_slow_prev:
            signal = 1
            strength = 0.8
            reason = "golden_cross"
        # Death Cross (bearish)
        elif ma_fast_current < ma_slow_current and ma_fast_prev >= ma_slow_prev:
            signal = -1
            strength = 0.8
            reason = "death_cross"
        # Trend continuation
        elif ma_fast_current > ma_slow_current:
            # Check if the gap is widening (strong uptrend)
            current_gap = (ma_fast_current - ma_slow_current) / ma_slow_current
            prev_gap = (ma_fast_prev - ma_slow_prev) / ma_slow_prev
            if current_gap > prev_gap and current_gap > 0.01:  # 1% gap
                signal = 1
                strength = 0.5
                reason = "uptrend_strengthening"
        elif ma_fast_current < ma_slow_current:
            # Check if the gap is widening (strong downtrend)
            current_gap = (ma_slow_current - ma_fast_current) / ma_slow_current
            prev_gap = (ma_slow_prev - ma_fast_prev) / ma_slow_prev
            if current_gap > prev_gap and current_gap > 0.01:  # 1% gap
                signal = -1
                strength = 0.5
                reason = "downtrend_strengthening"
                
        return {
            'signal': signal,
            'strength': strength,
            'reason': reason,
            'ma_fast': ma_fast_current,
            'ma_slow': ma_slow_current
        }
        
    def _rsi_signal(self, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Generate signal from RSI indicator."""
        rsi_current = indicators['rsi'][-1]
        rsi_prev = indicators['rsi'][-2]
        
        signal = 0
        strength = 0.0
        reason = "rsi_neutral"
        
        # RSI oversold reversal
        if rsi_current < self.rsi_oversold and rsi_prev >= rsi_current:
            signal = 1
            strength = min(0.8, (self.rsi_oversold - rsi_current) / self.rsi_oversold)
            reason = "rsi_oversold_reversal"
        # RSI overbought reversal
        elif rsi_current > self.rsi_overbought and rsi_prev <= rsi_current:
            signal = -1
            strength = min(0.8, (rsi_current - self.rsi_overbought) / (100 - self.rsi_overbought))
            reason = "rsi_overbought_reversal"
        # RSI divergence signals
        elif 30 < rsi_current < 70:
            # Momentum building
            if rsi_current > 60 and rsi_current > rsi_prev:
                signal = 1
                strength = 0.3
                reason = "rsi_momentum_up"
            elif rsi_current < 40 and rsi_current < rsi_prev:
                signal = -1
                strength = 0.3
                reason = "rsi_momentum_down"
                
        return {
            'signal': signal,
            'strength': strength,
            'reason': reason,
            'rsi': rsi_current
        }
        
    def _macd_signal(self, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Generate signal from MACD indicator."""
        macd_current = indicators['macd'][-1]
        macd_signal_current = indicators['macd_signal'][-1]
        macd_hist_current = indicators['macd_histogram'][-1]
        macd_hist_prev = indicators['macd_histogram'][-2]
        
        signal = 0
        strength = 0.0
        reason = "macd_neutral"
        
        # MACD line crosses above signal line (bullish)
        if macd_current > macd_signal_current and macd_hist_prev <= 0:
            signal = 1
            strength = 0.7
            reason = "macd_bullish_crossover"
        # MACD line crosses below signal line (bearish)
        elif macd_current < macd_signal_current and macd_hist_prev >= 0:
            signal = -1
            strength = 0.7
            reason = "macd_bearish_crossover"
        # MACD histogram momentum
        elif macd_hist_current > 0 and macd_hist_current > macd_hist_prev:
            signal = 1
            strength = 0.4
            reason = "macd_bullish_momentum"
        elif macd_hist_current < 0 and macd_hist_current < macd_hist_prev:
            signal = -1
            strength = 0.4
            reason = "macd_bearish_momentum"
            
        return {
            'signal': signal,
            'strength': strength,
            'reason': reason,
            'macd': macd_current,
            'macd_signal': macd_signal_current,
            'macd_histogram': macd_hist_current
        }
        
    def _volume_signal(self, df: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Generate signal from volume analysis."""
        current_volume = df['volume'].iloc[-1]
        volume_sma = indicators['volume_sma'].iloc[-1]
        price_change = df['close'].pct_change().iloc[-1]
        
        signal = 0
        strength = 0.0
        reason = "volume_neutral"
        
        volume_ratio = current_volume / volume_sma if volume_sma > 0 else 1
        
        # High volume breakouts
        if volume_ratio > 1.5:  # 50% above average volume
            if price_change > 0.005:  # 0.5% price increase
                signal = 1
                strength = min(0.6, volume_ratio / 3)
                reason = "volume_breakout_up"
            elif price_change < -0.005:  # 0.5% price decrease
                signal = -1
                strength = min(0.6, volume_ratio / 3)
                reason = "volume_breakout_down"
        # Low volume trends (less reliable)
        elif volume_ratio < 0.8:  # 20% below average volume
            strength = 0.2  # Low confidence for low volume signals
            if price_change > 0.002:
                signal = 1
                reason = "low_volume_drift_up"
            elif price_change < -0.002:
                signal = -1
                reason = "low_volume_drift_down"
                
        return {
            'signal': signal,
            'strength': strength,
            'reason': reason,
            'volume_ratio': volume_ratio,
            'price_change': price_change
        }
        
    def _combine_signals(self, ma_signal: Dict, rsi_signal: Dict, 
                        macd_signal: Dict, volume_signal: Dict) -> Dict[str, Any]:
        """Combine individual signals into a unified signal."""
        # Weighted signal combination
        signals = [ma_signal, rsi_signal, macd_signal, volume_signal]
        weights = [0.4, 0.3, 0.2, 0.1]  # MA gets highest weight
        
        weighted_signal = 0.0
        total_strength = 0.0
        reasons = []
        
        for signal_dict, weight in zip(signals, weights):
            signal_value = signal_dict['signal']
            strength = signal_dict['strength']
            
            weighted_signal += signal_value * strength * weight
            total_strength += strength * weight
            
            if signal_value != 0:
                reasons.append(signal_dict['reason'])
                
        # Determine final signal
        if weighted_signal > 0.3:
            final_signal = 1
        elif weighted_signal < -0.3:
            final_signal = -1
        else:
            final_signal = 0
            
        return {
            'signal': final_signal,
            'weighted_signal': weighted_signal,
            'total_strength': total_strength,
            'reasons': reasons,
            'component_count': sum(1 for s in signals if s['signal'] != 0)
        }
        
    def _apply_position_rules(self, combined_signal: Dict, current_position: float) -> Dict[str, Any]:
        """Apply position management rules."""
        signal = combined_signal['signal']
        
        # Don't reverse position immediately
        if current_position > 0.1 and signal < 0:
            if self.last_signal > 0:  # Was long in previous signal
                signal = 0  # Hold instead of immediate reversal
                reason = "avoid_immediate_reversal_from_long"
            else:
                reason = f"reversal_to_short: {', '.join(combined_signal['reasons'])}"
        elif current_position < -0.1 and signal > 0:
            if self.last_signal < 0:  # Was short in previous signal
                signal = 0  # Hold instead of immediate reversal
                reason = "avoid_immediate_reversal_from_short"
            else:
                reason = f"reversal_to_long: {', '.join(combined_signal['reasons'])}"
        else:
            reason = ', '.join(combined_signal['reasons']) if combined_signal['reasons'] else "neutral_market"
            
        return {
            'signal': signal,
            'reason': reason,
            'original_signal': combined_signal['signal']
        }
        
    def _calculate_confidence(self, ma_signal: Dict, rsi_signal: Dict, 
                            macd_signal: Dict, volume_signal: Dict) -> float:
        """Calculate overall confidence in the signal."""
        # Count agreeing signals
        signals = [ma_signal['signal'], rsi_signal['signal'], 
                  macd_signal['signal'], volume_signal['signal']]
        
        non_zero_signals = [s for s in signals if s != 0]
        
        if len(non_zero_signals) == 0:
            return 0.0
            
        # Check agreement
        agreement = len([s for s in non_zero_signals if s == non_zero_signals[0]]) / len(non_zero_signals)
        
        # Weight by individual strengths
        total_strength = (ma_signal['strength'] + rsi_signal['strength'] + 
                         macd_signal['strength'] + volume_signal['strength']) / 4
        
        # Final confidence
        confidence = agreement * total_strength
        
        return min(1.0, confidence)
        
    def _calculate_position_size(self, signal: int, confidence: float) -> float:
        """Calculate position size based on signal strength."""
        if signal == 0:
            return 0.0
            
        base_size = self.max_position_pct * confidence
        return min(base_size, self.max_position_pct)
        
    def _calculate_stop_loss(self, current_price: float, signal: int) -> Optional[float]:
        """Calculate stop loss level."""
        if signal == 0:
            return None
            
        if signal > 0:  # Long position
            return current_price * (1 - self.stop_loss_pct)
        else:  # Short position
            return current_price * (1 + self.stop_loss_pct)
            
    def _calculate_take_profit(self, current_price: float, signal: int) -> Optional[float]:
        """Calculate take profit level."""
        if signal == 0:
            return None
            
        if signal > 0:  # Long position
            return current_price * (1 + self.take_profit_pct)
        else:  # Short position
            return current_price * (1 - self.take_profit_pct)
            
    def _check_exit_conditions(self, current_price: float, 
                              entry_price: float, current_position: float) -> Dict[str, Any]:
        """Check stop-loss and take-profit conditions."""
        if current_position > 0:  # Long position
            pnl_pct = (current_price - entry_price) / entry_price
            
            if pnl_pct <= -self.stop_loss_pct:
                return {
                    'signal': -1,  # Close long position
                    'confidence': 1.0,
                    'position_size': abs(current_position),
                    'reason': f"stop_loss_triggered_{pnl_pct:.3f}",
                    'exit_type': 'stop_loss'
                }
            elif pnl_pct >= self.take_profit_pct:
                return {
                    'signal': -1,  # Close long position
                    'confidence': 1.0,
                    'position_size': abs(current_position),
                    'reason': f"take_profit_triggered_{pnl_pct:.3f}",
                    'exit_type': 'take_profit'
                }
                
        elif current_position < 0:  # Short position
            pnl_pct = (entry_price - current_price) / entry_price
            
            if pnl_pct <= -self.stop_loss_pct:
                return {
                    'signal': 1,  # Close short position
                    'confidence': 1.0,
                    'position_size': abs(current_position),
                    'reason': f"stop_loss_triggered_{pnl_pct:.3f}",
                    'exit_type': 'stop_loss'
                }
            elif pnl_pct >= self.take_profit_pct:
                return {
                    'signal': 1,  # Close short position
                    'confidence': 1.0,
                    'position_size': abs(current_position),
                    'reason': f"take_profit_triggered_{pnl_pct:.3f}",
                    'exit_type': 'take_profit'
                }
                
        return {'signal': 0}  # No exit condition met
        
    def _no_signal_response(self, reason: str) -> Dict[str, Any]:
        """Generate a no-signal response."""
        return {
            'signal': 0,
            'confidence': 0.0,
            'position_size': 0.0,
            'strategy': 'rule_based',
            'timestamp': datetime.now().isoformat(),
            'reason': reason
        }
        
    def get_strategy_status(self) -> Dict[str, Any]:
        """Get current strategy status and performance."""
        if not self.signal_history:
            return {'status': 'no_signals_generated'}
            
        recent_signals = self.signal_history[-10:]  # Last 10 signals
        
        signal_counts = {
            'buy': sum(1 for s in recent_signals if s['signal'] > 0),
            'sell': sum(1 for s in recent_signals if s['signal'] < 0),
            'hold': sum(1 for s in recent_signals if s['signal'] == 0)
        }
        
        avg_confidence = np.mean([s['confidence'] for s in recent_signals])
        
        return {
            'last_signal': self.last_signal,
            'signals_generated': len(self.signal_history),
            'recent_signal_counts': signal_counts,
            'average_confidence': avg_confidence,
            'last_update': recent_signals[-1]['timestamp']
        }


# Example usage and testing
if __name__ == "__main__":
    """
    Example usage of RuleBasedStrategy for testing.
    """
    # Mock configuration
    config = {
        'strategy': {
            'rule_based': {
                'ma_fast': 10,
                'ma_slow': 30,
                'rsi_period': 14,
                'rsi_oversold': 30,
                'rsi_overbought': 70
            }
        },
        'trading': {
            'risk_management': {
                'max_position_size_pct': 0.1,
                'stop_loss_pct': 0.02,
                'take_profit_pct': 0.04
            }
        }
    }
    
    # Create sample data
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1min')
    np.random.seed(42)
    
    # Generate trending price data
    trend = np.linspace(0, 0.1, 100)  # 10% uptrend
    noise = np.random.normal(0, 0.005, 100)  # 0.5% noise
    prices = 50000 * (1 + trend + noise.cumsum())
    
    df = pd.DataFrame({
        'open': prices,
        'high': prices * 1.002,
        'low': prices * 0.998,
        'close': prices,
        'volume': np.random.lognormal(10, 0.3, 100)
    }, index=dates)
    
    print("📊 Testing Rule-Based Trading Strategy")
    print(f"Sample data shape: {df.shape}")
    print(f"Price range: {df['close'].min():.2f} - {df['close'].max():.2f}")
    
    # Initialize strategy
    strategy = RuleBasedStrategy(config)
    
    # Generate signal
    print("\n📈 Generating rule-based signal...")
    signal = strategy.generate_signal(df)
    
    print("Signal result:")
    for key, value in signal.items():
        if key == 'indicators':
            print(f"  {key}: [computed]")
        elif key == 'component_signals':
            print(f"  {key}:")
            for comp_key, comp_val in value.items():
                print(f"    {comp_key}: {comp_val}")
        elif isinstance(value, (int, float)):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
            
    # Test with position
    print("\n🔄 Testing with existing position...")
    signal_with_position = strategy.generate_signal(df, current_position=0.5, entry_price=49000)
    
    print(f"Signal with position: {signal_with_position['signal']}")
    print(f"Reason: {signal_with_position['reason']}")
    
    # Get strategy status
    print("\n📊 Strategy status:")
    status = strategy.get_strategy_status()
    for key, value in status.items():
        print(f"  {key}: {value}")
        
    print("\n✅ Rule-Based Strategy test completed")
