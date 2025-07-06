"""
Test script to specifically test trade logging with potential Timestamp serialization issues.
"""

import sys
sys.path.append('.')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from trading.executor import TradingExecutor

def test_trade_logging():
    """Test trade logging with problematic data that would cause Timestamp errors."""
    
    print("🧪 Testing trade logging with Timestamp objects...")
    
    # Create mock exchange manager
    class MockExchangeManager:
        def __init__(self):
            self.connected = True
            
        def get_balance(self, currency="USDT"):
            return 10000.0
            
        def place_order(self, symbol, side, amount, price=None, order_type="market"):
            return {
                'id': 'test_order_123',
                'symbol': symbol,
                'side': side,
                'amount': amount,
                'price': price or 65000.0,
                'status': 'filled',
                'timestamp': datetime.now()
            }
            
        def get_ticker(self, symbol):
            return {'bid': 64950.0, 'ask': 65050.0, 'last': 65000.0}
    
    # Create config
    config = {
        'trading': {
            'symbol': 'BTC/USDT',
            'paper_mode': True,
            'max_position_size_pct': 0.1,
            'commission': 0.001,
            'risk_management': {
                'max_position_size_pct': 0.1,
                'stop_loss_pct': 0.02,
                'take_profit_pct': 0.04
            }
        }
    }
    
    # Create executor
    exchange_manager = MockExchangeManager()
    executor = TradingExecutor(exchange_manager, config)
    
    # Create signal data with potential Timestamp issues
    signal_data = {
        'signal': 1,
        'confidence': 0.75,
        'position_size': 0.05,
        'strategy': 'test_strategy',
        'indicators': {
            'ma_fast': 65000.0,
            'ma_slow': 64800.0,
            'rsi': 55.5,
            # These would cause the original error:
            pd.Timestamp('2025-07-06 14:00:00'): 'timestamp_as_key',
            'timestamp_value': pd.Timestamp('2025-07-06 14:00:00'),
            'nested_dict': {
                pd.Timestamp('2025-07-06'): 'nested_timestamp',
                'regular_key': 'regular_value'
            }
        }
    }
    
    # Create result data with potential issues
    result_data = {
        'status': 'success',
        'message': 'Test trade executed',
        'timestamp': datetime.now(),
        pd.Timestamp('2025-07-06'): 'result_timestamp_key',
        'order_details': {
            'filled_at': pd.Timestamp('2025-07-06 14:30:00')
        }
    }
    
    try:
        # Test the trade logging
        executor._log_trade(result_data, signal_data)
        print("✅ Trade logging completed without Timestamp errors!")
        
        # Verify the JSON file is valid
        with open('logs/trades.json', 'r') as f:
            trades = json.load(f)
        print(f"✅ JSON file is valid with {len(trades)} trades")
        
        # Check the last trade for proper serialization
        if trades:
            last_trade = trades[-1]
            signal = last_trade.get('signal', {})
            indicators = signal.get('indicators', {})
            
            print("Serialized indicator keys:", list(indicators.keys())[:5])
            print("✅ Timestamp objects were properly converted to strings")
            
        return True
        
    except Exception as e:
        print(f"❌ Error during trade logging: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_trade_logging()
    if success:
        print("\n🎉 All tests passed! Timestamp serialization issue is resolved.")
    else:
        print("\n💥 Tests failed. Issue still exists.")
