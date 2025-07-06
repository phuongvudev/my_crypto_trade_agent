#!/usr/bin/env python3
"""
Test script for Crypto Trading AI Agent
Verifies that all components are working correctly
"""

import sys
import os
import traceback
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def print_header(title):
    """Print a formatted header."""
    print("\n" + "="*60)
    print(f"🧪 {title}")
    print("="*60)

def print_test(test_name):
    """Print test name."""
    print(f"\n📋 Testing {test_name}...")

def print_success(message):
    """Print success message."""
    print(f"✅ {message}")

def print_error(message):
    """Print error message."""
    print(f"❌ {message}")

def print_warning(message):
    """Print warning message."""
    print(f"⚠️  {message}")

def test_imports():
    """Test all required imports."""
    print_test("Python Package Imports")
    
    # Core packages
    try:
        import pandas as pd
        import numpy as np
        import yaml
        print_success("Core packages (pandas, numpy, yaml)")
    except ImportError as e:
        print_error(f"Core packages: {e}")
        return False
    
    # TA-Lib
    try:
        import talib
        print_success("TA-Lib technical analysis library")
    except ImportError as e:
        print_error(f"TA-Lib: {e}")
        print_warning("Install TA-Lib: brew install ta-lib (macOS) or apt-get install libta-lib-dev (Ubuntu)")
        return False
    
    # CCXT
    try:
        import ccxt
        print_success("CCXT cryptocurrency exchange library")
    except ImportError as e:
        print_error(f"CCXT: {e}")
        return False
    
    # ML libraries
    try:
        import sklearn
        print_success("Scikit-learn")
    except ImportError as e:
        print_error(f"Scikit-learn: {e}")
        return False
    
    try:
        import xgboost
        print_success("XGBoost")
    except ImportError as e:
        print_warning(f"XGBoost: {e} (optional for ML strategy)")
    
    try:
        import tensorflow
        print_success("TensorFlow")
    except ImportError as e:
        print_warning(f"TensorFlow: {e} (optional for ML strategy)")
    
    # Visualization
    try:
        import plotly
        print_success("Plotly")
    except ImportError as e:
        print_error(f"Plotly: {e}")
        return False
    
    try:
        import streamlit
        print_success("Streamlit")
    except ImportError as e:
        print_error(f"Streamlit: {e}")
        return False
    
    return True

def test_config():
    """Test configuration loading."""
    print_test("Configuration Loading")
    
    try:
        import yaml
        
        # Check if config file exists
        if not os.path.exists('config.yaml'):
            print_error("config.yaml not found")
            return False
        
        # Load config
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Check required sections
        required_sections = ['exchange', 'trading', 'strategy']
        for section in required_sections:
            if section not in config:
                print_error(f"Missing required section: {section}")
                return False
        
        print_success("Configuration file loaded successfully")
        print_success(f"Exchange: {config['exchange']['name']}")
        print_success(f"Symbol: {config['trading']['symbol']}")
        print_success(f"Active strategy: {config['strategy']['active']}")
        
        return True
        
    except Exception as e:
        print_error(f"Configuration error: {e}")
        return False

def test_feature_engineering():
    """Test feature engineering module."""
    print_test("Feature Engineering")
    
    try:
        from features.feature_engineering import FeatureEngineer
        
        # Create sample data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1min')
        np.random.seed(42)
        
        # Generate realistic OHLCV data
        prices = np.random.random(100).cumsum() + 50000
        df = pd.DataFrame({
            'open': prices,
            'high': prices * 1.002,
            'low': prices * 0.998,
            'close': prices,
            'volume': np.random.lognormal(10, 0.3, 100)
        }, index=dates)
        
        # Test feature computation
        feature_engineer = FeatureEngineer()
        features = feature_engineer.compute_features(df)
        
        if len(features) > 0:
            print_success(f"Features computed successfully: {features.shape}")
            
            # Check for NaN values
            nan_count = np.isnan(features.values).sum()
            if nan_count == 0:
                print_success("No NaN values in features")
            else:
                print_warning(f"Found {nan_count} NaN values in features")
            
            return True
        else:
            print_error("No features computed")
            return False
            
    except Exception as e:
        print_error(f"Feature engineering error: {e}")
        traceback.print_exc()
        return False

def test_exchange_manager():
    """Test exchange manager (without API keys)."""
    print_test("Exchange Manager")
    
    try:
        from trading.exchange import ExchangeManager
        
        # Test initialization
        exchange_manager = ExchangeManager()
        print_success("Exchange manager initialized")
        
        # Test connection (will fail without API keys, but should not crash)
        try:
            status = exchange_manager.check_connection()
            if status:
                print_success("Exchange connection successful")
            else:
                print_warning("Exchange connection failed (expected without API keys)")
        except Exception as e:
            print_warning(f"Exchange connection failed: {e} (expected without API keys)")
        
        return True
        
    except Exception as e:
        print_error(f"Exchange manager error: {e}")
        traceback.print_exc()
        return False

def test_strategies():
    """Test trading strategies."""
    print_test("Trading Strategies")
    
    try:
        # Create sample config
        config = {
            'strategy': {
                'rule_based': {
                    'ma_fast': 10,
                    'ma_slow': 30,
                    'rsi_period': 14,
                    'rsi_oversold': 30,
                    'rsi_overbought': 70
                },
                'ml_model': {
                    'confidence_threshold': 0.6,
                    'lstm_lookback': 60,
                    'xgb_features': 20
                }
            },
            'trading': {
                'risk_management': {
                    'max_position_size_pct': 0.1,
                    'stop_loss_pct': 0.02,
                    'take_profit_pct': 0.04
                }
            },
            'models': {
                'lstm_path': 'models/lstm_price_forecast.h5',
                'xgb_path': 'models/xgb_trade_signal.pkl'
            }
        }
        
        # Test rule-based strategy
        from strategies.rule_based import RuleBasedStrategy
        rule_strategy = RuleBasedStrategy(config)
        print_success("Rule-based strategy initialized")
        
        # Test with sample data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1min')
        np.random.seed(42)
        prices = np.random.random(100).cumsum() + 50000
        df = pd.DataFrame({
            'open': prices,
            'high': prices * 1.002,
            'low': prices * 0.998,
            'close': prices,
            'volume': np.random.lognormal(10, 0.3, 100)
        }, index=dates)
        
        signal = rule_strategy.generate_signal(df)
        print_success(f"Rule-based signal generated: {signal['signal']}")
        
        # Test ML strategy (will use fallback if models not available)
        try:
            from strategies.ml_based import MLTradingStrategy
            ml_strategy = MLTradingStrategy(config)
            ml_signal = ml_strategy.generate_signal(df)
            print_success(f"ML-based signal generated: {ml_signal['signal']}")
        except Exception as e:
            print_warning(f"ML strategy test failed: {e} (models may not be trained)")
        
        return True
        
    except Exception as e:
        print_error(f"Strategy testing error: {e}")
        traceback.print_exc()
        return False

def test_backtester():
    """Test backtesting functionality."""
    print_test("Backtesting Engine")
    
    try:
        from backtest.backtester import Backtester
        
        # Create sample config
        config = {
            'backtest': {
                'initial_balance': 10000,
                'commission': 0.001
            },
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
                    'max_position_size_pct': 0.2,
                    'stop_loss_pct': 0.02,
                    'take_profit_pct': 0.04
                }
            }
        }
        
        # Initialize backtester
        backtester = Backtester(config)
        print_success("Backtester initialized")
        
        # Create sample data
        dates = pd.date_range(start='2024-01-01', periods=500, freq='1min')
        np.random.seed(42)
        
        # Generate trending price data
        trend = np.linspace(0, 0.1, 500)
        noise = np.random.normal(0, 0.01, 500)
        prices = 50000 * (1 + trend + noise.cumsum())
        
        df = pd.DataFrame({
            'open': prices,
            'high': prices * 1.002,
            'low': prices * 0.998,
            'close': prices,
            'volume': np.random.lognormal(10, 0.3, 500)
        }, index=dates)
        
        # Run backtest
        results = backtester.run_backtest(df, 'rule_based')
        
        if 'error' not in results:
            print_success("Backtest completed successfully")
            metrics = results['metrics']
            print_success(f"Total return: {metrics['total_return']:.2%}")
            print_success(f"Total trades: {metrics['total_trades']}")
        else:
            print_error(f"Backtest failed: {results['error']}")
            return False
        
        return True
        
    except Exception as e:
        print_error(f"Backtester error: {e}")
        traceback.print_exc()
        return False

def test_dashboard():
    """Test dashboard components."""
    print_test("Dashboard Components")
    
    try:
        # Test streamlit import
        import streamlit as st
        print_success("Streamlit imported")
        
        # Test plotly
        import plotly.graph_objects as go
        print_success("Plotly imported")
        
        # Check if dashboard file exists
        if os.path.exists('dashboard/app.py'):
            print_success("Dashboard app.py found")
        else:
            print_error("dashboard/app.py not found")
            return False
        
        return True
        
    except Exception as e:
        print_error(f"Dashboard test error: {e}")
        return False

def test_directories():
    """Test that all necessary directories exist."""
    print_test("Directory Structure")
    
    required_dirs = [
        'data',
        'data/historical',
        'data/live',
        'models',
        'logs',
        'strategies',
        'trading',
        'backtest',
        'dashboard',
        'features'
    ]
    
    all_exist = True
    for directory in required_dirs:
        if os.path.exists(directory):
            print_success(f"Directory exists: {directory}")
        else:
            print_error(f"Directory missing: {directory}")
            all_exist = False
    
    return all_exist

def main():
    """Run all tests."""
    print_header("Crypto Trading AI Agent - System Test")
    
    print("🔍 Running comprehensive system tests...")
    print("This will verify that all components are properly installed and configured.")
    
    # Track test results
    tests = []
    
    # Run tests
    tests.append(("Imports", test_imports()))
    tests.append(("Configuration", test_config()))
    tests.append(("Directories", test_directories()))
    tests.append(("Feature Engineering", test_feature_engineering()))
    tests.append(("Exchange Manager", test_exchange_manager()))
    tests.append(("Trading Strategies", test_strategies()))
    tests.append(("Backtesting", test_backtester()))
    tests.append(("Dashboard", test_dashboard()))
    
    # Print results summary
    print_header("Test Results Summary")
    
    passed = 0
    failed = 0
    
    for test_name, result in tests:
        if result:
            print_success(f"{test_name}: PASSED")
            passed += 1
        else:
            print_error(f"{test_name}: FAILED")
            failed += 1
    
    print(f"\n📊 Tests completed: {passed} passed, {failed} failed")
    
    if failed == 0:
        print_success("🎉 All tests passed! Your system is ready for trading.")
        print("\nNext steps:")
        print("1. Add your API credentials to .env file")
        print("2. Run: python main.py --mode paper --strategy rule_based")
        print("3. Launch dashboard: streamlit run dashboard/app.py")
    else:
        print_error("❌ Some tests failed. Please fix the issues before proceeding.")
        print("\nTroubleshooting tips:")
        print("1. Run: pip install -r requirements.txt")
        print("2. Install TA-Lib: brew install ta-lib (macOS)")
        print("3. Check that all files are present")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
