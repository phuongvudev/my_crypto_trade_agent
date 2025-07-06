# 🤖 Crypto Trading AI Agent

A modular, production-ready cryptocurrency trading agent powered by machine learning and technical analysis. This system combines XGBoost classification and LSTM price prediction with traditional technical indicators to make intelligent trading decisions.

## 🌟 Features

### 🔧 Core Components
- **Live Data Collection**: Real-time OHLCV data from Binance via CCXT
- **AI Models**: XGBoost classifier for trade signals + LSTM for price prediction
- **Technical Analysis**: 30+ indicators including RSI, MACD, Bollinger Bands
- **Risk Management**: Stop-loss, take-profit, position sizing, daily loss limits
- **Multiple Strategies**: Rule-based, ML-based, and hybrid approaches
- **Backtesting Engine**: Comprehensive historical strategy testing
- **Real-time Dashboard**: Streamlit web interface for monitoring
- **Paper Trading**: Safe testing mode with simulated execution

### 📊 Trading Strategies
1. **Rule-Based Strategy**: Moving averages, RSI, MACD crossovers
2. **ML-Based Strategy**: XGBoost + LSTM predictions with confidence scoring
3. **Hybrid Strategy**: Combines both approaches with weighted signals

### 🛡️ Risk Management
- Maximum position size limits (default: 10% of portfolio)
- Daily loss limits (default: 5% of portfolio)
- Stop-loss and take-profit automation
- Position exposure monitoring
- Drawdown protection

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd crypto_trading_ai

# Install dependencies
pip install -r requirements.txt

# Install TA-Lib (required for technical analysis)
# On macOS:
brew install ta-lib
pip install TA-Lib

# On Ubuntu/Debian:
sudo apt-get install libta-lib-dev
pip install TA-Lib

# On Windows:
# Download TA-Lib from https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
# pip install downloaded_file.whl
```

### 2. Configuration

Create a `.env` file for API credentials:
```bash
BINANCE_API_KEY=your_binance_api_key
BINANCE_SECRET=your_binance_secret
```

Edit `config.yaml` to customize trading parameters:
```yaml
# Exchange settings
exchange:
  name: "binance"
  sandbox: true  # Start with testnet!

# Trading settings
trading:
  symbol: "BTC/USDT"
  risk_management:
    max_position_size_pct: 0.1  # 10% max position
    daily_loss_limit_pct: 0.05  # 5% daily loss limit
    stop_loss_pct: 0.02        # 2% stop loss
    take_profit_pct: 0.04      # 4% take profit

# Strategy selection
strategy:
  active: "rule_based"  # Start with rule-based for safety
```

### 3. Running the Agent

#### Paper Trading (Recommended for testing)
```bash
python main.py --mode paper --strategy rule_based
```

#### Backtesting
```bash
# Test strategy on historical data
python main.py --mode backtest --strategy rule_based --start-date 2024-01-01 --end-date 2024-02-01
```

#### Live Trading (Use with caution!)
```bash
# Only after thorough testing!
python main.py --mode live --strategy rule_based
```

### 4. Dashboard

Launch the monitoring dashboard:
```bash
streamlit run dashboard/app.py
```

Open your browser to `http://localhost:8501` to view:
- Live portfolio performance
- Trade history and analysis
- AI model predictions
- Risk monitoring alerts

## 📁 Project Structure

```
crypto_trading_ai/
├── 📊 data/
│   ├── historical/          # Historical price data
│   └── live/               # Live data cache
├── 🤖 models/
│   ├── lstm_price_forecast.h5    # LSTM model for price prediction
│   └── xgb_trade_signal.json     # XGBoost model for trade signals
├── 🧠 strategies/
│   ├── rule_based.py       # Technical analysis strategy
│   └── ml_based.py         # Machine learning strategy
├── 💱 trading/
│   ├── exchange.py         # Binance connection & data fetching
│   └── executor.py         # Order execution & risk management
├── 🔙 backtest/
│   └── backtester.py       # Strategy backtesting engine
├── 📈 dashboard/
│   └── app.py              # Streamlit web dashboard
├── 🔧 features/
│   └── feature_engineering.py  # Technical indicator computation
├── ⚙️ config.yaml          # Main configuration
├── 🚀 main.py              # Main application runner
└── 📋 requirements.txt     # Python dependencies
```

## 🎯 Strategy Details

### Rule-Based Strategy
Uses traditional technical analysis:
- **Moving Average Crossover**: Golden/Death cross signals
- **RSI**: Oversold/overbought conditions (14-period)
- **MACD**: Momentum and trend changes
- **Bollinger Bands**: Volatility-based entry/exit
- **Volume Analysis**: Confirming breakouts

### ML-Based Strategy
Advanced machine learning approach:
- **Feature Engineering**: 60+ technical indicators
- **XGBoost Classifier**: Predicts BUY/HOLD/SELL signals
- **LSTM Neural Network**: Forecasts next price movements
- **Confidence Scoring**: Only trades high-confidence signals
- **Signal Combination**: Weighs multiple predictions

### Hybrid Strategy
Combines both approaches:
- Weights: 40% rule-based + 60% ML-based
- Agreement bonus: +20% confidence when strategies align
- Disagreement penalty: -30% confidence for conflicts

## 📊 Performance Metrics

The backtesting engine calculates comprehensive metrics:

### Returns & Risk
- Total Return, Annualized Return
- Sharpe Ratio, Sortino Ratio
- Maximum Drawdown
- Value at Risk (VaR), Conditional VaR

### Trade Analysis
- Win Rate, Average Win/Loss
- Profit Factor, Payoff Ratio
- Total Trades, Average Trade Duration

### Risk Metrics
- Calmar Ratio, Recovery Factor
- Daily loss tracking
- Position exposure monitoring

## 🛡️ Risk Management Features

### Position Management
- **Max Position Size**: Limits exposure per trade
- **Portfolio Diversification**: Multiple position limits
- **Position Scaling**: Size based on signal confidence

### Stop Loss & Take Profit
- **Automatic Stop Loss**: Configurable percentage-based
- **Take Profit Targets**: Lock in gains automatically
- **Trailing Stops**: Follow favorable price movements

### Daily Limits
- **Daily Loss Limit**: Stop trading if daily losses exceed threshold
- **Maximum Drawdown**: Emergency position closure
- **Trade Frequency**: Prevent overtrading

## 🔧 Configuration Guide

### Exchange Settings
```yaml
exchange:
  name: "binance"
  sandbox: true          # Use testnet for safety
  api_key: "${BINANCE_API_KEY}"
  secret: "${BINANCE_SECRET}"
```

### Trading Parameters
```yaml
trading:
  symbol: "BTC/USDT"
  timeframes: ["1m", "5m"]
  
  risk_management:
    max_position_size_pct: 0.1    # 10% max position
    daily_loss_limit_pct: 0.05    # 5% daily loss limit
    stop_loss_pct: 0.02           # 2% stop loss
    take_profit_pct: 0.04         # 4% take profit
    max_open_positions: 3
    
  order:
    type: "market"
    amount_type: "quote"
    min_amount: 10
```

### Strategy Configuration
```yaml
strategy:
  active: "rule_based"  # rule_based, ml_based, hybrid
  
  ml_model:
    lstm_lookback: 60
    xgb_features: 20
    confidence_threshold: 0.6
    
  rule_based:
    ma_fast: 10
    ma_slow: 30
    rsi_period: 14
    rsi_oversold: 30
    rsi_overbought: 70
```

## 🔬 Model Training

### Training XGBoost Model
```python
from strategies.ml_based import MLTradingStrategy
from trading.exchange import ExchangeManager

# Load historical data
exchange = ExchangeManager()
df = exchange.fetch_historical_data(days=90)

# Create target signals (simplified example)
signals = create_target_signals(df)  # Your signal generation logic

# Train models
strategy = MLTradingStrategy(config)
results = strategy.train_models(df, signals)

print(f"XGBoost Accuracy: {results['xgb_results']['accuracy']:.3f}")
print(f"LSTM Validation Loss: {results['lstm_results']['val_loss']:.6f}")
```

### Feature Engineering
The system automatically computes 60+ features:
- **Price Features**: Returns, spreads, momentum
- **Moving Averages**: SMA, EMA (multiple periods)
- **Momentum**: RSI, MACD, Stochastic, Williams %R
- **Volatility**: Bollinger Bands, ATR
- **Volume**: OBV, Volume ratios, MFI
- **Pattern Recognition**: Candlestick patterns
- **Time Features**: Hour, day, market sessions

## 📈 Dashboard Features

### Real-time Monitoring
- Live portfolio value and PnL
- Current positions and exposure
- Trade history with performance metrics
- Risk monitoring alerts

### Charts & Analysis
- **Price Charts**: Candlestick with volume
- **Portfolio Performance**: Equity curve and drawdown
- **Trade Analysis**: PnL distribution and statistics
- **AI Predictions**: Model confidence and signals

### Risk Dashboard
- Daily loss tracking
- Position size monitoring
- System health indicators
- Trading status controls

## 🧪 Testing Strategy

### 1. Backtest First
Always test strategies on historical data:
```bash
python main.py --mode backtest --strategy rule_based --start-date 2024-01-01
```

### 2. Paper Trading
Test with live data but simulated execution:
```bash
python main.py --mode paper --strategy rule_based
```

### 3. Small Live Test
Start with minimal amounts:
```python
# In config.yaml
trading:
  order:
    min_amount: 10  # Start with $10 orders
  risk_management:
    max_position_size_pct: 0.01  # 1% positions
```

### 4. Gradual Scale-Up
Only increase size after proven performance

## ⚠️ Important Warnings

### 🚨 Trading Risks
- **Cryptocurrency trading involves significant financial risk**
- **Past performance does not guarantee future results**
- **Only trade with money you can afford to lose**
- **Always start with paper trading and small amounts**

### 🔒 Security
- **Never share API keys or commit them to version control**
- **Use API keys with trading permissions only on trusted systems**
- **Enable IP whitelisting on exchange accounts**
- **Regularly rotate API credentials**

### 🐛 System Risks
- **Monitor system continuously during live trading**
- **Have emergency stop procedures ready**
- **Keep backups of configuration and data**
- **Test all changes thoroughly in paper mode**

## 🛠️ Development

### Adding New Strategies
1. Create new strategy class inheriting from base interface
2. Implement `generate_signal(df, current_position)` method
3. Add strategy to initialization in `main.py`
4. Test thoroughly with backtesting

### Custom Indicators
Add to `features/feature_engineering.py`:
```python
def _add_custom_features(self, features_df, df):
    # Your custom technical indicators
    features_df['custom_indicator'] = your_calculation(df)
    return features_df
```

### Exchange Integration
To add new exchanges:
1. Extend `ExchangeManager` class
2. Add exchange-specific configuration
3. Implement exchange-specific API calls
4. Test with paper trading first

## 🐛 Troubleshooting

### Common Issues

**Import Errors**
```bash
# Install missing dependencies
pip install -r requirements.txt

# TA-Lib installation issues
# See installation section above
```

**API Connection Issues**
- Check API credentials in `.env` file
- Verify exchange account permissions
- Confirm IP whitelisting settings
- Test with sandbox/testnet first

**Model Loading Errors**
- Pre-trained models need to be created first
- Run with `rule_based` strategy initially
- Train models using historical data

**Performance Issues**
- Reduce update frequency in config
- Limit historical data fetch size
- Use fewer technical indicators
- Monitor system resources

### Debug Mode
```bash
# Enable debug logging
python main.py --mode paper --strategy rule_based --debug
```

### Log Analysis
Check logs in `logs/trading_agent.log` for detailed information.

## 📚 Further Reading

### Technical Analysis
- [Technical Analysis of Financial Markets](https://www.amazon.com/Technical-Analysis-Financial-Markets-Comprehensive/dp/0735200661)
- [TA-Lib Documentation](https://ta-lib.org/)

### Machine Learning for Trading
- [Advances in Financial Machine Learning](https://www.amazon.com/Advances-Financial-Machine-Learning-Marcos/dp/1119482089)
- [Python for Finance](https://www.amazon.com/Python-Finance-Mastering-Data-Driven/dp/1492024333)

### Risk Management
- [Quantitative Risk Management](https://www.amazon.com/Quantitative-Risk-Management-Concepts-Techniques/dp/0691166277)
- [The Quants](https://www.amazon.com/Quants-Whizzes-Conquered-Street-Destroyed/dp/0307453383)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

### Code Standards
- Follow PEP 8 style guidelines
- Add type hints to all functions
- Include comprehensive docstrings
- Write unit tests for new features

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚖️ Disclaimer

This software is for educational and research purposes only. The authors are not responsible for any financial losses incurred through the use of this trading system. Cryptocurrency trading carries substantial risk of loss and is not suitable for all investors. Always consult with qualified financial advisors before making investment decisions.

---

**Happy Trading! 🚀📈**

*Remember: The best strategy is the one you understand and can sleep peacefully with.*
