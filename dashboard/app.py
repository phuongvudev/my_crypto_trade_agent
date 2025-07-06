"""
Streamlit dashboard for the crypto trading AI agent.
Displays live trades, PnL, equity curves, and model predictions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json
import os
import time
from typing import Dict, List, Any, Optional

# Streamlit page configuration
st.set_page_config(
    page_title="Crypto Trading AI Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        border: 1px solid #e1e5e9;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stAlert {
        margin: 1rem 0;
    }
    .trade-card {
        background-color: #ffffff;
        border: 1px solid #e1e5e9;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


class TradingDashboard:
    """Main dashboard class for the trading AI agent."""
    
    def __init__(self):
        self.config = self.load_config()
        self.refresh_interval = self.config.get('dashboard', {}).get('refresh_interval', 30)
        
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        try:
            with open('config.yaml', 'r') as f:
                import yaml
                return yaml.safe_load(f)
        except Exception as e:
            st.error(f"Error loading config: {e}")
            return {}
            
    def load_trade_log(self) -> List[Dict[str, Any]]:
        """Load trade log from file."""
        try:
            if os.path.exists('logs/trades.json'):
                with open('logs/trades.json', 'r') as f:
                    return json.load(f)
            return []
        except Exception as e:
            st.error(f"Error loading trade log: {e}")
            return []
            
    def load_portfolio_status(self) -> Dict[str, Any]:
        """Load current portfolio status."""
        try:
            # In a real implementation, this would connect to the trading executor
            # For now, return mock data
            return {
                'total_balance': 50000.0,
                'daily_pnl': 250.75,
                'total_pnl': 1500.50,
                'unrealized_pnl': 125.25,
                'open_positions': 1,
                'total_trades': 25,
                'trading_enabled': True,
                'paper_trading': True,
                'current_price': 50000.0,
                'last_trade_time': datetime.now().isoformat()
            }
        except Exception as e:
            st.error(f"Error loading portfolio status: {e}")
            return {}
            
    def load_market_data(self) -> pd.DataFrame:
        """Load current market data."""
        try:
            # In a real implementation, this would connect to the exchange manager
            # For now, generate mock data
            dates = pd.date_range(start=datetime.now() - timedelta(hours=24), 
                                 end=datetime.now(), freq='1min')
            np.random.seed(42)
            
            prices = 50000 + np.random.normal(0, 100, len(dates)).cumsum()
            
            return pd.DataFrame({
                'datetime': dates,
                'open': prices,
                'high': prices * 1.002,
                'low': prices * 0.998,
                'close': prices,
                'volume': np.random.lognormal(10, 0.3, len(dates))
            }).set_index('datetime')
        except Exception as e:
            st.error(f"Error loading market data: {e}")
            return pd.DataFrame()


def main():
    """Main dashboard function."""
    dashboard = TradingDashboard()
    
    # Sidebar
    with st.sidebar:
        st.title("🤖 Trading AI Control")
        
        # Status indicators
        st.subheader("System Status")
        
        # Auto-refresh toggle
        auto_refresh = st.checkbox("Auto Refresh", value=False)  # Changed default to False
        refresh_interval = st.slider("Refresh Interval (seconds)", 5, 60, 30)
        
        if auto_refresh:
            st.info(f"Auto-refresh enabled: {refresh_interval}s")
            # Use st.empty() and sleep instead of st.rerun() to avoid infinite loops
            placeholder = st.empty()
            time.sleep(refresh_interval)
            st.rerun()
            
        # Manual refresh button
        if st.button("🔄 Refresh Now"):
            st.rerun()
            
        # Trading controls
        st.subheader("Trading Controls")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶️ Start Trading"):
                st.success("Trading started!")
                
        with col2:
            if st.button("⏸️ Stop Trading"):
                st.warning("Trading stopped!")
                
        # Strategy selection
        st.subheader("Strategy Settings")
        strategy = st.selectbox(
            "Active Strategy",
            ["ml_based", "rule_based", "hybrid"],
            index=0
        )
        
        confidence_threshold = st.slider(
            "Confidence Threshold",
            0.0, 1.0, 0.6, 0.05
        )
        
        max_position = st.slider(
            "Max Position Size (%)",
            0.01, 0.5, 0.1, 0.01
        )
        
    # Main dashboard
    st.title("📈 Crypto Trading AI Dashboard")
    
    # Load data
    portfolio_status = dashboard.load_portfolio_status()
    trades = dashboard.load_trade_log()
    market_data = dashboard.load_market_data()
    
    # Top metrics row
    st.subheader("📊 Portfolio Overview")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Total Balance",
            f"${portfolio_status.get('total_balance', 0):,.2f}",
            delta=f"${portfolio_status.get('daily_pnl', 0):+,.2f}"
        )
        
    with col2:
        st.metric(
            "Daily PnL",
            f"${portfolio_status.get('daily_pnl', 0):+,.2f}",
            delta=f"{(portfolio_status.get('daily_pnl', 0) / portfolio_status.get('total_balance', 1) * 100):+.2f}%"
        )
        
    with col3:
        st.metric(
            "Total PnL",
            f"${portfolio_status.get('total_pnl', 0):+,.2f}",
            delta=f"{(portfolio_status.get('total_pnl', 0) / 10000 * 100):+.2f}%"  # Assuming 10k initial
        )
        
    with col4:
        st.metric(
            "Open Positions",
            portfolio_status.get('open_positions', 0),
            delta=f"Total Trades: {portfolio_status.get('total_trades', 0)}"
        )
        
    with col5:
        current_price = portfolio_status.get('current_price', 0)
        st.metric(
            "BTC Price",
            f"${current_price:,.2f}",
            delta=f"${np.random.normal(0, 50):+.2f}"  # Mock price change
        )
        
    # Status indicators
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if portfolio_status.get('trading_enabled', False):
            st.success("🟢 Trading Active")
        else:
            st.error("🔴 Trading Inactive")
            
    with col2:
        if portfolio_status.get('paper_trading', True):
            st.warning("📄 Paper Trading")
        else:
            st.info("💰 Live Trading")
            
    with col3:
        st.info(f"🤖 Strategy: {strategy.title()}")
        
    # Charts section
    st.subheader("📈 Market Data & Analysis")
    
    # Create tabs for different chart views
    tab1, tab2, tab3, tab4 = st.tabs(["Price Chart", "Portfolio Performance", "Trade Analysis", "Strategy Signals"])
    
    with tab1:
        if not market_data.empty:
            # Create candlestick chart
            fig = go.Figure(data=go.Candlestick(
                x=market_data.index,
                open=market_data['open'],
                high=market_data['high'],
                low=market_data['low'],
                close=market_data['close'],
                name="BTC/USDT"
            ))
            
            # Add volume as subplot
            fig.add_trace(go.Scatter(
                x=market_data.index,
                y=market_data['volume'],
                name="Volume",
                yaxis="y2",
                opacity=0.3
            ))
            
            fig.update_layout(
                title="BTC/USDT Price Chart (24H)",
                xaxis_title="Time",
                yaxis_title="Price (USDT)",
                yaxis2=dict(title="Volume", overlaying="y", side="right"),
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No market data available")
            
    with tab2:
        # Portfolio equity curve
        if trades:
            # Create mock equity curve
            dates = pd.date_range(start=datetime.now() - timedelta(days=30), 
                                 end=datetime.now(), freq='1h')
            equity_values = 10000 + np.random.normal(0, 50, len(dates)).cumsum()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates,
                y=equity_values,
                mode='lines',
                name='Portfolio Value',
                line=dict(color='blue', width=2)
            ))
            
            fig.update_layout(
                title="Portfolio Equity Curve (30 Days)",
                xaxis_title="Date",
                yaxis_title="Portfolio Value (USDT)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Drawdown chart
            running_max = pd.Series(equity_values).expanding().max()
            drawdown = (pd.Series(equity_values) - running_max) / running_max * 100
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates,
                y=drawdown,
                mode='lines',
                fill='tonexty',
                name='Drawdown',
                line=dict(color='red'),
                fillcolor='rgba(255,0,0,0.2)'
            ))
            
            fig.update_layout(
                title="Portfolio Drawdown",
                xaxis_title="Date",
                yaxis_title="Drawdown (%)",
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No portfolio data available yet")
            
    with tab3:
        # Trade analysis
        if trades:
            st.subheader("Recent Trades")
            
            # Display recent trades in a table
            recent_trades = trades[-10:] if len(trades) > 10 else trades
            
            for i, trade in enumerate(reversed(recent_trades)):
                with st.container():
                    col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 2, 2])
                    
                    with col1:
                        st.write(f"**Trade #{len(trades) - i}**")
                        st.write(f"Time: {datetime.now().strftime('%H:%M:%S')}")
                        
                    with col2:
                        signal = trade.get('signal', {}).get('signal', 0)
                        if signal > 0:
                            st.success("🟢 BUY")
                        elif signal < 0:
                            st.error("🔴 SELL")
                        else:
                            st.info("🟡 HOLD")
                            
                    with col3:
                        st.write(f"**Price:** ${np.random.uniform(49000, 51000):,.2f}")
                        st.write(f"**Quantity:** {np.random.uniform(0.001, 0.01):.6f} BTC")
                        
                    with col4:
                        pnl = np.random.normal(50, 100)
                        if pnl > 0:
                            st.success(f"**PnL:** +${pnl:.2f}")
                        else:
                            st.error(f"**PnL:** ${pnl:.2f}")
                            
                    with col5:
                        confidence = np.random.uniform(0.6, 0.95)
                        st.write(f"**Confidence:** {confidence:.1%}")
                        strategy_name = trade.get('signal', {}).get('strategy', 'unknown')
                        st.write(f"**Strategy:** {strategy_name}")
                        
                    st.divider()
                    
            # Trade statistics
            st.subheader("Trade Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Win Rate", "68.5%", "+2.1%")
            with col2:
                st.metric("Avg Win", "$127.50", "+$15.25")
            with col3:
                st.metric("Avg Loss", "-$89.25", "-$5.75")
            with col4:
                st.metric("Profit Factor", "1.85", "+0.12")
                
        else:
            st.info("No trades executed yet")
            
    with tab4:
        # Strategy signals and predictions
        st.subheader("AI Model Predictions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # XGBoost signal prediction
            st.write("**XGBoost Trade Signal**")
            
            signal_proba = np.random.dirichlet([2, 5, 2])  # [sell, hold, buy]
            signal_labels = ['SELL', 'HOLD', 'BUY']
            signal_colors = ['red', 'gray', 'green']
            
            fig = go.Figure(data=go.Bar(
                x=signal_labels,
                y=signal_proba,
                marker_color=signal_colors,
                opacity=0.7
            ))
            
            fig.update_layout(
                title="Signal Probability",
                xaxis_title="Action",
                yaxis_title="Probability",
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show actual prediction
            max_signal = signal_labels[np.argmax(signal_proba)]
            max_prob = np.max(signal_proba)
            
            if max_prob > 0.6:
                if max_signal == 'BUY':
                    st.success(f"🟢 **{max_signal}** ({max_prob:.1%} confidence)")
                elif max_signal == 'SELL':
                    st.error(f"🔴 **{max_signal}** ({max_prob:.1%} confidence)")
                else:
                    st.info(f"🟡 **{max_signal}** ({max_prob:.1%} confidence)")
            else:
                st.warning(f"⚠️ Low confidence signal: {max_signal} ({max_prob:.1%})")
                
        with col2:
            # LSTM price prediction
            st.write("**LSTM Price Prediction**")
            
            current_price = 50000
            predicted_prices = [current_price + np.random.normal(0, 100) for _ in range(10)]
            future_times = [datetime.now() + timedelta(minutes=i) for i in range(1, 11)]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=[datetime.now()] + future_times,
                y=[current_price] + predicted_prices,
                mode='lines+markers',
                name='Predicted Price',
                line=dict(dash='dash', color='blue')
            ))
            
            fig.update_layout(
                title="Next 10 Minutes Price Prediction",
                xaxis_title="Time",
                yaxis_title="Price (USDT)",
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Price change prediction
            price_change = (predicted_prices[-1] - current_price) / current_price * 100
            if price_change > 0.1:
                st.success(f"📈 Predicted increase: +{price_change:.2f}%")
            elif price_change < -0.1:
                st.error(f"📉 Predicted decrease: {price_change:.2f}%")
            else:
                st.info(f"➡️ Predicted stability: {price_change:.2f}%")
                
        # Technical indicators
        st.subheader("Technical Indicators")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            rsi = np.random.uniform(30, 70)
            st.metric("RSI (14)", f"{rsi:.1f}", 
                     "Oversold" if rsi < 30 else "Overbought" if rsi > 70 else "Neutral")
                     
        with col2:
            macd = np.random.normal(0, 50)
            st.metric("MACD", f"{macd:.2f}", 
                     "Bullish" if macd > 0 else "Bearish")
                     
        with col3:
            bb_position = np.random.uniform(0, 1)
            st.metric("BB Position", f"{bb_position:.2f}", 
                     "Upper" if bb_position > 0.8 else "Lower" if bb_position < 0.2 else "Middle")
                     
        with col4:
            volume_ratio = np.random.uniform(0.5, 2.0)
            st.metric("Volume Ratio", f"{volume_ratio:.2f}x", 
                     "High" if volume_ratio > 1.5 else "Low" if volume_ratio < 0.8 else "Normal")
                     
    # Risk monitoring section
    st.subheader("⚠️ Risk Monitoring")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Daily loss limit
        daily_loss_pct = portfolio_status.get('daily_pnl', 0) / portfolio_status.get('total_balance', 1) * 100
        daily_limit = -5  # -5% daily limit
        
        progress = max(0, (daily_limit - daily_loss_pct) / daily_limit)
        
        if daily_loss_pct <= daily_limit:
            st.error(f"🚨 Daily loss limit exceeded: {daily_loss_pct:.2f}%")
        elif daily_loss_pct <= daily_limit * 0.5:
            st.warning(f"⚠️ Approaching daily loss limit: {daily_loss_pct:.2f}%")
        else:
            st.success(f"✅ Daily PnL within limits: {daily_loss_pct:.2f}%")
            
    with col2:
        # Position size monitoring
        max_position_limit = 20  # 20% max position
        current_position_pct = np.random.uniform(5, 15)  # Mock current position
        
        if current_position_pct >= max_position_limit:
            st.error(f"🚨 Max position size exceeded: {current_position_pct:.1f}%")
        elif current_position_pct >= max_position_limit * 0.8:
            st.warning(f"⚠️ High position exposure: {current_position_pct:.1f}%")
        else:
            st.success(f"✅ Position size OK: {current_position_pct:.1f}%")
            
    with col3:
        # Connection status
        exchange_connected = True  # Mock status
        model_loaded = True  # Mock status
        
        if exchange_connected and model_loaded:
            st.success("✅ All systems operational")
        elif exchange_connected:
            st.warning("⚠️ Exchange OK, Model issues")
        elif model_loaded:
            st.warning("⚠️ Model OK, Exchange issues")
        else:
            st.error("🚨 System connectivity issues")
            
    # Footer
    st.markdown("---")
    st.markdown(
        "🤖 **Crypto Trading AI Agent** | "
        f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
        "⚡ Powered by ML & Technical Analysis"
    )


if __name__ == "__main__":
    main()
