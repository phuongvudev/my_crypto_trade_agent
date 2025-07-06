"""
Simplified Crypto Trading AI Dashboard - Debug Version
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import os

# Page config
st.set_page_config(
    page_title="Crypto Trading AI Dashboard",
    page_icon="📈",
    layout="wide"
)

def load_simple_portfolio_data():
    """Simple portfolio data loader."""
    return {
        'total_balance': 10500.75,
        'daily_pnl': 45.50,
        'total_pnl': 500.75,
        'open_positions': 1,
        'total_trades': 15,
        'current_price': 65430.50,
        'trading_enabled': True,
        'paper_trading': True
    }

def load_simple_trade_data():
    """Simple trade data loader."""
    if os.path.exists('logs/trades.json'):
        try:
            with open('logs/trades.json', 'r') as f:
                trades = json.load(f)
            return trades
        except:
            return []
    return []

def load_simple_market_data():
    """Simple market data loader."""
    # Generate realistic BTC price data
    dates = pd.date_range(start=datetime.now() - timedelta(hours=24), 
                         end=datetime.now(), freq='1min')
    
    initial_price = 65000
    prices = []
    current_price = initial_price
    
    for i in range(len(dates)):
        # Random walk with slight upward bias
        change = np.random.normal(0.0001, 0.001)
        current_price *= (1 + change)
        prices.append(current_price)
    
    df = pd.DataFrame({
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.001))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.001))) for p in prices],
        'close': prices,
        'volume': np.random.lognormal(15, 0.5, len(dates))
    }, index=dates)
    
    # Ensure OHLC integrity
    df['high'] = np.maximum(df['high'], np.maximum(df['open'], df['close']))
    df['low'] = np.minimum(df['low'], np.minimum(df['open'], df['close']))
    
    return df

def main():
    """Main dashboard function."""
    
    # Title
    st.title("📈 Crypto Trading AI Dashboard")
    st.markdown("---")
    
    # Load data
    try:
        portfolio_status = load_simple_portfolio_data()
        trades = load_simple_trade_data()
        market_data = load_simple_market_data()
        
        st.success("✅ Data loaded successfully")
        
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return
    
    # Sidebar
    with st.sidebar:
        st.title("🤖 Trading AI Control")
        
        # Status indicators
        st.subheader("System Status")
        st.success("🟢 System Online")
        st.info("📄 Paper Trading Mode")
        
        # Manual refresh button
        if st.button("🔄 Refresh Data"):
            st.rerun()
        
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
    
    # Top metrics row
    st.subheader("📊 Portfolio Overview")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Total Balance",
            f"${portfolio_status['total_balance']:,.2f}",
            delta=f"${portfolio_status['daily_pnl']:+,.2f}"
        )
        
    with col2:
        st.metric(
            "Daily PnL",
            f"${portfolio_status['daily_pnl']:+,.2f}",
            delta=f"{(portfolio_status['daily_pnl'] / portfolio_status['total_balance'] * 100):+.2f}%"
        )
        
    with col3:
        st.metric(
            "Total PnL",
            f"${portfolio_status['total_pnl']:+,.2f}",
            delta=f"{(portfolio_status['total_pnl'] / 10000 * 100):+.2f}%"
        )
        
    with col4:
        st.metric(
            "Open Positions",
            portfolio_status['open_positions'],
            delta=f"Total: {portfolio_status['total_trades']}"
        )
        
    with col5:
        st.metric(
            "BTC Price",
            f"${portfolio_status['current_price']:,.2f}",
            delta=f"${np.random.normal(0, 50):+.2f}"
        )
    
    # Status indicators
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if portfolio_status['trading_enabled']:
            st.success("🟢 Trading Active")
        else:
            st.error("🔴 Trading Inactive")
            
    with col2:
        if portfolio_status['paper_trading']:
            st.warning("📄 Paper Trading")
        else:
            st.info("💰 Live Trading")
            
    with col3:
        st.info(f"🤖 Strategy: {strategy.title()}")
    
    # Charts section
    st.subheader("📈 Market Data & Analysis")
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Price Chart", "Trade History", "Performance"])
    
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
            
            fig.update_layout(
                title="BTC/USDT Price Chart (24H)",
                xaxis_title="Time",
                yaxis_title="Price (USDT)",
                height=500,
                xaxis_rangeslider_visible=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No market data available")
    
    with tab2:
        st.subheader("📋 Recent Trades")
        if trades:
            # Display trade summary
            st.write(f"**Total Trades:** {len(trades)}")
            
            # Show first few trades in a table format
            trade_data = []
            for i, trade in enumerate(trades[:5]):  # Show last 5 trades
                trade_data.append({
                    'Time': trade.get('timestamp', 'N/A')[:19] if 'timestamp' in trade else 'N/A',
                    'Signal': trade.get('signal', {}).get('signal', 'N/A'),
                    'Status': trade.get('result', {}).get('status', 'N/A'),
                    'Message': trade.get('result', {}).get('message', 'N/A')[:50] + '...' if len(str(trade.get('result', {}).get('message', ''))) > 50 else trade.get('result', {}).get('message', 'N/A')
                })
            
            if trade_data:
                df_trades = pd.DataFrame(trade_data)
                st.dataframe(df_trades, use_container_width=True)
            else:
                st.info("No trade data to display")
        else:
            st.info("No trades found")
    
    with tab3:
        st.subheader("📊 Portfolio Performance")
        
        # Mock portfolio performance chart
        dates = pd.date_range(start=datetime.now() - timedelta(days=30), 
                             end=datetime.now(), freq='1d')
        portfolio_values = 10000 + np.random.normal(0, 100, len(dates)).cumsum()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=portfolio_values,
            mode='lines',
            name='Portfolio Value',
            line=dict(color='green', width=2)
        ))
        
        fig.update_layout(
            title="Portfolio Value Over Time (30 Days)",
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("30D Return", "+5.07%", delta="+1.2%")
        with col2:
            st.metric("Win Rate", "73.3%", delta="+5.1%")
        with col3:
            st.metric("Sharpe Ratio", "1.85", delta="+0.3")
        with col4:
            st.metric("Max Drawdown", "-2.1%", delta="+0.5%")
    
    # Footer
    st.markdown("---")
    st.markdown(
        f"🤖 **Crypto Trading AI Agent** | "
        f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
        "⚡ Powered by ML & Technical Analysis"
    )

if __name__ == "__main__":
    main()
