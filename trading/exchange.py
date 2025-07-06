"""
Exchange module for handling cryptocurrency exchange connections and data fetching.
Supports Binance via CCXT with live and historical data collection.
"""

import ccxt
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import time
import asyncio
import aiohttp
from loguru import logger
import yaml
import os


class ExchangeManager:
    """
    Manages cryptocurrency exchange connections and data operations.
    Supports live data streaming, historical data fetching, and order execution.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize exchange manager with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.exchange = None
        self.symbol = self.config['trading']['symbol']
        self.timeframes = self.config['trading']['timeframes']
        self.data_cache = {}
        
        # Initialize exchange connection
        self._initialize_exchange()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            
            # Replace environment variables
            self._replace_env_vars(config)
            return config
            
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing configuration file: {e}")
            raise
            
    def _replace_env_vars(self, config: Dict[str, Any]) -> None:
        """Replace environment variable placeholders in config."""
        def replace_recursive(obj):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    obj[key] = replace_recursive(value)
            elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
                env_var = obj[2:-1]
                obj = os.getenv(env_var, obj)
            return obj
        
        replace_recursive(config)
        
    def _initialize_exchange(self) -> None:
        """Initialize CCXT exchange connection."""
        try:
            exchange_config = self.config['exchange']
            
            # Initialize exchange class
            if exchange_config['name'].lower() == 'binance':
                self.exchange = ccxt.binance({
                    'apiKey': exchange_config.get('api_key', ''),
                    'secret': exchange_config.get('secret', ''),
                    'sandbox': exchange_config.get('sandbox', True),
                    'enableRateLimit': True,
                    'options': {
                        'defaultType': 'spot',  # spot, future, margin
                    }
                })
            else:
                raise ValueError(f"Unsupported exchange: {exchange_config['name']}")
                
            # Test connection - skip if using dummy credentials for paper trading
            api_key = exchange_config.get('api_key', '')
            api_secret = exchange_config.get('secret', '')
            
            if (api_key and api_secret and 
                not api_key.startswith('dummy') and 
                not api_secret.startswith('dummy') and
                len(api_key) > 10 and len(api_secret) > 10):
                try:
                    self.exchange.load_markets()
                    balance = self.exchange.fetch_balance()
                    logger.info(f"Successfully connected to {exchange_config['name']}")
                    logger.info(f"Account balance: {balance['total']}")
                except Exception as e:
                    logger.warning(f"Could not authenticate with exchange: {e}")
                    logger.info("Running in data-only mode")
            else:
                logger.info("Using dummy credentials or no API credentials - running in data-only mode")
                
                # For data-only mode, still load markets for public data access
                try:
                    self.exchange.load_markets()
                    logger.info("Successfully loaded market data for public access")
                except Exception as e:
                    logger.warning(f"Could not load market data: {e}")
                
        except Exception as e:
            logger.error(f"Failed to initialize exchange: {e}")
            raise
            
    def fetch_ohlcv(self, 
                    symbol: Optional[str] = None, 
                    timeframe: str = '1m', 
                    limit: int = 1000,
                    since: Optional[int] = None) -> pd.DataFrame:
        """
        Fetch OHLCV data from exchange.
        
        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT')
            timeframe: Timeframe ('1m', '5m', '1h', etc.)
            limit: Number of candles to fetch
            since: Timestamp to fetch data from
            
        Returns:
            DataFrame with OHLCV data
        """
        if symbol is None:
            symbol = self.symbol
            
        try:
            logger.info(f"Fetching {limit} {timeframe} candles for {symbol}")
            
            # Fetch data from exchange
            ohlcv = self.exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                limit=limit,
                since=since
            )
            
            if not ohlcv:
                logger.warning(f"No data received for {symbol}")
                return pd.DataFrame()
                
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Convert timestamp to datetime
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('datetime', inplace=True)
            
            # Ensure numeric types
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            df[numeric_columns] = df[numeric_columns].astype(float)
            
            logger.info(f"Successfully fetched {len(df)} candles")
            logger.info(f"Data range: {df.index.min()} to {df.index.max()}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching OHLCV data: {e}")
            raise
            
    def fetch_live_data(self, 
                       symbol: Optional[str] = None, 
                       timeframes: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        Fetch live data for multiple timeframes.
        
        Args:
            symbol: Trading symbol
            timeframes: List of timeframes to fetch
            
        Returns:
            Dictionary mapping timeframes to DataFrames
        """
        if symbol is None:
            symbol = self.symbol
        if timeframes is None:
            timeframes = self.timeframes
            
        live_data = {}
        
        for timeframe in timeframes:
            try:
                df = self.fetch_ohlcv(symbol=symbol, timeframe=timeframe, limit=100)
                live_data[timeframe] = df
                
                # Cache the data
                cache_key = f"{symbol}_{timeframe}"
                self.data_cache[cache_key] = {
                    'data': df,
                    'timestamp': datetime.now()
                }
                
            except Exception as e:
                logger.error(f"Error fetching live data for {timeframe}: {e}")
                
        return live_data
        
    def fetch_historical_data(self, 
                             symbol: Optional[str] = None,
                             timeframe: str = '1m',
                             days: int = 30) -> pd.DataFrame:
        """
        Fetch historical data for backtesting.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            days: Number of days of historical data
            
        Returns:
            DataFrame with historical OHLCV data
        """
        if symbol is None:
            symbol = self.symbol
            
        logger.info(f"Fetching {days} days of historical data for {symbol}")
        
        # Calculate time range
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        since = int(start_time.timestamp() * 1000)
        
        # Fetch data in chunks if needed
        all_data = []
        current_since = since
        limit = min(1000, self.config['data']['historical_limit'])
        
        while current_since < int(end_time.timestamp() * 1000):
            try:
                df = self.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    limit=limit,
                    since=current_since
                )
                
                if df.empty:
                    break
                    
                all_data.append(df)
                
                # Update since timestamp for next iteration
                current_since = int(df.index[-1].timestamp() * 1000) + 1
                
                # Rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error fetching historical data chunk: {e}")
                break
                
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=False)
            combined_df = combined_df.drop_duplicates().sort_index()
            
            # Save to file if configured
            if self.config['data']['save_data']:
                self._save_historical_data(combined_df, symbol, timeframe)
                
            logger.info(f"Fetched {len(combined_df)} historical candles")
            return combined_df
        else:
            logger.warning("No historical data fetched")
            return pd.DataFrame()
            
    def _save_historical_data(self, df: pd.DataFrame, symbol: str, timeframe: str) -> None:
        """Save historical data to file."""
        try:
            os.makedirs("data/historical", exist_ok=True)
            filename = f"data/historical/{symbol.replace('/', '_')}_{timeframe}.csv"
            df.to_csv(filename)
            logger.info(f"Saved historical data to {filename}")
        except Exception as e:
            logger.error(f"Error saving historical data: {e}")
            
    def get_ticker(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Get current ticker information.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Ticker information dictionary
        """
        if symbol is None:
            symbol = self.symbol
            
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return ticker
        except Exception as e:
            logger.error(f"Error fetching ticker for {symbol}: {e}")
            raise
            
    def get_order_book(self, symbol: Optional[str] = None, limit: int = 20) -> Dict[str, Any]:
        """
        Get order book data.
        
        Args:
            symbol: Trading symbol
            limit: Number of orders to fetch
            
        Returns:
            Order book dictionary
        """
        if symbol is None:
            symbol = self.symbol
            
        try:
            order_book = self.exchange.fetch_order_book(symbol, limit=limit)
            return order_book
        except Exception as e:
            logger.error(f"Error fetching order book for {symbol}: {e}")
            raise
            
    def get_account_balance(self) -> Dict[str, Any]:
        """
        Get account balance information.
        
        Returns:
            Balance dictionary
        """
        try:
            balance = self.exchange.fetch_balance()
            return balance
        except Exception as e:
            logger.error(f"Error fetching account balance: {e}")
            raise
            
    def check_connection(self) -> bool:
        """
        Check if exchange connection is working.
        
        Returns:
            True if connection is working
        """
        try:
            # Try to fetch exchange status first
            self.exchange.fetch_status()
            return True
        except Exception as e:
            # If status check fails (common with Binance testnet), try a simple ticker fetch
            try:
                logger.warning(f"Status check failed ({e}), trying ticker fetch as fallback")
                self.exchange.fetch_ticker('BTC/USDT')
                logger.info("Connection verified via ticker fetch")
                return True
            except Exception as e2:
                logger.error(f"Exchange connection check failed: {e2}")
                return False
            
    def get_cached_data(self, symbol: str, timeframe: str, max_age_seconds: int = 60) -> Optional[pd.DataFrame]:
        """
        Get cached data if available and not too old.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            max_age_seconds: Maximum age of cached data in seconds
            
        Returns:
            Cached DataFrame or None
        """
        cache_key = f"{symbol}_{timeframe}"
        
        if cache_key in self.data_cache:
            cached_item = self.data_cache[cache_key]
            age = (datetime.now() - cached_item['timestamp']).total_seconds()
            
            if age <= max_age_seconds:
                logger.debug(f"Using cached data for {cache_key} (age: {age:.1f}s)")
                return cached_item['data']
                
        return None


# Example usage and testing
if __name__ == "__main__":
    """
    Example usage of ExchangeManager for testing.
    """
    try:
        # Initialize exchange manager
        exchange_manager = ExchangeManager()
        
        # Test connection
        if exchange_manager.check_connection():
            print("✅ Exchange connection successful")
        else:
            print("❌ Exchange connection failed")
            
        # Fetch live data
        print("\n📊 Fetching live data...")
        live_data = exchange_manager.fetch_live_data()
        
        for timeframe, df in live_data.items():
            if not df.empty:
                latest_price = df['close'].iloc[-1]
                print(f"{timeframe}: Latest price = {latest_price}")
                
        # Fetch historical data
        print("\n📈 Fetching historical data...")
        historical_df = exchange_manager.fetch_historical_data(days=7)
        
        if not historical_df.empty:
            print(f"Historical data: {len(historical_df)} candles")
            print(f"Date range: {historical_df.index.min()} to {historical_df.index.max()}")
            
        # Get ticker
        print("\n💰 Getting ticker...")
        ticker = exchange_manager.get_ticker()
        print(f"Current price: {ticker['last']}")
        print(f"24h change: {ticker['percentage']}%")
        
    except Exception as e:
        logger.error(f"Error in example usage: {e}")
