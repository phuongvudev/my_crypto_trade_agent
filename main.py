"""
Main runner for the Crypto Trading AI Agent.
Orchestrates data collection, signal generation, trade execution, and monitoring.
"""

import asyncio
import signal
import sys
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import yaml
import os
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/trading_agent.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Import our modules
try:
    from trading.exchange import ExchangeManager
    from trading.executor import TradingExecutor
    from strategies.ml_based import MLTradingStrategy
    from strategies.rule_based import RuleBasedStrategy
    from backtest.backtester import Backtester
    from features.feature_engineering import FeatureEngineer
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.error("Make sure all dependencies are installed: pip install -r requirements.txt")
    sys.exit(1)


class TradingAgent:
    """
    Main trading agent that orchestrates the entire trading system.
    Handles data collection, signal generation, trade execution, and monitoring.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the trading agent.
        
        Args:
            config_path: Path to configuration file
        """
        logger.info("Initializing Crypto Trading AI Agent...")
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.exchange_manager = None
        self.trading_executor = None
        self.active_strategy = None
        self.strategies = {}
        
        # Operating mode
        self.mode = self.config.get('mode', 'paper')  # paper, live, backtest
        self.running = False
        self.error_count = 0
        self.max_errors = 10
        
        # Performance tracking
        self.loop_count = 0
        self.start_time = None
        self.last_signal_time = None
        
        # Initialize system
        self._initialize_system()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            if not os.path.exists(config_path):
                logger.error(f"Configuration file not found: {config_path}")
                raise FileNotFoundError(f"Configuration file not found: {config_path}")
                
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
                
            # Validate required configuration sections
            required_sections = ['exchange', 'trading', 'strategy']
            for section in required_sections:
                if section not in config:
                    raise ValueError(f"Missing required configuration section: {section}")
                    
            logger.info(f"Configuration loaded from {config_path}")
            return config
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
            
    def _initialize_system(self) -> None:
        """Initialize all system components."""
        try:
            # Create necessary directories
            os.makedirs("logs", exist_ok=True)
            os.makedirs("data/live", exist_ok=True)
            os.makedirs("data/historical", exist_ok=True)
            os.makedirs("models", exist_ok=True)
            
            # Initialize exchange manager
            logger.info("Initializing exchange manager...")
            self.exchange_manager = ExchangeManager(config_path="config.yaml")
            
            # Test exchange connection
            if not self.exchange_manager.check_connection():
                logger.warning("Exchange connection failed - some features may be limited")
                
            # Initialize trading executor
            logger.info("Initializing trading executor...")
            self.trading_executor = TradingExecutor(self.exchange_manager, self.config)
            
            # Initialize strategies
            logger.info("Initializing trading strategies...")
            self._initialize_strategies()
            
            # Set active strategy
            active_strategy_name = self.config['strategy']['active']
            if active_strategy_name in self.strategies:
                self.active_strategy = self.strategies[active_strategy_name]
                logger.info(f"Active strategy set to: {active_strategy_name}")
            else:
                logger.error(f"Unknown strategy: {active_strategy_name}")
                raise ValueError(f"Unknown strategy: {active_strategy_name}")
                
            logger.info("System initialization completed successfully")
            
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            raise
            
    def _initialize_strategies(self) -> None:
        """Initialize all available trading strategies."""
        try:
            # Rule-based strategy
            self.strategies['rule_based'] = RuleBasedStrategy(self.config)
            logger.info("Rule-based strategy initialized")
            
            # ML-based strategy
            try:
                self.strategies['ml_based'] = MLTradingStrategy(self.config)
                logger.info("ML-based strategy initialized")
            except Exception as e:
                logger.warning(f"ML-based strategy initialization failed: {e}")
                logger.warning("ML strategy will not be available")
                
            # Hybrid strategy (combines both)
            if 'rule_based' in self.strategies and 'ml_based' in self.strategies:
                self.strategies['hybrid'] = HybridStrategy(
                    self.strategies['rule_based'],
                    self.strategies['ml_based'],
                    self.config
                )
                logger.info("Hybrid strategy initialized")
                
        except Exception as e:
            logger.error(f"Strategy initialization failed: {e}")
            raise
            
    async def run_live_trading(self) -> None:
        """Run the main live trading loop."""
        logger.info(f"Starting live trading in {self.mode} mode...")
        
        self.running = True
        self.start_time = datetime.now()
        self.error_count = 0
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        try:
            update_interval = self.config['data']['update_interval']
            
            while self.running:
                loop_start_time = time.time()
                
                try:
                    # Main trading loop iteration
                    await self._trading_loop_iteration()
                    
                    # Reset error count on successful iteration
                    self.error_count = 0
                    self.loop_count += 1
                    
                    # Log progress periodically
                    if self.loop_count % 10 == 0:
                        runtime = datetime.now() - self.start_time
                        logger.info(f"Trading loop iteration {self.loop_count}, "
                                   f"runtime: {runtime}, errors: {self.error_count}")
                        
                except Exception as e:
                    self.error_count += 1
                    logger.error(f"Error in trading loop iteration {self.loop_count}: {e}")
                    
                    if self.error_count >= self.max_errors:
                        logger.error(f"Too many errors ({self.error_count}), stopping trading")
                        break
                        
                # Sleep for the remainder of the update interval
                loop_duration = time.time() - loop_start_time
                sleep_time = max(0, update_interval - loop_duration)
                
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                    
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received, shutting down...")
        except Exception as e:
            logger.error(f"Unexpected error in trading loop: {e}")
        finally:
            await self._shutdown()
            
    async def _trading_loop_iteration(self) -> None:
        """Single iteration of the trading loop."""
        try:
            # 1. Fetch latest market data
            logger.debug("Fetching latest market data...")
            live_data = self.exchange_manager.fetch_live_data()
            
            if not live_data:
                logger.warning("No market data received, skipping iteration")
                return
                
            # Use primary timeframe for signal generation
            primary_timeframe = self.config['trading']['timeframes'][0]
            market_df = live_data.get(primary_timeframe)
            
            if market_df is None or market_df.empty:
                logger.warning(f"No data for primary timeframe {primary_timeframe}")
                return
                
            # 2. Get current portfolio status
            portfolio_status = self.trading_executor.get_portfolio_status()
            current_position = self._calculate_current_position(portfolio_status)
            
            # 3. Generate trading signal
            logger.debug("Generating trading signal...")
            signal_data = self.active_strategy.generate_signal(
                df=market_df,
                current_position=current_position
            )
            
            self.last_signal_time = datetime.now()
            
            # Log signal information
            signal = signal_data.get('signal', 0)
            confidence = signal_data.get('confidence', 0.0)
            strategy_name = signal_data.get('strategy', 'unknown')
            
            logger.info(f"Signal generated: {signal} (confidence: {confidence:.3f}, "
                       f"strategy: {strategy_name})")
                       
            # 4. Execute trade if signal is strong enough
            if signal != 0 and confidence > 0.1:
                logger.info(f"Executing trade signal: {signal}")
                
                execution_result = self.trading_executor.execute_signal(signal_data)
                
                if execution_result['status'] == 'success':
                    logger.info(f"Trade executed successfully: {execution_result['message']}")
                else:
                    logger.warning(f"Trade execution failed: {execution_result['message']}")
                    
            else:
                logger.debug(f"No trade executed - weak signal or hold: {signal} "
                           f"(confidence: {confidence:.3f})")
                           
            # 5. Risk monitoring
            self._monitor_risk(portfolio_status, signal_data)
            
            # 6. Save data if configured
            if self.config['data']['save_data']:
                self._save_market_data(live_data)
                
        except Exception as e:
            logger.error(f"Error in trading loop iteration: {e}")
            raise
            
    def _calculate_current_position(self, portfolio_status: Dict[str, Any]) -> float:
        """Calculate current position size as a fraction of portfolio."""
        try:
            total_balance = portfolio_status.get('total_balance', 0)
            positions = portfolio_status.get('positions', {})
            
            if not positions or total_balance <= 0:
                return 0.0
                
            # Calculate net position value
            position_value = 0.0
            for position_data in positions.values():
                if isinstance(position_data, dict):
                    size = position_data.get('size', 0)
                    current_price = portfolio_status.get('current_price', 0)
                    side = position_data.get('side', 'long')
                    
                    if side == 'long':
                        position_value += size * current_price
                    else:  # short
                        position_value -= size * current_price
                        
            return position_value / total_balance
            
        except Exception as e:
            logger.error(f"Error calculating current position: {e}")
            return 0.0
            
    def _monitor_risk(self, portfolio_status: Dict[str, Any], signal_data: Dict[str, Any]) -> None:
        """Monitor risk metrics and take action if needed."""
        try:
            # Check daily loss limit
            daily_pnl = portfolio_status.get('daily_pnl', 0)
            total_balance = portfolio_status.get('total_balance', 1)
            daily_loss_pct = daily_pnl / total_balance
            
            daily_limit = self.config['trading']['risk_management']['daily_loss_limit_pct']
            
            if daily_loss_pct <= -daily_limit:
                logger.error(f"Daily loss limit exceeded: {daily_loss_pct:.2%}")
                self.trading_executor.disable_trading()
                
            # Check maximum drawdown
            total_pnl = portfolio_status.get('total_pnl', 0)
            total_return = total_pnl / 10000  # Assuming 10k initial balance
            
            if total_return <= -0.20:  # 20% max drawdown
                logger.error(f"Maximum drawdown exceeded: {total_return:.2%}")
                logger.info("Closing all positions and disabling trading")
                self.trading_executor.close_all_positions()
                self.trading_executor.disable_trading()
                
            # Log risk metrics periodically
            if self.loop_count % 100 == 0:
                logger.info(f"Risk metrics - Daily PnL: {daily_loss_pct:.2%}, "
                           f"Total Return: {total_return:.2%}")
                           
        except Exception as e:
            logger.error(f"Error in risk monitoring: {e}")
            
    def _save_market_data(self, live_data: Dict[str, Any]) -> None:
        """Save market data to files."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            for timeframe, df in live_data.items():
                if not df.empty:
                    filename = f"data/live/{self.config['trading']['symbol'].replace('/', '_')}_{timeframe}_{timestamp}.csv"
                    df.tail(100).to_csv(filename)  # Save only last 100 rows
                    
        except Exception as e:
            logger.error(f"Error saving market data: {e}")
            
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.running = False
        
    async def _shutdown(self) -> None:
        """Perform graceful shutdown."""
        logger.info("Performing graceful shutdown...")
        
        try:
            # Close all open positions
            if self.trading_executor:
                logger.info("Closing all open positions...")
                close_results = self.trading_executor.close_all_positions()
                for result in close_results:
                    logger.info(f"Position close result: {result}")
                    
            # Final portfolio status
            if self.trading_executor:
                final_status = self.trading_executor.get_portfolio_status()
                logger.info(f"Final portfolio status: {final_status}")
                
            # Runtime statistics
            if self.start_time:
                runtime = datetime.now() - self.start_time
                logger.info(f"Trading session statistics:")
                logger.info(f"  Runtime: {runtime}")
                logger.info(f"  Loop iterations: {self.loop_count}")
                logger.info(f"  Error count: {self.error_count}")
                logger.info(f"  Last signal time: {self.last_signal_time}")
                
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            
        logger.info("Shutdown completed")
        
    def run_backtest(self, 
                    start_date: Optional[str] = None, 
                    end_date: Optional[str] = None,
                    strategy: Optional[str] = None) -> Dict[str, Any]:
        """Run backtesting mode."""
        logger.info("Running backtesting mode...")
        
        try:
            # Initialize backtester
            backtester = Backtester(self.config)
            
            # Fetch historical data
            symbol = self.config['trading']['symbol']
            days = 30  # Default to 30 days
            
            if start_date and end_date:
                start = datetime.fromisoformat(start_date)
                end = datetime.fromisoformat(end_date)
                days = (end - start).days
                
            logger.info(f"Fetching {days} days of historical data for {symbol}")
            historical_data = self.exchange_manager.fetch_historical_data(days=days)
            
            if historical_data.empty:
                logger.error("No historical data available for backtesting")
                return {'error': 'No historical data available'}
                
            # Run backtest
            strategy_name = strategy or self.config['strategy']['active']
            results = backtester.run_backtest(
                df=historical_data,
                strategy_name=strategy_name,
                start_date=start_date,
                end_date=end_date
            )
            
            # Save results
            results_file = f"backtest_results_{strategy_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            backtester.save_results(results_file)
            
            logger.info(f"Backtest completed. Results saved to {results_file}")
            return results
            
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            return {'error': str(e)}


class HybridStrategy:
    """Hybrid strategy that combines rule-based and ML-based signals."""
    
    def __init__(self, rule_strategy, ml_strategy, config):
        self.rule_strategy = rule_strategy
        self.ml_strategy = ml_strategy
        self.config = config
        
    def generate_signal(self, df, current_position=0.0, entry_price=None):
        """Generate combined signal from both strategies."""
        try:
            # Get signals from both strategies
            rule_signal = self.rule_strategy.generate_signal(df, current_position, entry_price)
            ml_signal = self.ml_strategy.generate_signal(df, current_position)
            
            # Combine signals with weighted approach
            rule_weight = 0.4
            ml_weight = 0.6
            
            combined_confidence = (
                rule_signal.get('confidence', 0) * rule_weight +
                ml_signal.get('confidence', 0) * ml_weight
            )
            
            # Use ML signal if both agree, otherwise be conservative
            rule_sig = rule_signal.get('signal', 0)
            ml_sig = ml_signal.get('signal', 0)
            
            if rule_sig == ml_sig:
                final_signal = rule_sig
                combined_confidence *= 1.2  # Boost confidence when strategies agree
            elif abs(rule_sig) > abs(ml_sig):
                final_signal = rule_sig
                combined_confidence *= 0.7  # Reduce confidence for disagreement
            else:
                final_signal = ml_sig
                combined_confidence *= 0.7
                
            return {
                'signal': final_signal,
                'confidence': min(1.0, combined_confidence),
                'position_size': min(0.1, combined_confidence * 0.2),
                'strategy': 'hybrid',
                'rule_signal': rule_signal,
                'ml_signal': ml_signal,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in hybrid strategy: {e}")
            return {
                'signal': 0,
                'confidence': 0.0,
                'position_size': 0.0,
                'strategy': 'hybrid',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }


def main():
    """Main entry point for the trading agent."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Crypto Trading AI Agent')
    parser.add_argument('--mode', choices=['live', 'paper', 'backtest'], 
                       default='paper', help='Operating mode')
    parser.add_argument('--config', default='config.yaml', 
                       help='Configuration file path')
    parser.add_argument('--strategy', choices=['rule_based', 'ml_based', 'hybrid'],
                       help='Strategy to use (overrides config)')
    parser.add_argument('--start-date', help='Backtest start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='Backtest end date (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    try:
        # Initialize trading agent
        agent = TradingAgent(args.config)
        
        # Override strategy if specified
        if args.strategy and args.strategy in agent.strategies:
            agent.active_strategy = agent.strategies[args.strategy]
            logger.info(f"Strategy overridden to: {args.strategy}")
            
        # Run in specified mode
        if args.mode == 'backtest':
            results = agent.run_backtest(
                start_date=args.start_date,
                end_date=args.end_date,
                strategy=args.strategy
            )
            
            if 'error' not in results:
                print("\n=== BACKTEST RESULTS ===")
                metrics = results.get('metrics', {})
                print(f"Total Return: {metrics.get('total_return', 0):.2%}")
                print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}")
                print(f"Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
                print(f"Win Rate: {metrics.get('win_rate', 0):.2%}")
                print(f"Total Trades: {metrics.get('total_trades', 0)}")
            else:
                print(f"Backtest failed: {results['error']}")
                
        else:
            # Live or paper trading
            agent.mode = args.mode
            print(f"\n🚀 Starting Crypto Trading AI Agent in {args.mode.upper()} mode")
            print(f"Strategy: {agent.config['strategy']['active']}")
            print(f"Symbol: {agent.config['trading']['symbol']}")
            print("Press Ctrl+C to stop\n")
            
            # Run the trading loop
            asyncio.run(agent.run_live_trading())
            
    except KeyboardInterrupt:
        print("\n👋 Trading agent stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
