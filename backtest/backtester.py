"""
Backtesting module for testing trading strategies on historical data.
Supports multiple strategies, performance metrics, and risk analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, asdict
import json
import os
from loguru import logger

# Local imports
from strategies.ml_based import MLTradingStrategy
from strategies.rule_based import RuleBasedStrategy
from features.feature_engineering import FeatureEngineer


@dataclass
class BacktestTrade:
    """Individual trade record for backtesting."""
    timestamp: datetime
    signal: int
    entry_price: float
    exit_price: Optional[float]
    quantity: float
    side: str  # 'buy' or 'sell'
    pnl: float
    commission: float
    strategy: str
    confidence: float
    duration: Optional[timedelta] = None


@dataclass
class BacktestMetrics:
    """Comprehensive backtesting performance metrics."""
    # Basic metrics
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    
    # Risk metrics
    max_drawdown: float
    value_at_risk_95: float
    conditional_var_95: float
    
    # Trade metrics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    
    # Time metrics
    avg_trade_duration: timedelta
    max_trade_duration: timedelta
    
    # Additional metrics
    calmar_ratio: float
    recovery_factor: float
    payoff_ratio: float


class Backtester:
    """
    Comprehensive backtesting engine for crypto trading strategies.
    Supports multiple strategies, realistic transaction costs, and detailed analysis.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize backtester.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.backtest_config = config.get('backtest', {})
        
        # Backtesting parameters
        self.initial_balance = self.backtest_config.get('initial_balance', 10000)
        self.commission = self.backtest_config.get('commission', 0.001)  # 0.1%
        self.slippage = self.backtest_config.get('slippage', 0.0001)  # 0.01%
        
        # Initialize strategies
        self.strategies = {}
        self._initialize_strategies()
        
        # Results storage
        self.results = {}
        self.trades = {}
        self.equity_curves = {}
        
    def _initialize_strategies(self) -> None:
        """Initialize available trading strategies."""
        try:
            # Rule-based strategy
            self.strategies['rule_based'] = RuleBasedStrategy(self.config)
            logger.info("Rule-based strategy initialized for backtesting")
            
            # ML-based strategy (if models are available)
            try:
                self.strategies['ml_based'] = MLTradingStrategy(self.config)
                logger.info("ML-based strategy initialized for backtesting")
            except Exception as e:
                logger.warning(f"ML strategy not available for backtesting: {e}")
                
        except Exception as e:
            logger.error(f"Error initializing strategies: {e}")
            
    def run_backtest(self, 
                    df: pd.DataFrame, 
                    strategy_name: str,
                    start_date: Optional[str] = None,
                    end_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Run backtest for a specific strategy.
        
        Args:
            df: Historical OHLCV data
            strategy_name: Name of strategy to test
            start_date: Start date for backtest (YYYY-MM-DD)
            end_date: End date for backtest (YYYY-MM-DD)
            
        Returns:
            Backtest results dictionary
        """
        logger.info(f"Starting backtest for {strategy_name}")
        
        try:
            # Validate inputs
            if strategy_name not in self.strategies:
                raise ValueError(f"Strategy '{strategy_name}' not available")
                
            if len(df) < 100:
                raise ValueError("Insufficient data for backtesting")
                
            # Filter data by date range
            if start_date or end_date:
                df = self._filter_by_date_range(df, start_date, end_date)
                
            # Initialize backtest state
            balance = self.initial_balance
            position = 0.0  # Current position size
            entry_price = None
            trades = []
            equity_curve = []
            
            strategy = self.strategies[strategy_name]
            
            logger.info(f"Backtesting {len(df)} data points from {df.index[0]} to {df.index[-1]}")
            
            # Run backtest simulation
            for i in range(max(60, len(df) // 10), len(df)):  # Start after warm-up period
                try:
                    # Get data window for strategy
                    window_df = df.iloc[:i+1]
                    current_price = df['close'].iloc[i]
                    timestamp = df.index[i]
                    
                    # Generate signal
                    signal_data = strategy.generate_signal(window_df, position, entry_price)
                    signal = signal_data.get('signal', 0)
                    confidence = signal_data.get('confidence', 0.0)
                    
                    # Execute trade if signal is valid
                    if signal != 0 and confidence > 0.1:
                        trade_result = self._execute_backtest_trade(
                            signal, current_price, timestamp, balance, position, 
                            entry_price, strategy_name, confidence
                        )
                        
                        if trade_result['executed']:
                            trades.append(trade_result['trade'])
                            balance = trade_result['new_balance']
                            position = trade_result['new_position']
                            entry_price = trade_result['entry_price']
                            
                    # Calculate current portfolio value
                    portfolio_value = balance
                    if position != 0:
                        unrealized_pnl = position * (current_price - entry_price) if entry_price else 0
                        portfolio_value += unrealized_pnl
                        
                    equity_curve.append({
                        'timestamp': timestamp,
                        'portfolio_value': portfolio_value,
                        'position': position,
                        'balance': balance
                    })
                    
                except Exception as e:
                    logger.error(f"Error at index {i}: {e}")
                    continue
                    
            # Close final position if open
            if position != 0:
                final_trade = self._close_final_position(
                    position, df['close'].iloc[-1], df.index[-1], 
                    balance, entry_price, strategy_name
                )
                if final_trade:
                    trades.append(final_trade)
                    
            # Calculate metrics
            metrics = self._calculate_metrics(trades, equity_curve, df)
            
            # Store results
            results = {
                'strategy': strategy_name,
                'metrics': asdict(metrics),
                'trades': [asdict(trade) for trade in trades],
                'equity_curve': equity_curve,
                'data_points': len(df),
                'backtest_period': {
                    'start': df.index[0].isoformat(),
                    'end': df.index[-1].isoformat(),
                    'days': (df.index[-1] - df.index[0]).days
                }
            }
            
            self.results[strategy_name] = results
            self.trades[strategy_name] = trades
            self.equity_curves[strategy_name] = equity_curve
            
            logger.info(f"Backtest completed for {strategy_name}")
            logger.info(f"Total return: {metrics.total_return:.2%}")
            logger.info(f"Sharpe ratio: {metrics.sharpe_ratio:.3f}")
            logger.info(f"Max drawdown: {metrics.max_drawdown:.2%}")
            logger.info(f"Total trades: {metrics.total_trades}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            return {'error': str(e)}
            
    def _execute_backtest_trade(self, signal: int, price: float, timestamp: datetime,
                               balance: float, current_position: float, entry_price: Optional[float],
                               strategy: str, confidence: float) -> Dict[str, Any]:
        """Execute a trade in the backtest simulation."""
        try:
            executed = False
            trade = None
            new_balance = balance
            new_position = current_position
            new_entry_price = entry_price
            
            # Calculate position size based on signal confidence and balance
            max_position_value = balance * 0.95  # Use 95% of balance max
            position_size = min(confidence * 0.5, 0.2) * max_position_value  # Max 20% position
            
            if signal > 0 and current_position <= 0:  # Buy signal
                # Close short position if any
                if current_position < 0:
                    close_pnl = -current_position * (entry_price - price)
                    commission_cost = abs(current_position) * price * self.commission
                    new_balance += close_pnl - commission_cost
                    new_position = 0
                    
                # Open long position
                slippage_cost = price * self.slippage
                effective_price = price + slippage_cost
                quantity = position_size / effective_price
                commission_cost = quantity * effective_price * self.commission
                
                if new_balance >= quantity * effective_price + commission_cost:
                    new_balance -= quantity * effective_price + commission_cost
                    new_position += quantity
                    new_entry_price = effective_price
                    
                    trade = BacktestTrade(
                        timestamp=timestamp,
                        signal=signal,
                        entry_price=effective_price,
                        exit_price=None,
                        quantity=quantity,
                        side='buy',
                        pnl=0.0,  # Will be calculated on exit
                        commission=commission_cost,
                        strategy=strategy,
                        confidence=confidence
                    )
                    executed = True
                    
            elif signal < 0 and current_position >= 0:  # Sell signal
                # Close long position if any
                if current_position > 0:
                    slippage_cost = price * self.slippage
                    effective_price = price - slippage_cost
                    close_pnl = current_position * (effective_price - entry_price)
                    commission_cost = current_position * effective_price * self.commission
                    new_balance += current_position * effective_price - commission_cost
                    
                    trade = BacktestTrade(
                        timestamp=timestamp,
                        signal=signal,
                        entry_price=entry_price,
                        exit_price=effective_price,
                        quantity=current_position,
                        side='sell',
                        pnl=close_pnl,
                        commission=commission_cost,
                        strategy=strategy,
                        confidence=confidence,
                        duration=timestamp - getattr(trade, 'entry_timestamp', timestamp)
                    )
                    new_position = 0
                    new_entry_price = None
                    executed = True
                    
            return {
                'executed': executed,
                'trade': trade,
                'new_balance': new_balance,
                'new_position': new_position,
                'entry_price': new_entry_price
            }
            
        except Exception as e:
            logger.error(f"Error executing backtest trade: {e}")
            return {
                'executed': False,
                'trade': None,
                'new_balance': balance,
                'new_position': current_position,
                'entry_price': entry_price
            }
            
    def _close_final_position(self, position: float, price: float, timestamp: datetime,
                             balance: float, entry_price: float, strategy: str) -> Optional[BacktestTrade]:
        """Close final position at end of backtest."""
        try:
            if position == 0:
                return None
                
            slippage_cost = price * self.slippage
            effective_price = price - slippage_cost if position > 0 else price + slippage_cost
            
            pnl = position * (effective_price - entry_price) if position > 0 else position * (entry_price - effective_price)
            commission_cost = abs(position) * effective_price * self.commission
            
            return BacktestTrade(
                timestamp=timestamp,
                signal=-1 if position > 0 else 1,
                entry_price=entry_price,
                exit_price=effective_price,
                quantity=abs(position),
                side='sell' if position > 0 else 'buy',
                pnl=pnl,
                commission=commission_cost,
                strategy=strategy,
                confidence=1.0  # Force close
            )
            
        except Exception as e:
            logger.error(f"Error closing final position: {e}")
            return None
            
    def _calculate_metrics(self, trades: List[BacktestTrade], 
                          equity_curve: List[Dict], df: pd.DataFrame) -> BacktestMetrics:
        """Calculate comprehensive backtest metrics."""
        try:
            if not trades or not equity_curve:
                return self._empty_metrics()
                
            # Convert equity curve to pandas series
            eq_df = pd.DataFrame(equity_curve)
            eq_df.set_index('timestamp', inplace=True)
            
            # Calculate returns
            portfolio_values = eq_df['portfolio_value']
            returns = portfolio_values.pct_change().dropna()
            
            # Basic metrics
            total_return = (portfolio_values.iloc[-1] - self.initial_balance) / self.initial_balance
            
            # Annualized return
            days = (eq_df.index[-1] - eq_df.index[0]).days
            annualized_return = (1 + total_return) ** (365.25 / max(days, 1)) - 1 if days > 0 else 0
            
            # Volatility
            volatility = returns.std() * np.sqrt(365.25 * 24 * 60)  # Annualized for minute data
            
            # Risk-adjusted returns
            risk_free_rate = 0.02  # 2% annual risk-free rate
            excess_returns = returns - (risk_free_rate / (365.25 * 24 * 60))
            sharpe_ratio = excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0
            
            # Sortino ratio (downside deviation)
            downside_returns = returns[returns < 0]
            downside_deviation = downside_returns.std() if len(downside_returns) > 0 else 0.001
            sortino_ratio = excess_returns.mean() / downside_deviation if downside_deviation > 0 else 0
            
            # Drawdown calculation
            running_max = portfolio_values.expanding().max()
            drawdown = (portfolio_values - running_max) / running_max
            max_drawdown = drawdown.min()
            
            # VaR and CVaR
            var_95 = returns.quantile(0.05) if len(returns) > 0 else 0
            cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else 0
            
            # Trade analysis
            trade_pnls = [trade.pnl for trade in trades if trade.pnl is not None]
            winning_trades = len([pnl for pnl in trade_pnls if pnl > 0])
            losing_trades = len([pnl for pnl in trade_pnls if pnl < 0])
            
            win_rate = winning_trades / len(trade_pnls) if trade_pnls else 0
            avg_win = np.mean([pnl for pnl in trade_pnls if pnl > 0]) if winning_trades > 0 else 0
            avg_loss = np.mean([pnl for pnl in trade_pnls if pnl < 0]) if losing_trades > 0 else 0
            
            # Profit factor
            gross_profit = sum([pnl for pnl in trade_pnls if pnl > 0])
            gross_loss = abs(sum([pnl for pnl in trade_pnls if pnl < 0]))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # Trade duration analysis
            durations = [trade.duration for trade in trades if trade.duration is not None]
            avg_trade_duration = np.mean(durations) if durations else timedelta(0)
            max_trade_duration = max(durations) if durations else timedelta(0)
            
            # Additional ratios
            calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
            recovery_factor = total_return / abs(max_drawdown) if max_drawdown != 0 else 0
            payoff_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
            
            return BacktestMetrics(
                total_return=total_return,
                annualized_return=annualized_return,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                max_drawdown=max_drawdown,
                value_at_risk_95=var_95,
                conditional_var_95=cvar_95,
                total_trades=len(trades),
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                win_rate=win_rate,
                avg_win=avg_win,
                avg_loss=avg_loss,
                profit_factor=profit_factor,
                avg_trade_duration=avg_trade_duration,
                max_trade_duration=max_trade_duration,
                calmar_ratio=calmar_ratio,
                recovery_factor=recovery_factor,
                payoff_ratio=payoff_ratio
            )
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return self._empty_metrics()
            
    def _empty_metrics(self) -> BacktestMetrics:
        """Return empty metrics for failed backtests."""
        return BacktestMetrics(
            total_return=0.0, annualized_return=0.0, volatility=0.0,
            sharpe_ratio=0.0, sortino_ratio=0.0, max_drawdown=0.0,
            value_at_risk_95=0.0, conditional_var_95=0.0, total_trades=0,
            winning_trades=0, losing_trades=0, win_rate=0.0,
            avg_win=0.0, avg_loss=0.0, profit_factor=0.0,
            avg_trade_duration=timedelta(0), max_trade_duration=timedelta(0),
            calmar_ratio=0.0, recovery_factor=0.0, payoff_ratio=0.0
        )
        
    def _filter_by_date_range(self, df: pd.DataFrame, 
                             start_date: Optional[str], 
                             end_date: Optional[str]) -> pd.DataFrame:
        """Filter DataFrame by date range."""
        filtered_df = df.copy()
        
        if start_date:
            filtered_df = filtered_df[filtered_df.index >= start_date]
        if end_date:
            filtered_df = filtered_df[filtered_df.index <= end_date]
            
        return filtered_df
        
    def compare_strategies(self, df: pd.DataFrame, 
                          strategies: List[str]) -> Dict[str, Any]:
        """Compare multiple strategies on the same dataset."""
        logger.info(f"Comparing strategies: {strategies}")
        
        comparison_results = {}
        
        for strategy in strategies:
            if strategy in self.strategies:
                result = self.run_backtest(df, strategy)
                comparison_results[strategy] = result
            else:
                logger.warning(f"Strategy '{strategy}' not available")
                
        # Generate comparison summary
        summary = self._generate_comparison_summary(comparison_results)
        
        return {
            'individual_results': comparison_results,
            'comparison_summary': summary
        }
        
    def _generate_comparison_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary comparison of strategy results."""
        if not results:
            return {}
            
        metrics_comparison = {}
        
        for strategy, result in results.items():
            if 'metrics' in result:
                metrics = result['metrics']
                metrics_comparison[strategy] = {
                    'total_return': metrics.get('total_return', 0),
                    'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                    'max_drawdown': metrics.get('max_drawdown', 0),
                    'win_rate': metrics.get('win_rate', 0),
                    'total_trades': metrics.get('total_trades', 0)
                }
                
        # Find best strategy for each metric
        best_strategies = {}
        for metric in ['total_return', 'sharpe_ratio', 'win_rate']:
            best_strategy = max(metrics_comparison.keys(), 
                              key=lambda s: metrics_comparison[s][metric])
            best_strategies[f'best_{metric}'] = best_strategy
            
        # Best drawdown (minimum)
        best_strategies['best_drawdown'] = min(metrics_comparison.keys(),
                                             key=lambda s: abs(metrics_comparison[s]['max_drawdown']))
        
        return {
            'metrics_comparison': metrics_comparison,
            'best_strategies': best_strategies,
            'strategies_tested': list(results.keys())
        }
        
    def plot_results(self, strategy_name: str, save_path: Optional[str] = None) -> None:
        """Plot backtest results for a strategy."""
        if strategy_name not in self.results:
            logger.error(f"No results found for strategy: {strategy_name}")
            return
            
        try:
            equity_curve = self.equity_curves[strategy_name]
            trades = self.trades[strategy_name]
            
            # Create subplot figure
            fig, axes = plt.subplots(3, 2, figsize=(15, 12))
            fig.suptitle(f'Backtest Results: {strategy_name}', fontsize=16)
            
            # Equity curve
            eq_df = pd.DataFrame(equity_curve)
            eq_df.set_index('timestamp', inplace=True)
            
            axes[0, 0].plot(eq_df.index, eq_df['portfolio_value'])
            axes[0, 0].set_title('Portfolio Value Over Time')
            axes[0, 0].set_ylabel('Portfolio Value')
            axes[0, 0].grid(True)
            
            # Drawdown
            running_max = eq_df['portfolio_value'].expanding().max()
            drawdown = (eq_df['portfolio_value'] - running_max) / running_max * 100
            axes[0, 1].fill_between(eq_df.index, drawdown, 0, alpha=0.3, color='red')
            axes[0, 1].plot(eq_df.index, drawdown, color='red')
            axes[0, 1].set_title('Drawdown (%)')
            axes[0, 1].set_ylabel('Drawdown %')
            axes[0, 1].grid(True)
            
            # Trade PnL distribution
            trade_pnls = [trade.pnl for trade in trades if trade.pnl is not None]
            if trade_pnls:
                axes[1, 0].hist(trade_pnls, bins=20, alpha=0.7)
                axes[1, 0].axvline(x=0, color='red', linestyle='--')
                axes[1, 0].set_title('Trade PnL Distribution')
                axes[1, 0].set_xlabel('PnL')
                axes[1, 0].set_ylabel('Frequency')
                
            # Cumulative PnL
            cumulative_pnl = np.cumsum(trade_pnls) if trade_pnls else [0]
            axes[1, 1].plot(range(len(cumulative_pnl)), cumulative_pnl)
            axes[1, 1].set_title('Cumulative PnL by Trade')
            axes[1, 1].set_xlabel('Trade Number')
            axes[1, 1].set_ylabel('Cumulative PnL')
            axes[1, 1].grid(True)
            
            # Monthly returns heatmap
            eq_df['month'] = eq_df.index.to_period('M')
            monthly_returns = eq_df.groupby('month')['portfolio_value'].apply(
                lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] * 100
            )
            
            if len(monthly_returns) > 1:
                monthly_returns.index = monthly_returns.index.astype(str)
                axes[2, 0].bar(range(len(monthly_returns)), monthly_returns.values)
                axes[2, 0].set_title('Monthly Returns (%)')
                axes[2, 0].set_xlabel('Month')
                axes[2, 0].set_ylabel('Return %')
                axes[2, 0].tick_params(axis='x', rotation=45)
                
            # Performance metrics text
            metrics = self.results[strategy_name]['metrics']
            metrics_text = f"""
            Total Return: {metrics['total_return']:.2%}
            Sharpe Ratio: {metrics['sharpe_ratio']:.3f}
            Max Drawdown: {metrics['max_drawdown']:.2%}
            Win Rate: {metrics['win_rate']:.2%}
            Total Trades: {metrics['total_trades']}
            Profit Factor: {metrics['profit_factor']:.2f}
            """
            
            axes[2, 1].text(0.1, 0.9, metrics_text, transform=axes[2, 1].transAxes,
                           fontsize=10, verticalalignment='top', fontfamily='monospace')
            axes[2, 1].set_title('Performance Metrics')
            axes[2, 1].axis('off')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Plot saved to {save_path}")
            else:
                plt.show()
                
        except Exception as e:
            logger.error(f"Error plotting results: {e}")
            
    def save_results(self, file_path: str) -> None:
        """Save backtest results to JSON file."""
        try:
            # Convert datetime objects to strings for JSON serialization
            serializable_results = {}
            
            for strategy, result in self.results.items():
                serializable_result = {}
                
                # Copy most fields directly
                for key, value in result.items():
                    if key == 'trades':
                        # Convert trades
                        serializable_trades = []
                        for trade in value:
                            serializable_trade = trade.copy()
                            if 'timestamp' in serializable_trade:
                                serializable_trade['timestamp'] = serializable_trade['timestamp'].isoformat()
                            if 'duration' in serializable_trade and serializable_trade['duration']:
                                serializable_trade['duration'] = str(serializable_trade['duration'])
                            serializable_trades.append(serializable_trade)
                        serializable_result[key] = serializable_trades
                    elif key == 'equity_curve':
                        # Convert equity curve
                        serializable_equity = []
                        for point in value:
                            serializable_point = point.copy()
                            if 'timestamp' in serializable_point:
                                serializable_point['timestamp'] = serializable_point['timestamp'].isoformat()
                            serializable_equity.append(serializable_point)
                        serializable_result[key] = serializable_equity
                    elif key == 'metrics':
                        # Convert metrics object to dict and handle timedelta objects
                        if hasattr(value, '__dict__'):
                            # It's a dataclass, convert to dict
                            metrics_dict = {}
                            for field_name, field_value in value.__dict__.items():
                                if isinstance(field_value, timedelta):
                                    metrics_dict[field_name] = str(field_value)
                                else:
                                    metrics_dict[field_name] = field_value
                            serializable_result[key] = metrics_dict
                        else:
                            # It's already a dict, just handle timedelta values
                            metrics_dict = {}
                            for k, v in value.items():
                                if isinstance(v, timedelta):
                                    metrics_dict[k] = str(v)
                                else:
                                    metrics_dict[k] = v
                            serializable_result[key] = metrics_dict
                    else:
                        serializable_result[key] = value
                        
                serializable_results[strategy] = serializable_result
            
            # Add timestamp to results
            serializable_results['timestamp'] = datetime.now().isoformat()
                
            # Write to temporary file first, then rename for atomic operation
            temp_file_path = file_path + '.tmp'
            with open(temp_file_path, 'w') as f:
                json.dump(serializable_results, f, indent=2, ensure_ascii=False)
                f.flush()  # Ensure all data is written
                os.fsync(f.fileno())  # Force write to disk
            
            # Atomically replace the original file
            os.rename(temp_file_path, file_path)
                
            logger.info(f"Results saved to {file_path}")
            
            # Verify the file was written correctly
            try:
                with open(file_path, 'r') as f:
                    json.load(f)
                logger.info("JSON file integrity verified")
            except json.JSONDecodeError as verify_error:
                logger.error(f"Saved JSON file is corrupted: {verify_error}")
                raise
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            # Clean up temporary file if it exists
            temp_file_path = file_path + '.tmp'
            if os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except:
                    pass
            raise


# Example usage and testing
if __name__ == "__main__":
    """
    Example usage of Backtester for testing.
    """
    # Mock configuration
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
    
    # Create sample data
    dates = pd.date_range(start='2024-01-01', periods=1000, freq='1min')
    np.random.seed(42)
    
    # Generate realistic price data with trend
    price_changes = np.random.normal(0.0002, 0.01, 1000)  # Small upward drift
    prices = [50000]
    for change in price_changes[1:]:
        prices.append(prices[-1] * (1 + change))
        
    df = pd.DataFrame({
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.002))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.002))) for p in prices],
        'close': prices,
        'volume': np.random.lognormal(10, 0.3, 1000)
    }, index=dates)
    
    # Ensure OHLC data integrity
    df['high'] = np.maximum(df['high'], np.maximum(df['open'], df['close']))
    df['low'] = np.minimum(df['low'], np.minimum(df['open'], df['close']))
    
    print("📊 Testing Backtester")
    print(f"Sample data: {len(df)} points from {df.index[0]} to {df.index[-1]}")
    print(f"Price range: {df['close'].min():.2f} - {df['close'].max():.2f}")
    
    # Initialize backtester
    backtester = Backtester(config)
    
    # Run backtest
    print("\n🔙 Running backtest...")
    results = backtester.run_backtest(df, 'rule_based')
    
    if 'error' not in results:
        print("✅ Backtest completed successfully")
        print(f"Total return: {results['metrics']['total_return']:.2%}")
        print(f"Sharpe ratio: {results['metrics']['sharpe_ratio']:.3f}")
        print(f"Max drawdown: {results['metrics']['max_drawdown']:.2%}")
        print(f"Total trades: {results['metrics']['total_trades']}")
        print(f"Win rate: {results['metrics']['win_rate']:.2%}")
    else:
        print(f"❌ Backtest failed: {results['error']}")
        
    print("\n✅ Backtester test completed")
