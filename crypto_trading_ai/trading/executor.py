"""
Trading execution module for placing and managing orders.
Handles order placement, risk management, and position tracking.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import time
import json
import os
from dataclasses import dataclass, asdict
from enum import Enum
from loguru import logger

# CCXT for exchange operations
try:
    import ccxt
except ImportError:
    logger.warning("CCXT not available - trading execution will be limited")


class OrderType(Enum):
    """Order types supported by the executor."""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"


class OrderStatus(Enum):
    """Order status types."""
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"


class OrderSide(Enum):
    """Order side types."""
    BUY = "buy"
    SELL = "sell"


@dataclass
class Order:
    """Order data structure."""
    id: str
    symbol: str
    side: OrderSide
    type: OrderType
    amount: float
    price: Optional[float]
    status: OrderStatus
    timestamp: datetime
    filled: float = 0.0
    remaining: float = 0.0
    fee: float = 0.0
    average_price: Optional[float] = None
    stop_price: Optional[float] = None
    info: Dict[str, Any] = None


@dataclass
class Position:
    """Position data structure."""
    symbol: str
    side: str  # "long" or "short"
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    timestamp: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


class TradingExecutor:
    """
    Handles trade execution, risk management, and position tracking.
    Supports both live trading and paper trading modes.
    """
    
    def __init__(self, exchange_manager, config: Dict[str, Any]):
        """
        Initialize trading executor.
        
        Args:
            exchange_manager: ExchangeManager instance
            config: Configuration dictionary
        """
        self.exchange_manager = exchange_manager
        self.config = config
        
        # Trading configuration
        trading_config = config['trading']
        self.symbol = trading_config['symbol']
        self.base_currency = trading_config['base_currency']
        self.quote_currency = trading_config['quote_currency']
        
        # Risk management parameters
        risk_config = trading_config['risk_management']
        self.max_position_pct = risk_config['max_position_size_pct']
        self.daily_loss_limit_pct = risk_config['daily_loss_limit_pct']
        self.stop_loss_pct = risk_config['stop_loss_pct']
        self.take_profit_pct = risk_config['take_profit_pct']
        self.max_open_positions = risk_config['max_open_positions']
        
        # Order configuration
        order_config = trading_config['order']
        self.order_type = order_config['type']
        self.amount_type = order_config['amount_type']
        self.min_amount = order_config['min_amount']
        
        # Internal state
        self.positions: Dict[str, Position] = {}
        self.orders: Dict[str, Order] = {}
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        self.trade_count = 0
        self.last_reset_date = datetime.now().date()
        
        # Paper trading mode (default to True for safety)
        self.paper_trading = config.get('paper_trading', True)
        
        # Trading state
        self.trading_enabled = True
        self.last_trade_time = None
        
        # Initialize trade log
        self._initialize_trade_log()
        
    def _initialize_trade_log(self) -> None:
        """Initialize trade logging."""
        os.makedirs("logs", exist_ok=True)
        self.trade_log_file = "logs/trades.json"
        
        if not os.path.exists(self.trade_log_file):
            with open(self.trade_log_file, 'w') as f:
                json.dump([], f)
                
    def execute_signal(self, signal_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute trading signal with risk management.
        
        Args:
            signal_data: Signal dictionary from strategy
            
        Returns:
            Execution result dictionary
        """
        try:
            if not self.trading_enabled:
                return self._create_result("trading_disabled", "Trading is currently disabled")
                
            # Check daily loss limit
            if self._check_daily_loss_limit():
                return self._create_result("daily_loss_limit", "Daily loss limit exceeded")
                
            # Reset daily PnL if new day
            self._reset_daily_pnl_if_needed()
            
            signal = signal_data.get('signal', 0)
            confidence = signal_data.get('confidence', 0.0)
            position_size = signal_data.get('position_size', 0.0)
            
            # Validate signal
            if signal == 0 or confidence < 0.1:
                return self._create_result("no_signal", "No valid trading signal")
                
            # Get current price
            current_price = self._get_current_price()
            if current_price is None:
                return self._create_result("price_error", "Could not get current price")
                
            # Calculate order parameters
            order_params = self._calculate_order_params(signal, position_size, current_price)
            
            if order_params is None:
                return self._create_result("calculation_error", "Could not calculate order parameters")
                
            # Check risk limits
            risk_check = self._check_risk_limits(order_params, current_price)
            if not risk_check['allowed']:
                return self._create_result("risk_check_failed", risk_check['reason'])
                
            # Execute the order
            if signal > 0:  # Buy signal
                result = self._execute_buy_order(order_params, signal_data)
            else:  # Sell signal
                result = self._execute_sell_order(order_params, signal_data)
                
            # Log the trade
            self._log_trade(result, signal_data)
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing signal: {e}")
            return self._create_result("execution_error", f"Execution error: {e}")
            
    def _calculate_order_params(self, signal: int, position_size: float, 
                               current_price: float) -> Optional[Dict[str, Any]]:
        """Calculate order parameters based on signal and risk management."""
        try:
            # Get account balance
            balance = self._get_account_balance()
            available_balance = balance.get(self.quote_currency, 0.0)
            
            if available_balance < self.min_amount:
                logger.warning(f"Insufficient balance: {available_balance} < {self.min_amount}")
                return None
                
            # Calculate position size in quote currency
            max_amount = available_balance * self.max_position_pct
            signal_amount = available_balance * position_size
            
            # Use the smaller of the two
            order_amount = min(max_amount, signal_amount)
            
            # Ensure minimum order size
            if order_amount < self.min_amount:
                logger.warning(f"Order amount too small: {order_amount} < {self.min_amount}")
                return None
                
            # Calculate quantity in base currency for buy orders
            if signal > 0:  # Buy order
                quantity = order_amount / current_price
            else:  # Sell order - need to check available base currency
                base_balance = balance.get(self.base_currency, 0.0)
                quantity = min(base_balance * position_size, base_balance * self.max_position_pct)
                
                if quantity * current_price < self.min_amount:
                    logger.warning(f"Sell quantity too small: {quantity}")
                    return None
                    
            return {
                'quantity': quantity,
                'amount': order_amount,
                'price': current_price,
                'signal': signal
            }
            
        except Exception as e:
            logger.error(f"Error calculating order parameters: {e}")
            return None
            
    def _execute_buy_order(self, order_params: Dict[str, Any], 
                          signal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute buy order."""
        try:
            quantity = order_params['quantity']
            price = order_params['price']
            
            if self.paper_trading:
                # Simulate order execution
                order_id = f"paper_{int(time.time() * 1000)}"
                
                # Create paper order
                order = Order(
                    id=order_id,
                    symbol=self.symbol,
                    side=OrderSide.BUY,
                    type=OrderType.MARKET,
                    amount=quantity,
                    price=price,
                    status=OrderStatus.FILLED,
                    timestamp=datetime.now(),
                    filled=quantity,
                    remaining=0.0,
                    average_price=price,
                    fee=quantity * price * 0.001  # 0.1% fee
                )
                
                self.orders[order_id] = order
                
                # Update position
                self._update_position_from_order(order, "buy")
                
                return self._create_result("success", f"Paper buy order executed: {quantity:.6f} @ {price:.2f}")
                
            else:
                # Execute real order
                order_result = self.exchange_manager.exchange.create_market_buy_order(
                    self.symbol, quantity
                )
                
                order = self._create_order_from_exchange_result(order_result, OrderSide.BUY)
                self.orders[order.id] = order
                
                # Update position if order is filled
                if order.status == OrderStatus.FILLED:
                    self._update_position_from_order(order, "buy")
                    
                return self._create_result("success", f"Live buy order executed: {order.id}")
                
        except Exception as e:
            logger.error(f"Error executing buy order: {e}")
            return self._create_result("buy_error", f"Buy order failed: {e}")
            
    def _execute_sell_order(self, order_params: Dict[str, Any], 
                           signal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute sell order."""
        try:
            quantity = order_params['quantity']
            price = order_params['price']
            
            if self.paper_trading:
                # Simulate order execution
                order_id = f"paper_{int(time.time() * 1000)}"
                
                # Create paper order
                order = Order(
                    id=order_id,
                    symbol=self.symbol,
                    side=OrderSide.SELL,
                    type=OrderType.MARKET,
                    amount=quantity,
                    price=price,
                    status=OrderStatus.FILLED,
                    timestamp=datetime.now(),
                    filled=quantity,
                    remaining=0.0,
                    average_price=price,
                    fee=quantity * price * 0.001  # 0.1% fee
                )
                
                self.orders[order_id] = order
                
                # Update position
                self._update_position_from_order(order, "sell")
                
                return self._create_result("success", f"Paper sell order executed: {quantity:.6f} @ {price:.2f}")
                
            else:
                # Execute real order
                order_result = self.exchange_manager.exchange.create_market_sell_order(
                    self.symbol, quantity
                )
                
                order = self._create_order_from_exchange_result(order_result, OrderSide.SELL)
                self.orders[order.id] = order
                
                # Update position if order is filled
                if order.status == OrderStatus.FILLED:
                    self._update_position_from_order(order, "sell")
                    
                return self._create_result("success", f"Live sell order executed: {order.id}")
                
        except Exception as e:
            logger.error(f"Error executing sell order: {e}")
            return self._create_result("sell_error", f"Sell order failed: {e}")
            
    def _update_position_from_order(self, order: Order, side: str) -> None:
        """Update position tracking from executed order."""
        try:
            position_key = f"{self.symbol}_{side}"
            current_price = order.average_price or order.price
            
            if position_key in self.positions:
                position = self.positions[position_key]
                
                if side == "buy":
                    # Add to long position
                    total_cost = (position.size * position.entry_price) + (order.filled * current_price)
                    position.size += order.filled
                    position.entry_price = total_cost / position.size if position.size > 0 else current_price
                else:
                    # Reduce long position or add to short position
                    if position.side == "long":
                        # Calculate realized PnL
                        sell_amount = min(order.filled, position.size)
                        realized_pnl = sell_amount * (current_price - position.entry_price)
                        position.realized_pnl += realized_pnl
                        self.daily_pnl += realized_pnl
                        self.total_pnl += realized_pnl
                        
                        position.size -= sell_amount
                        
                        if position.size <= 0:
                            del self.positions[position_key]
                    else:
                        # Add to short position
                        total_cost = (position.size * position.entry_price) + (order.filled * current_price)
                        position.size += order.filled
                        position.entry_price = total_cost / position.size if position.size > 0 else current_price
                        
            else:
                # Create new position
                position = Position(
                    symbol=self.symbol,
                    side="long" if side == "buy" else "short",
                    size=order.filled,
                    entry_price=current_price,
                    current_price=current_price,
                    unrealized_pnl=0.0,
                    realized_pnl=0.0,
                    timestamp=datetime.now()
                )
                self.positions[position_key] = position
                
            # Update trade count
            self.trade_count += 1
            self.last_trade_time = datetime.now()
            
            logger.info(f"Position updated: {side} {order.filled:.6f} @ {current_price:.2f}")
            
        except Exception as e:
            logger.error(f"Error updating position: {e}")
            
    def _get_current_price(self) -> Optional[float]:
        """Get current market price."""
        try:
            ticker = self.exchange_manager.get_ticker(self.symbol)
            return ticker.get('last') or ticker.get('close')
        except Exception as e:
            logger.error(f"Error getting current price: {e}")
            return None
            
    def _get_account_balance(self) -> Dict[str, float]:
        """Get account balance."""
        try:
            if self.paper_trading:
                # Return simulated balance
                return {
                    self.base_currency: 1.0,  # 1 BTC
                    self.quote_currency: 50000.0  # 50,000 USDT
                }
            else:
                balance = self.exchange_manager.get_account_balance()
                return {
                    currency: info.get('free', 0.0) 
                    for currency, info in balance.items()
                    if currency in [self.base_currency, self.quote_currency]
                }
        except Exception as e:
            logger.error(f"Error getting account balance: {e}")
            return {}
            
    def _check_risk_limits(self, order_params: Dict[str, Any], 
                          current_price: float) -> Dict[str, Any]:
        """Check if order passes risk management rules."""
        try:
            # Check maximum open positions
            if len(self.positions) >= self.max_open_positions:
                return {'allowed': False, 'reason': 'max_positions_exceeded'}
                
            # Check position size limits
            order_value = order_params['quantity'] * current_price
            balance = self._get_account_balance()
            total_balance = sum(balance.values())
            
            if order_value > total_balance * self.max_position_pct:
                return {'allowed': False, 'reason': 'position_size_too_large'}
                
            # Check daily loss limit
            if self.daily_pnl < -total_balance * self.daily_loss_limit_pct:
                return {'allowed': False, 'reason': 'daily_loss_limit_exceeded'}
                
            # Check minimum time between trades (prevent overtrading)
            if self.last_trade_time and (datetime.now() - self.last_trade_time).seconds < 60:
                return {'allowed': False, 'reason': 'trade_frequency_limit'}
                
            return {'allowed': True, 'reason': 'risk_checks_passed'}
            
        except Exception as e:
            logger.error(f"Error in risk check: {e}")
            return {'allowed': False, 'reason': f'risk_check_error: {e}'}
            
    def _check_daily_loss_limit(self) -> bool:
        """Check if daily loss limit has been reached."""
        balance = self._get_account_balance()
        total_balance = sum(balance.values())
        return self.daily_pnl < -total_balance * self.daily_loss_limit_pct
        
    def _reset_daily_pnl_if_needed(self) -> None:
        """Reset daily PnL if it's a new day."""
        current_date = datetime.now().date()
        if current_date != self.last_reset_date:
            self.daily_pnl = 0.0
            self.last_reset_date = current_date
            logger.info("Daily PnL reset for new trading day")
            
    def _create_order_from_exchange_result(self, result: Dict[str, Any], 
                                          side: OrderSide) -> Order:
        """Create Order object from exchange API result."""
        return Order(
            id=result['id'],
            symbol=result['symbol'],
            side=side,
            type=OrderType(result['type']),
            amount=result['amount'],
            price=result.get('price'),
            status=OrderStatus(result['status']),
            timestamp=datetime.fromtimestamp(result['timestamp'] / 1000),
            filled=result.get('filled', 0.0),
            remaining=result.get('remaining', 0.0),
            fee=result.get('fee', {}).get('cost', 0.0),
            average_price=result.get('average'),
            info=result
        )
        
    def _create_result(self, status: str, message: str, **kwargs) -> Dict[str, Any]:
        """Create standardized result dictionary."""
        result = {
            'status': status,
            'message': message,
            'timestamp': datetime.now().isoformat(),
            'paper_trading': self.paper_trading,
            **kwargs
        }
        return result
        
    def _make_json_serializable(self, obj):
        """Convert objects to JSON serializable format."""
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        
        if isinstance(obj, dict):
            # Convert both keys and values to be JSON serializable
            serializable_dict = {}
            for k, v in obj.items():
                # Convert keys to strings if they're not already JSON serializable
                if isinstance(k, (pd.Timestamp, datetime)):
                    key = k.isoformat()
                elif isinstance(k, (np.integer, np.floating)):
                    key = k.item()
                elif isinstance(k, timedelta):
                    key = str(k)
                elif pd.isna(k):
                    key = "null"
                else:
                    key = str(k)  # Convert any other non-serializable keys to strings
                
                serializable_dict[key] = self._make_json_serializable(v)
            return serializable_dict
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, pd.Series):
            # Convert Series to dict with string keys
            return {str(k): self._make_json_serializable(v) for k, v in obj.to_dict().items()}
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        elif isinstance(obj, timedelta):
            return str(obj)
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif pd.isna(obj):
            return None
        else:
            return obj

    def _log_trade(self, result: Dict[str, Any], signal_data: Dict[str, Any]) -> None:
        """Log trade to file."""
        try:
            # Make all data JSON serializable with detailed error handling
            try:
                serializable_signal = self._make_json_serializable(signal_data)
            except Exception as e:
                logger.error(f"Error serializing signal data: {e}")
                logger.error(f"Signal data keys: {list(signal_data.keys()) if isinstance(signal_data, dict) else 'Not a dict'}")
                # Try with default str conversion as fallback
                serializable_signal = json.loads(json.dumps(signal_data, default=str))
            
            try:
                serializable_result = self._make_json_serializable(result)
            except Exception as e:
                logger.error(f"Error serializing result data: {e}")
                logger.error(f"Result data keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
                # Try with default str conversion as fallback
                serializable_result = json.loads(json.dumps(result, default=str))
            
            trade_record = {
                'timestamp': datetime.now().isoformat(),
                'result': serializable_result,
                'signal': serializable_signal,
                'positions': {k: asdict(v) for k, v in self.positions.items()},
                'daily_pnl': float(self.daily_pnl),
                'total_pnl': float(self.total_pnl)
            }
            
            # Test JSON serialization before writing
            try:
                test_json = json.dumps(trade_record, indent=2, ensure_ascii=False)
            except Exception as json_error:
                logger.error(f"JSON serialization test failed: {json_error}")
                # Try with default=str as ultimate fallback
                test_json = json.dumps(trade_record, indent=2, default=str, ensure_ascii=False)
            
            # Read existing trades
            try:
                with open(self.trade_log_file, 'r') as f:
                    trades = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                trades = []
                
            trades.append(trade_record)
            
            # Write updated trades with atomic operation
            temp_file = self.trade_log_file + '.tmp'
            try:
                with open(temp_file, 'w') as f:
                    json.dump(trades, f, indent=2, default=str, ensure_ascii=False)
                    f.flush()
                    os.fsync(f.fileno())  # Force write to disk
                
                # Atomically replace the original file
                os.rename(temp_file, self.trade_log_file)
                
                # Verify the file was written correctly
                with open(self.trade_log_file, 'r') as f:
                    json.load(f)
                    
            except Exception as write_error:
                # Clean up temp file if it exists
                if os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except:
                        pass
                raise write_error
                
        except Exception as e:
            logger.error(f"Error logging trade: {e}")
            # Log the actual problematic data for debugging
            logger.error(f"Signal data type: {type(signal_data)}")
            logger.error(f"Result data type: {type(result)}")
            if isinstance(signal_data, dict):
                for key, value in signal_data.items():
                    logger.error(f"Signal key '{key}' ({type(key)}): {type(value)}")
            if isinstance(result, dict):
                for key, value in result.items():
                    logger.error(f"Result key '{key}' ({type(key)}): {type(value)}")
            
    def get_portfolio_status(self) -> Dict[str, Any]:
        """Get current portfolio status."""
        current_price = self._get_current_price() or 0.0
        
        # Update unrealized PnL for all positions
        total_unrealized_pnl = 0.0
        for position in self.positions.values():
            if position.side == "long":
                position.unrealized_pnl = position.size * (current_price - position.entry_price)
            else:
                position.unrealized_pnl = position.size * (position.entry_price - current_price)
            total_unrealized_pnl += position.unrealized_pnl
            
        balance = self._get_account_balance()
        total_balance = sum(balance.values())
        
        return {
            'total_balance': total_balance,
            'daily_pnl': self.daily_pnl,
            'total_pnl': self.total_pnl,
            'unrealized_pnl': total_unrealized_pnl,
            'open_positions': len(self.positions),
            'total_trades': self.trade_count,
            'trading_enabled': self.trading_enabled,
            'paper_trading': self.paper_trading,
            'current_price': current_price,
            'positions': {k: asdict(v) for k, v in self.positions.items()},
            'last_trade_time': self.last_trade_time.isoformat() if self.last_trade_time else None
        }
        
    def close_all_positions(self) -> List[Dict[str, Any]]:
        """Close all open positions."""
        results = []
        current_price = self._get_current_price()
        
        if current_price is None:
            return [self._create_result("error", "Could not get current price for position closure")]
            
        for position_key, position in list(self.positions.items()):
            try:
                if position.side == "long":
                    # Sell to close long position
                    result = self._execute_sell_order({
                        'quantity': position.size,
                        'amount': position.size * current_price,
                        'price': current_price,
                        'signal': -1
                    }, {'signal': -1, 'reason': 'close_all_positions'})
                else:
                    # Buy to close short position
                    result = self._execute_buy_order({
                        'quantity': position.size,
                        'amount': position.size * current_price,
                        'price': current_price,
                        'signal': 1
                    }, {'signal': 1, 'reason': 'close_all_positions'})
                    
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error closing position {position_key}: {e}")
                results.append(self._create_result("error", f"Failed to close {position_key}: {e}"))
                
        return results
        
    def enable_trading(self) -> None:
        """Enable trading."""
        self.trading_enabled = True
        logger.info("Trading enabled")
        
    def disable_trading(self) -> None:
        """Disable trading."""
        self.trading_enabled = False
        logger.info("Trading disabled")


# Example usage and testing
if __name__ == "__main__":
    """
    Example usage of TradingExecutor for testing.
    """
    # Mock configuration
    config = {
        'trading': {
            'symbol': 'BTC/USDT',
            'base_currency': 'BTC',
            'quote_currency': 'USDT',
            'risk_management': {
                'max_position_size_pct': 0.1,
                'daily_loss_limit_pct': 0.05,
                'stop_loss_pct': 0.02,
                'take_profit_pct': 0.04,
                'max_open_positions': 3
            },
            'order': {
                'type': 'market',
                'amount_type': 'quote',
                'min_amount': 10
            }
        },
        'paper_trading': True
    }
    
    # Mock exchange manager
    class MockExchangeManager:
        def get_ticker(self, symbol):
            return {'last': 50000.0}
            
        def get_account_balance(self):
            return {'BTC': {'free': 1.0}, 'USDT': {'free': 50000.0}}
    
    print("💰 Testing Trading Executor")
    
    # Initialize executor
    exchange_manager = MockExchangeManager()
    executor = TradingExecutor(exchange_manager, config)
    
    # Test buy signal
    print("\n📈 Testing buy signal execution...")
    buy_signal = {
        'signal': 1,
        'confidence': 0.8,
        'position_size': 0.05,
        'strategy': 'test'
    }
    
    buy_result = executor.execute_signal(buy_signal)
    print(f"Buy result: {buy_result}")
    
    # Test portfolio status
    print("\n📊 Portfolio status:")
    status = executor.get_portfolio_status()
    for key, value in status.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")
            
    # Test sell signal
    print("\n📉 Testing sell signal execution...")
    sell_signal = {
        'signal': -1,
        'confidence': 0.7,
        'position_size': 0.03,
        'strategy': 'test'
    }
    
    sell_result = executor.execute_signal(sell_signal)
    print(f"Sell result: {sell_result}")
    
    # Final portfolio status
    print("\n📊 Final portfolio status:")
    final_status = executor.get_portfolio_status()
    for key, value in final_status.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")
            
    print("\n✅ Trading Executor test completed")
