"""
Portfolio Management System for Stock Trading Bot
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import sqlite3
import json
from typing import Dict, List, Optional, Tuple
from config import *
import os

class PortfolioManager:
    def __init__(self, initial_capital: float = INITIAL_CAPITAL):
        self.initial_capital = initial_capital
        self.available_cash = initial_capital
        self.total_value = initial_capital
        self.positions = {}  # {symbol: {'quantity': int, 'avg_price': float, 'current_price': float}}
        self.trade_history = []
        self.daily_portfolio_value = []
        self.logger = logging.getLogger(__name__)
        
        # Initialize database
        self.db_path = DATABASE_PATH
        self.init_database()
        
        # Load existing portfolio if available
        self.load_portfolio()
    
    def init_database(self):
        """Initialize SQLite database for portfolio tracking"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create tables
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS portfolio (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    available_cash REAL,
                    total_value REAL,
                    positions TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    symbol TEXT,
                    action TEXT,
                    quantity INTEGER,
                    price REAL,
                    commission REAL,
                    total_amount REAL,
                    reason TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS daily_values (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date DATE,
                    total_value REAL,
                    cash REAL,
                    invested_value REAL,
                    daily_return REAL,
                    cumulative_return REAL
                )
            ''')
            
            conn.commit()
            conn.close()
            
            self.logger.info("Database initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing database: {str(e)}")
    
    def load_portfolio(self):
        """Load existing portfolio from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get latest portfolio state
            cursor.execute('''
                SELECT available_cash, total_value, positions 
                FROM portfolio 
                ORDER BY timestamp DESC 
                LIMIT 1
            ''')
            
            result = cursor.fetchone()
            if result:
                self.available_cash = result[0]
                self.total_value = result[1]
                if result[2]:
                    self.positions = json.loads(result[2])
                
                self.logger.info(f"Portfolio loaded: Cash={self.available_cash}, Total={self.total_value}")
            
            # Load trade history
            cursor.execute('''
                SELECT timestamp, symbol, action, quantity, price, commission, total_amount, reason
                FROM trades
                ORDER BY timestamp
            ''')
            
            trades = cursor.fetchall()
            self.trade_history = []
            for trade in trades:
                self.trade_history.append({
                    'timestamp': trade[0],
                    'symbol': trade[1],
                    'action': trade[2],
                    'quantity': trade[3],
                    'price': trade[4],
                    'commission': trade[5],
                    'total_amount': trade[6],
                    'reason': trade[7]
                })
            
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error loading portfolio: {str(e)}")
    
    def save_portfolio(self):
        """Save current portfolio state to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO portfolio (available_cash, total_value, positions)
                VALUES (?, ?, ?)
            ''', (self.available_cash, self.total_value, json.dumps(self.positions)))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error saving portfolio: {str(e)}")
    
    def calculate_position_size(self, symbol: str, current_price: float, signal_strength: float = 1.0) -> int:
        """
        Calculate position size based on portfolio management rules
        
        Args:
            symbol: Stock symbol
            current_price: Current stock price
            signal_strength: Strength of the trading signal (0-1)
            
        Returns:
            Number of shares to trade
        """
        # Kelly criterion or fixed percentage approach
        if RISK_CONFIG["position_sizing"] == "kelly":
            # Simplified Kelly criterion
            max_position_value = self.total_value * MAX_POSITION_SIZE * signal_strength
        else:
            # Fixed percentage
            max_position_value = self.total_value * MAX_POSITION_SIZE
        
        # Check available cash
        available_for_position = min(max_position_value, self.available_cash * 0.95)  # 5% cash buffer
        
        # Calculate quantity
        quantity = int(available_for_position / current_price)
        
        return max(0, quantity)
    
    def buy_stock(self, symbol: str, quantity: int, price: float, reason: str = "AI Decision") -> bool:
        """
        Buy stock and update portfolio
        
        Args:
            symbol: Stock symbol
            quantity: Number of shares to buy
            price: Price per share
            reason: Reason for the trade
            
        Returns:
            True if trade executed successfully, False otherwise
        """
        if quantity <= 0:
            return False
        
        # Calculate costs
        total_cost = quantity * price
        commission = total_cost * COMMISSION_RATE
        total_amount = total_cost + commission
        
        # Check if we have enough cash
        if total_amount > self.available_cash:
            self.logger.warning(f"Insufficient funds for buying {quantity} shares of {symbol}")
            return False
        
        # Execute trade
        self.available_cash -= total_amount
        
        # Update positions
        if symbol in self.positions:
            # Update existing position
            current_quantity = self.positions[symbol]['quantity']
            current_avg_price = self.positions[symbol]['avg_price']
            
            new_quantity = current_quantity + quantity
            new_avg_price = ((current_quantity * current_avg_price) + (quantity * price)) / new_quantity
            
            self.positions[symbol]['quantity'] = new_quantity
            self.positions[symbol]['avg_price'] = new_avg_price
        else:
            # New position
            self.positions[symbol] = {
                'quantity': quantity,
                'avg_price': price,
                'current_price': price
            }
        
        # Record trade
        trade = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'symbol': symbol,
            'action': 'BUY',
            'quantity': quantity,
            'price': price,
            'commission': commission,
            'total_amount': total_amount,
            'reason': reason
        }
        
        self.trade_history.append(trade)
        self.save_trade(trade)
        
        self.logger.info(f"Bought {quantity} shares of {symbol} at ₹{price:.2f} (Total: ₹{total_amount:.2f})")
        
        return True
    
    def sell_stock(self, symbol: str, quantity: int, price: float, reason: str = "AI Decision") -> bool:
        """
        Sell stock and update portfolio
        
        Args:
            symbol: Stock symbol
            quantity: Number of shares to sell
            price: Price per share
            reason: Reason for the trade
            
        Returns:
            True if trade executed successfully, False otherwise
        """
        if quantity <= 0 or symbol not in self.positions:
            return False
        
        available_quantity = self.positions[symbol]['quantity']
        if quantity > available_quantity:
            quantity = available_quantity  # Sell all available shares
        
        # Calculate proceeds
        total_proceeds = quantity * price
        commission = total_proceeds * COMMISSION_RATE
        net_proceeds = total_proceeds - commission
        
        # Execute trade
        self.available_cash += net_proceeds
        
        # Update positions
        self.positions[symbol]['quantity'] -= quantity
        
        # Remove position if fully sold
        if self.positions[symbol]['quantity'] == 0:
            del self.positions[symbol]
        
        # Record trade
        trade = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'symbol': symbol,
            'action': 'SELL',
            'quantity': quantity,
            'price': price,
            'commission': commission,
            'total_amount': net_proceeds,
            'reason': reason
        }
        
        self.trade_history.append(trade)
        self.save_trade(trade)
        
        self.logger.info(f"Sold {quantity} shares of {symbol} at ₹{price:.2f} (Net: ₹{net_proceeds:.2f})")
        
        return True
    
    def save_trade(self, trade: Dict):
        """Save trade to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO trades (timestamp, symbol, action, quantity, price, commission, total_amount, reason)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade['timestamp'], trade['symbol'], trade['action'], trade['quantity'],
                trade['price'], trade['commission'], trade['total_amount'], trade['reason']
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error saving trade: {str(e)}")
    
    def update_positions(self, current_prices: Dict[str, float]):
        """
        Update current prices for all positions
        
        Args:
            current_prices: Dictionary of symbol: price
        """
        for symbol in self.positions:
            if symbol in current_prices:
                self.positions[symbol]['current_price'] = current_prices[symbol]
        
        # Update total portfolio value
        self.calculate_total_value()
    
    def calculate_total_value(self) -> float:
        """
        Calculate total portfolio value
        
        Returns:
            Total portfolio value
        """
        invested_value = 0
        
        for symbol, position in self.positions.items():
            current_price = position.get('current_price', position['avg_price'])
            invested_value += position['quantity'] * current_price
        
        self.total_value = self.available_cash + invested_value
        return self.total_value
    
    def get_portfolio_summary(self) -> Dict:
        """
        Get comprehensive portfolio summary
        
        Returns:
            Dictionary with portfolio metrics
        """
        invested_value = 0
        total_unrealized_pnl = 0
        
        positions_summary = []
        
        for symbol, position in self.positions.items():
            current_price = position.get('current_price', position['avg_price'])
            market_value = position['quantity'] * current_price
            cost_basis = position['quantity'] * position['avg_price']
            unrealized_pnl = market_value - cost_basis
            unrealized_pnl_pct = (unrealized_pnl / cost_basis) * 100 if cost_basis > 0 else 0
            
            invested_value += market_value
            total_unrealized_pnl += unrealized_pnl
            
            positions_summary.append({
                'symbol': symbol,
                'quantity': position['quantity'],
                'avg_price': position['avg_price'],
                'current_price': current_price,
                'market_value': market_value,
                'cost_basis': cost_basis,
                'unrealized_pnl': unrealized_pnl,
                'unrealized_pnl_pct': unrealized_pnl_pct,
                'weight': (market_value / self.total_value) * 100 if self.total_value > 0 else 0
            })
        
        # Calculate returns
        total_return = ((self.total_value - self.initial_capital) / self.initial_capital) * 100
        
        # Calculate realized P&L from trade history
        realized_pnl = self.calculate_realized_pnl()
        
        return {
            'initial_capital': self.initial_capital,
            'available_cash': self.available_cash,
            'invested_value': invested_value,
            'total_value': self.total_value,
            'total_return': total_return,
            'total_return_amount': self.total_value - self.initial_capital,
            'unrealized_pnl': total_unrealized_pnl,
            'realized_pnl': realized_pnl,
            'cash_percentage': (self.available_cash / self.total_value) * 100 if self.total_value > 0 else 0,
            'invested_percentage': (invested_value / self.total_value) * 100 if self.total_value > 0 else 0,
            'positions': positions_summary,
            'num_positions': len(self.positions),
            'num_trades': len(self.trade_history)
        }
    
    def calculate_realized_pnl(self) -> float:
        """
        Calculate realized profit/loss from trade history
        
        Returns:
            Realized P&L amount
        """
        symbol_trades = {}
        realized_pnl = 0
        
        # Group trades by symbol
        for trade in self.trade_history:
            symbol = trade['symbol']
            if symbol not in symbol_trades:
                symbol_trades[symbol] = []
            symbol_trades[symbol].append(trade)
        
        # Calculate realized P&L for each symbol using FIFO
        for symbol, trades in symbol_trades.items():
            buy_queue = []
            
            for trade in trades:
                if trade['action'] == 'BUY':
                    buy_queue.append({
                        'quantity': trade['quantity'],
                        'price': trade['price']
                    })
                elif trade['action'] == 'SELL':
                    sell_quantity = trade['quantity']
                    sell_price = trade['price']
                    
                    while sell_quantity > 0 and buy_queue:
                        buy_order = buy_queue[0]
                        
                        if buy_order['quantity'] <= sell_quantity:
                            # Sell entire buy order
                            pnl = buy_order['quantity'] * (sell_price - buy_order['price'])
                            realized_pnl += pnl
                            sell_quantity -= buy_order['quantity']
                            buy_queue.pop(0)
                        else:
                            # Partial sell
                            pnl = sell_quantity * (sell_price - buy_order['price'])
                            realized_pnl += pnl
                            buy_order['quantity'] -= sell_quantity
                            sell_quantity = 0
        
        return realized_pnl
    
    def check_risk_limits(self) -> Dict:
        """
        Check portfolio against risk limits
        
        Returns:
            Dictionary with risk metrics and alerts
        """
        alerts = []
        
        # Check maximum position size
        for symbol, position in self.positions.items():
            current_price = position.get('current_price', position['avg_price'])
            position_value = position['quantity'] * current_price
            position_weight = position_value / self.total_value if self.total_value > 0 else 0
            
            if position_weight > MAX_POSITION_SIZE:
                alerts.append(f"Position {symbol} exceeds maximum weight: {position_weight:.2%}")
        
        # Check daily loss limit
        daily_return = ((self.total_value - self.initial_capital) / self.initial_capital)
        if daily_return < -RISK_CONFIG["max_daily_loss"]:
            alerts.append(f"Daily loss limit exceeded: {daily_return:.2%}")
        
        # Check maximum drawdown
        max_value = max([self.total_value] + [pv['total_value'] for pv in self.daily_portfolio_value])
        current_drawdown = (max_value - self.total_value) / max_value if max_value > 0 else 0
        
        if current_drawdown > RISK_CONFIG["max_drawdown"]:
            alerts.append(f"Maximum drawdown exceeded: {current_drawdown:.2%}")
        
        return {
            'daily_return': daily_return,
            'current_drawdown': current_drawdown,
            'max_position_weight': max([pos['quantity'] * pos.get('current_price', pos['avg_price']) / self.total_value 
                                      for pos in self.positions.values()]) if self.positions else 0,
            'alerts': alerts,
            'risk_score': len(alerts)  # Simple risk score based on number of alerts
        }
    
    def record_daily_value(self, date: str = None):
        """
        Record daily portfolio value
        
        Args:
            date: Date string (YYYY-MM-DD), defaults to today
        """
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        invested_value = sum([pos['quantity'] * pos.get('current_price', pos['avg_price']) 
                             for pos in self.positions.values()])
        
        # Calculate returns
        if len(self.daily_portfolio_value) > 0:
            previous_value = self.daily_portfolio_value[-1]['total_value']
            daily_return = ((self.total_value - previous_value) / previous_value) * 100
        else:
            daily_return = ((self.total_value - self.initial_capital) / self.initial_capital) * 100
        
        cumulative_return = ((self.total_value - self.initial_capital) / self.initial_capital) * 100
        
        daily_record = {
            'date': date,
            'total_value': self.total_value,
            'cash': self.available_cash,
            'invested_value': invested_value,
            'daily_return': daily_return,
            'cumulative_return': cumulative_return
        }
        
        self.daily_portfolio_value.append(daily_record)
        
        # Save to database
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO daily_values 
                (date, total_value, cash, invested_value, daily_return, cumulative_return)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (date, self.total_value, self.available_cash, invested_value, daily_return, cumulative_return))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error saving daily value: {str(e)}")
    
    def get_performance_metrics(self, days: int = 30) -> Dict:
        """
        Calculate performance metrics
        
        Args:
            days: Number of days to calculate metrics for
            
        Returns:
            Dictionary with performance metrics
        """
        if len(self.daily_portfolio_value) < 2:
            return {}
        
        # Get recent data
        recent_data = self.daily_portfolio_value[-days:] if len(self.daily_portfolio_value) >= days else self.daily_portfolio_value
        
        # Calculate metrics
        returns = [record['daily_return'] for record in recent_data if record['daily_return'] is not None]
        
        if not returns:
            return {}
        
        avg_daily_return = np.mean(returns)
        volatility = np.std(returns)
        sharpe_ratio = (avg_daily_return / volatility) if volatility > 0 else 0
        
        # Win rate
        winning_days = len([r for r in returns if r > 0])
        win_rate = (winning_days / len(returns)) * 100
        
        # Maximum daily gain/loss
        max_daily_gain = max(returns)
        max_daily_loss = min(returns)
        
        return {
            'avg_daily_return': avg_daily_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'win_rate': win_rate,
            'max_daily_gain': max_daily_gain,
            'max_daily_loss': max_daily_loss,
            'total_trading_days': len(returns)
        }
    
    def rebalance_portfolio(self, target_weights: Dict[str, float]):
        """
        Rebalance portfolio to target weights
        
        Args:
            target_weights: Dictionary of symbol: target_weight
        """
        self.logger.info("Starting portfolio rebalancing")
        
        current_weights = {}
        for symbol, position in self.positions.items():
            current_price = position.get('current_price', position['avg_price'])
            position_value = position['quantity'] * current_price
            current_weights[symbol] = position_value / self.total_value
        
        # Calculate required trades
        for symbol, target_weight in target_weights.items():
            current_weight = current_weights.get(symbol, 0)
            weight_diff = target_weight - current_weight
            
            if abs(weight_diff) > 0.01:  # Only rebalance if difference > 1%
                target_value = self.total_value * target_weight
                current_value = self.total_value * current_weight
                trade_value = target_value - current_value
                
                if symbol in self.positions:
                    current_price = self.positions[symbol]['current_price']
                    
                    if trade_value > 0:
                        # Buy more
                        quantity = int(trade_value / current_price)
                        if quantity > 0:
                            self.buy_stock(symbol, quantity, current_price, "Rebalancing")
                    else:
                        # Sell some
                        quantity = int(abs(trade_value) / current_price)
                        if quantity > 0:
                            self.sell_stock(symbol, quantity, current_price, "Rebalancing")
        
        self.logger.info("Portfolio rebalancing completed")

# Example usage
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create portfolio manager
    portfolio = PortfolioManager(100000)  # 1 Lakh initial capital
    
    # Test buying stocks
    print("Testing stock purchases...")
    portfolio.buy_stock("RELIANCE.NS", 10, 2500.0, "Test Buy")
    portfolio.buy_stock("TCS.NS", 5, 3200.0, "Test Buy")
    
    # Update prices
    current_prices = {
        "RELIANCE.NS": 2550.0,
        "TCS.NS": 3150.0
    }
    portfolio.update_positions(current_prices)
    
    # Get portfolio summary
    summary = portfolio.get_portfolio_summary()
    print("\nPortfolio Summary:")
    for key, value in summary.items():
        if key != 'positions':
            print(f"{key}: {value}")
    
    print("\nPositions:")
    for pos in summary['positions']:
        print(f"{pos['symbol']}: {pos['quantity']} shares, P&L: ₹{pos['unrealized_pnl']:.2f}")
    
    # Test selling
    print("\nTesting stock sale...")
    portfolio.sell_stock("RELIANCE.NS", 5, 2600.0, "Test Sell")
    
    # Final summary
    final_summary = portfolio.get_portfolio_summary()
    print(f"\nFinal Portfolio Value: ₹{final_summary['total_value']:.2f}")
    print(f"Total Return: {final_summary['total_return']:.2f}%")