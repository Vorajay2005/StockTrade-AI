"""
Main Trading Bot that integrates all components
"""

import pandas as pd
import numpy as np
import time
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json
import os

from data_fetcher import DataFetcher
from technical_indicators import TechnicalIndicators
try:
    from lstm_model import LSTMPredictor
except ImportError:
    try:
        # Try XGBoost model
        from ml_model_simple import SimpleMLPredictor as LSTMPredictor
    except ImportError:
        # Fallback to sklearn-only model
        from ml_model_sklearn import SimpleMLPredictor as LSTMPredictor
from portfolio_manager import PortfolioManager
from config import *

class TradingBot:
    def __init__(self, symbols: List[str] = None):
        self.symbols = symbols or NSE_STOCKS[:10]  # Use top 10 stocks by default
        self.logger = self._setup_logging()
        
        # Initialize components
        self.data_fetcher = DataFetcher()
        self.technical_indicators = TechnicalIndicators()
        self.portfolio_manager = PortfolioManager(INITIAL_CAPITAL)
        
        # LSTM models for each symbol
        self.predictors = {}
        
        # Trading state
        self.is_running = False
        self.last_update = None
        self.market_data = {}
        self.trading_signals = {}
        
        # Performance tracking
        self.daily_stats = []
        self.trade_log = []
        
        self.logger.info(f"Trading bot initialized with {len(self.symbols)} symbols")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logging.basicConfig(
            level=getattr(logging, LOG_LEVEL),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(LOG_FILE),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def initialize_models(self, train_models: bool = True):
        """
        Initialize LSTM models for all symbols
        
        Args:
            train_models: Whether to train models from scratch
        """
        self.logger.info("Initializing LSTM models...")
        
        for symbol in self.symbols:
            self.logger.info(f"Initializing model for {symbol}")
            
            predictor = LSTMPredictor(symbol)
            
            if train_models:
                # Fetch historical data for training
                hist_data = self.data_fetcher.fetch_historical_data(
                    symbol, period="2y", interval="1d"
                )
                
                if not hist_data.empty:
                    # Calculate technical indicators
                    data_with_indicators = self.technical_indicators.calculate_all_indicators(hist_data)
                    
                    # Train model
                    metrics = predictor.train_model(data_with_indicators)
                    self.logger.info(f"Model trained for {symbol}: Val R2 = {metrics.get('val_r2', 0):.4f}")
                else:
                    self.logger.warning(f"No historical data available for {symbol}")
            else:
                # Try to load existing model
                if not predictor.load_model():
                    self.logger.warning(f"No existing model found for {symbol}")
            
            self.predictors[symbol] = predictor
        
        self.logger.info("Model initialization completed")
    
    def start_trading(self):
        """Start the trading bot"""
        self.logger.info("Starting trading bot...")
        
        if not self.predictors:
            self.logger.error("No models initialized. Please run initialize_models() first.")
            return
        
        self.is_running = True
        
        # Start data fetching
        self.data_fetcher.start_real_time_updates(
            self.symbols,
            callback=self._on_data_update
        )
        
        # Start main trading loop
        trading_thread = threading.Thread(target=self._trading_loop)
        trading_thread.daemon = True
        trading_thread.start()
        
        self.logger.info("Trading bot started successfully")
    
    def stop_trading(self):
        """Stop the trading bot"""
        self.logger.info("Stopping trading bot...")
        
        self.is_running = False
        self.data_fetcher.stop_real_time_updates()
        
        # Save final state
        self._save_daily_stats()
        self.portfolio_manager.save_portfolio()
        
        self.logger.info("Trading bot stopped")
    
    def _on_data_update(self, data: Dict[str, Dict]):
        """
        Callback function for data updates
        
        Args:
            data: Updated market data
        """
        self.market_data = data
        self.last_update = datetime.now()
        
        # Update portfolio with current prices
        current_prices = {symbol: info.get('current_price', 0) for symbol, info in data.items()}
        self.portfolio_manager.update_positions(current_prices)
    
    def _trading_loop(self):
        """Main trading loop"""
        last_analysis_time = None
        
        while self.is_running:
            try:
                current_time = datetime.now()
                
                # Check if market is open
                if not self.data_fetcher.is_market_open():
                    time.sleep(60)  # Wait 1 minute when market is closed
                    continue
                
                # Run analysis every minute during market hours
                if (last_analysis_time is None or 
                    (current_time - last_analysis_time).seconds >= 60):
                    
                    self._run_analysis()
                    last_analysis_time = current_time
                
                time.sleep(10)  # Main loop runs every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Error in trading loop: {str(e)}")
                time.sleep(30)  # Wait 30 seconds on error
    
    def _run_analysis(self):
        """Run complete market analysis and generate trading signals"""
        try:
            self.logger.info("Running market analysis...")
            
            for symbol in self.symbols:
                if symbol not in self.market_data:
                    continue
                
                # Get recent historical data for analysis
                hist_data = self.data_fetcher.fetch_historical_data(
                    symbol, period="3mo", interval="1d"
                )
                
                if hist_data.empty:
                    continue
                
                # Calculate technical indicators
                data_with_indicators = self.technical_indicators.calculate_all_indicators(hist_data)
                
                # Get technical signals
                data_with_signals = self.technical_indicators.get_trading_signals(data_with_indicators)
                
                # Get LSTM prediction
                if symbol in self.predictors:
                    prediction = self.predictors[symbol].predict(data_with_indicators)
                else:
                    prediction = {}
                
                # Combine signals and make trading decision
                trading_decision = self._make_trading_decision(symbol, data_with_signals, prediction)
                
                if trading_decision:
                    self._execute_trade(trading_decision)
            
            # Check risk limits
            risk_check = self.portfolio_manager.check_risk_limits()
            if risk_check['alerts']:
                self.logger.warning(f"Risk alerts: {risk_check['alerts']}")
            
        except Exception as e:
            self.logger.error(f"Error in analysis: {str(e)}")
    
    def _make_trading_decision(self, symbol: str, technical_data: pd.DataFrame, ml_prediction: Dict) -> Optional[Dict]:
        """
        Make trading decision based on technical analysis and ML prediction
        
        Args:
            symbol: Stock symbol
            technical_data: DataFrame with technical indicators and signals
            ml_prediction: LSTM prediction results
            
        Returns:
            Trading decision dictionary or None
        """
        if technical_data.empty:
            return None
        
        latest_data = technical_data.iloc[-1]
        current_price = self.market_data[symbol].get('current_price', 0)
        
        if current_price <= 0:
            return None
        
        # Technical analysis signal
        technical_signal = latest_data.get('signal', 0)
        technical_strength = latest_data.get('signal_strength', 0)
        
        # ML prediction signal
        ml_signal = 0
        ml_strength = 0
        
        if ml_prediction:
            ml_signal_str = ml_prediction.get('signal', 'HOLD')
            ml_strength = ml_prediction.get('signal_strength', 0)
            
            if ml_signal_str == 'BUY':
                ml_signal = 1
            elif ml_signal_str == 'SELL':
                ml_signal = -1
        
        # Combine signals (weighted average)
        combined_signal = (technical_signal * 0.4) + (ml_signal * 0.6)
        combined_strength = (technical_strength * 0.4) + (ml_strength * 0.6)
        
        # Decision thresholds
        buy_threshold = 0.5
        sell_threshold = -0.5
        min_strength = 0.3
        
        # Current position
        current_position = self.portfolio_manager.positions.get(symbol, {})
        has_position = current_position.get('quantity', 0) > 0
        
        # Risk management checks
        risk_check = self.portfolio_manager.check_risk_limits()
        if risk_check['risk_score'] > 2:  # High risk
            return None
        
        # Trading decision logic
        if combined_signal > buy_threshold and combined_strength > min_strength:
            if not has_position or len(self.portfolio_manager.positions) < 5:  # Max 5 positions
                quantity = self.portfolio_manager.calculate_position_size(symbol, current_price, combined_strength)
                
                if quantity > 0:
                    return {
                        'action': 'BUY',
                        'symbol': symbol,
                        'quantity': quantity,
                        'price': current_price,
                        'confidence': combined_strength,
                        'technical_signal': technical_signal,
                        'ml_signal': ml_signal,
                        'reason': f"Combined signal: {combined_signal:.2f}, Strength: {combined_strength:.2f}"
                    }
        
        elif combined_signal < sell_threshold and combined_strength > min_strength:
            if has_position:
                # Sell entire position
                quantity = current_position['quantity']
                
                return {
                    'action': 'SELL',
                    'symbol': symbol,
                    'quantity': quantity,
                    'price': current_price,
                    'confidence': combined_strength,
                    'technical_signal': technical_signal,
                    'ml_signal': ml_signal,
                    'reason': f"Combined signal: {combined_signal:.2f}, Strength: {combined_strength:.2f}"
                }
        
        # Stop loss and take profit checks
        if has_position:
            avg_price = current_position['avg_price']
            price_change = (current_price - avg_price) / avg_price
            
            # Stop loss
            if price_change < -STOP_LOSS_PERCENTAGE:
                return {
                    'action': 'SELL',
                    'symbol': symbol,
                    'quantity': current_position['quantity'],
                    'price': current_price,
                    'confidence': 1.0,
                    'technical_signal': technical_signal,
                    'ml_signal': ml_signal,
                    'reason': f"Stop loss triggered: {price_change:.2%}"
                }
            
            # Take profit
            if price_change > TAKE_PROFIT_PERCENTAGE:
                return {
                    'action': 'SELL',
                    'symbol': symbol,
                    'quantity': current_position['quantity'],
                    'price': current_price,
                    'confidence': 1.0,
                    'technical_signal': technical_signal,
                    'ml_signal': ml_signal,
                    'reason': f"Take profit triggered: {price_change:.2%}"
                }
        
        return None
    
    def _execute_trade(self, decision: Dict):
        """
        Execute trading decision
        
        Args:
            decision: Trading decision dictionary
        """
        try:
            symbol = decision['symbol']
            action = decision['action']
            quantity = decision['quantity']
            price = decision['price']
            reason = decision['reason']
            
            success = False
            
            if action == 'BUY':
                success = self.portfolio_manager.buy_stock(symbol, quantity, price, reason)
            elif action == 'SELL':
                success = self.portfolio_manager.sell_stock(symbol, quantity, price, reason)
            
            if success:
                # Log trade
                trade_log_entry = {
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'symbol': symbol,
                    'action': action,
                    'quantity': quantity,
                    'price': price,
                    'confidence': decision['confidence'],
                    'technical_signal': decision['technical_signal'],
                    'ml_signal': decision['ml_signal'],
                    'reason': reason
                }
                
                self.trade_log.append(trade_log_entry)
                
                self.logger.info(f"Trade executed: {action} {quantity} {symbol} @ ₹{price:.2f}")
                
                # Save portfolio state
                self.portfolio_manager.save_portfolio()
            
        except Exception as e:
            self.logger.error(f"Error executing trade: {str(e)}")
    
    def get_portfolio_status(self) -> Dict:
        """Get current portfolio status"""
        summary = self.portfolio_manager.get_portfolio_summary()
        
        # Add market status
        market_status = self.data_fetcher.get_market_status()
        
        # Add recent performance
        performance = self.portfolio_manager.get_performance_metrics()
        
        return {
            'portfolio': summary,
            'market_status': market_status,
            'performance': performance,
            'last_update': self.last_update.strftime('%Y-%m-%d %H:%M:%S') if self.last_update else None,
            'is_running': self.is_running,
            'active_symbols': len(self.symbols),
            'models_loaded': len(self.predictors)
        }
    
    def get_current_signals(self) -> Dict:
        """Get current trading signals for all symbols"""
        signals = {}
        
        for symbol in self.symbols:
            if symbol in self.market_data:
                # Get latest prediction
                if symbol in self.predictors:
                    hist_data = self.data_fetcher.fetch_historical_data(
                        symbol, period="3mo", interval="1d"
                    )
                    
                    if not hist_data.empty:
                        data_with_indicators = self.technical_indicators.calculate_all_indicators(hist_data)
                        prediction = self.predictors[symbol].predict(data_with_indicators)
                        
                        # Get technical analysis
                        trend_analysis = self.technical_indicators.analyze_trends(data_with_indicators)
                        
                        signals[symbol] = {
                            'current_price': self.market_data[symbol].get('current_price', 0),
                            'ml_prediction': prediction,
                            'trend_analysis': trend_analysis,
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        }
        
        return signals
    
    def retrain_models(self, symbols: List[str] = None):
        """
        Retrain LSTM models with latest data
        
        Args:
            symbols: List of symbols to retrain, default all
        """
        symbols_to_retrain = symbols or self.symbols
        
        self.logger.info(f"Retraining models for {len(symbols_to_retrain)} symbols")
        
        for symbol in symbols_to_retrain:
            if symbol in self.predictors:
                try:
                    # Fetch latest data
                    hist_data = self.data_fetcher.fetch_historical_data(
                        symbol, period="2y", interval="1d"
                    )
                    
                    if not hist_data.empty:
                        data_with_indicators = self.technical_indicators.calculate_all_indicators(hist_data)
                        metrics = self.predictors[symbol].retrain_model(data_with_indicators)
                        
                        self.logger.info(f"Model retrained for {symbol}: Val R2 = {metrics.get('val_r2', 0):.4f}")
                    
                except Exception as e:
                    self.logger.error(f"Error retraining model for {symbol}: {str(e)}")
    
    def _save_daily_stats(self):
        """Save daily trading statistics"""
        today = datetime.now().strftime('%Y-%m-%d')
        
        portfolio_summary = self.portfolio_manager.get_portfolio_summary()
        performance = self.portfolio_manager.get_performance_metrics()
        
        daily_stat = {
            'date': today,
            'portfolio_value': portfolio_summary['total_value'],
            'total_return': portfolio_summary['total_return'],
            'num_positions': portfolio_summary['num_positions'],
            'num_trades': len([t for t in self.trade_log if t['timestamp'].startswith(today)]),
            'performance': performance
        }
        
        self.daily_stats.append(daily_stat)
        
        # Save to file
        stats_file = os.path.join(REPORTS_DIR, f"daily_stats_{today}.json")
        with open(stats_file, 'w') as f:
            json.dump(daily_stat, f, indent=2)
    
    def generate_report(self) -> Dict:
        """Generate comprehensive trading report"""
        portfolio_summary = self.portfolio_manager.get_portfolio_summary()
        performance = self.portfolio_manager.get_performance_metrics()
        current_signals = self.get_current_signals()
        
        # Trade statistics
        buy_trades = [t for t in self.trade_log if t['action'] == 'BUY']
        sell_trades = [t for t in self.trade_log if t['action'] == 'SELL']
        
        report = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'portfolio': portfolio_summary,
            'performance': performance,
            'trading_stats': {
                'total_trades': len(self.trade_log),
                'buy_trades': len(buy_trades),
                'sell_trades': len(sell_trades),
                'avg_confidence': np.mean([t['confidence'] for t in self.trade_log]) if self.trade_log else 0
            },
            'current_signals': current_signals,
            'risk_metrics': self.portfolio_manager.check_risk_limits(),
            'market_status': self.data_fetcher.get_market_status()
        }
        
        return report

# Example usage and testing
if __name__ == "__main__":
    # Create trading bot with top 5 NSE stocks
    test_symbols = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "HINDUNILVR.NS"]
    
    bot = TradingBot(test_symbols)
    
    try:
        # Initialize models (training may take time)
        print("Initializing models...")
        bot.initialize_models(train_models=False)  # Set to True for fresh training
        
        # Get current status
        print("\nCurrent Portfolio Status:")
        status = bot.get_portfolio_status()
        print(f"Total Value: ₹{status['portfolio']['total_value']:,.2f}")
        print(f"Available Cash: ₹{status['portfolio']['available_cash']:,.2f}")
        print(f"Market Open: {status['market_status']['is_open']}")
        
        # Get current signals
        print("\nCurrent Trading Signals:")
        signals = bot.get_current_signals()
        for symbol, signal in signals.items():
            if signal.get('ml_prediction'):
                pred = signal['ml_prediction']
                print(f"{symbol}: {pred.get('signal', 'N/A')} (Confidence: {pred.get('signal_strength', 0):.2f})")
        
        # Start trading (commented out for demo)
        # print("\nStarting trading bot...")
        # bot.start_trading()
        # 
        # # Let it run for a while
        # time.sleep(300)  # 5 minutes
        # 
        # # Stop trading
        # bot.stop_trading()
        
        # Generate final report
        print("\nGenerating final report...")
        report = bot.generate_report()
        print(f"Report generated at: {report['timestamp']}")
        
    except KeyboardInterrupt:
        print("\nStopping trading bot...")
        bot.stop_trading()
    except Exception as e:
        print(f"Error: {str(e)}")
        bot.stop_trading()