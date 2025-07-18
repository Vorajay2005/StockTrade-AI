"""
Real-time data fetcher for Indian stocks from Yahoo Finance
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import time
from typing import Dict, List, Optional, Tuple
import threading
import pytz
from config import *

class DataFetcher:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.cache = {}
        self.last_update = {}
        self.is_running = False
        self.update_thread = None
        self.ist_timezone = pytz.timezone('Asia/Kolkata')
    
    def get_ist_time(self) -> datetime:
        """Get current time in Indian Standard Time (IST)"""
        return datetime.now(self.ist_timezone)
        
    def fetch_historical_data(self, symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        """
        Fetch historical data for a stock symbol
        
        Args:
            symbol: Stock symbol (e.g., 'RELIANCE.NS')
            period: Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
        
        Returns:
            DataFrame with OHLCV data
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                self.logger.warning(f"No data found for {symbol}")
                return pd.DataFrame()
            
            # Clean and prepare data
            data = data.dropna()
            data.columns = [col.lower() for col in data.columns]
            
            # Add additional columns
            data['symbol'] = symbol
            data['timestamp'] = data.index
            
            self.logger.info(f"Fetched {len(data)} records for {symbol}")
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def fetch_real_time_data(self, symbol: str) -> Dict:
        """
        Fetch real-time data for a stock symbol
        
        Args:
            symbol: Stock symbol (e.g., 'RELIANCE.NS')
            
        Returns:
            Dictionary with current stock data
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Get current price data
            current_data = ticker.history(period="1d", interval="1m").tail(1)
            
            if current_data.empty:
                return {}
            
            real_time_data = {
                'symbol': symbol,
                'current_price': current_data['Close'].iloc[0],
                'open': current_data['Open'].iloc[0],
                'high': current_data['High'].iloc[0],
                'low': current_data['Low'].iloc[0],
                'volume': current_data['Volume'].iloc[0],
                'timestamp': current_data.index[0],
                'previous_close': info.get('previousClose', 0),
                'change': 0,
                'change_percent': 0,
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'dividend_yield': info.get('dividendYield', 0),
                'fifty_two_week_high': info.get('fiftyTwoWeekHigh', 0),
                'fifty_two_week_low': info.get('fiftyTwoWeekLow', 0),
                'avg_volume': info.get('averageVolume', 0)
            }
            
            # Calculate change and change percentage
            if real_time_data['previous_close'] > 0:
                real_time_data['change'] = real_time_data['current_price'] - real_time_data['previous_close']
                real_time_data['change_percent'] = (real_time_data['change'] / real_time_data['previous_close']) * 100
            
            return real_time_data
            
        except Exception as e:
            self.logger.error(f"Error fetching real-time data for {symbol}: {str(e)}")
            return {}
    
    def fetch_multiple_stocks(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        Fetch real-time data for multiple stocks
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            Dictionary with stock data for each symbol
        """
        results = {}
        
        for symbol in symbols:
            try:
                data = self.fetch_real_time_data(symbol)
                if data:
                    results[symbol] = data
                time.sleep(0.1)  # Small delay to avoid rate limiting
            except Exception as e:
                self.logger.error(f"Error fetching data for {symbol}: {str(e)}")
                continue
        
        return results
    
    def start_real_time_updates(self, symbols: List[str], callback=None):
        """
        Start real-time data updates in a separate thread
        
        Args:
            symbols: List of stock symbols to monitor
            callback: Optional callback function to handle updates
        """
        self.is_running = True
        self.symbols = symbols
        self.callback = callback
        
        self.update_thread = threading.Thread(target=self._update_loop)
        self.update_thread.daemon = True
        self.update_thread.start()
        
        self.logger.info(f"Started real-time updates for {len(symbols)} symbols")
    
    def stop_real_time_updates(self):
        """Stop real-time data updates"""
        self.is_running = False
        if self.update_thread:
            self.update_thread.join()
        self.logger.info("Stopped real-time updates")
    
    def _update_loop(self):
        """Main update loop for real-time data"""
        while self.is_running:
            try:
                # Check if market is open
                if not self.is_market_open():
                    time.sleep(60)  # Check every minute when market is closed
                    continue
                
                # Fetch data for all symbols
                current_data = self.fetch_multiple_stocks(self.symbols)
                
                # Update cache
                self.cache = current_data
                self.last_update = self.get_ist_time()
                
                # Call callback if provided
                if self.callback:
                    self.callback(current_data)
                
                # Wait before next update
                time.sleep(UPDATE_FREQUENCY)
                
            except Exception as e:
                self.logger.error(f"Error in update loop: {str(e)}")
                time.sleep(UPDATE_FREQUENCY)
    
    def is_market_open(self) -> bool:
        """
        Check if NSE market is currently open
        
        Returns:
            True if market is open, False otherwise
        """
        now = self.get_ist_time()
        current_time = now.strftime("%H:%M")
        
        # Check if it's a weekday (Monday = 0, Sunday = 6)
        if now.weekday() > 4:  # Saturday or Sunday
            return False
        
        # Check if current time is within market hours
        if MARKET_OPEN <= current_time <= MARKET_CLOSE:
            return True
        
        return False
    
    def get_market_status(self) -> Dict:
        """
        Get current market status
        
        Returns:
            Dictionary with market status information
        """
        now = self.get_ist_time()
        current_time = now.strftime("%H:%M")
        current_date = now.strftime("%Y-%m-%d")
        
        status = {
            'is_open': self.is_market_open(),
            'current_time': f"{current_date} {current_time} IST",
            'market_open': MARKET_OPEN,
            'market_close': MARKET_CLOSE,
            'is_weekend': now.weekday() > 4,
            'next_open': None,
            'time_to_open': None,
            'time_to_close': None
        }
        
        if not status['is_open']:
            if now.weekday() > 4:  # Weekend (Saturday=5, Sunday=6)
                # Calculate days until next Monday
                if now.weekday() == 5:  # Saturday
                    days_until_monday = 2
                else:  # Sunday
                    days_until_monday = 1
                
                next_open = now + timedelta(days=days_until_monday)
                next_open = next_open.replace(hour=9, minute=15, second=0, microsecond=0)
                status['next_open'] = next_open.strftime("%Y-%m-%d %H:%M IST")
            else:
                # Weekday - check if market closed for the day or hasn't opened yet
                if current_time > MARKET_CLOSE:
                    # Market closed for the day, find next trading day
                    next_open = now + timedelta(days=1)
                    
                    # If tomorrow is weekend, skip to Monday
                    while next_open.weekday() > 4:  # Skip Saturday (5) and Sunday (6)
                        next_open = next_open + timedelta(days=1)
                    
                    next_open = next_open.replace(hour=9, minute=15, second=0, microsecond=0)
                else:
                    # Market hasn't opened yet today
                    next_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
                status['next_open'] = next_open.strftime("%Y-%m-%d %H:%M IST")
        
        return status
    
    def get_stock_info(self, symbol: str) -> Dict:
        """
        Get detailed information about a stock
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with stock information
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            stock_info = {
                'symbol': symbol,
                'name': info.get('longName', 'N/A'),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'pb_ratio': info.get('priceToBook', 0),
                'dividend_yield': info.get('dividendYield', 0),
                'beta': info.get('beta', 0),
                'fifty_two_week_high': info.get('fiftyTwoWeekHigh', 0),
                'fifty_two_week_low': info.get('fiftyTwoWeekLow', 0),
                'avg_volume': info.get('averageVolume', 0),
                'shares_outstanding': info.get('sharesOutstanding', 0),
                'float_shares': info.get('floatShares', 0),
                'description': info.get('longBusinessSummary', 'N/A')
            }
            
            return stock_info
            
        except Exception as e:
            self.logger.error(f"Error fetching stock info for {symbol}: {str(e)}")
            return {}
    
    def get_cached_data(self, symbol: str = None) -> Dict:
        """
        Get cached real-time data
        
        Args:
            symbol: Optional specific symbol to get data for
            
        Returns:
            Cached data for symbol or all symbols
        """
        if symbol:
            return self.cache.get(symbol, {})
        return self.cache
    
    def validate_symbol(self, symbol: str) -> bool:
        """
        Validate if a symbol exists and has data
        
        Args:
            symbol: Stock symbol to validate
            
        Returns:
            True if symbol is valid, False otherwise
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1d", interval="1d")
            return not data.empty
        except:
            return False

# Example usage
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create data fetcher
    fetcher = DataFetcher()
    
    # Test fetching historical data
    print("Fetching historical data for RELIANCE.NS...")
    hist_data = fetcher.fetch_historical_data("RELIANCE.NS", period="1mo", interval="1d")
    print(f"Fetched {len(hist_data)} records")
    print(hist_data.head())
    
    # Test fetching real-time data
    print("\nFetching real-time data for RELIANCE.NS...")
    real_time = fetcher.fetch_real_time_data("RELIANCE.NS")
    print(real_time)
    
    # Test market status
    print("\nMarket Status:")
    status = fetcher.get_market_status()
    print(status)
    
    # Test stock info
    print("\nStock Info:")
    info = fetcher.get_stock_info("RELIANCE.NS")
    print(info)