"""
Configuration file for Stock Trading Bot
"""

import os
from datetime import datetime, timedelta

# Trading Configuration
INITIAL_CAPITAL = 100000  # Starting with 1 Lakh rupees (dummy money)
COMMISSION_RATE = 0.001  # 0.1% commission per trade
MAX_POSITION_SIZE = 0.1  # Maximum 10% of portfolio per stock
STOP_LOSS_PERCENTAGE = 0.05  # 5% stop loss
TAKE_PROFIT_PERCENTAGE = 0.15  # 15% take profit

# Data Configuration
DATA_SOURCE = "yahoo"
MARKET = "NSE"
CURRENCY = "INR"

# NSE Stock Symbols (Top 50 stocks for demo)
NSE_STOCKS = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "HINDUNILVR.NS",
    "ICICIBANK.NS", "KOTAKBANK.NS", "BHARTIARTL.NS", "ITC.NS", "SBIN.NS",
    "BAJFINANCE.NS", "ASIANPAINT.NS", "MARUTI.NS", "HCLTECH.NS", "AXISBANK.NS",
    "LT.NS", "NESTLEIND.NS", "ULTRACEMCO.NS", "DMART.NS", "BAJAJFINSV.NS",
    "WIPRO.NS", "ADANIGREEN.NS", "ONGC.NS", "TITAN.NS", "SUNPHARMA.NS",
    "NTPC.NS", "POWERGRID.NS", "TECHM.NS", "JSWSTEEL.NS", "TATAMOTORS.NS",
    "INDUSINDBK.NS", "GRASIM.NS", "COALINDIA.NS", "HDFCLIFE.NS", "SBILIFE.NS",
    "DIVISLAB.NS", "BRITANNIA.NS", "CIPLA.NS", "DRREDDY.NS", "EICHERMOT.NS",
    "BPCL.NS", "TATACONSUM.NS", "HINDALCO.NS", "BAJAJ-AUTO.NS", "APOLLOHOSP.NS",
    "HEROMOTOCO.NS", "UPL.NS", "GODREJCP.NS", "PIDILITIND.NS", "ADANIENT.NS"
]

# Model Configuration
LSTM_CONFIG = {
    "sequence_length": 60,  # Use 60 days of data to predict next day
    "epochs": 100,
    "batch_size": 32,
    "validation_split": 0.2,
    "lstm_units": [50, 50],  # Two LSTM layers with 50 units each
    "dropout_rate": 0.2,
    "learning_rate": 0.001
}

# Technical Indicators
INDICATORS = {
    "SMA": [5, 10, 20, 50],  # Simple Moving Averages
    "EMA": [12, 26],  # Exponential Moving Averages
    "RSI": 14,  # Relative Strength Index
    "MACD": [12, 26, 9],  # MACD parameters
    "BB": [20, 2],  # Bollinger Bands (period, std dev)
    "STOCH": [14, 3, 3],  # Stochastic Oscillator
    "ADX": 14,  # Average Directional Index
    "CCI": 20,  # Commodity Channel Index
    "ROC": 10,  # Rate of Change
    "WILLIAMS": 14  # Williams %R
}

# Trading Hours (NSE)
MARKET_OPEN = "09:15"
MARKET_CLOSE = "15:30"
PRE_MARKET_OPEN = "09:00"
POST_MARKET_CLOSE = "16:00"

# Database Configuration
DATABASE_PATH = "trading_bot.db"

# API Configuration
YAHOO_FINANCE_INTERVAL = "1m"  # 1-minute data for real-time
UPDATE_FREQUENCY = 60  # Update every 60 seconds

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FILE = "trading_bot.log"

# Model Paths
MODEL_DIR = "models"
SCALER_DIR = "scalers"
REPORTS_DIR = "reports"
CHARTS_DIR = "charts"

# Create directories if they don't exist
for directory in [MODEL_DIR, SCALER_DIR, REPORTS_DIR, CHARTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Risk Management
RISK_CONFIG = {
    "max_daily_loss": 0.02,  # Maximum 2% daily loss
    "max_position_correlation": 0.7,  # Maximum correlation between positions
    "var_confidence": 0.95,  # Value at Risk confidence level
    "max_drawdown": 0.1,  # Maximum 10% drawdown
    "position_sizing": "kelly",  # Position sizing method
    "rebalance_frequency": "weekly"  # Portfolio rebalancing
}

# Prediction Thresholds
PREDICTION_THRESHOLDS = {
    "buy_threshold": 0.02,  # Buy if predicted return > 2%
    "sell_threshold": -0.02,  # Sell if predicted return < -2%
    "hold_threshold": 0.01,  # Hold if abs(predicted return) < 1%
    "confidence_threshold": 0.7  # Minimum confidence for trade execution
}

# Features for ML Model
FEATURES = [
    "open", "high", "low", "close", "volume",
    "sma_5", "sma_10", "sma_20", "sma_50",
    "ema_12", "ema_26", "rsi", "macd", "macd_signal",
    "bb_upper", "bb_middle", "bb_lower",
    "stoch_k", "stoch_d", "adx", "cci", "roc", "williams_r",
    "price_change", "volume_change", "volatility"
]