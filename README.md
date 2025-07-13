# 🤖 AI Stock Trading Bot for NSE

A comprehensive AI-powered stock trading bot for the Indian National Stock Exchange (NSE) that uses LSTM neural networks for price prediction, technical analysis, and automated trading with dummy money.

## 🚀 Features

### Core Features

- **Real-time NSE Data**: Live stock data from Yahoo Finance
- **LSTM Neural Networks**: Deep learning models for price prediction
- **Technical Analysis**: 15+ technical indicators (RSI, MACD, Bollinger Bands, etc.)
- **Automated Trading**: AI-driven buy/sell decisions with risk management
- **Portfolio Management**: Complete portfolio tracking with dummy money
- **Live Dashboard**: Interactive Streamlit dashboard with real-time charts

### AI & Machine Learning

- **LSTM Models**: Individual models for each stock symbol
- **Feature Engineering**: Technical indicators as input features
- **Model Retraining**: Continuous learning with new market data
- **Signal Confidence**: Confidence scoring for trading decisions
- **Ensemble Approach**: Combines technical analysis with ML predictions

### Risk Management

- **Position Sizing**: Kelly criterion and fixed percentage approaches
- **Stop Loss**: Automatic stop-loss at 5% decline
- **Take Profit**: Automatic profit-taking at 15% gain
- **Portfolio Limits**: Maximum position size and correlation controls
- **Daily Loss Limits**: Maximum daily loss protection

### Dashboard Features

- **Portfolio Overview**: Real-time portfolio value and P&L
- **Live Charts**: Candlestick charts with volume and indicators
- **Trading Signals**: Current AI predictions and confidence levels
- **Performance Analytics**: Detailed performance metrics and reports
- **Trade History**: Complete trade log with reasons and outcomes

## 📋 Requirements

### System Requirements

- Python 3.8 or higher
- 4GB+ RAM (for LSTM training)
- Internet connection for real-time data

### Python Dependencies

```
yfinance==0.2.28
pandas==2.1.4
numpy==1.24.3
scikit-learn==1.3.2
tensorflow==2.15.0
matplotlib==3.8.2
plotly==5.17.0
streamlit==1.28.2
```

## 🛠️ Installation

1. **Clone the repository**:

```bash
git clone <repository-url>
cd Stock-Trade_AI
```

2. **Install dependencies**:

```bash
pip install -r requirements.txt
```

3. **Create necessary directories**:

```bash
mkdir models scalers reports charts
```

## 🚀 Quick Start

### 1. Launch the Dashboard

```bash
python run_dashboard.py
```

Or directly with Streamlit:

```bash
streamlit run dashboard.py
```

### 2. Initialize the Bot

1. Open the dashboard in your browser (http://localhost:8501)
2. Select stocks from the sidebar (default: top 10 NSE stocks)
3. Click "Initialize Bot" to setup LSTM models
4. Wait for model training/loading to complete

### 3. Start Trading

1. Click "▶️ Start Trading" to begin automated trading
2. Monitor real-time portfolio performance
3. View live charts and trading signals
4. Stop trading anytime with "⏹️ Stop Trading"

## 📊 Usage Guide

### Dashboard Sections

#### 1. Portfolio Tab

- **Portfolio Composition**: Pie chart of current holdings
- **Portfolio Summary**: Key metrics and performance
- **Positions Table**: Detailed view of all positions

#### 2. Live Charts Tab

- **Candlestick Charts**: Price action with volume
- **Technical Indicators**: Overlaid indicators
- **Real-time Updates**: Live price movements

#### 3. Trading Signals Tab

- **Current Signals**: AI predictions for all stocks
- **Signal Confidence**: Confidence levels for each prediction
- **Detailed Analysis**: In-depth signal breakdown

#### 4. Reports Tab

- **Trading Statistics**: Performance metrics
- **Risk Analytics**: Risk assessment and alerts
- **Downloadable Reports**: Complete trading reports

### Bot Controls

#### Sidebar Controls

- **Stock Selection**: Choose up to 20 NSE stocks
- **Initialize Bot**: Setup LSTM models
- **Start/Stop Trading**: Control automated trading
- **Retrain Models**: Update models with latest data
- **Auto-refresh**: Automatic data updates

#### Trading Parameters

- **Initial Capital**: ₹1,00,000 (configurable in config.py)
- **Position Size**: Maximum 10% per stock
- **Stop Loss**: 5% automatic stop loss
- **Take Profit**: 15% automatic profit taking

## 🧠 AI Model Details

### LSTM Architecture

```
Input Layer: 60 days × features
LSTM Layer 1: 50 units with dropout
LSTM Layer 2: 50 units with dropout
Dense Layer 1: 50 units with ReLU
Dense Layer 2: 25 units with ReLU
Output Layer: 1 unit (price prediction)
```

### Features Used

- OHLCV data (Open, High, Low, Close, Volume)
- Simple Moving Averages (5, 10, 20, 50 days)
- Exponential Moving Averages (12, 26 days)
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Stochastic Oscillator
- ADX (Average Directional Index)
- Volume indicators
- Price action metrics

### Training Process

1. **Data Collection**: 2 years of historical data
2. **Feature Engineering**: Calculate technical indicators
3. **Data Preprocessing**: Normalization and sequence creation
4. **Model Training**: LSTM training with validation
5. **Performance Evaluation**: Metrics calculation and validation

## 📈 Trading Strategy

### Signal Generation

1. **Technical Analysis**: Multiple indicator analysis
2. **LSTM Prediction**: Neural network price forecast
3. **Signal Combination**: Weighted average of signals
4. **Confidence Scoring**: Reliability assessment

### Decision Making

- **Buy Signal**: Combined signal > 0.5 with confidence > 0.3
- **Sell Signal**: Combined signal < -0.5 with confidence > 0.3
- **Hold Signal**: Signals between thresholds
- **Risk Checks**: Portfolio limits and risk management

### Execution Logic

1. **Market Hours Check**: Only trade during NSE hours (9:15 AM - 3:30 PM)
2. **Position Sizing**: Calculate optimal position size
3. **Risk Validation**: Check all risk parameters
4. **Order Execution**: Simulate real trades with commission
5. **Portfolio Update**: Update holdings and cash

## 🛡️ Risk Management

### Portfolio Level

- **Maximum Position Size**: 10% of portfolio per stock
- **Maximum Daily Loss**: 2% of portfolio value
- **Maximum Drawdown**: 10% from peak value
- **Position Correlation**: Maximum 70% correlation between positions

### Trade Level

- **Stop Loss Orders**: Automatic 5% stop loss
- **Take Profit Orders**: Automatic 15% profit taking
- **Commission Costs**: 0.1% per trade simulation
- **Cash Buffer**: 5% minimum cash reserve

### Model Level

- **Confidence Thresholds**: Minimum confidence for trades
- **Model Validation**: Regular performance monitoring
- **Data Quality Checks**: Input data validation
- **Prediction Limits**: Reasonable prediction bounds

## 📁 File Structure

```
Stock-Trade_AI/
├── config.py              # Configuration and parameters
├── data_fetcher.py         # Real-time data collection
├── technical_indicators.py # Technical analysis indicators
├── lstm_model.py          # LSTM neural network implementation
├── portfolio_manager.py   # Portfolio and trade management
├── trading_bot.py         # Main trading bot logic
├── dashboard.py           # Streamlit dashboard
├── run_dashboard.py       # Dashboard launcher
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── models/               # Saved LSTM models
├── scalers/              # Data preprocessing scalers
├── reports/              # Trading reports and analytics
└── charts/               # Generated charts and visualizations
```

## 🔧 Configuration

### Key Settings (config.py)

```python
# Trading Configuration
INITIAL_CAPITAL = 100000          # Starting capital
COMMISSION_RATE = 0.001           # 0.1% commission
MAX_POSITION_SIZE = 0.1           # 10% max per stock
STOP_LOSS_PERCENTAGE = 0.05       # 5% stop loss
TAKE_PROFIT_PERCENTAGE = 0.15     # 15% take profit

# LSTM Configuration
LSTM_CONFIG = {
    "sequence_length": 60,         # 60 days lookback
    "epochs": 100,                 # Training epochs
    "batch_size": 32,              # Batch size
    "lstm_units": [50, 50],        # LSTM layer units
    "dropout_rate": 0.2,           # Dropout rate
    "learning_rate": 0.001         # Learning rate
}
```

### Customization Options

1. **Stock Selection**: Modify NSE_STOCKS list
2. **Trading Parameters**: Adjust risk and position sizing
3. **Model Architecture**: Change LSTM configuration
4. **Indicators**: Add/remove technical indicators
5. **Market Hours**: Customize trading schedule

## 📊 Performance Monitoring

### Dashboard Metrics

- **Portfolio Value**: Real-time portfolio valuation
- **Total Return**: Percentage and absolute returns
- **Sharpe Ratio**: Risk-adjusted returns
- **Win Rate**: Percentage of profitable trades
- **Maximum Drawdown**: Largest peak-to-trough decline

### Trade Analytics

- **Trade History**: Complete log of all trades
- **Signal Accuracy**: Prediction vs actual performance
- **Model Performance**: LSTM model metrics
- **Risk Metrics**: Portfolio risk assessment

## 🚨 Important Disclaimers

### Simulation Only

- **No Real Money**: This bot uses dummy money for simulation
- **Paper Trading**: All trades are simulated, not executed
- **Educational Purpose**: For learning and research only
- **No Financial Advice**: Not intended as investment advice

### Risk Warnings

- **Market Risk**: Stock prices can be volatile and unpredictable
- **Model Risk**: AI predictions may be inaccurate
- **Technical Risk**: Software bugs or failures possible
- **Data Risk**: Real-time data may have delays or errors

### Regulatory Compliance

- **Indian Regulations**: Ensure compliance with SEBI regulations
- **Data Usage**: Respect Yahoo Finance terms of service
- **Trading Permissions**: Obtain proper licenses for live trading
- **Tax Implications**: Understand tax obligations for trading

## 🤝 Contributing

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Areas for Contribution

- **New Features**: Additional indicators or strategies
- **Bug Fixes**: Identify and fix issues
- **Documentation**: Improve documentation
- **Testing**: Add unit tests and integration tests
- **Performance**: Optimize code and models

## 📞 Support

### Getting Help

- **Issues**: Report bugs on GitHub issues
- **Questions**: Ask questions in discussions
- **Documentation**: Check README and code comments
- **Community**: Join trading and ML communities

### Common Issues

1. **Module Import Errors**: Check Python path and dependencies
2. **Data Connection Issues**: Verify internet connection
3. **Model Training Failures**: Check memory and data availability
4. **Dashboard Loading Issues**: Ensure Streamlit is properly installed

## 📚 Learning Resources

### Recommended Reading

- **Technical Analysis**: "Technical Analysis of the Financial Markets" by Murphy
- **Machine Learning**: "Hands-On Machine Learning" by Aurélien Géron
- **Algorithmic Trading**: "Algorithmic Trading" by Ernest Chan
- **Python Finance**: "Python for Finance" by Yves Hilpisch

### Online Resources

- **NSE Website**: Official exchange information
- **Yahoo Finance API**: Data source documentation
- **TensorFlow/Keras**: Deep learning framework docs
- **Streamlit**: Dashboard framework documentation

## 📄 License

This project is open source and available under the MIT License. See LICENSE file for details.

## 🙏 Acknowledgments

- **Yahoo Finance**: For providing free market data
- **TensorFlow Team**: For the deep learning framework
- **Streamlit Team**: For the dashboard framework
- **Python Community**: For the amazing ecosystem
- **NSE**: For the transparent and efficient market

---

**Happy Trading! 📈🤖**

_Remember: This is for educational purposes only. Always do your own research and never invest more than you can afford to lose._
