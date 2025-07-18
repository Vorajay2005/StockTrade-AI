# 🤖 AI Stock Trading Bot

An intelligent stock trading bot that uses machine learning to analyze NSE (National Stock Exchange) stocks and provide automated trading signals with a comprehensive web dashboard.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![NSE](https://img.shields.io/badge/Market-NSE%20India-orange)](https://nseindia.com)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit%20Cloud-brightgreen)](https://stocktradeai.streamlit.app)

---

## 🌐 **Live Demo Available!**

### 🚀 **Try it now:** [https://stocktradeai.streamlit.app](https://stocktradeai.streamlit.app)

**No installation required!** Experience the full AI Stock Trading Bot dashboard instantly in your browser. All features are available including:

- ✅ Real-time NSE stock data
- ✅ AI-powered trading recommendations
- ✅ Interactive portfolio management
- ✅ Live charts and technical analysis
- ✅ Risk management tools

---

## 🌟 Features

### 🤖 **AI-Powered Trading**

- **Machine Learning Models**: XGBoost and Random Forest for price prediction
- **Technical Indicators**: 15+ indicators including RSI, MACD, Bollinger Bands
- **Confidence Scoring**: Each prediction comes with a confidence percentage
- **Multi-Stock Analysis**: Simultaneous analysis of 50+ NSE stocks

### 📊 **Professional Dashboard**

- **Real-time Portfolio Tracking**: Live P&L, positions, and performance metrics
- **Interactive Charts**: Candlestick charts with technical indicators overlay
- **AI Recommendations**: Buy/sell/hold signals with confidence scores
- **Manual Trading**: One-click buy/sell with real-time price data
- **Price Target Monitoring**: Automatic stop-loss and take-profit tracking

### 🧠 **Advanced Features**

- **Model Training Center**: Train AI models individually or in batches
- **Risk Management**: Built-in position sizing and risk assessment
- **Market Status**: Real-time NSE market open/close status
- **Performance Analytics**: Detailed trading reports and statistics

## 🚀 Quick Start

### 🌐 **Try Live Demo**

**🎯 Access the live dashboard instantly:** [https://stocktradeai.streamlit.app](https://stocktradeai.streamlit.app)

_No installation required! The app is hosted on Streamlit Community Cloud._

### 💻 **Local Installation**

#### Prerequisites

- Python 3.8 or higher
- Git

#### Installation Steps

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/Stock-Trade_AI.git
   cd Stock-Trade_AI
   ```

2. **Run the setup script**

   ```bash
   python3 setup_and_run.py
   ```

3. **Choose option 1** to start the dashboard

4. **Open your browser** and go to: `http://localhost:8501`

That's it! The setup script automatically:

- ✅ Creates a virtual environment
- ✅ Installs all dependencies
- ✅ Starts the web dashboard
- ✅ Opens in your browser

## 📱 Dashboard Overview

### **5 Main Tabs**

#### 1. 📈 **Portfolio Tab**

- Current portfolio value and cash balance
- Individual stock positions with P&L
- Portfolio composition pie chart
- Performance metrics

#### 2. 📊 **Live Charts Tab**

- Real-time candlestick charts
- Technical indicators overlay (RSI, MACD, SMA, EMA)
- Volume analysis
- Interactive price charts

#### 3. 🎯 **Trading Signals Tab**

- Current AI predictions for all stocks
- Buy/sell/hold recommendations
- Confidence scores and signal strength
- Technical analysis summary

#### 4. 📋 **Reports Tab**

- Trading performance analytics
- Historical trade logs
- Profit/loss reports
- Statistics and insights

#### 5. 🎯 **AI Trading Tab** _(Enhanced Features)_

##### **4 Sub-Features:**

###### 🤖 **AI Recommendations**

- Fresh AI predictions with confidence scores
- Ranked recommendations (Strong Buy, Buy, Hold, Sell)
- Real-time analysis of market conditions

###### 💼 **Manual Trading**

- **Buy Stocks**: Select stock, quantity, see total cost
- **Sell Stocks**: Manage positions, see P&L in real-time
- **Portfolio Management**: Track all trades automatically

###### 🎯 **Price Targets**

- Automatic stop-loss (5%) and take-profit (15%) levels
- Real-time monitoring of all positions
- Alerts when targets are reached

###### 🧠 **Model Training**

- **Individual Training**: Train models for specific stocks
- **Batch Training**: Train multiple models simultaneously
- **Model Performance**: View accuracy metrics and status

## 🛠️ Technology Stack

### **Backend**

- **Python 3.8+**: Core programming language
- **XGBoost**: Gradient boosting for ML predictions
- **Scikit-learn**: Random Forest and data preprocessing
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **yfinance**: Real-time stock data fetching

### **Frontend**

- **Streamlit**: Interactive web dashboard
- **Plotly**: Interactive charts and visualizations
- **Altair**: Statistical visualizations

### **Data Sources**

- **Yahoo Finance API**: Real-time stock prices and historical data
- **NSE (National Stock Exchange)**: Indian stock market data
- **Technical Indicators**: Custom implementation of 15+ indicators

## 📊 Supported Stocks

### **50+ NSE Stocks Including:**

- **Banking**: HDFCBANK.NS, ICICIBANK.NS, SBIN.NS, KOTAKBANK.NS
- **IT**: TCS.NS, INFY.NS, WIPRO.NS, HCLTECH.NS
- **Consumer**: RELIANCE.NS, ITC.NS, HINDUNILVR.NS, NESTLEIND.NS
- **Telecom**: BHARTIARTL.NS, ADANIPORTS.NS
- **And many more...**

## 🤖 AI Models

### **Machine Learning Algorithms**

1. **XGBoost Regressor**

   - Gradient boosting for price prediction
   - Feature importance analysis
   - Hyperparameter optimization

2. **Random Forest**
   - Ensemble learning for signal generation
   - Overfitting prevention
   - Feature selection

### **Technical Indicators (15+)**

- **Trend**: SMA, EMA, MACD
- **Momentum**: RSI, Stochastic, Williams %R
- **Volatility**: Bollinger Bands, ATR
- **Volume**: OBV, Volume SMA
- **Support/Resistance**: Pivot Points

### **Features**

- **Price Data**: Open, High, Low, Close, Volume
- **Technical Indicators**: All 15+ indicators as features
- **Lag Features**: Previous day's values
- **Rolling Statistics**: Moving averages and standard deviations

## 💰 Portfolio Management

### **Simulated Trading**

- **Starting Capital**: ₹1,00,000 (demo money)
- **Commission**: 0.1% per trade (realistic simulation)
- **Position Sizing**: Maximum 10% per stock
- **Risk Management**: Automatic stop-loss at 5%

### **Trading Features**

- **Real-time Prices**: Live NSE data via Yahoo Finance
- **P&L Tracking**: Unrealized and realized gains/losses
- **Portfolio Analytics**: Performance metrics and statistics
- **Trade History**: Complete log of all transactions

## 🚦 Getting Started Guide

### **Step 1: Access the Dashboard** ⏱️ _10 seconds_

**Option A: Live Demo (Recommended)**

1. Visit [https://stocktradeai.streamlit.app](https://stocktradeai.streamlit.app)
2. Start using immediately - no setup required!

**Option B: Local Installation**

1. Open the dashboard at `http://localhost:8501`
2. Go to the sidebar and click **"🚀 Initialize Bot"**
3. Wait for initialization to complete

### **Step 2: Train AI Models** ⏱️ _2-5 minutes_

1. Go to **"🎯 AI Trading"** → **"🧠 Model Training"**
2. Select 5-10 stocks from the list
3. Click **"🚀 Train All Selected Models"**
4. Wait for training to complete

### **Step 3: Get AI Recommendations** ⏱️ _30 seconds_

1. Go to **"🎯 AI Trading"** → **"🤖 AI Recommendations"**
2. Click **"🔄 Get Fresh AI Recommendations"**
3. View ranked predictions with confidence scores

### **Step 4: Start Trading** ⏱️ _1 minute per trade_

1. Go to **"🎯 AI Trading"** → **"💼 Manual Trading"**
2. Buy stocks based on AI recommendations
3. Monitor performance in **"🎯 Price Targets"**

## 📈 Performance Metrics

### **Model Accuracy**

- **RMSE**: Root Mean Square Error for price predictions
- **R² Score**: Coefficient of determination (>0.3 is good)
- **Signal Accuracy**: Percentage of correct buy/sell signals

### **Trading Performance**

- **Total Return**: Overall portfolio performance
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades

## 🛡️ Risk Management

### **Built-in Safety Features**

- ✅ **Demo Trading Only**: No real money at risk
- ✅ **Position Limits**: Maximum 10% allocation per stock
- ✅ **Stop-Loss**: Automatic 5% stop-loss on all positions
- ✅ **Take-Profit**: 15% profit targets
- ✅ **Risk Assessment**: Low/Medium/High risk scoring

### **Market Hours**

- **NSE Trading Hours**: 9:15 AM - 3:30 PM IST
- **Real-time Status**: Dashboard shows market open/close
- **After-hours**: Analysis continues with last available prices

## 🔧 Advanced Configuration

### **Model Parameters**

```python
# XGBoost Configuration
XGB_PARAMS = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'random_state': 42
}

# Risk Management
STOP_LOSS_PERCENTAGE = 0.05  # 5%
TAKE_PROFIT_PERCENTAGE = 0.15  # 15%
MAX_POSITION_SIZE = 0.10  # 10% of portfolio
```

### **Technical Indicator Settings**

```python
# Moving Averages
SMA_PERIODS = [5, 10, 20, 50]
EMA_PERIODS = [12, 26]

# Oscillators
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
```

## 📚 API Reference

### **Main Classes**

#### `TradingBot`

```python
bot = TradingBot()
bot.initialize()
signals = bot.get_current_signals()
portfolio = bot.get_portfolio_status()
```

#### `PortfolioManager`

```python
pm = PortfolioManager()
pm.buy_stock(symbol, quantity, price)
pm.sell_stock(symbol, quantity, price)
portfolio = pm.get_portfolio_status()
```

#### `DataFetcher`

```python
df = DataFetcher()
data = df.fetch_historical_data(symbol, period="1y")
price = df.get_current_price(symbol)
```

## 📊 Sample Output

### **AI Recommendations**

```
Symbol      | Recommendation | Confidence | Current Price | Target
------------|----------------|------------|---------------|--------
RELIANCE.NS | 🟢 STRONG BUY  | 92%        | ₹2,520       | ₹2,650
TCS.NS      | 🟢 BUY         | 78%        | ₹3,920       | ₹4,100
INFY.NS     | 🟡 HOLD        | 55%        | ₹1,680       | ₹1,700
HDFCBANK.NS | 🔴 SELL        | 83%        | ₹1,600       | ₹1,520
```

### **Portfolio Status**

```
Portfolio Value: ₹1,05,240 (+5.24%)
Available Cash:  ₹15,780
Active Positions: 8
Day's P&L:       +₹1,240 (+1.2%)
```

## 🔧 Troubleshooting

### **Common Issues**

#### **Dashboard Not Loading**

```bash
# Kill any existing Streamlit processes
pkill -f streamlit

# Restart the dashboard
python3 setup_and_run.py
```

#### **Data Fetching Errors**

- Check internet connection
- Verify NSE market hours
- Try refreshing the data

#### **Model Training Failures**

- Ensure sufficient historical data (>100 days)
- Check for network connectivity
- Try training individual stocks first

### **Performance Issues**

- Close other browser tabs
- Restart the dashboard
- Clear browser cache

## 🤝 Contributing

### **How to Contribute**

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Add tests if applicable
5. Commit: `git commit -am 'Add feature'`
6. Push: `git push origin feature-name`
7. Create a Pull Request

### **Areas for Contribution**

- 🔥 New technical indicators
- 🚀 Additional ML models (LSTM, Transformer)
- 📊 Enhanced visualizations
- 🛡️ Advanced risk management
- 🌐 Real broker integration
- 📱 Mobile-responsive design

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

**This software is for educational and research purposes only. It is not intended for real money trading. The predictions and signals generated are based on historical data and technical analysis, which do not guarantee future performance. Always consult with a qualified financial advisor before making investment decisions.**

## 📞 Support

### **Issues & Questions**

- 🐛 **Bug Reports**: Open an issue on GitHub
- 💡 **Feature Requests**: Submit via GitHub Issues
- ❓ **Questions**: Check existing issues or create a new one

### **Documentation**

- 📖 **Wiki**: Detailed guides and tutorials
- 🎥 **Videos**: Setup and usage demonstrations
- 📊 **Examples**: Sample configurations and use cases

---

## 🌟 **Key Highlights**

✅ **Professional-grade trading dashboard**  
✅ **AI-powered stock recommendations**  
✅ **Real-time portfolio management**  
✅ **15+ technical indicators**  
✅ **Risk management built-in**  
✅ **Easy one-command setup**  
✅ **50+ NSE stocks supported**  
✅ **No real money risk**

**Perfect for learning algorithmic trading and ML in finance!** 📈🤖

---

<div align="center">

**Made with ❤️ for the trading community**

[🚀 Live Demo](https://stocktradeai.streamlit.app) • [⭐ Star this repo](https://github.com/yourusername/Stock-Trade_AI) • [🍴 Fork it](https://github.com/yourusername/Stock-Trade_AI/fork) • [📝 Report Issues](https://github.com/yourusername/Stock-Trade_AI/issues)

</div>
