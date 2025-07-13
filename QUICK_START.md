# 🚀 Quick Start Guide - AI Stock Trading Bot

## ✅ Your AI Trading Bot is Ready!

The setup has been completed successfully and the dashboard is now running!

## 📱 Access Your Dashboard

**Dashboard URL:** http://localhost:8501

The dashboard should be open in your browser. If not, copy and paste the URL above.

## 🎯 Getting Started

### 1. **Select Stocks** (Left Sidebar)

- Choose up to 20 NSE stocks from the dropdown
- Default: Top 10 Indian stocks (RELIANCE, TCS, INFY, etc.)

### 2. **Initialize Bot**

- Click "Initialize Bot" in the sidebar
- This loads/trains AI models for each stock
- **First time:** Models will be trained (may take 5-10 minutes)
- **Subsequent runs:** Models load instantly

### 3. **Start Trading**

- Click "▶️ Start Trading"
- Bot begins automated trading simulation
- Uses ₹1,00,000 dummy money (no real money!)

### 4. **Monitor Performance**

- **Portfolio Tab:** View holdings and P&L
- **Live Charts Tab:** See real-time stock charts
- **Trading Signals Tab:** AI predictions and confidence
- **Reports Tab:** Performance analytics

## 🤖 How It Works

### AI-Powered Trading

- **Random Forest ML Models** instead of LSTM (compatible with Python 3.13)
- **Technical Analysis:** 15+ indicators (RSI, MACD, Bollinger Bands)
- **Real-time NSE Data:** Live prices from Yahoo Finance
- **Risk Management:** Automatic stop-loss and position sizing

### Trading Strategy

1. **Technical Analysis** (40% weight) + **ML Prediction** (60% weight)
2. **Buy Signal:** Combined confidence > 50%
3. **Sell Signal:** Combined confidence < -50%
4. **Risk Controls:** 5% stop-loss, 15% take-profit

## 🛡️ Safety Features

- ✅ **Dummy Money Only** - No real money at risk
- ✅ **Paper Trading** - All trades are simulated
- ✅ **NSE Market Hours** - Only trades 9:15 AM - 3:30 PM
- ✅ **Position Limits** - Max 10% per stock
- ✅ **Daily Loss Limits** - Built-in protection

## 🔧 Command Reference

### Start Dashboard

```bash
python3 setup_and_run.py
```

Then choose option 1.

### Alternative Methods

```bash
# Quick start (if already setup)
./venv/bin/python -m streamlit run dashboard.py

# Or using the shell script
./quick_start.sh
```

## 📊 Dashboard Tabs

### 1. Portfolio Tab

- Portfolio composition pie chart
- Current positions table
- P&L summary

### 2. Live Charts Tab

- Candlestick charts with volume
- Real-time price updates
- Technical indicators overlay

### 3. Trading Signals Tab

- Current AI predictions for all stocks
- Signal confidence levels
- Buy/Sell/Hold recommendations

### 4. Reports Tab

- Trading statistics
- Performance metrics
- Risk analytics
- Downloadable reports

## 🎮 Interactive Controls

### Sidebar Controls

- **Stock Selection:** Multi-select dropdown
- **Initialize Bot:** Setup AI models
- **Start/Stop Trading:** Control automation
- **Retrain Models:** Update with latest data
- **Auto-refresh:** Live data updates

### Main Dashboard

- **Portfolio Metrics:** Top-level KPIs
- **Tab Navigation:** Switch between views
- **Chart Interactions:** Zoom, pan, hover
- **Real-time Updates:** Live price feeds

## 🧠 AI Models

### Machine Learning

- **Model Type:** Random Forest (sklearn)
- **Features:** OHLCV + Technical indicators
- **Training:** 2 years historical data
- **Prediction:** Next-day price movement
- **Confidence:** Model uncertainty scoring

### Technical Analysis

- **Moving Averages:** SMA 5, 10, 20, 50
- **Momentum:** RSI, Stochastic
- **Trend:** MACD, ADX
- **Volatility:** Bollinger Bands
- **Volume:** Volume oscillators

## 📈 Performance Tracking

### Key Metrics

- **Total Return:** Portfolio vs benchmark
- **Sharpe Ratio:** Risk-adjusted returns
- **Win Rate:** Profitable trades %
- **Max Drawdown:** Largest decline
- **Trade Count:** Total executed trades

### Daily Reports

- Automatic report generation
- Trade log with reasons
- Signal accuracy tracking
- Model performance metrics

## 🔧 Troubleshooting

### Dashboard Won't Load

```bash
# Restart the setup
python3 setup_and_run.py
```

### No Data Showing

- Check internet connection
- Verify NSE market hours (9:15 AM - 3:30 PM IST)
- Try selecting different stocks

### Models Not Training

- Ensure you have internet access
- Wait for historical data download
- Check logs in terminal

### Performance Issues

- Reduce number of stocks
- Close other browser tabs
- Restart dashboard

## 💡 Tips for Best Experience

### 1. **First Time Setup**

- Let models train completely
- Start with 5-10 stocks initially
- Monitor for first few hours

### 2. **Daily Usage**

- Check signals before market open
- Monitor risk metrics
- Review end-of-day reports

### 3. **Performance Optimization**

- Use latest browser (Chrome/Safari)
- Close unnecessary tabs
- Ensure stable internet

### 4. **Learning the System**

- Start with "paper trading" mode
- Understand signal meanings
- Review trade reasons

## 🚨 Important Notes

### Educational Purpose

- This is for learning algorithmic trading
- No real money is used or at risk
- Results are simulated only

### Market Risks

- Stock prices can be volatile
- AI predictions may be wrong
- Past performance ≠ future results

### Technical Limitations

- Data may have delays
- Models need regular retraining
- Internet required for real-time data

## 🎯 Next Steps

1. **Explore the Dashboard** - Familiarize yourself with all tabs
2. **Train Models** - Let AI models learn from data
3. **Start Trading** - Begin automated simulation
4. **Monitor Performance** - Track your portfolio
5. **Analyze Results** - Learn from trading decisions

## 🆘 Need Help?

### Check The Logs

The terminal shows detailed logs of what's happening.

### Common Issues

- **No models found:** Click "Initialize Bot" first
- **No data:** Check internet and market hours
- **Slow performance:** Reduce number of stocks

### Dashboard Features

Every button and metric has tooltips - hover to learn more!

---

## 🎉 **You're All Set!**

Your AI Stock Trading Bot is now running at:
**http://localhost:8501**

**Happy Trading! 📈🤖**

_Remember: This is for educational purposes only. Always do your own research before making real investment decisions._
