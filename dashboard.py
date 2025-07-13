"""
Streamlit Dashboard for Stock Trading Bot
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
import json
import os

# Import our trading bot components
from trading_bot import TradingBot
from data_fetcher import DataFetcher
from portfolio_manager import PortfolioManager
from config import *

# Page config
st.set_page_config(
    page_title="AI Stock Trading Bot Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #ff6b6b;
    }
    .profit { border-left-color: #51cf66 !important; }
    .loss { border-left-color: #ff6b6b !important; }
    .neutral { border-left-color: #74c0fc !important; }
</style>
""", unsafe_allow_html=True)

class TradingDashboard:
    def __init__(self):
        self.bot = None
        self.data_fetcher = DataFetcher()
        
        # Initialize session state
        if 'bot_initialized' not in st.session_state:
            st.session_state.bot_initialized = False
        if 'bot_running' not in st.session_state:
            st.session_state.bot_running = False
        if 'selected_symbols' not in st.session_state:
            st.session_state.selected_symbols = NSE_STOCKS[:10]
    
    def initialize_bot(self, symbols, train_models=False):
        """Initialize the trading bot"""
        try:
            self.bot = TradingBot(symbols)
            
            with st.spinner("Initializing LSTM models... This may take a few minutes."):
                self.bot.initialize_models(train_models=train_models)
            
            st.session_state.bot_initialized = True
            st.success("Trading bot initialized successfully!")
            
        except Exception as e:
            st.error(f"Error initializing bot: {str(e)}")
            st.session_state.bot_initialized = False
    
    def run_dashboard(self):
        """Main dashboard function"""
        st.title("🤖 AI Stock Trading Bot Dashboard")
        st.markdown("---")
        
        # Sidebar
        self.render_sidebar()
        
        # Main content
        if not st.session_state.bot_initialized:
            self.render_setup_page()
        else:
            self.render_main_dashboard()
    
    def render_sidebar(self):
        """Render sidebar with controls"""
        st.sidebar.title("🎛️ Control Panel")
        
        # Market status
        market_status = self.data_fetcher.get_market_status()
        status_color = "🟢" if market_status['is_open'] else "🔴"
        st.sidebar.markdown(f"**Market Status:** {status_color} {'OPEN' if market_status['is_open'] else 'CLOSED'}")
        st.sidebar.markdown(f"**Current Time:** {market_status['current_time']}")
        
        if not market_status['is_open'] and market_status.get('next_open'):
            st.sidebar.markdown(f"**Next Open:** {market_status['next_open']}")
        
        st.sidebar.markdown("---")
        
        # Stock selection
        st.sidebar.subheader("📊 Stock Selection")
        available_stocks = NSE_STOCKS
        selected_stocks = st.sidebar.multiselect(
            "Select stocks to trade:",
            available_stocks,
            default=st.session_state.selected_symbols[:10],
            max_selections=20
        )
        st.session_state.selected_symbols = selected_stocks
        
        # Bot controls
        st.sidebar.subheader("🤖 Bot Controls")
        
        if not st.session_state.bot_initialized:
            train_models = st.sidebar.checkbox("Train new models (takes longer)", value=False)
            
            if st.sidebar.button("Initialize Bot", type="primary"):
                if selected_stocks:
                    self.initialize_bot(selected_stocks, train_models)
                else:
                    st.sidebar.error("Please select at least one stock")
        else:
            col1, col2 = st.sidebar.columns(2)
            
            with col1:
                if not st.session_state.bot_running:
                    if st.button("▶️ Start Trading"):
                        self.start_trading()
                else:
                    if st.button("⏹️ Stop Trading"):
                        self.stop_trading()
            
            with col2:
                if st.button("🔄 Retrain Models"):
                    self.retrain_models()
            
            if st.sidebar.button("Reset Bot"):
                self.reset_bot()
        
        # Settings
        st.sidebar.subheader("⚙️ Settings")
        auto_refresh = st.sidebar.checkbox("Auto-refresh data", value=True)
        refresh_interval = st.sidebar.slider("Refresh interval (seconds)", 10, 300, 60)
        
        if auto_refresh and st.session_state.bot_initialized:
            time.sleep(refresh_interval)
            st.rerun()
    
    def render_setup_page(self):
        """Render setup page"""
        st.header("🚀 Welcome to AI Stock Trading Bot")
        
        st.markdown("""
        ### Features:
        - **Real-time NSE stock data** from Yahoo Finance
        - **LSTM neural networks** for price prediction
        - **Technical analysis** with 15+ indicators
        - **Automated trading** with risk management
        - **Portfolio tracking** with dummy money
        - **Live dashboard** with charts and analytics
        """)
        
        # Quick start section
        st.subheader("Quick Start")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 1. Select Stocks")
            st.markdown("Choose NSE stocks from the sidebar")
        
        with col2:
            st.markdown("#### 2. Initialize Bot")
            st.markdown("Click 'Initialize Bot' to setup models")
        
        with col3:
            st.markdown("#### 3. Start Trading")
            st.markdown("Begin automated trading simulation")
        
        # Sample portfolio
        st.subheader("Sample Data Preview")
        sample_data = self.get_sample_market_data()
        if not sample_data.empty:
            st.dataframe(sample_data, use_container_width=True)
    
    def render_main_dashboard(self):
        """Render main trading dashboard"""
        if not self.bot:
            st.error("Bot not properly initialized")
            return
        
        # Get current data
        portfolio_status = self.bot.get_portfolio_status()
        current_signals = self.bot.get_current_signals()
        
        # Top metrics
        self.render_portfolio_metrics(portfolio_status)
        
        # Charts section
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Portfolio", "📊 Live Charts", "🎯 Trading Signals", "📋 Reports"])
        
        with tab1:
            self.render_portfolio_tab(portfolio_status)
        
        with tab2:
            self.render_charts_tab(current_signals)
        
        with tab3:
            self.render_signals_tab(current_signals)
        
        with tab4:
            self.render_reports_tab()
    
    def render_portfolio_metrics(self, portfolio_status):
        """Render top portfolio metrics"""
        portfolio = portfolio_status['portfolio']
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            total_value = portfolio['total_value']
            st.metric(
                "Portfolio Value",
                f"₹{total_value:,.2f}",
                delta=f"₹{portfolio['total_return_amount']:,.2f}"
            )
        
        with col2:
            return_pct = portfolio['total_return']
            color = "profit" if return_pct > 0 else "loss" if return_pct < 0 else "neutral"
            st.metric(
                "Total Return",
                f"{return_pct:.2f}%",
                delta=f"{return_pct:.2f}%"
            )
        
        with col3:
            st.metric(
                "Available Cash",
                f"₹{portfolio['available_cash']:,.2f}",
                delta=f"{portfolio['cash_percentage']:.1f}%"
            )
        
        with col4:
            st.metric(
                "Positions",
                portfolio['num_positions'],
                delta=f"{len(st.session_state.selected_symbols)} tracked"
            )
        
        with col5:
            unrealized_pnl = portfolio['unrealized_pnl']
            st.metric(
                "Unrealized P&L",
                f"₹{unrealized_pnl:,.2f}",
                delta=f"₹{unrealized_pnl:,.2f}"
            )
    
    def render_portfolio_tab(self, portfolio_status):
        """Render portfolio tab"""
        portfolio = portfolio_status['portfolio']
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Portfolio composition pie chart
            if portfolio['positions']:
                positions_df = pd.DataFrame(portfolio['positions'])
                
                fig = px.pie(
                    positions_df,
                    values='market_value',
                    names='symbol',
                    title="Portfolio Composition"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No positions currently held")
        
        with col2:
            # Portfolio summary
            st.subheader("Portfolio Summary")
            
            summary_data = {
                "Metric": [
                    "Initial Capital",
                    "Current Value",
                    "Cash",
                    "Invested",
                    "Total Return",
                    "Positions"
                ],
                "Value": [
                    f"₹{portfolio['initial_capital']:,.2f}",
                    f"₹{portfolio['total_value']:,.2f}",
                    f"₹{portfolio['available_cash']:,.2f}",
                    f"₹{portfolio['invested_value']:,.2f}",
                    f"{portfolio['total_return']:.2f}%",
                    portfolio['num_positions']
                ]
            }
            
            st.dataframe(pd.DataFrame(summary_data), hide_index=True)
        
        # Positions table
        if portfolio['positions']:
            st.subheader("Current Positions")
            positions_df = pd.DataFrame(portfolio['positions'])
            
            # Format columns
            positions_df['avg_price'] = positions_df['avg_price'].apply(lambda x: f"₹{x:.2f}")
            positions_df['current_price'] = positions_df['current_price'].apply(lambda x: f"₹{x:.2f}")
            positions_df['market_value'] = positions_df['market_value'].apply(lambda x: f"₹{x:,.2f}")
            positions_df['unrealized_pnl'] = positions_df['unrealized_pnl'].apply(lambda x: f"₹{x:,.2f}")
            positions_df['unrealized_pnl_pct'] = positions_df['unrealized_pnl_pct'].apply(lambda x: f"{x:.2f}%")
            positions_df['weight'] = positions_df['weight'].apply(lambda x: f"{x:.1f}%")
            
            st.dataframe(positions_df, use_container_width=True)
    
    def render_charts_tab(self, current_signals):
        """Render live charts tab"""
        st.subheader("📊 Live Stock Charts")
        
        # Stock selector
        if current_signals:
            selected_stock = st.selectbox(
                "Select stock to view:",
                list(current_signals.keys())
            )
            
            if selected_stock:
                self.render_stock_chart(selected_stock)
        else:
            st.info("No signal data available. Please ensure the bot is running.")
    
    def render_stock_chart(self, symbol):
        """Render detailed stock chart"""
        try:
            # Fetch historical data
            hist_data = self.data_fetcher.fetch_historical_data(
                symbol, period="1mo", interval="1d"
            )
            
            if hist_data.empty:
                st.error(f"No data available for {symbol}")
                return
            
            # Create candlestick chart with volume
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=[f'{symbol} Price', 'Volume'],
                row_width=[0.7, 0.3]
            )
            
            # Candlestick chart
            fig.add_trace(
                go.Candlestick(
                    x=hist_data.index,
                    open=hist_data['open'],
                    high=hist_data['high'],
                    low=hist_data['low'],
                    close=hist_data['close'],
                    name='Price'
                ),
                row=1, col=1
            )
            
            # Volume chart
            fig.add_trace(
                go.Bar(
                    x=hist_data.index,
                    y=hist_data['volume'],
                    name='Volume',
                    marker_color='rgba(0,100,80,0.6)'
                ),
                row=2, col=1
            )
            
            # Update layout
            fig.update_layout(
                title=f"{symbol} - Live Chart",
                xaxis_rangeslider_visible=False,
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Current price info
            current_price = hist_data['close'].iloc[-1]
            price_change = hist_data['close'].iloc[-1] - hist_data['close'].iloc[-2]
            price_change_pct = (price_change / hist_data['close'].iloc[-2]) * 100
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current Price", f"₹{current_price:.2f}")
            with col2:
                st.metric("Change", f"₹{price_change:.2f}", f"{price_change_pct:.2f}%")
            with col3:
                st.metric("Volume", f"{hist_data['volume'].iloc[-1]:,}")
                
        except Exception as e:
            st.error(f"Error rendering chart: {str(e)}")
    
    def render_signals_tab(self, current_signals):
        """Render trading signals tab"""
        st.subheader("🎯 Current Trading Signals")
        
        if not current_signals:
            st.info("No signals available. Please ensure the bot is initialized and running.")
            return
        
        # Signals summary
        signals_data = []
        for symbol, signal_data in current_signals.items():
            ml_pred = signal_data.get('ml_prediction', {})
            trend = signal_data.get('trend_analysis', {})
            
            signals_data.append({
                'Symbol': symbol,
                'Current Price': f"₹{signal_data.get('current_price', 0):.2f}",
                'ML Signal': ml_pred.get('signal', 'N/A'),
                'ML Confidence': f"{ml_pred.get('signal_strength', 0):.2f}",
                'Trend': trend.get('short_term', 'N/A'),
                'RSI': f"{trend.get('rsi', 0):.1f}",
                'Last Update': signal_data.get('timestamp', 'N/A')
            })
        
        signals_df = pd.DataFrame(signals_data)
        
        # Color coding for signals
        def color_signal(val):
            if val == 'BUY':
                return 'background-color: #d4edda'
            elif val == 'SELL':
                return 'background-color: #f8d7da'
            return ''
        
        styled_df = signals_df.style.applymap(color_signal, subset=['ML Signal'])
        st.dataframe(styled_df, use_container_width=True)
        
        # Detailed signal analysis
        if signals_data:
            st.subheader("Detailed Signal Analysis")
            selected_signal = st.selectbox(
                "Select stock for detailed analysis:",
                [s['Symbol'] for s in signals_data]
            )
            
            if selected_signal in current_signals:
                signal_detail = current_signals[selected_signal]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**ML Prediction:**")
                    ml_pred = signal_detail.get('ml_prediction', {})
                    st.json(ml_pred)
                
                with col2:
                    st.write("**Trend Analysis:**")
                    trend = signal_detail.get('trend_analysis', {})
                    st.json(trend)
    
    def render_reports_tab(self):
        """Render reports tab"""
        st.subheader("📋 Trading Reports")
        
        if not self.bot:
            st.info("Bot not initialized")
            return
        
        # Generate comprehensive report
        with st.spinner("Generating report..."):
            report = self.bot.generate_report()
        
        # Report summary
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Trading Statistics:**")
            trading_stats = report['trading_stats']
            st.json(trading_stats)
        
        with col2:
            st.write("**Risk Metrics:**")
            risk_metrics = report['risk_metrics']
            st.json(risk_metrics)
        
        # Performance metrics
        st.subheader("Performance Metrics")
        performance = report.get('performance', {})
        if performance:
            perf_df = pd.DataFrame([performance]).T
            perf_df.columns = ['Value']
            st.dataframe(perf_df)
        
        # Download report
        report_json = json.dumps(report, indent=2, default=str)
        st.download_button(
            label="Download Full Report",
            data=report_json,
            file_name=f"trading_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    def get_sample_market_data(self):
        """Get sample market data for preview"""
        try:
            sample_stocks = NSE_STOCKS[:5]
            data = []
            
            for symbol in sample_stocks:
                stock_data = self.data_fetcher.fetch_real_time_data(symbol)
                if stock_data:
                    data.append({
                        'Symbol': symbol,
                        'Price': f"₹{stock_data.get('current_price', 0):.2f}",
                        'Change': f"{stock_data.get('change_percent', 0):.2f}%",
                        'Volume': f"{stock_data.get('volume', 0):,}"
                    })
            
            return pd.DataFrame(data)
        except:
            return pd.DataFrame()
    
    def start_trading(self):
        """Start the trading bot"""
        try:
            self.bot.start_trading()
            st.session_state.bot_running = True
            st.success("Trading bot started!")
        except Exception as e:
            st.error(f"Error starting bot: {str(e)}")
    
    def stop_trading(self):
        """Stop the trading bot"""
        try:
            self.bot.stop_trading()
            st.session_state.bot_running = False
            st.success("Trading bot stopped!")
        except Exception as e:
            st.error(f"Error stopping bot: {str(e)}")
    
    def retrain_models(self):
        """Retrain LSTM models"""
        try:
            with st.spinner("Retraining models... This may take several minutes."):
                self.bot.retrain_models()
            st.success("Models retrained successfully!")
        except Exception as e:
            st.error(f"Error retraining models: {str(e)}")
    
    def reset_bot(self):
        """Reset the trading bot"""
        self.bot = None
        st.session_state.bot_initialized = False
        st.session_state.bot_running = False
        st.success("Bot reset successfully!")
        st.rerun()

# Main app
def main():
    dashboard = TradingDashboard()
    dashboard.run_dashboard()

if __name__ == "__main__":
    main()