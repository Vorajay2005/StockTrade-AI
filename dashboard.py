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
# Import enhanced portfolio - with error handling
try:
    from enhanced_portfolio import create_enhanced_portfolio_interface, EnhancedPortfolioManager
    ENHANCED_FEATURES_AVAILABLE = True
except ImportError as e:
    ENHANCED_FEATURES_AVAILABLE = False
    print(f"Enhanced portfolio features not available: {e}")
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
        
        # Always show the main dashboard with tabs
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
        
        # Always show top metrics and tabs, even if bot is not initialized
        if self.bot and st.session_state.get('bot_initialized'):
            # Get current data from bot
            portfolio_status = self.bot.get_portfolio_status()
            current_signals = self.bot.get_current_signals()
            
            # Top metrics
            self.render_portfolio_metrics(portfolio_status)
        else:
            # Show sample/empty metrics when bot is not initialized
            self.render_sample_metrics()
            portfolio_status = None
            current_signals = None
        
        # Always show tabs - this is the key fix!
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Portfolio", "📊 Live Charts", "🎯 Trading Signals", "📋 Reports", "🎯 AI Trading"])
        
        with tab1:
            if portfolio_status:
                self.render_portfolio_tab(portfolio_status)
            else:
                self.render_setup_portfolio_tab()
        
        with tab2:
            if current_signals:
                self.render_charts_tab(current_signals)
            else:
                self.render_setup_charts_tab()
        
        with tab3:
            if current_signals:
                self.render_signals_tab(current_signals)
            else:
                self.render_setup_signals_tab()
        
        with tab4:
            self.render_reports_tab()
        
        with tab5:
            self.render_enhanced_trading_tab()
    
    def render_sample_metrics(self):
        """Render sample metrics when bot is not initialized"""
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Portfolio Value", "₹1,00,000", "₹0")
        with col2:
            st.metric("Available Cash", "₹1,00,000", "₹0")
        with col3:
            st.metric("Total Return", "0.0%", "0.0%")
        with col4:
            st.metric("Active Positions", "0", "0")
        with col5:
            st.metric("Day's P&L", "₹0", "0.0%")
        
        st.info("💡 **Initialize the bot** from the sidebar to see real portfolio data!")
    
    def render_setup_portfolio_tab(self):
        """Render portfolio tab when bot is not initialized"""
        st.header("📈 Portfolio Management")
        st.warning("⚠️ **Bot not initialized yet**")
        st.info("Please initialize the trading bot from the sidebar to:")
        st.write("- ✅ View your current portfolio")
        st.write("- ✅ See individual stock positions")
        st.write("- ✅ Track profit & loss")
        st.write("- ✅ Monitor portfolio performance")
        
        # Show sample portfolio structure
        st.subheader("📊 Sample Portfolio Structure")
        sample_portfolio = {
            'Symbol': ['RELIANCE.NS', 'TCS.NS', 'INFY.NS'],
            'Quantity': [10, 5, 15],
            'Avg Price': ['₹2,450', '₹3,890', '₹1,650'],
            'Current Price': ['₹2,520', '₹3,920', '₹1,680'],
            'P&L': ['+₹700', '+₹150', '+₹450']
        }
        import pandas as pd
        st.dataframe(pd.DataFrame(sample_portfolio), use_container_width=True)
    
    def render_setup_charts_tab(self):
        """Render charts tab when bot is not initialized"""
        st.header("📊 Live Stock Charts")
        st.warning("⚠️ **Bot not initialized yet**")
        st.info("Please initialize the trading bot from the sidebar to:")
        st.write("- ✅ View real-time stock charts")
        st.write("- ✅ See technical indicators")
        st.write("- ✅ Monitor price movements")
        st.write("- ✅ Analyze trading patterns")
        
        # Show sample chart
        st.subheader("📈 Sample Chart")
        import numpy as np
        chart_data = pd.DataFrame(
            np.random.randn(20, 3),
            columns=['RELIANCE.NS', 'TCS.NS', 'INFY.NS']
        )
        st.line_chart(chart_data)
        st.caption("This is sample data. Real charts will appear after bot initialization.")
    
    def render_setup_signals_tab(self):
        """Render signals tab when bot is not initialized"""
        st.header("🎯 AI Trading Signals")
        st.warning("⚠️ **Bot not initialized yet**")
        st.info("Please initialize the trading bot from the sidebar to:")
        st.write("- ✅ Get AI-powered buy/sell signals")
        st.write("- ✅ See confidence scores")
        st.write("- ✅ View ML predictions")
        st.write("- ✅ Monitor signal strength")
        
        # Show sample signals
        st.subheader("🤖 Sample AI Signals")
        sample_signals = {
            'Symbol': ['RELIANCE.NS', 'TCS.NS', 'INFY.NS'],
            'Signal': ['🟢 BUY', '🟡 HOLD', '🔴 SELL'],
            'Confidence': ['85%', '60%', '75%'],
            'Current Price': ['₹2,520', '₹3,920', '₹1,680'],
            'Target': ['₹2,650', '₹4,000', '₹1,620']
        }
        st.dataframe(pd.DataFrame(sample_signals), use_container_width=True)
        st.caption("These are sample signals. Real AI predictions will appear after bot initialization.")
    
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
    
    def render_enhanced_trading_tab(self):
        """Render the enhanced AI trading interface"""
        st.header("🎯 AI-Powered Trading Center")
        
        if not self.bot or not st.session_state.get('bot_initialized'):
            st.warning("⚠️ Please initialize the trading bot first!")
            st.info("Go to the sidebar and click 'Initialize Bot' to get started.")
            return
        
        # Create sub-tabs for different features
        ai_tab1, ai_tab2, ai_tab3, ai_tab4 = st.tabs([
            "🤖 AI Recommendations",
            "💼 Manual Trading", 
            "🎯 Price Targets",
            "🧠 Model Training"
        ])
        
        with ai_tab1:
            self.render_ai_recommendations()
        
        with ai_tab2:
            self.render_manual_trading()
        
        with ai_tab3:
            self.render_price_targets()
        
        with ai_tab4:
            self.render_model_training()
    
    def render_ai_recommendations(self):
        """Render AI recommendations sub-tab"""
        st.subheader("🤖 AI Stock Recommendations")
        
        if st.button("🔄 Get Fresh AI Recommendations", type="primary"):
            with st.spinner("Getting AI recommendations..."):
                try:
                    # Get current signals from the bot
                    current_signals = self.bot.get_current_signals()
                    
                    if current_signals:
                        st.success(f"Found {len(current_signals)} stock recommendations!")
                        
                        # Display recommendations in a table
                        recommendations = []
                        for symbol, signal_data in current_signals.items():
                            ml_pred = signal_data.get('ml_prediction', {})
                            current_price = signal_data.get('current_price', 0)
                            signal = ml_pred.get('signal', 'HOLD')
                            confidence = ml_pred.get('signal_strength', 0)
                            
                            # Determine recommendation color
                            if signal == 'BUY' and confidence > 0.6:
                                recommendation = "🟢 STRONG BUY"
                            elif signal == 'BUY':
                                recommendation = "🟢 BUY"
                            elif signal == 'SELL' and confidence > 0.6:
                                recommendation = "🔴 STRONG SELL"
                            elif signal == 'SELL':
                                recommendation = "🟠 SELL"
                            else:
                                recommendation = "🟡 HOLD"
                            
                            recommendations.append({
                                'Symbol': symbol,
                                'Current Price': f"₹{current_price:.2f}",
                                'Recommendation': recommendation,
                                'Confidence': f"{confidence:.1%}",
                                'Signal Strength': f"{confidence:.3f}"
                            })
                        
                        # Sort by confidence
                        recommendations.sort(key=lambda x: float(x['Signal Strength']), reverse=True)
                        
                        # Display top 10 recommendations
                        import pandas as pd
                        df = pd.DataFrame(recommendations[:10])
                        st.dataframe(df, use_container_width=True)
                        
                    else:
                        st.info("No recommendations available. Try starting the trading bot first.")
                        
                except Exception as e:
                    st.error(f"Error getting recommendations: {str(e)}")
        
        st.info("💡 **Tip:** Higher confidence scores (>60%) indicate stronger signals!")
    
    def render_manual_trading(self):
        """Render manual trading sub-tab"""
        st.subheader("💼 Manual Stock Trading")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Buy Stocks**")
            
            # Stock selection
            buy_symbol = st.selectbox("Select Stock to Buy", NSE_STOCKS[:20], key="manual_buy_symbol")
            
            # Get current price
            try:
                current_signals = self.bot.get_current_signals()
                current_price = current_signals.get(buy_symbol, {}).get('current_price', 0)
                
                if current_price > 0:
                    st.metric("Current Price", f"₹{current_price:.2f}")
                    
                    # Quantity input
                    buy_quantity = st.number_input("Quantity to Buy", min_value=1, value=10, key="manual_buy_qty")
                    
                    # Calculate total cost
                    commission = 0.001  # 0.1% commission
                    total_cost = buy_quantity * current_price * (1 + commission)
                    
                    st.write(f"**Total Cost:** ₹{total_cost:.2f} (including 0.1% commission)")
                    
                    # Get portfolio status
                    portfolio_status = self.bot.get_portfolio_status()
                    available_cash = portfolio_status['portfolio']['available_cash']
                    st.write(f"**Available Cash:** ₹{available_cash:.2f}")
                    
                    if st.button("💰 Execute Buy Order", type="primary"):
                        if total_cost <= available_cash:
                            # Execute buy order (simulated)
                            success = self.bot.portfolio_manager.buy_stock(
                                buy_symbol, buy_quantity, current_price, 
                                reason="Manual buy order from dashboard"
                            )
                            if success:
                                st.success(f"✅ Successfully bought {buy_quantity} shares of {buy_symbol}")
                                st.rerun()
                            else:
                                st.error("❌ Failed to execute buy order")
                        else:
                            st.error("❌ Insufficient funds")
                else:
                    st.warning("Price data not available. Please ensure the bot is running.")
                    
            except Exception as e:
                st.error(f"Error getting price data: {str(e)}")
        
        with col2:
            st.write("**Sell Stocks**")
            
            try:
                # Get current positions
                portfolio_status = self.bot.get_portfolio_status()
                positions = portfolio_status['positions']
                
                if positions:
                    # Position selection
                    position_symbols = list(positions.keys())
                    sell_symbol = st.selectbox("Select Stock to Sell", position_symbols, key="manual_sell_symbol")
                    
                    if sell_symbol in positions:
                        position = positions[sell_symbol]
                        current_signals = self.bot.get_current_signals()
                        current_price = current_signals.get(sell_symbol, {}).get('current_price', 0)
                        
                        st.metric("Holdings", f"{position['quantity']} shares")
                        st.metric("Avg Price", f"₹{position['avg_price']:.2f}")
                        st.metric("Current Price", f"₹{current_price:.2f}")
                        
                        # P&L calculation
                        if current_price > 0:
                            unrealized_pnl = (current_price - position['avg_price']) * position['quantity']
                            pnl_pct = (unrealized_pnl / (position['avg_price'] * position['quantity'])) * 100
                            
                            pnl_color = "green" if unrealized_pnl >= 0 else "red"
                            st.markdown(f"**Unrealized P&L:** <span style='color:{pnl_color}'>₹{unrealized_pnl:.2f} ({pnl_pct:+.1f}%)</span>", 
                                       unsafe_allow_html=True)
                            
                            # Quantity to sell
                            sell_quantity = st.number_input("Quantity to Sell", 
                                                          min_value=1, 
                                                          max_value=position['quantity'],
                                                          value=min(10, position['quantity']),
                                                          key="manual_sell_qty")
                            
                            # Calculate proceeds
                            commission = 0.001  # 0.1% commission
                            total_proceeds = sell_quantity * current_price * (1 - commission)
                            st.write(f"**Total Proceeds:** ₹{total_proceeds:.2f} (after 0.1% commission)")
                            
                            if st.button("💸 Execute Sell Order", type="primary"):
                                success = self.bot.portfolio_manager.sell_stock(
                                    sell_symbol, sell_quantity, current_price,
                                    reason="Manual sell order from dashboard"
                                )
                                if success:
                                    st.success(f"✅ Successfully sold {sell_quantity} shares of {sell_symbol}")
                                    st.rerun()
                                else:
                                    st.error("❌ Failed to execute sell order")
                else:
                    st.info("No positions to sell. Buy some stocks first!")
                    
            except Exception as e:
                st.error(f"Error getting position data: {str(e)}")
    
    def render_price_targets(self):
        """Render price targets sub-tab"""
        st.subheader("🎯 Price Targets & Stop Loss Monitoring")
        
        try:
            # Get current positions and prices
            portfolio_status = self.bot.get_portfolio_status()
            positions = portfolio_status['positions']
            current_signals = self.bot.get_current_signals()
            
            if positions:
                st.write("**Active Positions with Price Targets:**")
                
                for symbol, position in positions.items():
                    current_price = current_signals.get(symbol, {}).get('current_price', 0)
                    
                    if current_price > 0:
                        with st.container():
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.write(f"**{symbol}**")
                                st.write(f"Holdings: {position['quantity']} shares")
                                st.write(f"Avg Price: ₹{position['avg_price']:.2f}")
                            
                            with col2:
                                st.write(f"**Current Price**")
                                st.write(f"₹{current_price:.2f}")
                                price_change = current_price - position['avg_price']
                                price_change_pct = (price_change / position['avg_price']) * 100
                                color = "green" if price_change >= 0 else "red"
                                st.markdown(f"<span style='color:{color}'>{price_change_pct:+.1f}%</span>", 
                                           unsafe_allow_html=True)
                            
                            with col3:
                                # Calculate targets
                                stop_loss = position['avg_price'] * 0.95  # 5% stop loss
                                take_profit = position['avg_price'] * 1.15  # 15% take profit
                                
                                st.write("**Stop Loss (5%)**")
                                st.write(f"₹{stop_loss:.2f}")
                                if current_price <= stop_loss:
                                    st.error("🚨 Stop Loss Triggered!")
                                
                                st.write("**Take Profit (15%)**")
                                st.write(f"₹{take_profit:.2f}")
                                if current_price >= take_profit:
                                    st.success("🎯 Target Reached!")
                            
                            with col4:
                                # P&L
                                total_pnl = (current_price - position['avg_price']) * position['quantity']
                                total_pnl_pct = (total_pnl / (position['avg_price'] * position['quantity'])) * 100
                                
                                st.write("**Unrealized P&L**")
                                color = "green" if total_pnl >= 0 else "red"
                                st.markdown(f"<span style='color:{color}'>₹{total_pnl:.2f}</span>", 
                                           unsafe_allow_html=True)
                                st.markdown(f"<span style='color:{color}'>({total_pnl_pct:+.1f}%)</span>", 
                                           unsafe_allow_html=True)
                            
                            st.divider()
            else:
                st.info("No active positions to monitor. Buy some stocks to see price targets!")
                
        except Exception as e:
            st.error(f"Error loading price targets: {str(e)}")
    
    def render_model_training(self):
        """Render model training sub-tab"""
        st.subheader("🧠 AI Model Training Center")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Individual Stock Training**")
            
            train_symbol = st.selectbox("Select Stock for Training", NSE_STOCKS[:20], key="train_symbol")
            training_period = st.selectbox("Training Data Period", ["6mo", "1y", "2y", "5y"], index=2, key="train_period")
            
            if st.button("🚀 Train Model", type="primary"):
                with st.spinner(f"Training AI model for {train_symbol}..."):
                    try:
                        # Retrain the specific model
                        if self.bot and hasattr(self.bot, 'models') and train_symbol in self.bot.models:
                            # Get historical data
                            historical_data = self.bot.data_fetcher.fetch_historical_data(train_symbol, period=training_period)
                            
                            if not historical_data.empty:
                                # Add technical indicators
                                historical_data = self.bot.technical_indicators.add_all_indicators(historical_data)
                                
                                # Train the model
                                self.bot.models[train_symbol].train_model(historical_data)
                                st.success(f"✅ Model trained successfully for {train_symbol}!")
                                
                                # Show some basic metrics
                                st.info(f"📊 Training completed with {len(historical_data)} data points")
                            else:
                                st.error(f"❌ No historical data available for {train_symbol}")
                        else:
                            st.error("❌ Model not found. Please initialize the bot first.")
                            
                    except Exception as e:
                        st.error(f"❌ Training failed: {str(e)}")
        
        with col2:
            st.write("**Batch Training**")
            
            selected_stocks = st.multiselect("Select Multiple Stocks", 
                                           NSE_STOCKS[:20], 
                                           default=NSE_STOCKS[:5],
                                           key="batch_train_stocks")
            
            if st.button("🚀 Train All Selected Models", type="primary"):
                if selected_stocks:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i, symbol in enumerate(selected_stocks):
                        status_text.text(f"Training model for {symbol}...")
                        
                        try:
                            if self.bot and hasattr(self.bot, 'models') and symbol in self.bot.models:
                                historical_data = self.bot.data_fetcher.fetch_historical_data(symbol, period="2y")
                                
                                if not historical_data.empty:
                                    historical_data = self.bot.technical_indicators.add_all_indicators(historical_data)
                                    self.bot.models[symbol].train_model(historical_data)
                                    st.write(f"✅ {symbol} - Model trained successfully")
                                else:
                                    st.write(f"⚠️ {symbol} - No data available")
                            else:
                                st.write(f"❌ {symbol} - Model not found")
                        
                        except Exception as e:
                            st.write(f"❌ {symbol} - Training failed: {str(e)}")
                        
                        progress_bar.progress((i + 1) / len(selected_stocks))
                    
                    status_text.text("✅ Batch training completed!")
                    st.success("🎉 All selected models trained!")
                else:
                    st.warning("Please select at least one stock for training")
        
        # Model status
        st.write("**Current Model Status**")
        if self.bot and hasattr(self.bot, 'models'):
            model_info = []
            for symbol in NSE_STOCKS[:10]:
                if symbol in self.bot.models:
                    model_info.append({
                        'Symbol': symbol,
                        'Status': '✅ Trained',
                        'Type': 'Random Forest'
                    })
                else:
                    model_info.append({
                        'Symbol': symbol,
                        'Status': '❌ Not Trained',
                        'Type': 'N/A'
                    })
            
            import pandas as pd
            df = pd.DataFrame(model_info)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("Initialize the trading bot to see model status")
    
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