"""
Enhanced Portfolio Management with Advanced Trading Features
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
from typing import Dict, List, Optional, Tuple
from portfolio_manager import PortfolioManager
from data_fetcher import DataFetcher
from trading_bot import TradingBot
from config import NSE_STOCKS

class EnhancedPortfolioManager(PortfolioManager):
    def __init__(self, initial_capital: float = 100000):
        super().__init__(initial_capital)
        self.prediction_cache = {}
        self.price_targets = {}
        
    def get_price_predictions(self, symbol: str, data: pd.DataFrame, model) -> Dict:
        """Get detailed price predictions with buy/sell/hold recommendations"""
        try:
            # Get model prediction
            prediction = model.predict(data)
            
            if not prediction:
                return {}
            
            current_price = prediction.get('current_price', 0)
            predicted_price = prediction.get('predicted_price', 0)
            confidence = prediction.get('signal_strength', 0)
            price_change_percent = prediction.get('price_change_percent', 0)
            
            # Calculate detailed recommendations
            recommendation = self._calculate_detailed_recommendation(
                current_price, predicted_price, price_change_percent, confidence
            )
            
            return {
                'symbol': symbol,
                'current_price': current_price,
                'predicted_price': predicted_price,
                'price_change_percent': price_change_percent,
                'confidence': confidence,
                'recommendation': recommendation,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            st.error(f"Error getting predictions for {symbol}: {str(e)}")
            return {}
    
    def _calculate_detailed_recommendation(self, current_price: float, predicted_price: float, 
                                         change_percent: float, confidence: float) -> Dict:
        """Calculate detailed buy/sell/hold recommendation with price targets"""
        
        # Base recommendation logic
        if change_percent > 3 and confidence > 0.6:
            action = "STRONG BUY"
            color = "#00C851"  # Green
        elif change_percent > 1 and confidence > 0.4:
            action = "BUY"
            color = "#4CAF50"  # Light green
        elif change_percent < -3 and confidence > 0.6:
            action = "STRONG SELL"
            color = "#FF4444"  # Red
        elif change_percent < -1 and confidence > 0.4:
            action = "SELL"
            color = "#FF8A80"  # Light red
        else:
            action = "HOLD"
            color = "#FFB74D"  # Orange
        
        # Calculate price targets
        stop_loss = current_price * 0.95  # 5% stop loss
        take_profit = current_price * 1.15  # 15% take profit
        
        # Adjust targets based on prediction
        if predicted_price > current_price:
            take_profit = max(take_profit, predicted_price * 1.05)
        else:
            stop_loss = min(stop_loss, predicted_price * 0.95)
        
        return {
            'action': action,
            'color': color,
            'confidence_level': confidence,
            'price_target': predicted_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'expected_return': change_percent,
            'risk_level': self._calculate_risk_level(change_percent, confidence)
        }
    
    def _calculate_risk_level(self, change_percent: float, confidence: float) -> str:
        """Calculate risk level based on prediction volatility and confidence"""
        volatility = abs(change_percent)
        
        if volatility > 5 or confidence < 0.3:
            return "HIGH"
        elif volatility > 2 or confidence < 0.5:
            return "MEDIUM"
        else:
            return "LOW"
    
    def execute_trade_with_predictions(self, symbol: str, action: str, quantity: int, 
                                     current_price: float, prediction: Dict) -> bool:
        """Execute trade with prediction-based reasoning"""
        
        if action.upper() == "BUY":
            return self._execute_buy_with_prediction(symbol, quantity, current_price, prediction)
        elif action.upper() == "SELL":
            return self._execute_sell_with_prediction(symbol, quantity, current_price, prediction)
        
        return False
    
    def _execute_buy_with_prediction(self, symbol: str, quantity: int, 
                                   current_price: float, prediction: Dict) -> bool:
        """Execute buy order with prediction reasoning"""
        
        total_cost = quantity * current_price * (1 + COMMISSION_RATE)
        
        if total_cost > self.available_cash:
            st.warning(f"Insufficient funds for {symbol}. Need ₹{total_cost:.2f}, have ₹{self.available_cash:.2f}")
            return False
        
        # Execute the trade
        success = self.buy_stock(symbol, quantity, current_price, 
                               reason=f"AI Prediction: {prediction.get('action', 'BUY')} - "
                                     f"Target: ₹{prediction.get('price_target', 0):.2f}")
        
        if success:
            # Set price targets
            self.price_targets[symbol] = {
                'entry_price': current_price,
                'target_price': prediction.get('price_target', current_price * 1.1),
                'stop_loss': prediction.get('stop_loss', current_price * 0.95),
                'take_profit': prediction.get('take_profit', current_price * 1.15),
                'timestamp': datetime.now()
            }
            
            st.success(f"✅ Bought {quantity} shares of {symbol} at ₹{current_price:.2f}")
            
        return success
    
    def _execute_sell_with_prediction(self, symbol: str, quantity: int, 
                                    current_price: float, prediction: Dict) -> bool:
        """Execute sell order with prediction reasoning"""
        
        if symbol not in self.positions or self.positions[symbol]['quantity'] < quantity:
            st.warning(f"Insufficient shares of {symbol} to sell")
            return False
        
        # Execute the trade
        success = self.sell_stock(symbol, quantity, current_price,
                                reason=f"AI Prediction: {prediction.get('action', 'SELL')} - "
                                      f"Expected decline to ₹{prediction.get('price_target', 0):.2f}")
        
        if success:
            st.success(f"✅ Sold {quantity} shares of {symbol} at ₹{current_price:.2f}")
            
            # Remove price targets if position closed
            if symbol in self.positions and self.positions[symbol]['quantity'] == 0:
                self.price_targets.pop(symbol, None)
        
        return success
    
    def get_portfolio_recommendations(self, bot: TradingBot) -> List[Dict]:
        """Get AI recommendations for entire portfolio"""
        recommendations = []
        
        for symbol in NSE_STOCKS[:20]:  # Top 20 stocks
            try:
                # Get latest data
                stock_data = bot.data_fetcher.fetch_real_time_data(symbol)
                if not stock_data:
                    continue
                
                # Get historical data for prediction
                historical_data = bot.data_fetcher.fetch_historical_data(symbol, period="1y")
                if historical_data.empty:
                    continue
                
                # Add technical indicators
                historical_data = bot.technical_indicators.add_all_indicators(historical_data)
                
                # Get prediction from model
                if symbol in bot.models:
                    prediction = self.get_price_predictions(symbol, historical_data, bot.models[symbol])
                    
                    if prediction:
                        # Add current position info
                        position_info = self.positions.get(symbol, {'quantity': 0, 'avg_price': 0})
                        prediction.update({
                            'current_quantity': position_info['quantity'],
                            'avg_price': position_info.get('avg_price', 0),
                            'current_value': position_info['quantity'] * prediction['current_price'],
                            'unrealized_pnl': (prediction['current_price'] - position_info.get('avg_price', 0)) * position_info['quantity']
                        })
                        
                        recommendations.append(prediction)
                        
            except Exception as e:
                st.warning(f"Error getting recommendation for {symbol}: {str(e)}")
                continue
        
        # Sort by confidence and expected return
        recommendations.sort(key=lambda x: x['confidence'] * abs(x['price_change_percent']), reverse=True)
        
        return recommendations

def create_enhanced_portfolio_interface():
    """Create the enhanced portfolio management interface"""
    
    st.header("🎯 AI-Powered Portfolio Management")
    
    # Initialize enhanced portfolio manager
    if 'enhanced_portfolio' not in st.session_state:
        st.session_state.enhanced_portfolio = EnhancedPortfolioManager()
    
    portfolio = st.session_state.enhanced_portfolio
    
    # Portfolio Summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("💰 Total Portfolio Value", f"₹{portfolio.total_value:,.2f}", 
                 delta=f"₹{portfolio.total_value - portfolio.initial_capital:,.2f}")
    
    with col2:
        st.metric("💵 Available Cash", f"₹{portfolio.available_cash:,.2f}")
    
    with col3:
        invested_value = portfolio.total_value - portfolio.available_cash
        st.metric("📈 Invested Value", f"₹{invested_value:,.2f}")
    
    with col4:
        return_pct = ((portfolio.total_value - portfolio.initial_capital) / portfolio.initial_capital) * 100
        st.metric("📊 Total Return", f"{return_pct:.2f}%")
    
    # Tabs for different portfolio functions
    tab1, tab2, tab3, tab4 = st.tabs(["📊 AI Recommendations", "💼 Manual Trading", "🎯 Price Targets", "🧠 Model Training"])
    
    with tab1:
        create_ai_recommendations_tab(portfolio)
    
    with tab2:
        create_manual_trading_tab(portfolio)
    
    with tab3:
        create_price_targets_tab(portfolio)
    
    with tab4:
        create_model_training_tab()

def create_ai_recommendations_tab(portfolio):
    """Create AI recommendations interface"""
    
    st.subheader("🤖 AI Stock Recommendations")
    
    if st.button("🔄 Get Fresh AI Recommendations", type="primary"):
        with st.spinner("Getting AI recommendations..."):
            try:
                # Get bot instance
                if 'bot' in st.session_state and st.session_state.bot:
                    recommendations = portfolio.get_portfolio_recommendations(st.session_state.bot)
                    st.session_state.recommendations = recommendations
                else:
                    st.error("Please initialize the trading bot first!")
                    return
            except Exception as e:
                st.error(f"Error getting recommendations: {str(e)}")
                return
    
    # Display recommendations
    if 'recommendations' in st.session_state and st.session_state.recommendations:
        recommendations = st.session_state.recommendations
        
        st.write(f"**Found {len(recommendations)} AI recommendations:**")
        
        for i, rec in enumerate(recommendations[:10]):  # Show top 10
            with st.container():
                col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 2, 2])
                
                with col1:
                    st.write(f"**{rec['symbol']}**")
                    st.write(f"₹{rec['current_price']:.2f}")
                
                with col2:
                    action_color = rec['recommendation']['color']
                    st.markdown(f"""
                    <div style="background-color: {action_color}; color: white; padding: 8px; border-radius: 4px; text-align: center;">
                        <strong>{rec['recommendation']['action']}</strong>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.write(f"**Target:** ₹{rec['recommendation']['price_target']:.2f}")
                    st.write(f"**Expected:** {rec['price_change_percent']:.1f}%")
                
                with col4:
                    st.write(f"**Confidence:** {rec['confidence']:.1%}")
                    st.write(f"**Risk:** {rec['recommendation']['risk_level']}")
                
                with col5:
                    if rec['recommendation']['action'] in ['BUY', 'STRONG BUY']:
                        if st.button(f"Buy {rec['symbol']}", key=f"buy_{i}"):
                            st.session_state[f'manual_trade_symbol'] = rec['symbol']
                            st.session_state[f'manual_trade_action'] = 'BUY'
                            st.session_state[f'suggested_price'] = rec['current_price']
                    
                    elif rec['recommendation']['action'] in ['SELL', 'STRONG SELL']:
                        if rec['current_quantity'] > 0:
                            if st.button(f"Sell {rec['symbol']}", key=f"sell_{i}"):
                                st.session_state[f'manual_trade_symbol'] = rec['symbol']
                                st.session_state[f'manual_trade_action'] = 'SELL'
                                st.session_state[f'suggested_price'] = rec['current_price']
                
                st.divider()

def create_manual_trading_tab(portfolio):
    """Create manual trading interface"""
    
    st.subheader("📱 Manual Stock Trading")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Buy Stocks**")
        
        buy_symbol = st.selectbox("Select Stock to Buy", 
                                 options=NSE_STOCKS[:20], 
                                 key="buy_symbol",
                                 index=NSE_STOCKS.index(st.session_state.get('manual_trade_symbol', NSE_STOCKS[0])) 
                                 if st.session_state.get('manual_trade_symbol') in NSE_STOCKS else 0)
        
        # Get current price
        data_fetcher = DataFetcher()
        current_data = data_fetcher.fetch_real_time_data(buy_symbol)
        current_price = current_data.get('current_price', 0) if current_data else 0
        
        if current_price > 0:
            st.write(f"Current Price: ₹{current_price:.2f}")
            
            buy_quantity = st.number_input("Quantity to Buy", min_value=1, value=10, key="buy_quantity")
            total_cost = buy_quantity * current_price * (1 + COMMISSION_RATE)
            
            st.write(f"Total Cost: ₹{total_cost:.2f} (including commission)")
            st.write(f"Available Cash: ₹{portfolio.available_cash:.2f}")
            
            if st.button("💰 Execute Buy Order", type="primary"):
                if total_cost <= portfolio.available_cash:
                    success = portfolio.buy_stock(buy_symbol, buy_quantity, current_price, 
                                                reason="Manual buy order")
                    if success:
                        st.success(f"✅ Successfully bought {buy_quantity} shares of {buy_symbol}")
                        st.rerun()
                    else:
                        st.error("❌ Failed to execute buy order")
                else:
                    st.error("❌ Insufficient funds")
    
    with col2:
        st.write("**Sell Stocks**")
        
        # Show current positions
        if portfolio.positions:
            sell_symbol = st.selectbox("Select Stock to Sell", 
                                     options=list(portfolio.positions.keys()),
                                     key="sell_symbol")
            
            if sell_symbol in portfolio.positions:
                position = portfolio.positions[sell_symbol]
                st.write(f"Current Holdings: {position['quantity']} shares")
                st.write(f"Average Price: ₹{position['avg_price']:.2f}")
                st.write(f"Current Price: ₹{current_price:.2f}")
                
                unrealized_pnl = (current_price - position['avg_price']) * position['quantity']
                pnl_color = "green" if unrealized_pnl >= 0 else "red"
                st.markdown(f"**Unrealized P&L:** <span style='color:{pnl_color}'>₹{unrealized_pnl:.2f}</span>", 
                           unsafe_allow_html=True)
                
                sell_quantity = st.number_input("Quantity to Sell", 
                                              min_value=1, 
                                              max_value=position['quantity'],
                                              value=min(10, position['quantity']),
                                              key="sell_quantity")
                
                total_proceeds = sell_quantity * current_price * (1 - COMMISSION_RATE)
                st.write(f"Total Proceeds: ₹{total_proceeds:.2f} (after commission)")
                
                if st.button("💸 Execute Sell Order", type="primary"):
                    success = portfolio.sell_stock(sell_symbol, sell_quantity, current_price,
                                                 reason="Manual sell order")
                    if success:
                        st.success(f"✅ Successfully sold {sell_quantity} shares of {sell_symbol}")
                        st.rerun()
                    else:
                        st.error("❌ Failed to execute sell order")
        else:
            st.info("No positions to sell")

def create_price_targets_tab(portfolio):
    """Create price targets monitoring interface"""
    
    st.subheader("🎯 Price Targets & Stop Loss")
    
    if portfolio.price_targets:
        for symbol, targets in portfolio.price_targets.items():
            with st.container():
                col1, col2, col3, col4 = st.columns(4)
                
                # Get current price
                data_fetcher = DataFetcher()
                current_data = data_fetcher.fetch_real_time_data(symbol)
                current_price = current_data.get('current_price', 0) if current_data else 0
                
                with col1:
                    st.write(f"**{symbol}**")
                    st.write(f"Entry: ₹{targets['entry_price']:.2f}")
                    st.write(f"Current: ₹{current_price:.2f}")
                
                with col2:
                    target_diff = ((current_price - targets['target_price']) / targets['target_price']) * 100
                    target_color = "green" if current_price >= targets['target_price'] else "orange"
                    st.markdown(f"**Target:** ₹{targets['target_price']:.2f}")
                    st.markdown(f"<span style='color:{target_color}'>({target_diff:+.1f}%)</span>", 
                               unsafe_allow_html=True)
                
                with col3:
                    sl_diff = ((current_price - targets['stop_loss']) / targets['stop_loss']) * 100
                    sl_color = "red" if current_price <= targets['stop_loss'] else "green"
                    st.markdown(f"**Stop Loss:** ₹{targets['stop_loss']:.2f}")
                    st.markdown(f"<span style='color:{sl_color}'>({sl_diff:+.1f}%)</span>", 
                               unsafe_allow_html=True)
                
                with col4:
                    current_pnl = current_price - targets['entry_price']
                    pnl_pct = (current_pnl / targets['entry_price']) * 100
                    pnl_color = "green" if current_pnl >= 0 else "red"
                    st.markdown(f"**P&L:** <span style='color:{pnl_color}'>₹{current_pnl:.2f}</span>", 
                               unsafe_allow_html=True)
                    st.markdown(f"<span style='color:{pnl_color}'>({pnl_pct:+.1f}%)</span>", 
                               unsafe_allow_html=True)
                
                st.divider()
    else:
        st.info("No active price targets. Set targets by making trades through AI recommendations.")

def create_model_training_tab():
    """Create AI model training interface"""
    
    st.subheader("🧠 AI Model Training Center")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Individual Stock Model Training**")
        
        train_symbol = st.selectbox("Select Stock for Training", NSE_STOCKS[:20], key="train_symbol")
        
        training_period = st.selectbox("Training Data Period", 
                                     ["6mo", "1y", "2y", "5y"], 
                                     index=2, key="training_period")
        
        if st.button("🚀 Train Model", type="primary"):
            with st.spinner(f"Training AI model for {train_symbol}..."):
                try:
                    # Get bot instance
                    if 'bot' in st.session_state and st.session_state.bot:
                        bot = st.session_state.bot
                        
                        # Fetch historical data
                        historical_data = bot.data_fetcher.fetch_historical_data(train_symbol, period=training_period)
                        
                        if not historical_data.empty:
                            # Add technical indicators
                            historical_data = bot.technical_indicators.add_all_indicators(historical_data)
                            
                            # Train model
                            if train_symbol not in bot.models:
                                from ml_model_sklearn import SimpleMLPredictor
                                bot.models[train_symbol] = SimpleMLPredictor(train_symbol)
                            
                            metrics = bot.models[train_symbol].train_model(historical_data)
                            
                            st.success(f"✅ Model trained successfully for {train_symbol}!")
                            
                            # Display training metrics
                            col_a, col_b, col_c = st.columns(3)
                            with col_a:
                                st.metric("Training RMSE", f"{metrics.get('train_rmse', 0):.4f}")
                            with col_b:
                                st.metric("Validation RMSE", f"{metrics.get('val_rmse', 0):.4f}")
                            with col_c:
                                st.metric("R² Score", f"{metrics.get('val_r2', 0):.4f}")
                        else:
                            st.error(f"❌ No data available for {train_symbol}")
                    else:
                        st.error("Please initialize the trading bot first!")
                        
                except Exception as e:
                    st.error(f"❌ Training failed: {str(e)}")
    
    with col2:
        st.write("**Batch Model Training**")
        
        selected_stocks = st.multiselect("Select Multiple Stocks", 
                                       NSE_STOCKS[:20], 
                                       default=NSE_STOCKS[:5],
                                       key="batch_stocks")
        
        if st.button("🚀 Train All Selected Models", type="primary"):
            if selected_stocks:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, symbol in enumerate(selected_stocks):
                    status_text.text(f"Training model for {symbol}...")
                    
                    try:
                        if 'bot' in st.session_state and st.session_state.bot:
                            bot = st.session_state.bot
                            
                            # Fetch and train
                            historical_data = bot.data_fetcher.fetch_historical_data(symbol, period="2y")
                            
                            if not historical_data.empty:
                                historical_data = bot.technical_indicators.add_all_indicators(historical_data)
                                
                                if symbol not in bot.models:
                                    from ml_model_sklearn import SimpleMLPredictor
                                    bot.models[symbol] = SimpleMLPredictor(symbol)
                                
                                bot.models[symbol].train_model(historical_data)
                                st.write(f"✅ {symbol} model trained")
                            else:
                                st.write(f"⚠️ {symbol} - no data available")
                        
                    except Exception as e:
                        st.write(f"❌ {symbol} - training failed: {str(e)}")
                    
                    progress_bar.progress((i + 1) / len(selected_stocks))
                
                status_text.text("✅ Batch training completed!")
                st.success("🎉 All models trained successfully!")
            else:
                st.warning("Please select at least one stock for training")
        
        # Model status
        st.write("**Current Model Status**")
        if 'bot' in st.session_state and st.session_state.bot and hasattr(st.session_state.bot, 'models'):
            models = st.session_state.bot.models
            if models:
                model_data = []
                for symbol, model in models.items():
                    model_info = model.get_model_info() if hasattr(model, 'get_model_info') else {}
                    model_data.append({
                        'Symbol': symbol,
                        'Type': model_info.get('model_type', 'Unknown'),
                        'Status': '✅ Trained' if model else '❌ Not Trained'
                    })
                
                st.dataframe(pd.DataFrame(model_data), use_container_width=True)
            else:
                st.info("No models trained yet")
        else:
            st.info("Initialize trading bot to see model status")