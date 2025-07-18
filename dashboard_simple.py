"""
Simplified Streamlit Dashboard for debugging
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import time

# Page config
st.set_page_config(
    page_title="AI Stock Trading Bot Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("🤖 AI Stock Trading Bot Dashboard")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("🎛️ Control Panel")
    st.sidebar.success("✅ Dashboard loaded successfully!")
    
    # Test the data fetcher import
    try:
        from data_fetcher import DataFetcher
        fetcher = DataFetcher()
        status = fetcher.get_market_status()
        
        st.sidebar.markdown(f"**Market Status:** {'🟢 OPEN' if status['is_open'] else '🔴 CLOSED'}")
        st.sidebar.markdown(f"**Current Time:** {status['current_time']}")
        
        if not status['is_open'] and status.get('next_open'):
            st.sidebar.markdown(f"**Next Open:** {status['next_open']}")
            
    except Exception as e:
        st.sidebar.error(f"Error loading market status: {str(e)}")
    
    # Main content
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
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📈 Portfolio", "📊 Charts", "🎯 Signals"])
    
    with tab1:
        st.header("📈 Portfolio Management")
        st.info("✅ Simplified dashboard is working!")
        
        # Sample data
        sample_data = pd.DataFrame({
            'Symbol': ['RELIANCE.NS', 'TCS.NS', 'INFY.NS'],
            'Quantity': [10, 5, 15],
            'Avg Price': [2450, 3890, 1650],
            'Current Price': [2520, 3920, 1680],
            'P&L': [700, 150, 450]
        })
        st.dataframe(sample_data, use_container_width=True)
    
    with tab2:
        st.header("📊 Live Charts")
        # Sample chart
        chart_data = pd.DataFrame(
            np.random.randn(20, 3),
            columns=['RELIANCE.NS', 'TCS.NS', 'INFY.NS']
        )
        st.line_chart(chart_data)
    
    with tab3:
        st.header("🎯 Trading Signals")
        st.info("AI signals will appear here when the bot is initialized.")

if __name__ == "__main__":
    main()