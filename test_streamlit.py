import streamlit as st

st.title("🧪 Streamlit Test")
st.write("If you can see this, Streamlit is working!")
st.success("✅ Basic Streamlit functionality is working")

# Test some basic components
col1, col2 = st.columns(2)
with col1:
    st.metric("Test Metric", "100", "10")
with col2:
    st.write("Test column")

st.info("This is a test to verify Streamlit is working properly.")