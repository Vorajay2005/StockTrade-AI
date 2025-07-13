#!/bin/bash
# Quick Start Script for AI Stock Trading Bot

echo "🤖 AI Stock Trading Bot - Quick Start"
echo "===================================="

# Check if python3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Please install Python 3.8+ from https://python.org"
    exit 1
fi

echo "✅ Python3 found: $(python3 --version)"

# Install requirements
echo "📦 Installing requirements..."
python3 -m pip install -r requirements.txt

# Create directories
echo "📁 Creating directories..."
mkdir -p models scalers reports charts

# Launch dashboard
echo "🚀 Launching Dashboard..."
echo "📊 Opening at http://localhost:8501"
python3 -m streamlit run dashboard.py