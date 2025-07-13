#!/usr/bin/env python3
"""
Launch script for the Trading Bot Dashboard
"""

import streamlit as streamlit_main
import sys
import os
from pathlib import Path

# Add the current directory to the Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def main():
    """Main function to run the Streamlit dashboard"""
    
    # Set the main script path
    dashboard_path = current_dir / "dashboard.py"
    
    # Run Streamlit
    sys.argv = [
        "streamlit",
        "run",
        str(dashboard_path),
        "--server.port=8501",
        "--server.address=localhost",
        "--browser.gatherUsageStats=false"
    ]
    
    streamlit_main.main()

if __name__ == "__main__":
    main()