#!/usr/bin/env python3
"""
Simple launcher for the Stock Trading Bot
"""

import os
import sys
import subprocess
from pathlib import Path

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'yfinance', 'pandas', 'numpy', 'tensorflow', 
        'streamlit', 'plotly', 'scikit-learn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing required packages: {', '.join(missing_packages)}")
        print("📥 Installing missing packages...")
        
        for package in missing_packages:
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
                print(f"✅ Installed {package}")
            except subprocess.CalledProcessError:
                print(f"❌ Failed to install {package}")
                return False
    
    print("✅ All dependencies satisfied!")
    return True

def setup_directories():
    """Create necessary directories"""
    directories = ['models', 'scalers', 'reports', 'charts']
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 Created directory: {directory}")

def main():
    """Main launcher function"""
    print("🤖 AI Stock Trading Bot Launcher")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        print("❌ Dependency check failed. Please install required packages manually.")
        return
    
    # Setup directories
    setup_directories()
    
    # Show menu
    print("\n🚀 Choose an option:")
    print("1. Launch Dashboard (Recommended)")
    print("2. Run Command Line Bot")
    print("3. Test Data Connection")
    print("4. Train Models Only")
    print("5. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-5): ").strip()
            
            if choice == '1':
                print("🌐 Launching Streamlit Dashboard...")
                print("📊 Dashboard will open in your browser at http://localhost:8501")
                os.system(f"{sys.executable} run_dashboard.py")
                break
                
            elif choice == '2':
                print("💻 Running Command Line Bot...")
                os.system(f"{sys.executable} trading_bot.py")
                break
                
            elif choice == '3':
                print("🔗 Testing Data Connection...")
                os.system(f"{sys.executable} -c \"from data_fetcher import DataFetcher; df = DataFetcher(); print('Testing RELIANCE.NS...'); data = df.fetch_real_time_data('RELIANCE.NS'); print('Success!' if data else 'Failed!')\"")
                
            elif choice == '4':
                print("🧠 Training Models...")
                print("This may take several minutes...")
                os.system(f"{sys.executable} -c \"from trading_bot import TradingBot; bot = TradingBot(['RELIANCE.NS', 'TCS.NS']); bot.initialize_models(train_models=True)\"")
                
            elif choice == '5':
                print("👋 Goodbye!")
                break
                
            else:
                print("❌ Invalid choice. Please enter 1-5.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()