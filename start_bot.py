#!/usr/bin/env python3
"""
Simple and reliable launcher for the Stock Trading Bot
Compatible with macOS and uses python3 directly
"""

import os
import sys
import subprocess
import time

def check_python():
    """Check Python installation"""
    try:
        result = subprocess.run(['python3', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Python found: {result.stdout.strip()}")
            return True
        else:
            print("❌ Python3 not found")
            return False
    except FileNotFoundError:
        print("❌ Python3 not found in PATH")
        return False

def install_requirements():
    """Install required packages"""
    print("📦 Installing requirements...")
    try:
        subprocess.check_call(['python3', '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        print("💡 Try running: python3 -m pip install --upgrade pip")
        return False

def create_directories():
    """Create necessary directories"""
    directories = ['models', 'scalers', 'reports', 'charts']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 Directory ready: {directory}")

def test_imports():
    """Test if key packages can be imported"""
    print("🔍 Testing package imports...")
    packages = ['yfinance', 'pandas', 'numpy', 'streamlit', 'tensorflow']
    
    for package in packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - not installed")
            return False
    return True

def launch_dashboard():
    """Launch the Streamlit dashboard"""
    print("🚀 Launching AI Stock Trading Bot Dashboard...")
    print("📊 Dashboard will open at: http://localhost:8501")
    print("⏱️  Please wait for the dashboard to load...")
    
    try:
        # Use subprocess to run streamlit
        subprocess.run(['python3', '-m', 'streamlit', 'run', 'dashboard.py', 
                       '--server.port=8501', '--server.address=localhost'])
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")
        print("\n💡 Alternative: Try running manually:")
        print("   python3 -m streamlit run dashboard.py")

def quick_test():
    """Quick test of data connection"""
    print("🔗 Testing data connection...")
    try:
        from data_fetcher import DataFetcher
        fetcher = DataFetcher()
        data = fetcher.fetch_real_time_data('RELIANCE.NS')
        if data and 'current_price' in data:
            print(f"✅ Data connection successful!")
            print(f"📈 RELIANCE.NS: ₹{data['current_price']:.2f}")
            return True
        else:
            print("❌ No data received")
            return False
    except Exception as e:
        print(f"❌ Data connection failed: {e}")
        return False

def main():
    """Main launcher"""
    print("🤖 AI Stock Trading Bot - macOS Launcher")
    print("=" * 50)
    
    # Step 1: Check Python
    if not check_python():
        print("\n💡 Please install Python 3.8+ from https://python.org")
        return
    
    # Step 2: Create directories
    create_directories()
    
    # Step 3: Install requirements
    if not os.path.exists('requirements.txt'):
        print("❌ requirements.txt not found!")
        return
    
    print("\n📦 Checking/Installing dependencies...")
    install_choice = input("Install/update requirements? (y/n): ").lower().strip()
    if install_choice in ['y', 'yes', '']:
        if not install_requirements():
            print("❌ Installation failed. Please check your internet connection.")
            return
    
    # Step 4: Test imports
    if not test_imports():
        print("❌ Some packages are missing. Please install requirements first.")
        return
    
    # Step 5: Quick test
    print("\n🔍 Testing system...")
    if quick_test():
        print("✅ System ready!")
    else:
        print("⚠️  Data connection issues, but you can still use the dashboard")
    
    # Step 6: Launch options
    print("\n🚀 Ready to launch!")
    print("1. 📊 Launch Dashboard (Recommended)")
    print("2. 🔗 Test Data Connection Only")
    print("3. 📖 Show Help")
    print("4. 🚪 Exit")
    
    while True:
        try:
            choice = input("\nChoose option (1-4): ").strip()
            
            if choice == '1' or choice == '':
                launch_dashboard()
                break
            elif choice == '2':
                quick_test()
            elif choice == '3':
                show_help()
            elif choice == '4':
                print("👋 Goodbye!")
                break
            else:
                print("❌ Please enter 1, 2, 3, or 4")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break

def show_help():
    """Show help information"""
    print("\n📖 Help - AI Stock Trading Bot")
    print("-" * 40)
    print("🎯 Purpose: Simulate stock trading with AI predictions")
    print("💰 Money: Uses dummy money (₹1,00,000 starting capital)")
    print("📈 Market: NSE (National Stock Exchange) Indian stocks")
    print("🤖 AI: LSTM neural networks for price prediction")
    print("⏰ Hours: Follows NSE trading hours (9:15 AM - 3:30 PM)")
    print("\n🚀 Getting Started:")
    print("1. Launch dashboard (option 1)")
    print("2. Select stocks from sidebar")
    print("3. Click 'Initialize Bot'")
    print("4. Wait for models to load/train")
    print("5. Click 'Start Trading'")
    print("6. Monitor your portfolio!")
    print("\n⚠️  Disclaimer: This is for educational purposes only!")
    print("   No real money is used or at risk.")

if __name__ == "__main__":
    main()