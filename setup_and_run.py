#!/usr/bin/env python3
"""
Setup script that creates a virtual environment and runs the trading bot
Handles macOS externally-managed-environment issues
"""

import os
import sys
import subprocess
import venv
from pathlib import Path

def create_virtual_environment():
    """Create a virtual environment for the project"""
    venv_path = Path("venv")
    
    if venv_path.exists():
        print("✅ Virtual environment already exists")
        return venv_path
    
    print("🔧 Creating virtual environment...")
    try:
        venv.create(venv_path, with_pip=True)
        print("✅ Virtual environment created successfully")
        return venv_path
    except Exception as e:
        print(f"❌ Failed to create virtual environment: {e}")
        return None

def get_venv_python(venv_path):
    """Get path to Python executable in virtual environment"""
    if sys.platform == "win32":
        return venv_path / "Scripts" / "python.exe"
    else:
        return venv_path / "bin" / "python"

def install_requirements(venv_python):
    """Install requirements in virtual environment"""
    print("📦 Installing requirements in virtual environment...")
    try:
        subprocess.check_call([str(venv_python), "-m", "pip", "install", "--upgrade", "pip"])
        subprocess.check_call([str(venv_python), "-m", "pip", "install", "-r", "requirements_compatible.txt"])
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    directories = ['models', 'scalers', 'reports', 'charts']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 Directory ready: {directory}")

def test_installation(venv_python):
    """Test if packages are properly installed"""
    print("🔍 Testing installation...")
    test_code = """
try:
    import yfinance
    import pandas
    import numpy
    import streamlit
    import sklearn
    print("✅ All packages imported successfully!")
except ImportError as e:
    print(f"❌ Import error: {e}")
    exit(1)
"""
    
    try:
        result = subprocess.run([str(venv_python), "-c", test_code], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(result.stdout.strip())
            return True
        else:
            print(result.stderr.strip())
            return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def launch_dashboard(venv_python):
    """Launch the Streamlit dashboard"""
    print("🚀 Launching AI Stock Trading Bot Dashboard...")
    print("📊 Dashboard will open at: http://localhost:8501")
    print("🔄 Press Ctrl+C to stop the dashboard")
    
    try:
        subprocess.run([str(venv_python), "-m", "streamlit", "run", "dashboard.py", 
                       "--server.port=8501", "--server.address=localhost"])
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")

def main():
    """Main setup and run function"""
    print("🤖 AI Stock Trading Bot - Setup & Launch")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("dashboard.py").exists():
        print("❌ Please run this script from the Stock-Trade_AI directory")
        return
    
    # Create virtual environment
    venv_path = create_virtual_environment()
    if not venv_path:
        return
    
    # Get Python executable
    venv_python = get_venv_python(venv_path)
    if not venv_python.exists():
        print("❌ Virtual environment Python not found")
        return
    
    # Install requirements
    if not install_requirements(venv_python):
        return
    
    # Create directories
    create_directories()
    
    # Test installation
    if not test_installation(venv_python):
        print("❌ Installation test failed")
        return
    
    print("\n✅ Setup completed successfully!")
    
    # Ask user what to do
    print("\n🚀 What would you like to do?")
    print("1. 📊 Launch Dashboard")
    print("2. 🔗 Test Data Connection")
    print("3. 📖 Show Virtual Environment Info")
    print("4. 🚪 Exit")
    
    while True:
        try:
            choice = input("\nChoose option (1-4): ").strip()
            
            if choice == '1' or choice == '':
                launch_dashboard(venv_python)
                break
            elif choice == '2':
                test_data_connection(venv_python)
            elif choice == '3':
                show_venv_info(venv_path, venv_python)
            elif choice == '4':
                print("👋 Goodbye!")
                break
            else:
                print("❌ Please enter 1, 2, 3, or 4")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break

def test_data_connection(venv_python):
    """Test data connection"""
    print("🔗 Testing data connection...")
    test_code = """
from data_fetcher import DataFetcher
fetcher = DataFetcher()
data = fetcher.fetch_real_time_data('RELIANCE.NS')
if data and 'current_price' in data:
    print(f"✅ Data connection successful!")
    print(f"📈 RELIANCE.NS: ₹{data['current_price']:.2f}")
else:
    print("❌ No data received")
"""
    
    try:
        subprocess.run([str(venv_python), "-c", test_code])
    except Exception as e:
        print(f"❌ Test failed: {e}")

def show_venv_info(venv_path, venv_python):
    """Show virtual environment information"""
    print(f"\n📋 Virtual Environment Info")
    print(f"Path: {venv_path.absolute()}")
    print(f"Python: {venv_python}")
    print(f"\n💡 To manually activate:")
    if sys.platform == "win32":
        print(f"   {venv_path}\\Scripts\\activate")
    else:
        print(f"   source {venv_path}/bin/activate")
    
    print(f"\n💡 To run dashboard manually:")
    print(f"   {venv_python} -m streamlit run dashboard.py")

if __name__ == "__main__":
    main()