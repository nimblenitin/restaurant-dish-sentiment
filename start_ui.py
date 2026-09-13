#!/usr/bin/env python3
"""
Quick Start Script for Restaurant Food Recommendation System
Launches the Streamlit UI
"""

import subprocess
import sys
import os


def check_streamlit():
    """Check if Streamlit is installed"""
    try:
        import streamlit
        return True
    except ImportError:
        return False


def install_streamlit():
    """Install Streamlit"""
    print("Installing Streamlit...")
    subprocess.run([sys.executable, "-m", "pip", "install", "streamlit"], check=True)
    print("Streamlit installed successfully!")


def main():
    print("\n" + "="*60)
    print("🍽️  RESTAURANT FOOD RECOMMENDATION SYSTEM")
    print("="*60)
    
    # Check if Streamlit is installed
    if not check_streamlit():
        print("\nStreamlit is not installed. Installing...")
        install_streamlit()
    
    # Check if model exists
    model_path = "models/mt5-finetuned-restaurant-food"
    if not os.path.exists(model_path):
        print("\n⚠️  Model not found!")
        print("Please train the model first:")
        print("  python run.py")
        print("\nOr use sample data:")
        print("  python src/scraper.py --sample")
        print("  python src/label_generator.py")
        print("  python src/train.py --epochs 5")
        return
    
    # Launch Streamlit
    print("\n🚀 Launching Streamlit UI...")
    print("="*60)
    print("Open your browser and go to: http://localhost:8501")
    print("="*60)
    print("\nPress Ctrl+C to stop the server\n")
    
    # Run Streamlit
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", "app.py",
        "--server.headless", "true",
        "--server.port", "8501"
    ])


if __name__ == "__main__":
    main()
