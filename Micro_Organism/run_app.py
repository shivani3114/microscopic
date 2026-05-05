#!/usr/bin/env python3
"""
Simple script to test and run the Flask application
"""

import os
import sys

def test_imports():
    """Test if all required modules can be imported"""
    try:
        from flask import Flask
        print("✓ Flask imported successfully")
    except ImportError as e:
        print(f"✗ Flask import failed: {e}")
        return False

    try:
        from PIL import Image
        print("✓ Pillow imported successfully")
    except ImportError as e:
        print(f"✗ Pillow import failed: {e}")
        return False

    try:
        import numpy as np
        print("✓ NumPy imported successfully")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False

    try:
        import joblib
        print("✓ Joblib imported successfully")
    except ImportError as e:
        print(f"✗ Joblib import failed: {e}")
        return False

    return True

def check_files():
    """Check if required files exist"""
    required_files = [
        'templates/index.html',
        'outputs/microscopic_classifier.joblib'
    ]

    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path} exists")
        else:
            print(f"✗ {file_path} missing")

def main():
    print("Testing Morphological Species Identifier setup...\n")

    # Test imports
    print("Testing imports:")
    if not test_imports():
        print("\n❌ Import tests failed. Please install missing packages:")
        print("pip install -r requirements.txt")
        return

    print("\nChecking required files:")
    check_files()

    print("\nStarting Flask application...")
    try:
        from app import app
        print("✓ Flask app imported successfully")
        print("\n🚀 Starting server on http://localhost:5000")
        print("Press Ctrl+C to stop the server")
        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        print(f"❌ Failed to start Flask app: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()