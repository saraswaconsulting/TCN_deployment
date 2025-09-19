#!/usr/bin/env python3
"""
Setup verification script for ISL Streamlit deployment
Run this to verify all dependencies and components work correctly
"""

import sys
import os

def check_python_version():
    """Check if Python version is compatible"""
    print("🔍 Checking Python version...")
    
    version = sys.version_info
    print(f"📍 Current Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor == 10:
        print("✅ Python 3.10 - Perfect for Streamlit Cloud!")
        return True
    elif version.major == 3 and version.minor >= 8:
        print("⚠️  Python 3.8+ detected - Should work, but 3.10 is recommended")
        return True
    else:
        print("❌ Python 3.8+ required. Please upgrade your Python version.")
        return False

def check_dependencies():
    """Check if all required packages are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'streamlit', 'torch', 'cv2', 'mediapipe', 
        'google.generativeai', 'numpy'
    ]
    
    missing = []
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            else:
                __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing.append(package)
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("\n✅ All dependencies installed!")
    return True

def check_model():
    """Check if model file exists and is valid"""
    print("\n🔍 Checking model file...")
    
    model_path = 'checkpoints/best_gru.pt'
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return False
    
    file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
    print(f"✅ Model file found: {file_size:.1f} MB")
    
    # Try loading the model
    try:
        import torch
        from common import GRUClassifier
        
        ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
        
        if 'model' not in ckpt or 'class_to_idx' not in ckpt:
            print("❌ Invalid model file format")
            return False
        
        num_classes = len(ckpt['class_to_idx'])
        print(f"✅ Model loaded successfully: {num_classes} classes")
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def check_streamlit():
    """Check if Streamlit can import the main app"""
    print("\n🔍 Checking Streamlit app...")
    
    try:
        # Check if main script exists
        if not os.path.exists('streamlit_gemini_demo.py'):
            print("❌ Main app file not found: streamlit_gemini_demo.py")
            return False
        
        # Try importing main components
        sys.path.insert(0, '.')
        from streamlit_gemini_demo import get_predictor
        
        print("✅ Streamlit app can be imported")
        return True
        
    except Exception as e:
        print(f"❌ Error importing Streamlit app: {e}")
        return False

def main():
    """Run all verification checks"""
    print("🚀 ISL Streamlit Deployment Verification\n")
    
    checks = [
        check_python_version,
        check_dependencies,
        check_model, 
        check_streamlit
    ]
    
    results = []
    for check in checks:
        results.append(check())
    
    print("\n" + "="*50)
    
    if all(results):
        print("🎉 All checks passed! Ready for deployment!")
        print("\nNext steps:")
        print("1. Test locally: streamlit run streamlit_gemini_demo.py")
        print("2. Push to GitHub")
        print("3. Deploy on Streamlit Cloud")
        return 0
    else:
        print("❌ Some checks failed. Please fix the issues above.")
        return 1

if __name__ == "__main__":
    exit(main())