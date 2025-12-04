#!/usr/bin/env python
"""
Diagnostic script for Streamlit Cloud deployment issues
Run this to identify import or configuration problems
"""

import sys
import traceback

def test_imports():
    """Test all required imports"""
    print("🔍 Testing imports...\n")
    
    imports_to_test = [
        ('streamlit', 'streamlit as st'),
        ('pathlib', 'Path from pathlib'),
        ('os', 'os'),
        ('io', 'io'),
        ('base64', 'base64'),
        ('pandas', 'pandas as pd'),
        ('plotly.express', 'plotly.express as px'),
        ('requests', 'requests'),
        ('PIL', 'PIL.Image'),
        ('numpy', 'numpy'),
        ('seaborn', 'seaborn'),
    ]
    
    failed = []
    
    for module_name, display_name in imports_to_test:
        try:
            __import__(module_name)
            print(f"  ✅ {display_name}")
        except ImportError as e:
            print(f"  ❌ {display_name}: {e}")
            failed.append((display_name, str(e)))
    
    if failed:
        print(f"\n⚠️ {len(failed)} imports failed:")
        for name, error in failed:
            print(f"   - {name}: {error}")
        return False
    
    print("\n✅ All imports successful\n")
    return True

def test_app_utils():
    """Test app_utils module"""
    print("🔍 Testing app_utils...\n")
    
    try:
        from app_utils import (
            is_streamlit_cloud,
            is_local_deployment,
            load_pil_image,
            list_images_in_folder,
        )
        
        print(f"  ✅ is_streamlit_cloud: {is_streamlit_cloud()}")
        print(f"  ✅ is_local_deployment: {is_local_deployment()}")
        print(f"  ✅ load_pil_image: available")
        print(f"  ✅ list_images_in_folder: available")
        
        print("\n✅ app_utils working\n")
        return True
    except Exception as e:
        print(f"  ❌ Error: {e}")
        traceback.print_exc()
        return False

def test_streamlit_app():
    """Test streamlit_app can be imported"""
    print("🔍 Testing streamlit_app...\n")
    
    try:
        # We can't actually run it without streamlit runtime,
        # but we can check the syntax
        import ast
        with open('streamlit_app.py', 'r') as f:
            ast.parse(f.read())
        
        print("  ✅ streamlit_app.py syntax is valid")
        
        # Check for main function
        import streamlit_app
        if hasattr(streamlit_app, 'main'):
            print("  ✅ main() function found")
        
        print("\n✅ streamlit_app ready\n")
        return True
    except SyntaxError as e:
        print(f"  ❌ Syntax Error at line {e.lineno}: {e.msg}")
        return False
    except Exception as e:
        print(f"  ❌ Error: {e}")
        traceback.print_exc()
        return False

def main():
    print("=" * 70)
    print("STREAMLIT CLOUD DEPLOYMENT DIAGNOSTIC")
    print("=" * 70 + "\n")
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("app_utils", test_app_utils()))
    results.append(("streamlit_app", test_streamlit_app()))
    
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name}: {status}")
    
    all_passed = all(passed for _, passed in results)
    
    if all_passed:
        print("\n✅ All diagnostics passed!")
        print("Your app should work correctly on Streamlit Cloud.")
        return 0
    else:
        print("\n❌ Some diagnostics failed.")
        print("Check the errors above and install missing dependencies:")
        print("   pip install -r requirements.txt")
        return 1

if __name__ == "__main__":
    sys.exit(main())
