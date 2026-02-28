
try:
    print("Checking core libraries...")
    import pandas as pd
    print(f"✅ Pandas: {pd.__version__}")
    
    import numpy as np
    print(f"✅ Numpy: {np.__version__}")
    
    import xgboost as xgb
    print(f"✅ XGBoost: {xgb.__version__}")
    
    print("Checking tensorflow (optional)...")
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow: {tf.__version__}")
    except ImportError:
        print("⚠️ TensorFlow not available (ok for phase 1)")
        
    print("Checking transformers (optional)...")
    try:
        import transformers
        print(f"✅ Transformers: {transformers.__version__}")
    except ImportError:
        print("⚠️ Transformers not available (ok for phase 1)")

    print("\nREADY TO LAUNCH VN-QUANT PRO!")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    exit(1)
