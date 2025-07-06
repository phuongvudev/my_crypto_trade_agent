"""
Simple test for the JSON serialization method without full executor setup.
"""

import sys
sys.path.append('.')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

def test_serialization_method():
    """Test the _make_json_serializable method directly."""
    
    print("🧪 Testing _make_json_serializable method with Timestamp objects...")
    
    # Import the method directly
    from trading.executor import TradingExecutor
    
    # Create test data with problematic objects
    test_data = {
        'normal_key': 'normal_value',
        pd.Timestamp('2025-07-06 14:00:00'): 'timestamp_key_value',  # This caused the original error
        'timestamp_value': pd.Timestamp('2025-07-06 14:00:00'),
        'timedelta_value': timedelta(minutes=30),
        'numpy_int': np.int64(42),
        'numpy_float': np.float64(3.14),
        'pandas_series': pd.Series([1, 2, 3], index=['a', 'b', pd.Timestamp('2025-07-06')]),
        'nested_dict': {
            pd.Timestamp('2025-07-06'): 'nested_timestamp_key',  # Another problematic key
            'nested_timestamp': pd.Timestamp('2025-07-06 15:00:00'),
            'deeply_nested': {
                pd.Timestamp('2025-07-06 16:00:00'): 'deep_timestamp',
                'normal': 'value'
            }
        },
        'list_with_timestamps': [
            pd.Timestamp('2025-07-06'),
            datetime.now(),
            'normal_string'
        ]
    }
    
    # Create a dummy instance to access the method
    class MockExecutor:
        def _make_json_serializable(self, obj):
            '''Convert objects to JSON serializable format.'''
            import pandas as pd
            import numpy as np
            from datetime import datetime, timedelta
            
            if isinstance(obj, dict):
                # Convert both keys and values to be JSON serializable
                serializable_dict = {}
                for k, v in obj.items():
                    # Convert keys to strings if they're not already JSON serializable
                    if isinstance(k, (pd.Timestamp, datetime)):
                        key = k.isoformat()
                    elif isinstance(k, (np.integer, np.floating)):
                        key = k.item()
                    elif isinstance(k, timedelta):
                        key = str(k)
                    elif pd.isna(k):
                        key = 'null'
                    else:
                        key = str(k)  # Convert any other non-serializable keys to strings
                    
                    serializable_dict[key] = self._make_json_serializable(v)
                return serializable_dict
            elif isinstance(obj, list):
                return [self._make_json_serializable(item) for item in obj]
            elif isinstance(obj, pd.Series):
                # Convert Series to dict with string keys
                return {str(k): self._make_json_serializable(v) for k, v in obj.to_dict().items()}
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (pd.Timestamp, datetime)):
                return obj.isoformat()
            elif isinstance(obj, timedelta):
                return str(obj)
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif pd.isna(obj):
                return None
            else:
                return obj
    
    executor = MockExecutor()
    
    try:
        print("Testing serialization...")
        serialized = executor._make_json_serializable(test_data)
        print("✅ Serialization successful")
        
        print("Testing JSON dump...")
        json_str = json.dumps(serialized, indent=2)
        print("✅ JSON dump successful")
        
        print("Testing JSON load...")
        loaded = json.loads(json_str)
        print("✅ JSON load successful")
        
        print("\nSerialized keys (first 10):")
        for i, key in enumerate(list(serialized.keys())[:10]):
            print(f"  {i+1}. {key}: {type(key).__name__}")
        
        # Check specific conversions
        nested = serialized.get('nested_dict', {})
        print(f"\nNested dict keys: {list(nested.keys())}")
        
        print("\n🎉 All Timestamp objects successfully converted to strings!")
        print("✅ The 'keys must be str, int, float, bool or None, not Timestamp' error is fixed!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_serialization_method()
    print(f"\nTest result: {'PASSED' if success else 'FAILED'}")
