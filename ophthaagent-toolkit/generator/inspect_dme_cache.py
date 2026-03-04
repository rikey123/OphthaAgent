import sys
import os
import logging
import json
import pickle
from pathlib import Path
from collections import Counter

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("InspectCache")

# Add current directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from real_tool_executor import RealToolExecutor
except ImportError:
    print("Error: Could not import RealToolExecutor.")
    sys.exit(1)

def inspect_cache():
    # 1. Configuration
    # 使用默认的缓存目录 ./tool_cache
    cache_dir = "./tool_cache"
    
    print(f"Initializing RealToolExecutor with cache_dir={cache_dir}...")
    # enable_cache=True 会触发 _load_persistent_cache
    executor = RealToolExecutor(enable_cache=True, cache_dir=cache_dir)
    
    print(f"Total Cache Entries Loaded: {len(executor.tool_cache)}")
    
    # 2. Define Parameters for dme_risk_assessment
    image_path = "<ANON_ABS_PATH>"
    tool_name = "dme_risk_assess" 
    
    # Check both possible tool names
    tool_names = ["dme_risk_assess", "dme_risk_assessment"]
    
    for tool_name in tool_names:
        print(f"\n{'='*40}")
        print(f"Checking Tool Name: {tool_name}")
        print(f"{'='*40}")
        
        args = {"image_path": image_path}
        
        # 3. Generate Cache Key
        cache_key = executor._generate_cache_key(tool_name, args)
        print(f"Target Cache Key: {cache_key}")
        print(f"Args: {json.dumps(args, sort_keys=True)}")

        # 4. Check if it exists
        if cache_key in executor.tool_cache:
            print(f"✅ FOUND in cache for '{tool_name}'!")
            cached_value = executor.tool_cache[cache_key]
            print("Cached Value Summary:")
            print(json.dumps(cached_value, indent=2, ensure_ascii=False)[:2000] + "...")
            
            # Check if it is a mock result
            is_mock = False
            if isinstance(cached_value, dict):
                if cached_value.get("result", {}).get("mock") is True:
                    is_mock = True
                # Also check if it lacks expected keys or has placeholder text
                if "risk_level" not in str(cached_value) and "mock" in str(cached_value).lower():
                    is_mock = True
                    
            if is_mock:
                print(f"\n⚠️ WARNING: This is a MOCK result!")
                print("The system is reading this cached mock result instead of executing the real tool.")
                
                # Ask to delete
                print(f"Removing this invalid cache entry...")
                del executor.tool_cache[cache_key]
                
                # Save back to disk
                try:
                    executor._save_to_persistent_cache(cache_key, cached_value) # Value doesn't matter here as it dumps self.tool_cache
                    print("✅ Successfully REMOVED mock result from cache file.")
                    print("Next run will force real tool execution.")
                except Exception as e:
                    print(f"❌ Failed to save cache file: {e}")
            else:
                print("\nThis appears to be a VALID (non-mock) result.")
                
        else:
            print(f"❌ NOT FOUND in cache for '{tool_name}'.")
            
            # 5. Debugging: List all keys for this tool
            print(f"Listing all cached entries starting with '{tool_name}:':")
            found_count = 0
            for k, v in executor.tool_cache.items():
                if k.startswith(f"{tool_name}:"):
                    found_count += 1
                    print(f"  - Key: {k}")
                    print(f"    Status: {v.get('status', 'unknown')}")
            if found_count == 0:
                print("  (No entries found)")
                
        # 6. Check absolute path variant
        abs_path = os.path.abspath(image_path)
        args_abs = {"image_path": abs_path}
        key_abs = executor._generate_cache_key(tool_name, args_abs)
        
        if key_abs in executor.tool_cache:
            print(f"✅ FOUND with absolute path for '{tool_name}'!")
            print(json.dumps(executor.tool_cache[key_abs], indent=2, ensure_ascii=False)[:500])

if __name__ == "__main__":
    inspect_cache()
