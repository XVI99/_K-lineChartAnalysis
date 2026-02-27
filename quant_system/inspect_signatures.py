from pattern_loader import PatternLoader
import inspect
import os
import pandas as pd

def inspect_signatures():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    k_line_dir = os.path.join(base_dir, "k_line_code")
    
    loader = PatternLoader(k_line_dir)
    patterns = loader.load_patterns()
    
    print(f"\n{'Pattern Name':<30} | {'Arguments'}")
    print("-" * 80)
    
    for name, func in patterns.items():
        sig = inspect.signature(func)
        print(f"{name:<30} | {sig}")

if __name__ == "__main__":
    inspect_signatures()
