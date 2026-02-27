import importlib.util
import os
import inspect
import pandas as pd
import sys

class PatternLoader:
    def __init__(self, patterns_dir):
        self.patterns_dir = patterns_dir
        self.patterns = {}  # {name: function}
        
    def load_patterns(self):
        """
        Scans the directory and loads functions starting with 'identify_' or 'calculate_'.
        """
        if not os.path.exists(self.patterns_dir):
            print(f"Error: Directory {self.patterns_dir} not found.")
            return

        # Add directory to sys.path to allow imports
        if self.patterns_dir not in sys.path:
            sys.path.append(self.patterns_dir)

        for filename in os.listdir(self.patterns_dir):
            if filename.endswith(".py") and filename != "__init__.py":
                module_name = filename[:-3]
                file_path = os.path.join(self.patterns_dir, filename)
                
                try:
                    spec = importlib.util.spec_from_file_location(module_name, file_path)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        sys.modules[module_name] = module # caching
                        spec.loader.exec_module(module)
                        
                        # Inspect functions
                        for name, func in inspect.getmembers(module, inspect.isfunction):
                            # Criteria: starts with specific prefixes and defined in this module
                            if (name.startswith("identify_") or name.startswith("calculate_")) and \
                               func.__module__ == module_name:
                                
                                # Optional: Check if first arg is 'df' or type annotated as DataFrame
                                # For now, we trust the naming convention.
                                
                                # Use a clean key name (e.g., 'belt_hold' from 'identify_belt_hold')
                                key_name = name.replace("identify_", "").replace("calculate_", "")
                                self.patterns[key_name] = func
                                print(f"Loaded pattern: {key_name} from {filename}")
                                
                except Exception as e:
                    print(f"Failed to load {filename}: {e}")

        print(f"Total patterns loaded: {len(self.patterns)}")
        return self.patterns

if __name__ == "__main__":
    # Test execution
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    k_line_dir = os.path.join(base_dir, "k_line_code")
    
    loader = PatternLoader(k_line_dir)
    patterns = loader.load_patterns()
    
    # Print discovered patterns
    print("\nAvailable Pattern Functions:")
    for name, func in patterns.items():
        print(f" - {name}: {func.__name__}")
