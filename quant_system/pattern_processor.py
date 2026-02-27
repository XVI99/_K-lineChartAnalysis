try:
    from pattern_loader import PatternLoader
except ImportError:
    from quant_system.pattern_loader import PatternLoader
import pandas as pd
import os
import inspect

class StandardizedPatternProcessor:
    def __init__(self, patterns_dir):
        self.loader = PatternLoader(patterns_dir)
        self.patterns = self.loader.load_patterns()
        
    def list_available_patterns(self):
        """Returns a list of available pattern names."""
        return list(self.patterns.keys())

    def run_pattern(self, pattern_name, df, **kwargs):
        """
        Runs a specific pattern function on the DataFrame.
        
        Args:
            pattern_name (str): The name of the pattern to run.
            df (pd.DataFrame): The input market data.
            **kwargs: Specific parameters for the pattern function.
            
        Returns:
            pd.DataFrame: The DataFrame with added signal columns.
        """
        if pattern_name not in self.patterns:
            print(f"Warning: Pattern '{pattern_name}' not found.")
            return df
            
        func = self.patterns[pattern_name]
        
        # Filter kwargs to only include valid arguments for the function
        sig = inspect.signature(func)
        valid_params = {
            k: v for k, v in kwargs.items() 
            if k in sig.parameters
        }
        
        # execution
        try:
            # We assume the first argument is always the dataframe
            result_df = func(df, **valid_params)
            return result_df
        except Exception as e:
            print(f"Error running pattern '{pattern_name}': {e}")
            return df

    def run_all_patterns(self, df, config=None):
        """
        Runs all available patterns.
        
        Args:
            df (pd.DataFrame): Input data.
            config (dict): Configuration dictionary {pattern_name: {param: value}}.
            
        Returns:
            pd.DataFrame: DataFrame with all signals.
        """
        if config is None:
            config = {}
            
        results = df.copy()
        
        for name in self.patterns.keys():
            # Get specific config for this pattern
            pattern_kwargs = config.get(name, {})
            
            # Run pattern
            # Note: We pass the accumulating 'results' df, so new columns are added sequentially.
            # CAUTION: If patterns rely on 'clean' input, this might be risky if previous patterns modified OHLC.
            # Looking at the code, deep copies are usually made inside the functions, but return includes new cols.
            # So passing 'results' is correct to accumulate columns.
            
            try:
                # Some functions might return a tuple or list (based on inspection 'tower_patterns' returns list?)
                # We need to handle return types.
                # Inspecting 'tower_patterns': returns list. We need to standardize this.
                
                temp_res = self.run_pattern(name, results, **pattern_kwargs)
                
                if isinstance(temp_res, pd.DataFrame):
                    # Update results with new columns
                    # We avoid overwriting existing columns like Open, High, Low, Close unless intended
                    new_cols = temp_res.columns.difference(results.columns)
                    if not new_cols.empty:
                        results = results.join(temp_res[new_cols])
                else:
                    print(f"Skipping '{name}': returned {type(temp_res)}, expected DataFrame.")
                    
            except Exception as e:
                print(f"Failed to process '{name}': {e}")
                
        return results

if __name__ == "__main__":
    # Test
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    k_line_dir = os.path.join(base_dir, "k_line_code")
    
    processor = StandardizedPatternProcessor(k_line_dir)
    print("Patterns initialized.")
    
    # Create dummy data
    dates = pd.date_range(start="2023-01-01", periods=100)
    data = {
        "Open": [10, 11, 10, 12] * 25,
        "High": [12, 13, 11, 14] * 25,
        "Low": [9, 10, 9, 11] * 25,
        "Close": [11, 10, 11, 13] * 25,
        "Volume": [1000, 1200, 900, 1500] * 25
    }
    df = pd.DataFrame(data, index=dates)
    
    print("\nRunning 'engulfing' pattern...")
    res = processor.run_pattern("engulfing", df, ma_len=5)
    print("Columns:", res.columns)
    
    if "Bull_Engulf" in res.columns:
        print("Success: Bull_Engulf found.")
