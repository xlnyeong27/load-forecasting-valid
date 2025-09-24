#!/usr/bin/env python3
"""
Test Split P90 Forecast Tables
===============================

Test the updated P90 forecasting with split tables for 1-minute and 10-minute forecasts.

Usage:
    python test_split_tables.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(__file__))

# Import the functions from md_shaving_solution_v2
from md_shaving_solution_v2 import (
    convert_roc_backtest_to_long_format,
    compute_residual_quantiles_by_horizon,
    generate_p90_forecast_bands
)

def create_test_data_with_multiple_horizons():
    """Create sample forecast data with 1-minute and 10-minute horizons."""
    print("🔧 Creating test data with 1-minute and 10-minute horizons...")
    
    # Create timestamps (2-hour period with 1-minute intervals)
    start_time = datetime(2024, 1, 1, 10, 0, 0)
    timestamps = pd.date_range(start_time, periods=120, freq='1T')  # 2 hours
    
    # Create actual power data
    np.random.seed(42)
    base_load = 1000  # kW
    actual_data = base_load + 200 * np.sin(np.arange(120) * 2 * np.pi / 60) + np.random.normal(0, 50, 120)
    actual_series = pd.Series(actual_data, index=timestamps, name='Power (kW)')
    
    # Create forecast data for 1-minute and 10-minute horizons
    forecast_dict = {}
    horizons = [1, 10]  # Focus on 1-min and 10-min
    
    for horizon in horizons:
        # Different error characteristics for each horizon
        if horizon == 1:
            bias = 2  # Small bias for 1-minute
            noise_std = 15  # Lower noise for 1-minute
        else:  # 10-minute
            bias = 25  # Higher bias for 10-minute 
            noise_std = 40  # Higher noise for 10-minute
        
        forecast_data = actual_data + bias + np.random.normal(0, noise_std, 120)
        forecast_series = pd.Series(forecast_data, index=timestamps, name=f'Forecast_{horizon}min')
        forecast_dict[horizon] = forecast_series
    
    # Create actual dictionary
    actual_dict = {horizon: actual_series for horizon in horizons}
    
    return forecast_dict, actual_dict, horizons

def test_split_table_functionality():
    """Test the complete split table workflow."""
    print("\n📊 Testing split table functionality...")
    
    # Create test data
    forecast_dict, actual_dict, horizons = create_test_data_with_multiple_horizons()
    
    # Convert to long format
    df_long = convert_roc_backtest_to_long_format(forecast_dict, actual_dict, horizons)
    print(f"✓ Long format DataFrame created with {len(df_long)} rows")
    
    # Compute residual quantiles
    residual_quantiles = compute_residual_quantiles_by_horizon(df_long)
    print(f"✓ Residual quantiles computed for {len(residual_quantiles)} horizons")
    
    # Generate P90 bands
    df_with_bands = generate_p90_forecast_bands(df_long, residual_quantiles)
    print(f"✓ P90 bands added to {len(df_with_bands)} rows")
    
    return df_with_bands, residual_quantiles

def analyze_split_tables(df_with_bands):
    """Analyze the data that would be shown in split tables."""
    print("\n🔍 Analyzing split table data...")
    
    # Filter data for each horizon
    df_1min = df_with_bands[df_with_bands['horizon_min'] == 1]
    df_10min = df_with_bands[df_with_bands['horizon_min'] == 10]
    
    print(f"\n📊 1-Minute Forecast Table:")
    print(f"  • Data points: {len(df_1min)}")
    if not df_1min.empty:
        band_width_1min = (df_1min['forecast_p90'] - df_1min['forecast_p10']).mean()
        mae_1min = abs(df_1min['forecast_p50'] - df_1min['actual']).mean()
        print(f"  • Average uncertainty band: {band_width_1min:.1f} kW")
        print(f"  • Mean Absolute Error: {mae_1min:.1f} kW")
        
        # Show sample data
        print(f"\n  Sample 1-minute data (first 5 rows):")
        sample_cols = ['t', 'actual', 'forecast_p10', 'forecast_p50', 'forecast_p90']
        print(df_1min[sample_cols].head().to_string(index=False))
    
    print(f"\n📊 10-Minute Forecast Table:")
    print(f"  • Data points: {len(df_10min)}")
    if not df_10min.empty:
        band_width_10min = (df_10min['forecast_p90'] - df_10min['forecast_p10']).mean()
        mae_10min = abs(df_10min['forecast_p50'] - df_10min['actual']).mean()
        print(f"  • Average uncertainty band: {band_width_10min:.1f} kW")
        print(f"  • Mean Absolute Error: {mae_10min:.1f} kW")
        
        # Show sample data
        print(f"\n  Sample 10-minute data (first 5 rows):")
        sample_cols = ['t', 'actual', 'forecast_p10', 'forecast_p50', 'forecast_p90']
        print(df_10min[sample_cols].head().to_string(index=False))
    
    # Comparison
    if not df_1min.empty and not df_10min.empty:
        print(f"\n🔍 Horizon Comparison:")
        print(f"  • 1-min vs 10-min uncertainty: {band_width_1min:.1f} vs {band_width_10min:.1f} kW")
        print(f"  • 1-min vs 10-min accuracy: {mae_1min:.1f} vs {mae_10min:.1f} kW MAE")
        
        uncertainty_ratio = band_width_10min / band_width_1min
        accuracy_ratio = mae_10min / mae_1min
        print(f"  • Uncertainty increases by {uncertainty_ratio:.1f}x for longer horizon")
        print(f"  • Error increases by {accuracy_ratio:.1f}x for longer horizon")

def main():
    """Run split table tests."""
    print("🚀 Testing P90 Forecast Split Tables")
    print("=" * 50)
    
    try:
        # Test split table functionality
        df_with_bands, residual_quantiles = test_split_table_functionality()
        
        # Analyze split table data
        analyze_split_tables(df_with_bands)
        
        print("\n" + "=" * 50)
        print("✅ Split table tests completed successfully!")
        
        # Show residual quantiles summary
        print(f"\n📈 Residual Quantiles Summary:")
        for _, row in residual_quantiles.iterrows():
            horizon = row['horizon_min']
            p10 = row.get('residual_p10', np.nan)
            p50 = row.get('residual_p50', np.nan)
            p90 = row.get('residual_p90', np.nan)
            print(f"  {horizon}min: P10={p10:.2f}, P50={p50:.2f}, P90={p90:.2f} kW")
        
        # Export test results
        output_file = "test_split_tables_results.csv"
        df_with_bands.to_csv(output_file, index=False)
        print(f"\n💾 Results saved to: {output_file}")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return False
        
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)