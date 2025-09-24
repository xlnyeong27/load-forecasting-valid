#!/usr/bin/env python3
"""
Test P90 Forecast Functions
============================

Test the new P90 forecasting functions:
1. convert_roc_backtest_to_long_format()
2. compute_residual_quantiles_by_horizon() 
3. generate_p90_forecast_bands()

Usage:
    python test_p90_forecast.py
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

def create_test_data():
    """Create sample forecast and actual data for testing."""
    print("🔧 Creating test data...")
    
    # Create timestamps (1-hour period with 1-minute intervals)
    start_time = datetime(2024, 1, 1, 10, 0, 0)
    timestamps = pd.date_range(start_time, periods=60, freq='1T')
    
    # Create actual power data (with some realistic pattern)
    np.random.seed(42)  # For reproducible results
    base_load = 1000  # kW
    actual_data = base_load + 200 * np.sin(np.arange(60) * 2 * np.pi / 30) + np.random.normal(0, 50, 60)
    actual_series = pd.Series(actual_data, index=timestamps, name='Power (kW)')
    
    # Create forecast data with some error patterns
    forecast_dict = {}
    horizons = [1, 5, 10]
    
    for horizon in horizons:
        # Forecast has increasing bias and variance with horizon
        bias = horizon * 5  # kW bias increases with horizon
        noise_std = 20 + horizon * 3  # Noise increases with horizon
        
        forecast_data = actual_data + bias + np.random.normal(0, noise_std, 60)
        forecast_series = pd.Series(forecast_data, index=timestamps, name=f'Forecast_{horizon}min')
        forecast_dict[horizon] = forecast_series
    
    # Create actual dictionary (same series for all horizons)
    actual_dict = {horizon: actual_series for horizon in horizons}
    
    return forecast_dict, actual_dict, horizons

def test_long_format_conversion():
    """Test conversion to long format."""
    print("\n📊 Testing long format conversion...")
    
    forecast_dict, actual_dict, horizons = create_test_data()
    
    # Test conversion
    df_long = convert_roc_backtest_to_long_format(forecast_dict, actual_dict, horizons)
    
    print(f"✓ Long format DataFrame created with {len(df_long)} rows")
    print(f"✓ Columns: {list(df_long.columns)}")
    print(f"✓ Unique horizons: {sorted(df_long['horizon_min'].unique())}")
    print(f"✓ Time range: {df_long['t'].min()} to {df_long['t'].max()}")
    
    # Show sample data
    print("\nSample long format data:")
    print(df_long.head(10))
    
    return df_long

def test_residual_quantiles(df_long):
    """Test residual quantiles computation."""
    print("\n📈 Testing residual quantiles computation...")
    
    residual_quantiles = compute_residual_quantiles_by_horizon(df_long)
    
    print(f"✓ Residual quantiles computed for {len(residual_quantiles)} horizons")
    print(f"✓ Columns: {list(residual_quantiles.columns)}")
    
    # Show results
    print("\nResidual quantiles by horizon:")
    for _, row in residual_quantiles.iterrows():
        horizon = row['horizon_min']
        p10 = row.get('residual_p10', np.nan)
        p50 = row.get('residual_p50', np.nan)  
        p90 = row.get('residual_p90', np.nan)
        print(f"  {horizon}min: P10={p10:.2f}kW, P50={p50:.2f}kW, P90={p90:.2f}kW")
    
    return residual_quantiles

def test_p90_bands(df_long, residual_quantiles):
    """Test P90 band generation."""
    print("\n🎯 Testing P90 band generation...")
    
    df_with_bands = generate_p90_forecast_bands(df_long, residual_quantiles)
    
    print(f"✓ P90 bands added to {len(df_with_bands)} rows")
    print(f"✓ Columns: {list(df_with_bands.columns)}")
    
    # Check for new columns
    has_p10 = 'forecast_p10' in df_with_bands.columns and df_with_bands['forecast_p10'].notna().sum() > 0
    has_p90 = 'forecast_p90' in df_with_bands.columns and df_with_bands['forecast_p90'].notna().sum() > 0
    
    print(f"✓ P10 forecasts available: {has_p10}")
    print(f"✓ P90 forecasts available: {has_p90}")
    
    # Show sample with bands
    print("\nSample data with P10/P50/P90 bands:")
    display_cols = ['t', 'horizon_min', 'actual', 'forecast_p10', 'forecast_p50', 'forecast_p90']
    available_cols = [col for col in display_cols if col in df_with_bands.columns]
    print(df_with_bands[available_cols].head(12))  # Show 4 rows per horizon
    
    return df_with_bands

def test_band_properties(df_with_bands):
    """Test properties of the P90 bands."""
    print("\n🔍 Testing P90 band properties...")
    
    for horizon in sorted(df_with_bands['horizon_min'].unique()):
        horizon_data = df_with_bands[df_with_bands['horizon_min'] == horizon].copy()
        
        if len(horizon_data) > 0:
            # Calculate band widths
            horizon_data['band_width'] = horizon_data['forecast_p90'] - horizon_data['forecast_p10']
            
            # Check ordering: P10 <= P50 <= P90
            p10_le_p50 = (horizon_data['forecast_p10'] <= horizon_data['forecast_p50']).all()
            p50_le_p90 = (horizon_data['forecast_p50'] <= horizon_data['forecast_p90']).all()
            
            avg_width = horizon_data['band_width'].mean()
            
            print(f"  {horizon}min horizon:")
            print(f"    ✓ P10 <= P50: {p10_le_p50}")
            print(f"    ✓ P50 <= P90: {p50_le_p90}")
            print(f"    ✓ Avg band width: {avg_width:.2f}kW")

def main():
    """Run all P90 forecast tests."""
    print("🚀 Starting P90 Forecast Function Tests")
    print("=" * 50)
    
    try:
        # Test 1: Long format conversion
        df_long = test_long_format_conversion()
        
        # Test 2: Residual quantiles
        residual_quantiles = test_residual_quantiles(df_long)
        
        # Test 3: P90 band generation
        df_with_bands = test_p90_bands(df_long, residual_quantiles)
        
        # Test 4: Band properties validation
        test_band_properties(df_with_bands)
        
        print("\n" + "=" * 50)
        print("✅ All P90 forecast tests completed successfully!")
        print(f"📊 Final result: {len(df_with_bands)} rows with P10/P50/P90 bands")
        
        # Export test results
        output_file = "test_p90_forecast_results.csv"
        df_with_bands.to_csv(output_file, index=False)
        print(f"💾 Results saved to: {output_file}")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return False
        
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)