#!/usr/bin/env python3
"""
Test script for ROC forecasting functions
Demonstrates usage of roc_forecast and roc_forecast_with_validation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add current directory to path to import our module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from md_shaving_solution_v2 import roc_forecast, roc_forecast_with_validation

def create_sample_data():
    """Create sample power data for testing"""
    # Create 1-hour of 1-minute interval data
    start_time = datetime(2024, 1, 1, 10, 0, 0)
    timestamps = [start_time + timedelta(minutes=i) for i in range(60)]
    
    # Create sample power pattern (base load + some variation)
    base_power = 100  # kW
    power_values = []
    
    for i, ts in enumerate(timestamps):
        # Add some realistic patterns
        if i < 10:
            # Gradual increase (ramp up)
            power = base_power + i * 2
        elif i < 20:
            # Stable period
            power = base_power + 20 + np.random.normal(0, 1)
        elif i < 30:
            # Quick increase (step change)
            power = base_power + 30 + (i-20) * 3
        elif i < 40:
            # Gradual decrease
            power = base_power + 60 - (i-30) * 1.5
        else:
            # Another stable period
            power = base_power + 45 + np.random.normal(0, 2)
        
        power_values.append(max(0, power))  # Ensure non-negative
    
    # Create Series with datetime index
    power_series = pd.Series(power_values, index=pd.DatetimeIndex(timestamps))
    
    return power_series

def test_basic_roc_forecast():
    """Test basic ROC forecast functionality"""
    print("=" * 60)
    print("Testing Basic ROC Forecast (1-minute horizon)")
    print("=" * 60)
    
    # Create sample data
    power_series = create_sample_data()
    
    # Test Series input
    forecast_df = roc_forecast(power_series, horizon=1)
    
    print(f"Original data points: {len(power_series)}")
    print(f"Forecast data points: {len(forecast_df)}")
    print(f"Forecasts available: {forecast_df['Forecast_Available'].sum()}")
    
    # Display first 10 rows
    print("\nFirst 10 forecast results:")
    print(forecast_df[['Timestamp', 'Power_Actual (kW)', 'ROC (kW/min)', 'Power_Forecast (kW)', 'Forecast_Available']].head(10).to_string(index=False))
    
    return forecast_df

def test_multi_horizon_forecast():
    """Test forecasts with different horizons"""
    print("\n" + "=" * 60)
    print("Testing Multi-Horizon Forecasts")
    print("=" * 60)
    
    power_series = create_sample_data()
    
    horizons = [1, 5, 10]
    
    for horizon in horizons:
        forecast_df = roc_forecast(power_series, horizon=horizon)
        available_forecasts = forecast_df['Forecast_Available'].sum()
        
        print(f"\nHorizon {horizon} minutes:")
        print(f"  Forecasts available: {available_forecasts}")
        
        if available_forecasts > 0:
            # Show some sample forecasts
            valid_forecasts = forecast_df[forecast_df['Forecast_Available']].head(3)
            for _, row in valid_forecasts.iterrows():
                print(f"  {row['Timestamp']:%H:%M} | Current: {row['Power_Actual (kW)']:.1f} kW | ROC: {row['ROC (kW/min)']:.2f} kW/min | Forecast (+{horizon}min): {row['Power_Forecast (kW)']:.1f} kW")

def test_validated_forecast():
    """Test forecast with validation against actual future values"""
    print("\n" + "=" * 60)
    print("Testing ROC Forecast with Validation")
    print("=" * 60)
    
    power_series = create_sample_data()
    
    # Test with validation
    validated_df, metrics = roc_forecast_with_validation(power_series, horizon=1, return_metrics=True)
    
    print("Validation Metrics:")
    print(f"  Total points: {metrics['total_points']}")
    print(f"  Forecasts made: {metrics['forecasts_made']}")
    print(f"  Validations available: {metrics['validations_available']}")
    print(f"  Validation rate: {metrics['validation_rate']:.1f}%")
    
    if metrics['validations_available'] > 0:
        print(f"  MAE: {metrics['mae_kw']:.3f} kW")
        print(f"  RMSE: {metrics['rmse_kw']:.3f} kW")
        print(f"  Mean %Error: {metrics['mean_pct_error']:.2f}%")
        print(f"  Median %Error: {metrics['median_pct_error']:.2f}%")
        print(f"  Bias: {metrics['bias_kw']:.3f} kW")
    
    # Show sample validated results
    print("\nSample validation results (first 5 validated forecasts):")
    valid_results = validated_df[
        (validated_df['Forecast_Available']) & 
        (validated_df['Validation_Available'])
    ].head(5)
    
    if len(valid_results) > 0:
        display_cols = ['Timestamp', 'Power_Actual (kW)', 'Power_Forecast (kW)', 
                       'Power_Actual_Future (kW)', 'Forecast_Error (kW)', 'Percentage_Error (%)']
        print(valid_results[display_cols].to_string(index=False))
    
    return validated_df, metrics

def test_dataframe_input():
    """Test with DataFrame input instead of Series"""
    print("\n" + "=" * 60)
    print("Testing DataFrame Input")
    print("=" * 60)
    
    power_series = create_sample_data()
    
    # Convert to DataFrame
    df = pd.DataFrame({
        'Power_kW': power_series.values,
        'Other_Data': range(len(power_series))
    }, index=power_series.index)
    
    # Test DataFrame input
    forecast_df = roc_forecast(df, horizon=1, power_col='Power_kW')
    
    print(f"DataFrame input - Forecasts available: {forecast_df['Forecast_Available'].sum()}")
    
    # Show first few results
    print("\nFirst 5 DataFrame forecast results:")
    print(forecast_df[['Timestamp', 'Power_Actual (kW)', 'ROC (kW/min)', 'Power_Forecast (kW)']].head(5).to_string(index=False))

def main():
    """Run all tests"""
    print("ROC Forecast Function Tests")
    print("Reusing logic from load_forecasting.py")
    
    try:
        # Basic functionality test
        basic_forecast = test_basic_roc_forecast()
        
        # Multi-horizon test
        test_multi_horizon_forecast()
        
        # Validation test
        validated_forecast, metrics = test_validated_forecast()
        
        # DataFrame input test
        test_dataframe_input()
        
        print("\n" + "=" * 60)
        print("✅ All tests completed successfully!")
        print("=" * 60)
        
        print("\nFunction Usage Examples:")
        print("# Basic forecast (Series input)")
        print("forecast_df = roc_forecast(power_series, horizon=1)")
        
        print("\n# DataFrame input with validation")
        print("forecast_df, metrics = roc_forecast_with_validation(df, horizon=5, power_col='Power_kW', return_metrics=True)")
        
        print("\n# Multi-horizon comparison")
        print("for horizon in [1, 5, 10]:")
        print("    forecast_df = roc_forecast(power_series, horizon=horizon)")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()