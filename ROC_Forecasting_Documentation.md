# ROC Forecasting Functions Documentation

## Overview

The ROC (Rate of Change) forecasting functions provide a simple yet effective method for generating short-term power demand forecasts. These functions reuse the existing ROC calculation logic from `load_forecasting.py` and wrap it in convenient interfaces for generating forecasts for all data points.

## Key Functions

### 1. `roc_forecast(series, horizon=1, power_col=None)`

Generates horizon-minute ahead forecasts for ALL data points using the Rate of Change method.

**Parameters:**
- `series`: pandas Series (with datetime index) or DataFrame containing power data
- `horizon`: forecast horizon in minutes (default: 1)
- `power_col`: column name if using DataFrame input (auto-detected if None)

**Returns:**
- DataFrame with columns: Timestamp, Power_Actual (kW), ROC (kW/min), Forecast_Timestamp, Power_Forecast (kW), Forecast_Available

**Example:**
```python
# Using Series input
forecast_df = roc_forecast(power_series, horizon=5)

# Using DataFrame input
forecast_df = roc_forecast(df, horizon=1, power_col='Power_kW')
```

### 2. `roc_forecast_with_validation(series, horizon=1, power_col=None, return_metrics=False)`

Extended version that includes validation against actual future values and calculates error metrics.

**Parameters:**
- Same as `roc_forecast()` plus:
- `return_metrics`: if True, returns (forecast_df, metrics_dict)

**Returns:**
- DataFrame with additional validation columns: Power_Actual_Future (kW), Forecast_Error (kW), Absolute_Error (kW), Percentage_Error (%), Validation_Available
- Optional metrics dictionary with MAE, RMSE, bias, etc.

**Example:**
```python
# Basic validation
validated_df = roc_forecast_with_validation(power_series, horizon=1)

# With metrics
validated_df, metrics = roc_forecast_with_validation(
    power_series, horizon=5, return_metrics=True
)
print(f"MAE: {metrics['mae_kw']:.2f} kW")
```

### 3. `_calculate_roc_from_series(series, power_col=None)`

Helper function that calculates ROC values from time series data (reuses load_forecasting.py logic).

## Forecasting Method

The ROC method uses this simple formula:
```
P_forecast(t+h) = P_actual(t) + ROC(t) × h
```

Where:
- `P_forecast(t+h)`: forecasted power at time t+horizon
- `P_actual(t)`: actual power at time t  
- `ROC(t)`: rate of change at time t (kW/min)
- `h`: forecast horizon in minutes

## Key Features

### ✅ Advantages
- **Simple and Fast**: Computationally efficient for real-time applications
- **No Training Required**: Uses only current and previous data points
- **Reuses Existing Logic**: Built on proven ROC calculations from load_forecasting.py
- **All Points Forecasted**: Generates forecasts for every available data point
- **Validation Included**: Built-in validation against actual future values
- **Flexible Input**: Handles both Series and DataFrame inputs

### ⚠️ Limitations
- **Linear Assumption**: Assumes constant rate of change over forecast horizon
- **Short-term Only**: Most effective for horizons ≤ 10-15 minutes
- **First Point Issue**: Cannot forecast for the first data point (no ROC available)
- **Trend Sensitivity**: May over-extrapolate during rapid changes

## Usage Patterns

### 1. Single Point Forecast
```python
# Forecast 1 minute ahead for all points
forecast_df = roc_forecast(power_data, horizon=1)

# Get forecasts where available
valid_forecasts = forecast_df[forecast_df['Forecast_Available']]
```

### 2. Multi-Horizon Analysis
```python
horizons = [1, 5, 10, 15]
results = {}

for h in horizons:
    forecast_df, metrics = roc_forecast_with_validation(
        power_data, horizon=h, return_metrics=True
    )
    results[h] = metrics
    
# Compare accuracy across horizons
for h, m in results.items():
    print(f"{h}min: MAE={m['mae_kw']:.2f}kW, RMSE={m['rmse_kw']:.2f}kW")
```

### 3. Real-time Forecasting Integration
```python
def get_next_minute_forecast(current_data):
    """Get 1-minute ahead forecast for latest data point"""
    forecast_df = roc_forecast(current_data, horizon=1)
    latest_forecast = forecast_df.iloc[-1]
    
    if latest_forecast['Forecast_Available']:
        return {
            'timestamp': latest_forecast['Forecast_Timestamp'],
            'forecast_kw': latest_forecast['Power_Forecast (kW)'],
            'roc_kw_per_min': latest_forecast['ROC (kW/min)']
        }
    return None
```

### 4. Performance Monitoring
```python
# Validate forecasts and track performance
validated_df, metrics = roc_forecast_with_validation(
    historical_data, horizon=5, return_metrics=True
)

# Monitor key metrics
print(f"Validation Rate: {metrics['validation_rate']:.1f}%")
print(f"Mean Absolute Error: {metrics['mae_kw']:.2f} kW")
print(f"Root Mean Square Error: {metrics['rmse_kw']:.2f} kW")
print(f"Bias (over/under forecast): {metrics['bias_kw']:.2f} kW")
```

## Error Metrics

The validation function provides these key metrics:

- **MAE (Mean Absolute Error)**: Average magnitude of forecast errors in kW
- **RMSE (Root Mean Square Error)**: Square root of mean squared errors in kW
- **Bias**: Average forecast error (positive = over-forecast, negative = under-forecast)
- **Mean/Median Percentage Error**: Error as percentage of actual values
- **Validation Rate**: Percentage of forecasts that could be validated

## Integration with MD Shaving V2

These functions are designed to integrate seamlessly with the MD Shaving V2 forecasting system:

```python
# In MD Shaving V2 forecasting section
if enable_forecasting and selected_method == "Rate of Change (ROC)":
    # Generate forecasts for analysis
    forecast_results = roc_forecast_with_validation(
        processed_data, 
        horizon=user_selected_horizon,
        power_col=power_column,
        return_metrics=True
    )
    
    # Display results and metrics
    display_forecast_results(forecast_results)
```

## Best Practices

1. **Horizon Selection**: Use 1-5 minutes for best accuracy
2. **Data Quality**: Ensure consistent time intervals in input data  
3. **Validation**: Always validate forecasts against actual values when available
4. **Multiple Horizons**: Compare performance across different forecast horizons
5. **Error Monitoring**: Track forecast accuracy over time to detect degradation
6. **Ramp Detection**: ROC method works well during steady periods, use caution during rapid changes