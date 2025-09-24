# P90 Forecasting Implementation
## Enhanced ROC Forecasting with Uncertainty Bands

### Overview
The P90 forecasting enhancement adds uncertainty quantification to the MD Shaving V2 ROC forecasting system. This implementation generates P10, P50, and P90 forecast bands using historical residual distributions.

### Key Features

#### 1. **Long Format Conversion**
- Converts ROC backtest results from wide format (separate series per horizon) to long format table
- Structure: `[t, horizon_min, actual, forecast_p50]`
- Enables unified analysis across multiple forecast horizons

#### 2. **Residual Quantile Analysis** 
- Computes P10, P50, P90 quantiles of historical forecast residuals by horizon
- Residuals calculated as: `residual = forecast - actual`
- Accounts for horizon-specific error characteristics

#### 3. **P90 Band Generation**
- Generates uncertainty bands by adding residual quantiles to P50 forecasts:
  - `forecast_p10 = forecast_p50 + residual_p10`
  - `forecast_p90 = forecast_p50 + residual_p90`
- Maintains proper ordering: P10 ≤ P50 ≤ P90

#### 4. **Data Export Options**
- CSV export for compatibility with external analysis tools
- Parquet export for efficient storage of large datasets
- Timestamped filenames for version control

### Function Reference

#### `convert_roc_backtest_to_long_format(forecast_series_dict, actual_series_dict, horizons)`
**Purpose:** Convert ROC backtest results to long format table

**Parameters:**
- `forecast_series_dict`: Dictionary of {horizon: forecast_series}
- `actual_series_dict`: Dictionary of {horizon: actual_series}
- `horizons`: List of forecast horizons in minutes

**Returns:** DataFrame with columns [t, horizon_min, actual, forecast_p50]

**Example:**
```python
forecast_dict = {1: series_1min, 10: series_10min}
actual_dict = {1: actual_series, 10: actual_series}
df_long = convert_roc_backtest_to_long_format(forecast_dict, actual_dict, [1, 10])
```

#### `compute_residual_quantiles_by_horizon(df_long, quantiles=[0.1, 0.5, 0.9])`
**Purpose:** Compute residual quantiles by horizon from long format data

**Parameters:**
- `df_long`: Long format DataFrame with columns [t, horizon_min, actual, forecast_p50]
- `quantiles`: List of quantiles to compute (default: [0.1, 0.5, 0.9])

**Returns:** DataFrame with columns [horizon_min, residual_p10, residual_p50, residual_p90]

**Example:**
```python
residual_quantiles = compute_residual_quantiles_by_horizon(df_long)
print(residual_quantiles)
#   horizon_min  residual_p10  residual_p50  residual_p90
# 0           1        -21.94          4.30         31.78
# 1           5        -16.83         33.04         74.28
# 2          10         -7.66         58.17        102.84
```

#### `generate_p90_forecast_bands(df_long, residual_quantiles)`
**Purpose:** Generate P10/P90 forecast bands by adding residual quantiles to P50 forecasts

**Parameters:**
- `df_long`: Long format DataFrame with P50 forecasts
- `residual_quantiles`: DataFrame with quantiles by horizon

**Returns:** DataFrame with added columns [forecast_p10, forecast_p90]

**Example:**
```python
df_with_bands = generate_p90_forecast_bands(df_long, residual_quantiles)
# Result includes: t, horizon_min, actual, forecast_p10, forecast_p50, forecast_p90
```

### Integration with MD Shaving V2

The P90 forecasting is seamlessly integrated into the existing MD Shaving V2 workflow:

1. **Automatic Activation:** P90 bands are generated automatically after ROC forecasting completes
2. **Session State Storage:** Results stored in `st.session_state` for downstream analysis:
   - `roc_long_format`: Long format table with P10/P50/P90 bands
   - `residual_quantiles`: Quantiles by horizon for reference
3. **Visual Integration:** Ready for integration with existing visualization components

### Usage in Streamlit App

1. Navigate to **MD Shaving V2** tab
2. Upload load data and configure analysis settings
3. Go to **Forecasting** section
4. Select **Rate of Change (ROC)** method
5. Choose forecast horizons and enable backtesting
6. P90 bands are automatically generated and displayed

### Technical Implementation Details

#### Error Handling
- Graceful handling of empty data sets
- Validation of required columns in DataFrames
- Comprehensive exception catching with user-friendly error messages

#### Performance Optimizations
- Efficient pandas operations using vectorized calculations
- Memory-efficient handling of large time series datasets
- Lazy evaluation prevents unnecessary computations

#### Data Validation
- Automatic ordering validation: P10 ≤ P50 ≤ P90
- Timestamp alignment across different forecast horizons
- Handling of missing values and irregular time series

### Example Output Structure

**Long Format Table:**
```
                    t  horizon_min    actual  forecast_p10  forecast_p50  forecast_p90
0 2024-01-01 10:00:00            1  1024.84        996.87      1018.81      1050.59
1 2024-01-01 10:00:00            5  1024.84       1060.69      1077.52      1151.80
2 2024-01-01 10:00:00           10  1024.84       1098.46      1106.12      1208.96
```

**Residual Quantiles:**
```
  horizon_min  residual_p10  residual_p50  residual_p90
0           1        -21.94          4.30         31.78
1           5        -16.83         33.04         74.28
2          10         -7.66         58.17        102.84
```

### Testing and Validation

The implementation includes comprehensive testing via `test_p90_forecast.py`:
- ✅ Long format conversion validation
- ✅ Residual quantile computation testing  
- ✅ P90 band generation verification
- ✅ Band ordering properties validation
- ✅ Sample data export testing

### Future Enhancements

Potential extensions to the P90 forecasting system:
- **Dynamic Quantiles:** User-configurable quantile levels
- **Rolling Quantiles:** Time-varying quantile estimation
- **Multi-variate Bands:** Uncertainty bands for multiple variables
- **Probabilistic Validation:** Coverage probability analysis
- **Real-time Updates:** Live quantile updates as new data arrives

---

*This documentation covers the P90 forecasting implementation as of September 2024. For the latest updates and enhancements, refer to the code comments and version history.*