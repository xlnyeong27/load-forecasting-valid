# 311025_README_METRICS_DISCREPANCY_DEBUGGING

## Variables of Interest and Their Functions

This document lists the key variables being checked for formula consistency between `md_shaving_solution_v2.py` and `md_shaving_solution_v2_copy.py` files to resolve metrics discrepancy issues.

## Battery Simulation Performance Metrics

### 1. **Peak Reduction (kW)**
- **Variable**: `peak_reduction_kw`
- **Function**: `_simulate_battery_operation_v2()` - lines 3364-3378
- **Calculation**: V2 daily MD window analysis using monthly targets

### 2. **Success Rate (%)**
- **Variable**: `success_rate_percent`
- **Function**: `_calculate_success_rate_from_shaving_status()` - lines 4344-4450
- **Calculation**: Daily aggregation method with MD period gating

### 3. **Total Discharge (kWh)**
- **Variable**: `total_discharge_kwh`
- **Function**: `_simulate_battery_operation_v2()`
- **Calculation**: `sum([p * interval_hours for p in battery_power if p > 0])`

### 4. **Average SOC (%)**
- **Variable**: `avg_soc_percent`
- **Function**: `_simulate_battery_operation_v2()`
- **Calculation**: `float(df_sim['Battery_SOC_Percent'].mean())`

## Conservation Mode Results (SOC-Aware Strategy)

### 5. **Conservation Periods**
- **Variable**: `conservation_periods`
- **Function**: `_simulate_battery_operation_v2()`
- **Calculation**: `int(np.sum(conservation_activated))`

### 6. **Conservation Rate (%)**
- **Variable**: `conservation_rate_percent`
- **Function**: `_simulate_battery_operation_v2()`
- **Calculation**: `(conservation_periods / total_periods * 100)`

### 7. **Min Exceedance Observed (kW)**
- **Variable**: `min_exceedance_observed_kw`
- **Function**: `_simulate_battery_operation_v2()`
- **Calculation**: `float(np.min(running_min_exceedance[running_min_exceedance != np.inf]))`

### 8. **SOC Threshold Used (%)**
- **Variable**: `soc_threshold_used`
- **Function**: `_simulate_battery_operation_v2()`
- **Calculation**: Parameter passed to function (default: 50%)

## Monthly Summary Calculations

### 9. **Max Monthly MD Excess (kW)**
- **Variable**: `max_md_excess`
- **Function**: `_generate_monthly_summary_table()` - lines 320-400
- **Calculation**: Clustering-based intermediate calculation → monthly aggregation

### 10. **Max Monthly Required Energy (kWh)**
- **Variable**: `max_energy_required`
- **Function**: `_generate_monthly_summary_table()` - lines 320-400
- **Calculation**: Clustering-based intermediate calculation → monthly aggregation

## Battery Quantity Recommendations

### 11. **Power-Based Quantity**
- **Variable**: `qty_for_power_rounded`
- **Function**: `_render_battery_quantity_recommendation()`
- **Calculation**: `int(np.ceil(max_power_shaving_required / battery_power_kw))`

### 12. **Energy-Based Quantity**
- **Variable**: `qty_for_energy_rounded`
- **Function**: `_render_battery_quantity_recommendation()`
- **Calculation**: `int(np.ceil(recommended_energy_capacity / battery_energy_kwh / 0.9 / 0.93))`

### 13. **Recommended Quantity**
- **Variable**: `recommended_qty`
- **Function**: `_render_battery_quantity_recommendation()`
- **Calculation**: `max(qty_for_power_rounded, qty_for_energy_rounded)`

---

## Debugging Notes

### Current Issues
- Values reported in UI don't match expected calculations from copy file
- Need to verify formula consistency between both files
- Monthly summary calculations showing discrepancies

### Investigation Areas
1. Monthly Summary: Direct vs clustering-based approaches
2. Success Rate: Daily vs interval-based methodologies  
3. Conservation Mode: Simple vs comprehensive cascade workflow
4. Peak Reduction: Error handling differences

### Status
- **Definitely Different**: Monthly Summary, Success Rate calculation methods
- **Identical**: Peak Reduction V2 methodology, Total Discharge, Average SOC
- **Needs Validation**: Conservation Mode implementation details

---

*Created: October 31, 2025*  
*Purpose: Debug metrics discrepancy between md_shaving_solution_v2.py versions*