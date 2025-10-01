"""
Forecast to Operations Converter
===============================

Pure function to convert prediction-only data into the full operational schema
required by the MD shaving simulation. Synthesizes missing operational fields
deterministically to solve infinite loop issues.

Author: Energy Analytics Team
Version: 1.0
Date: October 2025
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Any


def prepare_forecast_for_md_shaving(
    forecast_df: pd.DataFrame, 
    *, 
    historical_ref: Optional[pd.DataFrame] = None,
    md_config: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Convert prediction-only data into the full operational schema required by MD shaving simulation.
    
    This function deterministically synthesizes all missing operational fields that the MD shaving
    algorithm expects, preventing infinite loops caused by data structure mismatches.
    
    Args:
        forecast_df: DataFrame containing forecast data with columns like:
            - anchor_ts (timestamp)
            - P_hat_kW (forecasted power)
            - P_now_kW (current power, optional)
            - ROC_now_kW_per_min (rate of change, optional)
            
        historical_ref: Optional historical operational data for pattern reference
        
        md_config: Optional configuration dict with:
            - md_target_kw: Maximum demand target (default: 200.0)
            - battery_capacity_kwh: Battery energy capacity (default: 100.0)
            - battery_power_kw: Battery power rating (default: 50.0)
            - md_window_hours: MD window hours as tuple (default: (14, 22))
            - holidays: Set of holiday dates (default: empty set)
    
    Returns:
        pd.DataFrame: Operational data with all required columns:
            - timestamp (datetime index)
            - Original_Demand (kW)
            - Battery_Power_kW (kW, positive=discharge, negative=charge)
            - Battery_SOC_Percent (%)
            - Net_Demand_kW (kW)
            - MD_Window (boolean)
            - Holiday (boolean)
            - Weekday (0=Monday, 6=Sunday)
            - Hour (0-23)
    """
    
    # Default configuration
    default_config = {
        'md_target_kw': 200.0,
        'battery_capacity_kwh': 100.0,
        'battery_power_kw': 50.0,
        'md_window_hours': (14, 22),  # 2 PM to 10 PM
        'holidays': set(),
        'initial_soc': 50.0,  # Starting SOC %
        'min_soc': 10.0,     # Minimum SOC %
        'max_soc': 90.0,     # Maximum SOC %
        'efficiency': 0.95,   # Round-trip efficiency
    }
    
    # Merge with user config
    config = {**default_config, **(md_config or {})}
    
    # Prepare working dataframe
    ops_df = forecast_df.copy()
    
    # 1. TIMESTAMP PROCESSING
    # Convert anchor_ts to proper datetime index
    if 'anchor_ts' in ops_df.columns:
        if ops_df['anchor_ts'].dtype == 'object':
            # String format, parse it
            ops_df['timestamp'] = pd.to_datetime(ops_df['anchor_ts'])
        else:
            ops_df['timestamp'] = ops_df['anchor_ts']
    elif ops_df.index.name in ['timestamp', 'datetime'] or isinstance(ops_df.index, pd.DatetimeIndex):
        ops_df['timestamp'] = ops_df.index
    else:
        # Generate synthetic timestamps (15-min intervals)
        start_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        ops_df['timestamp'] = [start_time + timedelta(minutes=15*i) for i in range(len(ops_df))]
    
    ops_df.set_index('timestamp', inplace=True)
    
    # 2. DEMAND SYNTHESIS
    # Use forecasted power as original demand
    if 'P_hat_kW' in ops_df.columns:
        ops_df['Original_Demand'] = ops_df['P_hat_kW']
    elif 'P_now_kW' in ops_df.columns:
        ops_df['Original_Demand'] = ops_df['P_now_kW']
    else:
        # Synthesize realistic demand pattern
        ops_df['Original_Demand'] = _synthesize_demand_pattern(ops_df.index, config)
    
    # 3. TIME-BASED FEATURES
    ops_df['Hour'] = ops_df.index.hour
    ops_df['Weekday'] = ops_df.index.weekday  # 0=Monday, 6=Sunday
    ops_df['Holiday'] = pd.Series([d.date() in config['holidays'] for d in ops_df.index], 
                                  index=ops_df.index)
    
    # 4. MD WINDOW DETECTION
    md_start_hour, md_end_hour = config['md_window_hours']
    ops_df['MD_Window'] = (
        (ops_df['Hour'] >= md_start_hour) & 
        (ops_df['Hour'] < md_end_hour) & 
        (ops_df['Weekday'] < 5) &  # Weekdays only
        (~ops_df['Holiday'])  # Non-holidays only
    )
    
    # 5. BATTERY OPERATION SYNTHESIS
    # Synthesize realistic battery operation based on MD shaving strategy
    ops_df = _synthesize_battery_operation(ops_df, config)
    
    # 6. NET DEMAND CALCULATION
    ops_df['Net_Demand_kW'] = ops_df['Original_Demand'] + ops_df['Battery_Power_kW']
    
    # 7. ADDITIONAL OPERATIONAL FIELDS
    # Add fields that MD shaving algorithm may reference
    ops_df['Peak_Period'] = ops_df['MD_Window']  # Alias for peak detection
    ops_df['Tariff_Period'] = ops_df.apply(_determine_tariff_period, axis=1)
    ops_df['MD_Target_kW'] = config['md_target_kw']
    ops_df['MD_Excess_kW'] = np.maximum(0, ops_df['Net_Demand_kW'] - config['md_target_kw'])
    
    # 8. DATA QUALITY FLAGS
    ops_df['Has_Excess'] = ops_df['MD_Excess_kW'] > 0
    ops_df['In_MD_Window'] = ops_df['MD_Window']
    ops_df['Shaving_Opportunity'] = ops_df['Has_Excess'] & ops_df['In_MD_Window']
    
    # 9. CLEAN UP AND RETURN
    # Remove temporary columns and ensure required columns exist
    required_columns = [
        'Original_Demand', 'Battery_Power_kW', 'Battery_SOC_Percent', 'Net_Demand_kW',
        'MD_Window', 'Holiday', 'Weekday', 'Hour', 'Has_Excess', 'In_MD_Window'
    ]
    
    # Ensure all required columns exist with defaults
    for col in required_columns:
        if col not in ops_df.columns:
            if col == 'Battery_SOC_Percent':
                ops_df[col] = config['initial_soc']
            elif col in ['MD_Window', 'Holiday', 'Has_Excess', 'In_MD_Window']:
                ops_df[col] = False
            elif col in ['Weekday', 'Hour']:
                ops_df[col] = 0
            else:
                ops_df[col] = 0.0
    
    return ops_df[required_columns + ['Peak_Period', 'Tariff_Period', 'MD_Target_kW', 'MD_Excess_kW', 'Shaving_Opportunity']]


def _synthesize_demand_pattern(timestamps: pd.DatetimeIndex, config: Dict[str, Any]) -> np.ndarray:
    """Synthesize realistic demand pattern based on timestamps."""
    base_demand = config['md_target_kw'] * 0.7  # 70% of MD target as base
    
    # Daily pattern (higher during business hours)
    hourly_factors = np.array([
        0.6, 0.55, 0.5, 0.5, 0.55, 0.65,  # 0-5 AM: Low
        0.75, 0.9, 1.1, 1.2, 1.25, 1.3,   # 6-11 AM: Rising
        1.35, 1.4, 1.45, 1.4, 1.35, 1.3,  # 12-5 PM: Peak
        1.2, 1.1, 0.95, 0.85, 0.75, 0.65  # 6-11 PM: Declining
    ])
    
    demand = []
    for ts in timestamps:
        hour_factor = hourly_factors[ts.hour]
        
        # Weekend reduction
        if ts.weekday() >= 5:
            hour_factor *= 0.8
        
        # Random variation (±10%)
        variation = np.random.normal(1.0, 0.1)
        
        demand.append(base_demand * hour_factor * variation)
    
    return np.array(demand)


def _synthesize_battery_operation(ops_df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Synthesize realistic battery operation for MD shaving."""
    
    battery_power_kw = config['battery_power_kw']
    battery_capacity_kwh = config['battery_capacity_kwh']
    efficiency = config['efficiency']
    min_soc = config['min_soc']
    max_soc = config['max_soc']
    md_target = config['md_target_kw']
    
    # Initialize battery state
    current_soc = config['initial_soc']
    
    battery_power = []
    battery_soc = []
    
    for idx, row in ops_df.iterrows():
        original_demand = row['Original_Demand']
        in_md_window = row['MD_Window']
        
        # MD Shaving Strategy:
        # 1. If demand > target AND in MD window: discharge to reduce peak
        # 2. If demand < target AND not in MD window: charge when excess capacity available
        # 3. Otherwise: standby (no action)
        
        if in_md_window and original_demand > md_target:
            # DISCHARGE: Reduce peak during MD window
            excess_demand = original_demand - md_target
            # Discharge up to battery limit or excess demand
            discharge_power = min(battery_power_kw, excess_demand)
            
            # Check SOC constraints
            energy_needed = discharge_power * 0.25  # 15-min interval
            if current_soc - (energy_needed / battery_capacity_kwh * 100) >= min_soc:
                battery_action = discharge_power  # Positive = discharge
                current_soc -= energy_needed / battery_capacity_kwh * 100
            else:
                battery_action = 0.0
                
        elif not in_md_window and original_demand < md_target * 0.8:
            # CHARGE: Store energy during low demand periods
            available_capacity = md_target * 0.8 - original_demand
            charge_power = min(battery_power_kw, available_capacity)
            
            # Check SOC constraints
            energy_stored = charge_power * 0.25 * efficiency  # 15-min interval with efficiency
            if current_soc + (energy_stored / battery_capacity_kwh * 100) <= max_soc:
                battery_action = -charge_power  # Negative = charge
                current_soc += energy_stored / battery_capacity_kwh * 100
            else:
                battery_action = 0.0
        else:
            # STANDBY: No action
            battery_action = 0.0
        
        # Ensure SOC stays within bounds
        current_soc = np.clip(current_soc, min_soc, max_soc)
        
        battery_power.append(battery_action)
        battery_soc.append(current_soc)
    
    ops_df['Battery_Power_kW'] = battery_power
    ops_df['Battery_SOC_Percent'] = battery_soc
    
    return ops_df


def _determine_tariff_period(row) -> str:
    """Determine tariff period based on time and day."""
    if row['Holiday'] or row['Weekday'] >= 5:
        return 'Off-Peak'
    elif 14 <= row['Hour'] < 22:  # 2 PM to 10 PM weekdays
        return 'Peak'
    else:
        return 'Off-Peak'


# Example usage function for testing
def test_forecast_conversion():
    """Test function to demonstrate the conversion."""
    
    # Create sample forecast data (what load_forecasting.py produces)
    sample_forecast = pd.DataFrame({
        'anchor_ts': pd.date_range('2025-01-01 00:00', periods=96, freq='15min'),
        'P_hat_kW': np.random.normal(150, 30, 96),
        'P_now_kW': np.random.normal(145, 25, 96),
        'ROC_now_kW_per_min': np.random.normal(0, 2, 96),
        'horizon_min': [5] * 96,
        'md_risk': [False] * 96
    })
    
    # Convert to operational format
    ops_data = prepare_forecast_for_md_shaving(
        sample_forecast,
        md_config={
            'md_target_kw': 180.0,
            'battery_capacity_kwh': 200.0,
            'battery_power_kw': 100.0
        }
    )
    
    print("Forecast data columns:", sample_forecast.columns.tolist())
    print("Operational data columns:", ops_data.columns.tolist())
    print(f"Sample operational data:\n{ops_data.head()}")
    
    return ops_data


if __name__ == "__main__":
    test_forecast_conversion()