"""
MD Shaving Solution V2 - Enhanced MD Shaving Analysis
=====================================================

This module provides next-generation Maximum Demand (MD) shaving analysis with:
- Monthly-based target calculation with dynamic user settings
- Battery database integration with vendor specifications
- Enhanced timeline visualization with peak events
- Interactive battery capacity selection interface

Author: Enhanced MD Shaving Team
Version: 2.0
Date: August 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import io

# Import V1 components for reuse
from md_shaving_solution import (
    read_uploaded_file,
    _configure_data_inputs,
    _process_dataframe,
    _configure_tariff_selection,
    create_conditional_demand_line_with_peak_logic,
    _detect_peak_events,
    _detect_peak_events_tou_aware,
    _display_battery_simulation_chart,
    _simulate_battery_operation,
    _get_tariff_description
)
from tariffs.peak_logic import (
    is_peak_rp4, 
    get_period_classification,
    get_malaysia_holidays,
    detect_holidays_from_data
)


def _infer_interval_hours(datetime_index, fallback=0.25):
    """
    Infer sampling interval from datetime index using mode of timestamp differences.
    
    Args:
        datetime_index: pandas DatetimeIndex
        fallback: fallback interval in hours (default: 0.25 = 15 minutes)
        
    Returns:
        float: Interval in hours
    """
    try:
        if len(datetime_index) > 1:
            diffs = datetime_index.to_series().diff().dropna()
            if len(diffs) > 0 and not diffs.mode().empty:
                interval_hours = diffs.mode()[0].total_seconds() / 3600
                return interval_hours
    except Exception:
        pass
    return fallback


def _get_dynamic_interval_hours(df_or_index):
    """
    Centralized function to get dynamic interval hours with consistent fallback logic.
    
    This ensures all energy conversion calculations use the same detected interval
    throughout V2, preventing inconsistencies between hardcoded and dynamic intervals.
    
    Args:
        df_or_index: DataFrame or DatetimeIndex to detect interval from
        
    Returns:
        float: Detected interval in hours
    """
    # First try to get from session state (already detected and stored)
    try:
        import streamlit as st
        if hasattr(st.session_state, 'data_interval_hours'):
            return st.session_state.data_interval_hours
    except (ImportError, AttributeError):
        pass
    
    # Fallback to detection from provided data
    if hasattr(df_or_index, 'index'):
        return _infer_interval_hours(df_or_index.index, fallback=0.25)
    else:
        return _infer_interval_hours(df_or_index, fallback=0.25)


def infer_base_interval(series):
    """
    Infer the base interval of the input series using native timestamp frequency.
    
    Args:
        series: pd.Series or pd.DataFrame with datetime index
        
    Returns:
        pd.Timedelta: Base interval of the series
        
    Raises:
        ValueError: If interval is irregular or outside valid range (1-60 minutes)
    """
    import numpy as np
    
    # Get the datetime index
    if hasattr(series, 'index'):
        dt_index = series.index
    else:
        dt_index = series
        
    if len(dt_index) < 2:
        raise ValueError("Need at least 2 timestamps to infer interval")
    
    try:
        # Try pandas infer_freq first
        inferred_freq = pd.infer_freq(dt_index)
        if inferred_freq and inferred_freq != 'T':  # 'T' is sometimes problematic
            try:
                base_interval = pd.Timedelta(inferred_freq)
            except (ValueError, TypeError):
                inferred_freq = None
        else:
            inferred_freq = None
            
        if not inferred_freq:
            # Fallback: median of timestamp differences
            diffs = np.diff(dt_index.values).astype('timedelta64[s]')
            median_seconds = np.median(diffs.astype(float))
            base_interval = pd.Timedelta(seconds=median_seconds)
        
        # Validate interval is between 1 minute and 60 minutes
        min_interval = pd.Timedelta(minutes=1)
        max_interval = pd.Timedelta(minutes=60)
        
        if base_interval < min_interval or base_interval > max_interval:
            raise ValueError(
                f"Base interval {base_interval} is outside valid range (1-60 minutes). "
                f"Please ensure your data has regular intervals between 1 minute and 1 hour."
            )
            
        return base_interval
        
    except Exception as e:
        # Check for irregular intervals
        if len(dt_index) > 2:
            diffs = np.diff(dt_index.values).astype('timedelta64[s]')
            diff_std = np.std(diffs.astype(float))
            diff_mean = np.mean(diffs.astype(float))
            
            if diff_std > diff_mean * 0.1:  # More than 10% variation
                raise ValueError(
                    f"Irregular time intervals detected (std/mean ratio: {diff_std/diff_mean:.2f}). "
                    f"ROC forecasting requires regular time intervals. "
                    f"Consider resampling your data to a fixed frequency."
                )
        
        raise ValueError(f"Could not infer base interval: {str(e)}")


def generate_forecast_horizons(base_interval):
    """
    Generate forecast horizons based on the base interval of the input series.
    
    Rules:
    - If base ≤ 5min, use multipliers [1, 10, 30]  
    - If base > 5min, use multipliers [1, 2, 3, 4]
    
    Args:
        base_interval: pd.Timedelta representing the base interval
        
    Returns:
        list: List of pd.Timedelta objects representing forecast horizons
    """
    # Convert to minutes for comparison
    base_minutes = base_interval.total_seconds() / 60
    
    if base_minutes <= 5:
        multipliers = [1, 10, 30]
    else:
        multipliers = [1, 2, 3, 4]
    
    # Generate horizons as Timedelta objects
    horizons = [base_interval * multiplier for multiplier in multipliers]
    
    return horizons


def get_adaptive_forecast_horizons(series):
    """
    Convenience function to infer base interval and generate appropriate horizons.
    
    Args:
        series: pd.Series or pd.DataFrame with datetime index
        
    Returns:
        tuple: (base_interval, horizons_list, horizons_minutes_list)
    """
    base_interval = infer_base_interval(series)
    horizons = generate_forecast_horizons(base_interval)
    
    # Also return horizon values in minutes for backwards compatibility
    horizons_minutes = [int(h.total_seconds() / 60) for h in horizons]
    
    return base_interval, horizons, horizons_minutes


def _calculate_tariff_specific_monthly_peaks(df, power_col, selected_tariff, holidays):
    """
    Calculate monthly peak demands based on tariff type:
    - General Tariff: Uses 24/7 peak demand (highest demand anytime)
    - TOU Tariff: Uses peak period demand only (2PM-10PM weekdays)
    
    Args:
        df: DataFrame with power data
        power_col: Column name containing power values
        selected_tariff: Selected tariff configuration
        holidays: Set of holiday dates
    
    Returns:
        tuple: (monthly_general_peaks, monthly_tou_peaks, tariff_type)
    """
    # Determine tariff type
    tariff_type = 'General'  # Default
    if selected_tariff:
        tariff_name = selected_tariff.get('Tariff', '').lower()
        tariff_type_field = selected_tariff.get('Type', '').lower()
        
        # Check if it's a TOU tariff
        if 'tou' in tariff_name or 'tou' in tariff_type_field or tariff_type_field == 'tou':
            tariff_type = 'TOU'
    
    # Calculate monthly peaks
    df_monthly = df.copy()
    df_monthly['Month'] = df_monthly.index.to_period('M')
    
    # General peaks (24/7 maximum demand)
    monthly_general_peaks = df_monthly.groupby('Month')[power_col].max()
    
    # TOU peaks (peak period maximum demand only - 2PM-10PM weekdays)
    monthly_tou_peaks = {}
    
    for month_period in monthly_general_peaks.index:
        month_start = month_period.start_time
        month_end = month_period.end_time
        month_mask = (df_monthly.index >= month_start) & (df_monthly.index <= month_end)
        month_data = df_monthly[month_mask]
        
        if not month_data.empty:
            # Filter for TOU peak periods only
            tou_peak_data = []
            
            for timestamp in month_data.index:
                power_value = month_data.loc[timestamp, power_col]
                if is_peak_rp4(timestamp, holidays if holidays else set()):
                    tou_peak_data.append(power_value)
            
            if tou_peak_data:
                monthly_tou_peaks[month_period] = max(tou_peak_data)
            else:
                monthly_tou_peaks[month_period] = monthly_general_peaks[month_period]
        else:
            monthly_tou_peaks[month_period] = 0
    
    monthly_tou_peaks = pd.Series(monthly_tou_peaks)
    
    return monthly_general_peaks, monthly_tou_peaks, tariff_type


def _calculate_monthly_targets_v2(df, power_col, selected_tariff, holidays, target_method, shave_percent, target_percent, target_manual_kw):
    """
    Calculate monthly targets based on tariff-specific peak demands.
    
    Returns:
        tuple: (monthly_targets, reference_peaks, tariff_type, target_description)
    """
    # Get tariff-specific monthly peaks
    monthly_general_peaks, monthly_tou_peaks, tariff_type = _calculate_tariff_specific_monthly_peaks(
        df, power_col, selected_tariff, holidays
    )
    
    # Select appropriate reference peaks based on tariff type
    if tariff_type == 'TOU':
        reference_peaks = monthly_tou_peaks
        peak_description = "TOU Peak Period (2PM-10PM weekdays)"
    else:
        reference_peaks = monthly_general_peaks
        peak_description = "General (24/7)"
    
    # Calculate targets based on reference peaks
    if target_method == "Manual Target (kW)":
        # For manual target, use the same value for all months
        monthly_targets = pd.Series(index=reference_peaks.index, data=target_manual_kw)
        target_description = f"{target_manual_kw:.0f} kW manual target ({peak_description})"
    elif target_method == "Percentage to Shave":
        # Calculate target as percentage reduction from each month's reference peak
        target_multiplier = 1 - (shave_percent / 100)
        monthly_targets = reference_peaks * target_multiplier
        target_description = f"{shave_percent}% shaving from {peak_description}"
    else:  # Percentage of Current Max
        # Calculate target as percentage of each month's reference peak
        target_multiplier = target_percent / 100
        monthly_targets = reference_peaks * target_multiplier
        target_description = f"{target_percent}% of {peak_description}"
    
    return monthly_targets, reference_peaks, tariff_type, target_description


def _generate_clustering_summary_table(all_monthly_events, selected_tariff, holidays):
    """
    Generate date-based clustering summary table for Section B2.
    
    Args:
        all_monthly_events: List of peak events from peak events detection
        selected_tariff: Selected tariff configuration
        holidays: Set of holiday dates
        
    Returns:
        pd.DataFrame: Summary table with columns: Date, Total Peak Events, General/TOU MD Excess, 
                     General/TOU Total Energy Required, Cost Impact
    """
    if not all_monthly_events or len(all_monthly_events) == 0:
        return pd.DataFrame()
    
    # Group events by date
    daily_events = {}
    for event in all_monthly_events:
        event_date = event.get('Start Date')
        if event_date not in daily_events:
            daily_events[event_date] = []
        daily_events[event_date].append(event)
    
    # Determine tariff type for MD cost calculation
    tariff_type = 'General'  # Default
    if selected_tariff:
        tariff_name = selected_tariff.get('Tariff', '').lower()
        tariff_type_field = selected_tariff.get('Type', '').lower()
        
        # Check if it's a TOU tariff
        if 'tou' in tariff_name or 'tou' in tariff_type_field or tariff_type_field == 'tou':
            tariff_type = 'TOU'
    
    # Get MD rate from tariff for cost calculation
    md_rate_rm_per_kw = 0
    if selected_tariff and isinstance(selected_tariff, dict):
        rates = selected_tariff.get('Rates', {})
        md_rate_rm_per_kw = rates.get('Capacity Rate', 0) + rates.get('Network Rate', 0)
    
    # Create summary data
    summary_data = []
    for date, events in daily_events.items():
        # Count total events for this date
        total_events = len(events)
        
        # Calculate MD excess values based on tariff type
        if tariff_type == 'TOU':
            # For TOU: Use TOU-specific values
            md_excess_values = [event.get('TOU Excess (kW)', 0) or 0 for event in events]
            energy_required_values = [event.get('TOU Required Energy (kWh)', 0) or 0 for event in events]
            max_md_excess = max(md_excess_values) if md_excess_values else 0
        else:
            # For General: Use General values (24/7 MD impact)
            md_excess_values = [event.get('General Excess (kW)', 0) or 0 for event in events]
            energy_required_values = [event.get('General Required Energy (kWh)', 0) or 0 for event in events]
            max_md_excess = max(md_excess_values) if md_excess_values else 0
        
        # Sum total energy required for the date
        total_energy_required = sum(energy_required_values)
        
        # Calculate cost impact using the maximum MD excess for the date
        # This follows the MD charging methodology where only the highest peak matters
        cost_impact_rm = max_md_excess * md_rate_rm_per_kw if max_md_excess > 0 and md_rate_rm_per_kw > 0 else 0
        
        summary_data.append({
            'Date': date,
            'Total Peak Events (count)': total_events,
            f'{tariff_type} MD Excess (Max kW)': round(max_md_excess, 2),
            f'{tariff_type} Total Energy Required (sum kWh)': round(total_energy_required, 2),
            'Cost Impact (RM/month)': round(cost_impact_rm, 2)
        })
    
    # Create DataFrame and sort by date
    df_summary = pd.DataFrame(summary_data)
    if not df_summary.empty:
        df_summary = df_summary.sort_values('Date')
    
    return df_summary


def _generate_monthly_summary_table(all_monthly_events, selected_tariff, holidays):
    """
    Generate monthly summary table for Section B2.
    
    CORRECTED METHODOLOGY: Uses daily clustering as intermediary step to match reference calculations.
    
    Args:
        all_monthly_events: List of peak events from peak events detection
        selected_tariff: Selected tariff configuration
        holidays: Set of holiday dates
        
    Returns:
        pd.DataFrame: Summary table with columns: Month, General/TOU MD Excess (Max kW), 
                     General/TOU Required Energy (Max kWh)
    """
    if not all_monthly_events or len(all_monthly_events) == 0:
        return pd.DataFrame()
    
    # STEP 1: Generate daily clustering summary first (intermediary calculation)
    daily_clustering_df = _generate_clustering_summary_table(all_monthly_events, selected_tariff, holidays)
    
    if daily_clustering_df.empty:
        return pd.DataFrame()
    
    # Determine tariff type for column naming
    tariff_type = 'General'  # Default
    if selected_tariff:
        tariff_name = selected_tariff.get('Tariff', '').lower()
        tariff_type_field = selected_tariff.get('Type', '').lower()
        
        # Check if it's a TOU tariff
        if 'tou' in tariff_name or 'tou' in tariff_type_field or tariff_type_field == 'tou':
            tariff_type = 'TOU'
    
    # STEP 2: Group daily results by month and take maximums
    # Add month column to daily clustering data
    daily_clustering_df['Month'] = pd.to_datetime(daily_clustering_df['Date']).dt.strftime('%Y-%m')
    
    # Define column names based on tariff type
    md_excess_col = f'{tariff_type} MD Excess (Max kW)'
    energy_required_col = f'{tariff_type} Total Energy Required (sum kWh)'
    
    # Group by month and calculate maximums from daily values
    monthly_summary = daily_clustering_df.groupby('Month').agg({
        md_excess_col: 'max',  # Maximum daily MD excess becomes monthly MD excess
        energy_required_col: 'max'  # Maximum daily energy required becomes monthly energy required
    }).reset_index()
    
    # Rename energy column to match expected format
    monthly_summary = monthly_summary.rename(columns={
        energy_required_col: f'{tariff_type} Required Energy (Max kWh)'
    })
    
    # Round values for display
    monthly_summary[md_excess_col] = monthly_summary[md_excess_col].round(2)
    monthly_summary[f'{tariff_type} Required Energy (Max kWh)'] = monthly_summary[f'{tariff_type} Required Energy (Max kWh)'].round(2)
    
    # Sort by month
    monthly_summary = monthly_summary.sort_values('Month')
    
    return monthly_summary


def build_daily_simulator_structure(df, threshold_kw, clusters_df, selected_tariff=None):
    """
    Build day-level structure for battery dispatch simulation.
    
    Args:
        df: Cleaned DataFrame with DateTimeIndex and 'kW' column
        threshold_kw: Power threshold for shaving analysis
        clusters_df: DataFrame from cluster_peak_events() function
        selected_tariff: Tariff configuration dict (optional)
    
    Returns:
        dict: days_struct[date] = {timeline, kW, above_threshold_kW, clusters, recharge_windows, dt_hours}
    """
    if df.empty or clusters_df.empty:
        return {}
        
    # Infer sampling interval from DataFrame index
    dt_hours = (df.index[1] - df.index[0]).total_seconds() / 3600 if len(df) > 1 else 0.5
    
    # Get TOU configuration from selected tariff
    md_hours = (14, 22)  # Default 2PM-10PM
    working_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    tou_windows = {'off_peak': [(0, 8), (22, 24)], 'peak': [(14, 22)]}  # Default TOU
    charge_allowed_windows = [(0, 8), (22, 24)]  # Default: off-peak charging only
    max_site_charge_kw = None  # No default grid limit
    
    if selected_tariff:
        # Extract TOU configuration from tariff if available
        tariff_config = selected_tariff.get('config', {})
        if 'md_hours' in tariff_config:
            md_hours = tariff_config['md_hours']
        if 'working_days' in tariff_config:
            working_days = tariff_config['working_days']
        if 'tou_windows' in tariff_config:
            tou_windows = tariff_config['tou_windows']
        if 'charge_windows' in tariff_config:
            charge_allowed_windows = tariff_config['charge_windows']
        if 'max_site_charge_kw' in tariff_config:
            max_site_charge_kw = tariff_config['max_site_charge_kw']
    
    # Get unique dates from clusters and DataFrame
    cluster_dates = set()
    for _, cluster in clusters_df.iterrows():
        cluster_dates.add(cluster['cluster_start'].date())
        cluster_dates.add(cluster['cluster_end'].date())
    
    # Also include dates from DataFrame to ensure complete coverage
    df_dates = set(df.index.date)
    all_dates = cluster_dates.union(df_dates)
    
    days_struct = {}
    
    for date in sorted(all_dates):
        # Define day timeline (handle overnight MD window if needed)
        day_start = pd.Timestamp.combine(date, pd.Timestamp.min.time())
        day_end = day_start + pd.Timedelta(days=1)
        
        # Extract day's timeline from DataFrame
        day_mask = (df.index >= day_start) & (df.index < day_end)
        day_df = df[day_mask].copy()
        
        if day_df.empty:
            continue
            
        timeline = day_df.index
        kW_series = day_df['kW']
        above_threshold_kW = pd.Series(
            data=np.maximum(kW_series.values - threshold_kw, 0),
            index=timeline,
            name='above_threshold_kW'
        )
        
        # Find clusters intersecting this day
        day_clusters = []
        for _, cluster in clusters_df.iterrows():
            cluster_start = cluster['cluster_start']
            cluster_end = cluster['cluster_end']
            
            # Check if cluster intersects with this day
            if (cluster_start.date() == date or cluster_end.date() == date or
                (cluster_start.date() < date < cluster_end.date())):
                
                # Find timeline slice for this cluster
                cluster_mask = (timeline >= cluster_start) & (timeline <= cluster_end)
                if cluster_mask.any():
                    start_idx = np.where(cluster_mask)[0][0] if cluster_mask.any() else None
                    end_idx = np.where(cluster_mask)[0][-1] + 1 if cluster_mask.any() else None
                    
                    day_clusters.append({
                        'cluster_id': int(cluster['cluster_id']),
                        'start': cluster_start,
                        'end': cluster_end,
                        'duration_hr': float(cluster['cluster_duration_hr']),
                        'num_events': int(cluster['num_events_in_cluster']),
                        'peak_abs_kw_in_cluster': float(cluster['peak_abs_kw_in_cluster']),
                        'total_energy_above_threshold_kwh': float(cluster['total_energy_above_threshold_kwh']),
                        'slice': (start_idx, end_idx) if start_idx is not None else None
                    })
        
        # Sort clusters by start time
        day_clusters.sort(key=lambda x: x['start'])
        
        # Generate recharge windows between clusters
        recharge_windows = []
        
        if len(day_clusters) > 1:
            for i in range(len(day_clusters) - 1):
                current_cluster = day_clusters[i]
                next_cluster = day_clusters[i + 1]
                
                gap_start = current_cluster['end']
                gap_end = next_cluster['start']
                gap_duration = (gap_end - gap_start).total_seconds() / 3600
                
                if gap_duration > 0:
                    # Determine TOU label for this time period
                    gap_hour = gap_start.hour
                    gap_day = gap_start.strftime('%A')
                    
                    tou_label = 'off_peak'  # Default
                    for label, windows in tou_windows.items():
                        for window_start, window_end in windows:
                            if window_start <= gap_hour < window_end:
                                tou_label = label
                                break
                    
                    # Check if charging is allowed during this window
                    is_charging_allowed = False
                    for charge_start, charge_end in charge_allowed_windows:
                        if charge_start <= gap_hour < charge_end:
                            is_charging_allowed = True
                            break
                    
                    # Additional check: only allow charging on working days if specified
                    if gap_day not in working_days and 'working_days_only_charge' in (selected_tariff or {}):
                        is_charging_allowed = False
                    
                    recharge_windows.append({
                        'start': gap_start,
                        'end': gap_end,
                        'duration_hr': gap_duration,
                        'tou_label': tou_label,
                        'is_charging_allowed': is_charging_allowed,
                        'max_site_charge_kw': max_site_charge_kw
                    })
        
        # Add recharge windows at beginning and end of day if needed
        if day_clusters:
            # Before first cluster
            first_cluster = day_clusters[0]
            if first_cluster['start'] > timeline[0]:
                gap_start = timeline[0]
                gap_end = first_cluster['start']
                gap_duration = (gap_end - gap_start).total_seconds() / 3600
                
                gap_hour = gap_start.hour
                tou_label = 'off_peak'
                for label, windows in tou_windows.items():
                    for window_start, window_end in windows:
                        if window_start <= gap_hour < window_end:
                            tou_label = label
                            break
                
                is_charging_allowed = any(
                    charge_start <= gap_hour < charge_end 
                    for charge_start, charge_end in charge_allowed_windows
                )
                
                recharge_windows.insert(0, {
                    'start': gap_start,
                    'end': gap_end,
                    'duration_hr': gap_duration,
                    'tou_label': tou_label,
                    'is_charging_allowed': is_charging_allowed,
                    'max_site_charge_kw': max_site_charge_kw
                })
            
            # After last cluster
            last_cluster = day_clusters[-1]
            if last_cluster['end'] < timeline[-1]:
                gap_start = last_cluster['end']
                gap_end = timeline[-1]
                gap_duration = (gap_end - gap_start).total_seconds() / 3600
                
                gap_hour = gap_start.hour
                tou_label = 'off_peak'
                for label, windows in tou_windows.items():
                    for window_start, window_end in windows:
                        if window_start <= gap_hour < window_end:
                            tou_label = label
                            break
                
                is_charging_allowed = any(
                    charge_start <= gap_hour < charge_end 
                    for charge_start, charge_end in charge_allowed_windows
                )
                
                recharge_windows.append({
                    'start': gap_start,
                    'end': gap_end,
                    'duration_hr': gap_duration,
                    'tou_label': tou_label,
                    'is_charging_allowed': is_charging_allowed,
                    'max_site_charge_kw': max_site_charge_kw
                })
        
        # Store day structure
        days_struct[date] = {
            'timeline': timeline,
            'kW': kW_series,
            'above_threshold_kW': above_threshold_kW,
            'clusters': day_clusters,
            'recharge_windows': recharge_windows,
            'dt_hours': dt_hours
            }
        
        
        def _calculate_c_rate_limited_power_simple(current_soc_percent, max_power_rating_kw, battery_capacity_kwh, c_rate=1.0):
            """
            Simple C-rate power limitation for charging/discharging.
            
            Args:
                current_soc_percent: Current state of charge percentage
                max_power_rating_kw: Battery's rated power
                battery_capacity_kwh: Battery's energy capacity
                c_rate: Battery's C-rate (default 1.0C)
                
            Returns:
                Dictionary with power limits
            """
            # Calculate C-rate based power limits
            c_rate_power_limit = battery_capacity_kwh * c_rate
            
            # SOC-based derating (power reduces at extreme SOC levels)
            if current_soc_percent > 95:
                soc_factor = 0.8  # Reduce power at high SOC
            elif current_soc_percent < 5:
                soc_factor = 0.7  # Reduce power at low SOC (5% minimum safety limit)
            else:
                soc_factor = 1.0  # Full power in normal SOC range
            
            # Final power limit is minimum of C-rate limit and rated power
            effective_max_discharge_kw = min(c_rate_power_limit, max_power_rating_kw) * soc_factor
            effective_max_charge_kw = min(c_rate_power_limit, max_power_rating_kw) * soc_factor * 0.8  # Charging typically slower
            
            return {
                'max_discharge_power_kw': effective_max_discharge_kw,
                'max_charge_power_kw': effective_max_charge_kw,
                'c_rate_power_limit_kw': c_rate_power_limit,
                'soc_derating_factor': soc_factor,
                'limiting_factor': 'C-rate' if c_rate_power_limit < max_power_rating_kw else 'Power Rating'
            }
    
    return days_struct


def validate_and_clean_data(series, power_col=None, fill_method='drop'):
    """
    Validate and clean input data by handling NaN values and "null" strings.
    
    Args:
        series: pandas Series or DataFrame with datetime index and power values
        power_col: column name if series is a DataFrame (optional if Series)
        fill_method: method to handle NaN values ('drop', 'interpolate', 'forward', 'backward', 'raise')
        
    Returns:
        tuple: (cleaned_data, validation_report)
        
    Raises:
        ValueError: If data quality is insufficient or fill_method='raise' and NaNs found
    """
    # Handle both Series and DataFrame inputs
    if isinstance(series, pd.Series):
        data = series.copy()
        col_name = power_col or 'Power'
    else:
        if power_col is None:
            # Use first numeric column if power_col not specified
            numeric_cols = series.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                raise ValueError("No numeric columns found in DataFrame")
            power_col = numeric_cols[0]
        data = series[power_col].copy()
        col_name = power_col
    
    # Convert string "null" values to actual NaN
    if data.dtype == 'object':
        data = data.replace(['null', 'NULL', 'Null', 'nan', 'NaN', 'NAN', ''], np.nan)
        # Try to convert to numeric
        data = pd.to_numeric(data, errors='coerce')
    
    # Validation report
    total_points = len(data)
    nan_count = data.isna().sum()
    nan_percentage = (nan_count / total_points) * 100 if total_points > 0 else 0
    
    validation_report = {
        'total_points': total_points,
        'nan_count': nan_count,
        'nan_percentage': nan_percentage,
        'fill_method_used': fill_method,
        'data_quality': 'good' if nan_percentage < 5 else 'fair' if nan_percentage < 15 else 'poor'
    }
    
    # Check data quality
    if total_points < 2:
        raise ValueError("Need at least 2 data points for analysis")
    
    if nan_percentage > 50:
        raise ValueError(f"Too many missing values ({nan_percentage:.1f}%). Data quality insufficient for analysis.")
    
    # Handle NaN values based on fill_method - Default to 'drop' for peak/target calculations
    if nan_count > 0:
        if fill_method == 'raise':
            raise ValueError(f"Found {nan_count} NaN values in {col_name} column. Clean data required.")
        
        elif fill_method == 'drop':
            # Drop NaN rows completely (preserving datetime index alignment)
            data_filled = data.dropna()
            validation_report['points_after_drop'] = len(data_filled)
            validation_report['fill_method_used'] = 'drop'
            if len(data_filled) < 2:
                raise ValueError("Insufficient data points remaining after dropping NaNs")
        
        elif fill_method == 'interpolate':
            # Use linear interpolation with limit to avoid excessive extrapolation
            data_filled = data.interpolate(method='linear', limit=5)
            # Fill any remaining NaNs at edges with forward/backward fill
            data_filled = data_filled.fillna(method='ffill').fillna(method='bfill')
            validation_report['fill_method_used'] = 'interpolate + edge_fill'
            
        elif fill_method == 'forward':
            data_filled = data.fillna(method='ffill')
            
        elif fill_method == 'backward':
            data_filled = data.fillna(method='bfill')
            
        else:
            raise ValueError(f"Unknown fill_method: {fill_method}")
        
        # Check if fill was successful
        remaining_nans = data_filled.isna().sum()
        if remaining_nans > 0 and fill_method != 'drop':
            validation_report['remaining_nans'] = remaining_nans
            validation_report['data_quality'] = 'poor'
    else:
        data_filled = data
        validation_report['fill_method_used'] = 'none_needed'
    
    return data_filled, validation_report


def _calculate_roc_from_series(series, power_col=None):
    """
    Calculate Rate of Change (ROC) in kW per minute from a pandas Series or DataFrame.
    Reuses logic from load_forecasting.py _calculate_roc function.
    Now includes comprehensive NaN handling and data validation.
    
    Args:
        series: pandas Series with datetime index and power values, or DataFrame with power column
        power_col: column name if series is a DataFrame (optional if Series)
    
    Returns:
        pandas DataFrame with Timestamp, Power (kW), and ROC (kW/min) columns
    """
    # Validate and clean input data using DROP method for data integrity
    try:
        cleaned_data, validation_report = validate_and_clean_data(series, power_col, fill_method='drop')
    except Exception as e:
        raise ValueError(f"Data validation failed: {str(e)}")
    
    # Handle both Series and DataFrame inputs with cleaned data
    if isinstance(series, pd.Series):
        df_processed = pd.DataFrame({power_col or 'Power': cleaned_data})
        power_col = power_col or 'Power'
    else:
        # Use only the valid timestamps from cleaned data
        df_processed = series.loc[cleaned_data.index].copy()
        df_processed[power_col] = cleaned_data  # Use cleaned data
        if power_col is None:
            numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                raise ValueError("No numeric columns found in DataFrame")
            power_col = numeric_cols[0]
    
    # Ensure datetime index
    if not isinstance(df_processed.index, pd.DatetimeIndex):
        raise ValueError("Series/DataFrame must have datetime index")
    
    df_roc = df_processed.copy()
    
    # Calculate time differences in minutes (reusing load_forecasting.py logic)
    df_roc['time_diff_min'] = df_roc.index.to_series().diff().dt.total_seconds() / 60
    
    # Calculate power differences
    df_roc['power_diff_kw'] = df_roc[power_col].diff()
    
    # Calculate ROC (kW per minute)
    df_roc['roc_kw_per_min'] = df_roc['power_diff_kw'] / df_roc['time_diff_min']
    
    # Create clean output dataframe (matching load_forecasting.py format)
    roc_df = pd.DataFrame({
        'Timestamp': df_roc.index,
        'Power (kW)': df_roc[power_col],
        'ROC (kW/min)': df_roc['roc_kw_per_min']
    })
    
    # Add validation report as metadata
    roc_df._validation_report = validation_report
    
    return roc_df


def roc_forecast(series, horizon=1, power_col=None):
    """
    Generate horizon-minute ahead forecasts for all data points using Rate of Change (ROC) method.
    Reuses existing ROC calculation logic from load_forecasting.py.
    
    This function generates forecasts for ALL points in the series, not rolling forecasts.
    For each point t, it forecasts the value at t+horizon using the ROC at time t.
    
    Args:
        series: pandas Series with datetime index and power values, or DataFrame with power column
        horizon: forecast horizon in minutes (default: 1)
        power_col: column name if series is a DataFrame (optional if Series)
    
    Returns:
        pandas DataFrame with columns:
        - Timestamp: original timestamp
        - Power_Actual (kW): actual power at timestamp
        - ROC (kW/min): rate of change at timestamp
        - Forecast_Timestamp: timestamp + horizon
        - Power_Forecast (kW): forecasted power value
        - Forecast_Available: boolean indicating if forecast could be made
    
    Examples:
        # Using Series
        forecast_df = roc_forecast(power_series, horizon=5)
        
        # Using DataFrame
        forecast_df = roc_forecast(df, horizon=1, power_col='Power_kW')
    """
    
    # Validate inputs
    if not isinstance(series, (pd.Series, pd.DataFrame)):
        raise ValueError("Input must be a pandas Series or DataFrame")
    
    if horizon <= 0:
        raise ValueError("Horizon must be positive")
    
    # Calculate ROC using existing logic
    try:
        roc_df = _calculate_roc_from_series(series, power_col)
    except Exception as e:
        raise ValueError(f"Error calculating ROC: {str(e)}")
    
    # Generate forecasts for all points
    forecasts = []
    
    for idx, row in roc_df.iterrows():
        timestamp = row['Timestamp']
        power_actual = row['Power (kW)']
        roc_value = row['ROC (kW/min)']
        
        # Calculate forecast timestamp
        forecast_timestamp = timestamp + pd.Timedelta(minutes=horizon)
        
        # Generate forecast using ROC method (reusing load_forecasting.py logic)
        # P_hat = P_now + ROC_now * h
        if pd.isna(roc_value):
            # Cannot make forecast without ROC (typically first data point)
            power_forecast = np.nan
            forecast_available = False
        else:
            power_forecast = power_actual + roc_value * horizon
            forecast_available = True
        
        forecasts.append({
            'Timestamp': timestamp,
            'Power_Actual (kW)': power_actual,
            'ROC (kW/min)': roc_value,
            'Forecast_Timestamp': forecast_timestamp,
            'Power_Forecast (kW)': power_forecast,
            'Forecast_Available': forecast_available
        })
    
    forecast_df = pd.DataFrame(forecasts)
    
    return forecast_df


def roc_forecast_with_validation(series, horizon=1, power_col=None, return_metrics=False):
    """
    Generate ROC forecasts and validate against actual future values where available.
    Extended version of roc_forecast that includes validation metrics.
    
    Args:
        series: pandas Series with datetime index and power values, or DataFrame with power column
        horizon: forecast horizon in minutes (default: 1)
        power_col: column name if series is a DataFrame (optional if Series)
        return_metrics: if True, return summary metrics along with forecast DataFrame
    
    Returns:
        pandas DataFrame with forecast + validation columns, optionally metrics dict
    """
    
    # Generate basic forecasts
    forecast_df = roc_forecast(series, horizon, power_col)
    
    # Prepare original data for validation
    if isinstance(series, pd.Series):
        original_data = pd.DataFrame({'Power': series})
        power_col = power_col or 'Power'
    else:
        original_data = series.copy()
        if power_col is None:
            numeric_cols = original_data.select_dtypes(include=[np.number]).columns
            power_col = numeric_cols[0]
    
    # Add validation columns
    validation_results = []
    
    for idx, row in forecast_df.iterrows():
        result = row.to_dict()
        
        if row['Forecast_Available']:
            forecast_timestamp = row['Forecast_Timestamp']
            
            # Find actual value at forecast timestamp
            try:
                # Try exact match first
                if forecast_timestamp in original_data.index:
                    actual_future = original_data.loc[forecast_timestamp, power_col]
                    validation_available = True
                else:
                    # Try nearest time match
                    time_diffs = (original_data.index - forecast_timestamp).abs()
                    nearest_idx = time_diffs.idxmin()
                    
                    # Only use if within reasonable tolerance (e.g., half the horizon)
                    tolerance = pd.Timedelta(minutes=horizon/2)
                    if time_diffs.min() <= tolerance:
                        actual_future = original_data.loc[nearest_idx, power_col]
                        validation_available = True
                    else:
                        actual_future = np.nan
                        validation_available = False
            except:
                actual_future = np.nan
                validation_available = False
            
            # Calculate validation metrics
            if validation_available and not pd.isna(actual_future):
                forecast_error = row['Power_Forecast (kW)'] - actual_future
                abs_error = abs(forecast_error)
                if actual_future != 0:
                    pct_error = (abs_error / abs(actual_future)) * 100
                else:
                    pct_error = np.nan
            else:
                actual_future = np.nan
                forecast_error = np.nan
                abs_error = np.nan
                pct_error = np.nan
                validation_available = False
        else:
            # No forecast available
            actual_future = np.nan
            forecast_error = np.nan
            abs_error = np.nan
            pct_error = np.nan
            validation_available = False
        
        # Add validation columns
        result.update({
            'Power_Actual_Future (kW)': actual_future,
            'Forecast_Error (kW)': forecast_error,
            'Absolute_Error (kW)': abs_error,
            'Percentage_Error (%)': pct_error,
            'Validation_Available': validation_available
        })
        
        validation_results.append(result)
    
    validated_forecast_df = pd.DataFrame(validation_results)
    
    # Calculate summary metrics if requested
    if return_metrics:
        valid_forecasts = validated_forecast_df[
            (validated_forecast_df['Forecast_Available']) & 
            (validated_forecast_df['Validation_Available'])
        ].copy()
        
        if len(valid_forecasts) > 0:
            metrics = {
                'total_points': len(forecast_df),
                'forecasts_made': len(validated_forecast_df[validated_forecast_df['Forecast_Available']]),
                'validations_available': len(valid_forecasts),
                'validation_rate': len(valid_forecasts) / len(validated_forecast_df[validated_forecast_df['Forecast_Available']]) * 100,
                'mae_kw': valid_forecasts['Absolute_Error (kW)'].mean(),
                'rmse_kw': np.sqrt((valid_forecasts['Forecast_Error (kW)'] ** 2).mean()),
                'mean_pct_error': valid_forecasts['Percentage_Error (%)'].mean(),
                'median_pct_error': valid_forecasts['Percentage_Error (%)'].median(),
                'bias_kw': valid_forecasts['Forecast_Error (kW)'].mean()
            }
        else:
            metrics = {
                'total_points': len(forecast_df),
                'forecasts_made': 0,
                'validations_available': 0,
                'validation_rate': 0,
                'mae_kw': np.nan,
                'rmse_kw': np.nan,
                'mean_pct_error': np.nan,
                'median_pct_error': np.nan,
                'bias_kw': np.nan
            }
        
        return validated_forecast_df, metrics
    
    return validated_forecast_df


def convert_roc_backtest_to_long_format(forecast_series_dict, actual_series_dict, horizons):
    """
    Convert ROC backtest results to long format table with columns: t, horizon_min, actual, forecast_p50
    
    Args:
        forecast_series_dict: Dictionary of {horizon: forecast_series}
        actual_series_dict: Dictionary of {horizon: actual_series} 
        horizons: List of forecast horizons in minutes
        
    Returns:
        pd.DataFrame: Long format table with columns [t, horizon_min, actual, forecast_p50]
    """
    try:
        long_data = []
        
        for horizon in horizons:
            if horizon in forecast_series_dict and horizon in actual_series_dict:
                forecast_series = forecast_series_dict[horizon]
                actual_series = actual_series_dict[horizon]
                
                # Align series by index (timestamp)
                aligned_forecast, aligned_actual = forecast_series.align(actual_series, join='inner')
                
                for timestamp in aligned_forecast.index:
                    if pd.notna(aligned_forecast.loc[timestamp]) and pd.notna(aligned_actual.loc[timestamp]):
                        long_data.append({
                            't': timestamp,
                            'horizon_min': horizon,
                            'actual': aligned_actual.loc[timestamp],
                            'forecast_p50': aligned_forecast.loc[timestamp]
                        })
        
        if long_data:
            df_long = pd.DataFrame(long_data)
            df_long = df_long.sort_values(['t', 'horizon_min']).reset_index(drop=True)
            return df_long
        else:
            return pd.DataFrame(columns=['t', 'horizon_min', 'actual', 'forecast_p50'])
    
    except Exception as e:
        print(f"Error converting ROC backtest to long format: {e}")
        return pd.DataFrame(columns=['t', 'horizon_min', 'actual', 'forecast_p50'])


def compute_residual_quantiles_by_horizon(df_long, quantiles=[0.1, 0.5, 0.9]):
    """
    Compute residual quantiles by horizon from long format backtest results
    
    Args:
        df_long: DataFrame with columns [t, horizon_min, actual, forecast_p50]
        quantiles: List of quantiles to compute (default: [0.1, 0.5, 0.9] for P10, P50, P90)
        
    Returns:
        pd.DataFrame: Quantiles by horizon with columns [horizon_min, residual_p10, residual_p50, residual_p90]
    """
    try:
        if df_long.empty or 'actual' not in df_long.columns or 'forecast_p50' not in df_long.columns:
            return pd.DataFrame(columns=['horizon_min', 'residual_p10', 'residual_p50', 'residual_p90'])
        
        # Calculate residuals (forecast - actual)
        df_long = df_long.copy()
        df_long['residual'] = df_long['forecast_p50'] - df_long['actual']
        
        # Group by horizon and compute quantiles
        quantile_results = []
        for horizon in sorted(df_long['horizon_min'].unique()):
            horizon_data = df_long[df_long['horizon_min'] == horizon]
            residuals = horizon_data['residual'].dropna()
            
            if len(residuals) > 0:
                quantile_values = residuals.quantile(quantiles).values
                result = {'horizon_min': horizon}
                
                # Map quantiles to column names
                for i, q in enumerate(quantiles):
                    if q == 0.1:
                        result['residual_p10'] = quantile_values[i]
                    elif q == 0.5:
                        result['residual_p50'] = quantile_values[i]
                    elif q == 0.9:
                        result['residual_p90'] = quantile_values[i]
                
                quantile_results.append(result)
        
        if quantile_results:
            return pd.DataFrame(quantile_results)
        else:
            return pd.DataFrame(columns=['horizon_min', 'residual_p10', 'residual_p50', 'residual_p90'])
    
    except Exception as e:
        print(f"Error computing residual quantiles: {e}")
        return pd.DataFrame(columns=['horizon_min', 'residual_p10', 'residual_p50', 'residual_p90'])


def generate_p90_forecast_bands(df_long, residual_quantiles):
    """
    Generate P90 forecast bands by adding residual quantiles to P50 forecasts
    
    Args:
        df_long: DataFrame with columns [t, horizon_min, actual, forecast_p50]
        residual_quantiles: DataFrame with quantiles by horizon
        
    Returns:
        pd.DataFrame: Long format with added columns [forecast_p10, forecast_p90]
    """
    try:
        if df_long.empty or residual_quantiles.empty:
            return df_long.copy()
        
        df_result = df_long.copy()
        
        # Initialize new columns
        df_result['forecast_p10'] = np.nan
        df_result['forecast_p90'] = np.nan
        
        # Add quantiles by horizon
        for _, row in residual_quantiles.iterrows():
            horizon = row['horizon_min']
            horizon_mask = df_result['horizon_min'] == horizon
            
            if 'residual_p10' in row and pd.notna(row['residual_p10']):
                df_result.loc[horizon_mask, 'forecast_p10'] = (
                    df_result.loc[horizon_mask, 'forecast_p50'] + row['residual_p10']
                )
            
            if 'residual_p90' in row and pd.notna(row['residual_p90']):
                df_result.loc[horizon_mask, 'forecast_p90'] = (
                    df_result.loc[horizon_mask, 'forecast_p50'] + row['residual_p90']
                )
        
        return df_result
    
    except Exception as e:
        print(f"Error generating P90 forecast bands: {e}")
        return df_long.copy()


def load_vendor_battery_database():
    """Load vendor battery database from JSON file."""
    try:
        with open('vendor_battery_database.json', 'r') as f:
            battery_db = json.load(f)
        return battery_db
    except FileNotFoundError:
        st.error("❌ Battery database file 'vendor_battery_database.json' not found")
        return None
    except json.JSONDecodeError:
        st.error("❌ Error parsing battery database JSON file")
        return None
    except Exception as e:
        st.error(f"❌ Error loading battery database: {str(e)}")
        return None


def get_battery_capacity_range(battery_db):
    """Get the capacity range from battery database."""
    if not battery_db:
        return 200, 250, 225  # Default fallback values (all int)
    
    capacities = []
    for battery_id, spec in battery_db.items():
        capacity = spec.get('energy_kWh', 0)
        if capacity > 0:
            capacities.append(int(capacity))  # Ensure int type
    
    if capacities:
        min_cap = min(capacities)
        max_cap = max(capacities)
        default_cap = int(np.mean(capacities))
        return min_cap, max_cap, default_cap
    else:
        return 200, 250, 225  # Default fallback (all int)


def _render_battery_selection_dropdown():
    """
    Render independent battery selection dropdown that's always visible when data is available.
    This function should be called when a file is uploaded and data is available.
    """
    with st.container():
        st.markdown("#### 7. 📋 Tabled Analysis")
        
        # Battery selection dropdown
        battery_db = load_vendor_battery_database()
        
        if battery_db:
            # Create battery options for dropdown
            battery_options = {}
            battery_list = []
            
            for battery_id, spec in battery_db.items():
                company = spec.get('company', 'Unknown')
                model = spec.get('model', battery_id)
                capacity = spec.get('energy_kWh', 0)
                power = spec.get('power_kW', 0)
                
                label = f"{company} {model} ({capacity}kWh, {power}kW)"
                battery_options[label] = {
                    'id': battery_id,
                    'spec': spec,
                    'capacity_kwh': capacity,
                    'power_kw': power
                }
                battery_list.append(label)
            
            # Sort battery list for better UX
            battery_list.sort()
            battery_list.insert(0, "-- Select a Battery --")
            
            # Battery selection dropdown
            selected_battery_label = st.selectbox(
                "🔋 Select Battery for Analysis:",
                options=battery_list,
                index=0,
                key="independent_battery_selection",
                help="Choose a battery from the vendor database to view specifications and analysis"
            )
            
            # Display selected battery information
            if selected_battery_label != "-- Select a Battery --":
                selected_battery_data = battery_options[selected_battery_label]
                battery_spec = selected_battery_data['spec']
                
                # Display battery specifications in a table format
                st.markdown("**📊 Battery Specifications:**")
                spec_data = {
                    'Parameter': ['Company', 'Model', 'Energy Capacity', 'Power Rating', 'C-Rate', 'Voltage', 'Lifespan', 'Cooling'],
                    'Value': [
                        battery_spec.get('company', 'N/A'),
                        battery_spec.get('model', 'N/A'),
                        f"{battery_spec.get('energy_kWh', 0)} kWh",
                        f"{battery_spec.get('power_kW', 0)} kW",
                        f"{battery_spec.get('c_rate', 0)}C",
                        f"{battery_spec.get('voltage_V', 0)} V",
                        f"{battery_spec.get('lifespan_years', 0)} years",
                        battery_spec.get('cooling', 'N/A')
                    ]
                }
                df_specs = pd.DataFrame(spec_data)
                st.dataframe(df_specs, use_container_width=True, hide_index=True)
                
                # Store selected battery in session state for use in other parts of the analysis
                st.session_state.tabled_analysis_selected_battery = {
                    'id': selected_battery_data['id'],
                    'spec': battery_spec,
                    'capacity_kwh': selected_battery_data['capacity_kwh'],
                    'power_kw': selected_battery_data['power_kw'],
                    'label': selected_battery_label
                }
                
                return selected_battery_data
            else:
                st.info("💡 Select a battery from the dropdown above to view detailed specifications and analysis.")
                return None
        else:
            st.error("❌ Battery database not available")
            return None


def _render_battery_quantity_recommendation(max_power_shaving_required, recommended_energy_capacity):
    """
    Render battery quantity recommendation section between Tabled Analysis and Battery Sizing Analysis.
    
    Args:
        max_power_shaving_required: Maximum power shaving required (kW)
        recommended_energy_capacity: Maximum required energy (kWh)
    """
    st.markdown("#### 7.1 🔢 Battery Quantity Recommendation")
    
    # Check if user has selected a battery from the tabled analysis dropdown
    if hasattr(st.session_state, 'tabled_analysis_selected_battery') and st.session_state.tabled_analysis_selected_battery:
        selected_battery = st.session_state.tabled_analysis_selected_battery
        battery_spec = selected_battery['spec']
        battery_name = selected_battery['label']
        
        # Extract battery specifications
        battery_power_kw = battery_spec.get('power_kW', 0)
        battery_energy_kwh = battery_spec.get('energy_kWh', 0)
        
        if battery_power_kw > 0 and battery_energy_kwh > 0:
            # Calculate recommended quantities
            # Power Rating based quantity: roundup(Max Power Shaving Required / Battery Power Rating)
            qty_for_power = max_power_shaving_required / battery_power_kw if battery_power_kw > 0 else 0
            qty_for_power_rounded = int(np.ceil(qty_for_power))
            
            # Energy Capacity based quantity: roundup(Max Required Energy / Battery Energy Capacity / DOD / Efficiency) 
            qty_for_energy = recommended_energy_capacity / battery_energy_kwh / 0.9 / 0.93 if battery_energy_kwh > 0 else 0
            qty_for_energy_rounded = int(np.ceil(qty_for_energy))
            
            # Recommended quantity: maximum of the two
            recommended_qty = max(qty_for_power_rounded, qty_for_energy_rounded)
            
            # Display metrics showing the calculation
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Power-Based Qty", 
                    f"{qty_for_power_rounded} units",
                    help=f"Based on {max_power_shaving_required:.1f} kW ÷ {battery_power_kw} kW"
                )
                st.caption(f"Calculation: ⌈{max_power_shaving_required:.1f} ÷ {battery_power_kw}⌉")
            
            with col2:
                st.metric(
                    "Energy-Based Qty", 
                    f"{qty_for_energy_rounded} units",
                    help=f"Based on {recommended_energy_capacity:.1f} kWh ÷ {battery_energy_kwh} kWh ÷ 0.9 ÷ 0.93"
                )
                st.caption(f"Calculation: ⌈{recommended_energy_capacity:.1f} ÷ {battery_energy_kwh} ÷ 0.9 ÷ 0.93⌉")
            
            with col3:
                st.metric(
                    "Recommended Qty", 
                    f"{recommended_qty} units",
                    delta=f"{recommended_qty} units",
                    help="Maximum of power-based and energy-based quantities"
                )
                st.caption("Auto-recommended based on max requirement")
            
            # Allow user to override the recommended quantity
            st.markdown("**🎛️ Battery Quantity Configuration:**")
            
            # User input for quantity with recommended as default
            user_selected_qty = st.number_input(
                "Select Battery Quantity:",
                min_value=1,
                max_value=200,
                value=recommended_qty,
                step=1,
                key="v2_battery_quantity_selection",
                help=f"Auto-recommended: {recommended_qty} units. You can adjust this value if needed."
            )
            
            # Show impact of user selection
            total_power_capacity = user_selected_qty * battery_power_kw
            total_energy_capacity = user_selected_qty * battery_energy_kwh
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Total Power Capacity",
                    f"{total_power_capacity:.1f} kW",
                    f"{user_selected_qty} × {battery_power_kw} kW"
                )
            
            with col2:
                st.metric(
                    "Total Energy Capacity", 
                    f"{total_energy_capacity:.1f} kWh",
                    f"{user_selected_qty} × {battery_energy_kwh} kWh"
                )
            
            with col3:
                # Calculate coverage percentages
                power_coverage = (total_power_capacity / max_power_shaving_required * 100) if max_power_shaving_required > 0 else 100
                energy_coverage = (total_energy_capacity / recommended_energy_capacity * 100) if recommended_energy_capacity > 0 else 100
                
                overall_coverage = min(power_coverage, energy_coverage)
                
                coverage_color = "normal" if overall_coverage >= 100 else "inverse"
                st.metric(
                    "Coverage",
                    f"{overall_coverage:.1f}%",
                    delta_color=coverage_color,
                    help="Minimum of power and energy coverage percentages"
                )
            
            # Store the selected quantity in session state for use in sizing analysis
            st.session_state.tabled_analysis_battery_quantity = user_selected_qty
            
            # Provide guidance on the selection and integration information
            if user_selected_qty == recommended_qty:
                st.success(f"✅ **Optimal Configuration**: Using auto-recommended quantity of {recommended_qty} units based on your requirements.")
            elif user_selected_qty > recommended_qty:
                st.info(f"ℹ️ **Oversized Configuration**: You've selected {user_selected_qty} units, which is {user_selected_qty - recommended_qty} units more than the recommended {recommended_qty} units. This provides extra capacity margin.")
            else:
                st.warning(f"⚠️ **Undersized Configuration**: You've selected {user_selected_qty} units, which is {recommended_qty - user_selected_qty} units less than the recommended {recommended_qty} units. This may not fully meet your requirements.")
            
            # Integration feedback
            st.info(f"🔄 **Integration Active**: The selected quantity ({user_selected_qty} units) will be automatically used in the '📊 Battery Operation Simulation' section below, replacing any auto-calculated values.")
            
        else:
            st.error("❌ Selected battery has invalid power or energy specifications")
    
    else:
        st.warning("⚠️ **No Battery Selected**: Please select a battery from the '📋 Tabled Analysis' dropdown above to see quantity recommendations.")
        st.info("💡 Battery quantity will be automatically calculated based on your requirements once a battery is selected.")


def _render_battery_sizing_analysis(max_power_shaving_required, recommended_energy_capacity, total_md_cost):
    """
    Render comprehensive battery sizing and financial analysis table.
    
    Args:
        max_power_shaving_required: Maximum power shaving required (kW)
        recommended_energy_capacity: Maximum TOU excess power requirement (kW)  
        total_md_cost: Total MD cost impact (RM)
    """
    st.markdown("#### 7.2 🔋 Battery Sizing & Financial Analysis")
    
    # Check if user has selected a battery from the tabled analysis dropdown
    if hasattr(st.session_state, 'tabled_analysis_selected_battery') and st.session_state.tabled_analysis_selected_battery:
        selected_battery = st.session_state.tabled_analysis_selected_battery
        battery_spec = selected_battery['spec']
        battery_name = selected_battery['label']
        
        st.info(f"🔋 **Analysis based on selected battery:** {battery_name}")
        
        # Extract battery specifications
        battery_power_kw = battery_spec.get('power_kW', 0)
        battery_energy_kwh = battery_spec.get('energy_kWh', 0)
        battery_lifespan_years = battery_spec.get('lifespan_years', 15)
        
        if battery_power_kw > 0 and battery_energy_kwh > 0:
            # Use the user-selected quantity from the quantity recommendation section
            bess_quantity = getattr(st.session_state, 'tabled_analysis_battery_quantity', 1)
            
            # Calculate quantities that would be needed (for reference only)
            qty_for_power = max_power_shaving_required / battery_power_kw if battery_power_kw > 0 else 0
            qty_for_power_rounded = int(np.ceil(qty_for_power))
            qty_for_excess = recommended_energy_capacity / battery_energy_kwh if battery_energy_kwh > 0 else 0
            qty_for_excess_rounded = int(np.ceil(qty_for_excess))
            
            # Calculate total system specifications based on user-selected quantity
            total_power_kw = bess_quantity * battery_power_kw
            total_energy_kwh = bess_quantity * battery_energy_kwh
            
            # Column 4: MD shaved (actual impact with this battery configuration)
            # Use the total power capacity from the larger battery quantity (BESS quantity)
            md_shaved_kw = total_power_kw  # Total power from the BESS system
            md_shaving_percentage = (md_shaved_kw / max_power_shaving_required * 100) if max_power_shaving_required > 0 else 0

            # Column 5: Cost of batteries
            estimated_cost_per_kwh = 1400  # RM per kWh (consistent with main app)
            total_battery_cost = total_energy_kwh * estimated_cost_per_kwh
            
            # Create analysis table
            analysis_data = {
                'Analysis Parameter': [
                    'Units for Selected Power Requirement',
                    'Units for Selected Energy Capacity',
                    'Total BESS Quantity Required',
                    'Total System Power Capacity',
                    'Total System Energy Capacity',
                    'Actual MD Shaved',
                    'MD Shaving Coverage',
                    'Total Battery Investment'
                ],
                'Value': [
                    f"{qty_for_power_rounded} units (for {max_power_shaving_required:.1f} kW)",
                    f"{qty_for_excess_rounded} units (for {recommended_energy_capacity:.1f} kWh)", 
                    f"{bess_quantity} units",
                    f"{total_power_kw:.1f} kW",
                    f"{total_energy_kwh:.1f} kWh",
                    f"{md_shaved_kw:.1f} kW",
                    f"{md_shaving_percentage:.1f}%",
                    f"RM {total_battery_cost:,.0f}"
                ],
                'Calculation Basis': [
                    f"Selected Power Requirement: {max_power_shaving_required:.1f} kW ÷ {battery_power_kw} kW/unit",
                    f"Selected Energy Capacity: {recommended_energy_capacity:.1f} kWh ÷ {battery_energy_kwh} kWh/unit",
                    "Higher of power or energy requirement",
                    f"{bess_quantity} units × {battery_power_kw} kW/unit",
                    f"{bess_quantity} units × {battery_energy_kwh} kWh/unit", 
                    f"{bess_quantity} units × {battery_power_kw} kW/unit = {total_power_kw:.1f} kW",
                    f"MD Shaved ÷ Selected Power Requirement × 100%",
                    f"{total_energy_kwh:.1f} kWh × RM {estimated_cost_per_kwh}/kWh"
                ]
            }
            
            df_analysis = pd.DataFrame(analysis_data)
            
            # Display the dataframe without styling for consistent formatting
            st.dataframe(df_analysis, use_container_width=True, hide_index=True)
            
            # Key insights - only showing total investment
            col1, col2, col3 = st.columns(3)
            
            with col2:  # Center the single metric
                st.metric(
                    "💰 Total Investment", 
                    f"RM {total_battery_cost:,.0f}",
                    help="Total cost for complete BESS installation"
                )
            
            # Analysis insights
            if bess_quantity > 0:
                st.success(f"""
                **📊 Analysis Summary:**
                - **Battery Selection**: {battery_name}
                - **System Configuration**: {bess_quantity} units providing {total_power_kw:.1f} kW / {total_energy_kwh:.1f} kWh
                - **MD Shaving Capability**: {md_shaving_percentage:.1f}% coverage of maximum demand events
                - **Investment Required**: RM {total_battery_cost:,.0f} for complete BESS installation
                """)
                
                if md_shaving_percentage < 100:
                    st.warning(f"""
                    ⚠️ **Partial Coverage Notice**: 
                    This battery configuration covers {md_shaving_percentage:.1f}% of maximum power shaving requirements.
                    Additional {max_power_shaving_required - md_shaved_kw:.1f} kW capacity may be needed for complete coverage.
                    """)
            else:
                st.error("❌ Invalid battery configuration - no units required")
                
        else:
            st.error("❌ Selected battery has invalid power or energy specifications")
            
    else:
        st.warning("⚠️ **No Battery Selected**: Please select a battery from the '📋 Tabled Analysis' dropdown above to perform sizing analysis.")
        st.info("💡 Navigate to the top of this page and select a battery from the dropdown to see detailed sizing and financial analysis.")


def get_battery_options_for_capacity(battery_db, target_capacity, tolerance=5):
    """Get batteries that match the target capacity within tolerance."""
    if not battery_db:
        return []
    
    matching_batteries = []
    for battery_id, spec in battery_db.items():
        battery_capacity = spec.get('energy_kWh', 0)
        if abs(battery_capacity - target_capacity) <= tolerance:
            matching_batteries.append({
                'id': battery_id,
                'spec': spec,
                'capacity_kwh': battery_capacity,
                'power_kw': spec.get('power_kW', 0),
                'c_rate': spec.get('c_rate', 0)
            })
    
    # Sort by closest match to target capacity
    matching_batteries.sort(key=lambda x: abs(x['capacity_kwh'] - target_capacity))
    return matching_batteries


def _render_v2_battery_controls(max_power_shaving_required=None, max_required_energy=None):
    """
    Render battery capacity controls in the main content area (right side).
    
    Args:
        max_power_shaving_required: Maximum power shaving required from analysis (kW)
        max_required_energy: Maximum required energy from analysis (kWh)
    """
    
    st.markdown("### 🔋 Battery Configuration")
    st.markdown("**Configure battery specifications for MD shaving analysis.**")
    
    # Load battery database
    battery_db = load_vendor_battery_database()
    
    if not battery_db:
        st.error("❌ Battery database not available")
        return None
    
    # Get capacity range
    min_cap, max_cap, default_cap = get_battery_capacity_range(battery_db)
    
    # Selection method
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("**Battery Selection Method:** By Specific Model")
        st.caption("Choose specific battery model from database")
    
    with col2:
        st.metric("Available Range", f"{min_cap}-{max_cap} kWh")
    
    # Battery selection by specific model only
    # Create battery options
    battery_options = {}
    for battery_id, spec in battery_db.items():
        label = f"{spec.get('company', 'Unknown')} {spec.get('model', 'Unknown')} ({spec.get('energy_kWh', 0)}kWh)"
        battery_options[label] = {
            'id': battery_id,
            'spec': spec,
            'capacity': spec.get('energy_kWh', 0)
        }
    
    selected_battery_label = st.selectbox(
        "Select Battery Model:",
        options=list(battery_options.keys()),
        key="v2_main_battery_model",
        help="Choose specific battery model from database"
    )
    
    if selected_battery_label:
        selected_battery_data = battery_options[selected_battery_label]
        active_battery_spec = selected_battery_data['spec']
        selected_capacity = selected_battery_data['capacity']
        
        # Display selected battery specs
        st.markdown("#### 📊 Battery Specifications:")
        
        # Create specifications table
        specs_data = [
            ["Company", active_battery_spec.get('company', 'Unknown')],
            ["Model", active_battery_spec.get('model', 'Unknown')],
            ["Energy Capacity", f"{active_battery_spec.get('energy_kWh', 0)} kWh"],
            ["Power Rating", f"{active_battery_spec.get('power_kW', 0)} kW"],
            ["C Rate", f"{active_battery_spec.get('c_rate', 0)}C"],
            ["Voltage", f"{active_battery_spec.get('voltage_V', 0)} V"],
            ["Lifespan", f"{active_battery_spec.get('lifespan_years', 0)} years"],
            ["Cooling", active_battery_spec.get('cooling', 'N/A')]
        ]
        
        specs_df = pd.DataFrame(specs_data, columns=["Parameter", "Value"])
        st.table(specs_df)
        
        # Battery Quantity Recommendation section
        st.markdown("#### 7.1 🔢 Battery Quantity Recommendation")
        
        # Use actual calculated values from analysis, fallback to default if not provided
        max_power_required = max_power_shaving_required if max_power_shaving_required is not None else 1734.4  # kW
        max_energy_required = max_required_energy if max_required_energy is not None else 7884.8  # kWh
        
        # Display source of values
        if max_power_shaving_required is not None and max_required_energy is not None:
            st.info("📊 **Values sourced from Section 6.5 Battery Sizing Analysis**")
        else:
            st.warning("⚠️ **Using default values** - Run analysis in Section 6 to get accurate calculations")
        
        # Extract battery specifications for calculations
        battery_power_kw = active_battery_spec.get('power_kW', 0)
        battery_energy_kwh = active_battery_spec.get('energy_kWh', 0)
        
        # Initialize variables that will be needed later
        total_power_capacity = 0
        total_energy_capacity = 0
        overall_coverage = 0
        
        if battery_power_kw > 0 and battery_energy_kwh > 0:
            # Calculate recommended quantities
            # Power-based quantity: ceiling(Max Power Required / Battery Power Rating)
            qty_power = np.ceil(max_power_required / battery_power_kw) if battery_power_kw > 0 else 0
            
            # Energy-based quantity: ceiling(Max Energy Required / Battery Energy / DOD / Efficiency)
            dod = 0.9  # Depth of Discharge
            efficiency = 0.93  # Battery efficiency
            qty_energy = np.ceil(max_energy_required / battery_energy_kwh / dod / efficiency) if battery_energy_kwh > 0 else 0
            
            # Recommended quantity: maximum of the two
            recommended_qty = max(int(qty_power), int(qty_energy))
            
            # Display quantity recommendations
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Power-Based Qty ℹ️", f"{int(qty_power)} units")
                st.caption(f"Calculation: ⌈{max_power_required} ÷ {battery_power_kw}⌉")
            
            with col2:
                st.metric("Energy-Based Qty ℹ️", f"{int(qty_energy)} units") 
                st.caption(f"Calculation: ⌈{max_energy_required} ÷ {battery_energy_kwh} ÷ {dod} ÷ {efficiency}⌉")
            
            with col3:
                st.metric("Recommended Qty ℹ️", f"{recommended_qty} units", delta=f"↑ {recommended_qty} units")
                st.caption("Auto-recommended based on max requirement")
        else:
            st.warning("⚠️ Battery specifications incomplete for quantity calculation")
    else:
        active_battery_spec = None
        selected_capacity = default_cap
    
    # Battery Quantity Configuration
    st.markdown("#### 🔢 Battery Quantity Configuration:")
    
    # Get the recommended quantity from the previous calculation (if available)
    if active_battery_spec and battery_power_kw > 0 and battery_energy_kwh > 0:
        default_qty = recommended_qty
    else:
        default_qty = 37  # Fallback default
    
    # Battery quantity selector with +/- controls
    st.markdown("**Select Battery Quantity:**")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.button("➖", key="decrease_qty", help="Decrease quantity by 1"):
            if "battery_quantity" not in st.session_state:
                st.session_state.battery_quantity = default_qty
            if st.session_state.battery_quantity > 1:
                st.session_state.battery_quantity -= 1
    
    with col2:
        # Initialize session state if not exists
        if "battery_quantity" not in st.session_state:
            st.session_state.battery_quantity = default_qty
            
        # Display current quantity
        selected_quantity = st.number_input(
            "",
            min_value=1,
            max_value=1000,
            value=st.session_state.battery_quantity,
            key="qty_input",
            label_visibility="collapsed"
        )
        st.session_state.battery_quantity = selected_quantity
    
    with col3:
        if st.button("➕", key="increase_qty", help="Increase quantity by 1"):
            if "battery_quantity" not in st.session_state:
                st.session_state.battery_quantity = default_qty
            if st.session_state.battery_quantity < 1000:
                st.session_state.battery_quantity += 1
    
    # Calculate total capacities and coverage based on selected quantity
    if active_battery_spec and battery_power_kw > 0 and battery_energy_kwh > 0:
        total_power_capacity = st.session_state.battery_quantity * battery_power_kw
        total_energy_capacity = st.session_state.battery_quantity * battery_energy_kwh
        
        # Calculate coverage percentage (based on max requirements)
        power_coverage = (total_power_capacity / max_power_required) * 100 if max_power_required > 0 else 0
        energy_coverage = (total_energy_capacity / max_energy_required) * 100 if max_energy_required > 0 else 0
        overall_coverage = min(power_coverage, energy_coverage)  # Limiting factor
        
        # Display capacity metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Total Power Capacity",
                f"{total_power_capacity:.1f} kW",
                delta=f"↑ {st.session_state.battery_quantity} × {battery_power_kw} kW"
            )
        
        with col2:
            st.metric(
                "Total Energy Capacity", 
                f"{total_energy_capacity:.1f} kWh",
                delta=f"↑ {st.session_state.battery_quantity} × {battery_energy_kwh} kWh"
            )
        
        with col3:
            coverage_color = "normal"
            if overall_coverage >= 100:
                coverage_color = "normal"
            elif overall_coverage >= 80:
                coverage_color = "normal" 
            else:
                coverage_color = "inverse"
                
            st.metric(
                "Coverage",
                f"{overall_coverage:.1f}%",
                delta=None
            )
        
        # Configuration status messages
        if st.session_state.battery_quantity == default_qty:
            st.success("✅ **Optimal Configuration:** Using auto-recommended quantity of {} units based on your requirements.".format(default_qty))
        else:
            st.info(f"🔧 **Custom Configuration:** Using {st.session_state.battery_quantity} units (recommended: {default_qty} units)")
        
        # Integration notice
        st.info("📊 **Integration Active:** The selected quantity ({} units) will be automatically used in the '📊 Battery Operation Simulation' section below, replacing any auto-calculated values.".format(st.session_state.battery_quantity))
        
        # Battery Sizing & Financial Analysis Section
        st.markdown("#### 7.2 💰 Battery Sizing & Financial Analysis")
        
        # Analysis header with battery info
        battery_model = active_battery_spec.get('model', 'Unknown')
        battery_company = active_battery_spec.get('company', 'Unknown')
        st.info(f"🔋 **Analysis based on selected battery:** {battery_company} {battery_model} ({battery_energy_kwh}kWh, {battery_power_kw}kW)")
        
        # Show data source
        if max_power_shaving_required is not None and max_required_energy is not None:
            st.success("📊 **Values sourced from Section 6.5 Battery Sizing Analysis** - Using actual calculated requirements")
        else:
            st.warning("⚠️ **Using default values** - Run analysis in Section 6 to get accurate calculations")
        
        # Analysis calculations - use dynamic values passed to function
        selected_qty = st.session_state.battery_quantity
        
        # Calculate units needed based on power and energy
        units_for_power = int(np.ceil(max_power_required / battery_power_kw)) if battery_power_kw > 0 else 0
        units_for_energy = int(np.ceil(max_energy_required / battery_energy_kwh)) if battery_energy_kwh > 0 else 0
        
        # System capacities based on selected quantity
        total_system_power = selected_qty * battery_power_kw
        total_system_energy = selected_qty * battery_energy_kwh
        
        # MD shaving calculations
        actual_md_shaved = total_system_power  # Assuming full power utilization
        md_shaving_coverage = (actual_md_shaved / max_power_required) * 100 if max_power_required > 0 else 0
        
        # Financial calculations (using RM 1400/kWh as per screenshot)
        cost_per_kwh = 1400  # RM per kWh
        total_battery_investment = total_system_energy * cost_per_kwh
        
        # Create analysis table
        analysis_data = [
            ["Units for Selected Power Requirement", f"{units_for_power} units (for {max_power_required} kW)", f"Selected Power Requirement: {max_power_required} kW ÷ {battery_power_kw} kW/unit"],
            ["Units for Selected Energy Capacity", f"{units_for_energy} units (for {max_energy_required} kWh)", f"Selected Energy Capacity: {max_energy_required} kWh ÷ {battery_energy_kwh} kWh/unit"],
            ["Total BESS Quantity Required", f"{selected_qty} units", "Higher of power or energy requirement"],
            ["Total System Power Capacity", f"{total_system_power:.1f} kW", f"{selected_qty} units × {battery_power_kw} kW/unit"],
            ["Total System Energy Capacity", f"{total_system_energy:.1f} kWh", f"{selected_qty} units × {battery_energy_kwh} kWh/unit"],
            ["Actual MD Shaved", f"{actual_md_shaved:.1f} kW", f"{selected_qty} units × {battery_power_kw} kW/unit = {actual_md_shaved:.1f} kW"],
            ["MD Shaving Coverage", f"{md_shaving_coverage:.1f}%", f"MD Shaved ÷ Selected Power Requirement × 100%"],
            ["Total Battery Investment", f"RM {total_battery_investment:,.0f}", f"{total_system_energy:.1f} kWh × RM {cost_per_kwh}/kWh"]
        ]
        
        analysis_df = pd.DataFrame(analysis_data, columns=["Analysis Parameter", "Value", "Calculation Basis"])
        st.table(analysis_df)
        
        # Total Investment Display
        st.markdown("### 💰 Total Investment ℹ️")
        st.markdown(f"<h1 style='text-align: center; color: #2E8B57; font-size: 3em; margin: 0.5em 0;'>RM {total_battery_investment:,}</h1>", unsafe_allow_html=True)
        
        run_analysis = True  # Always enable analysis when battery is properly configured
    else:
        st.warning("⚠️ Battery specifications incomplete for quantity configuration")
        run_analysis = False
    
    # Return the selected battery configuration with consistent key names
    battery_config = {
        'selection_method': 'By Specific Model',
        'selected_capacity': selected_capacity if 'selected_capacity' in locals() else default_cap,
        'active_battery_spec': active_battery_spec if 'active_battery_spec' in locals() else {},
        'selected_quantity': st.session_state.get('battery_quantity', default_qty),
        'total_power_capacity': total_power_capacity if 'total_power_capacity' in locals() else 0,
        'total_energy_capacity': total_energy_capacity if 'total_energy_capacity' in locals() else 0,
        'coverage_percentage': overall_coverage if 'overall_coverage' in locals() else 0,
        'run_analysis': run_analysis
    }
    
    return battery_config


def _get_soc_aware_discharge_strategy(current_soc_percent, demand_excess_kw, battery_power_kw):
    """
    SOC-aware conservative discharge strategy for battery management.
    
    Args:
        current_soc_percent (float): Current state of charge as percentage
        demand_excess_kw (float): Excess demand above target that needs shaving
        battery_power_kw (float): Maximum battery discharge power
        
    Returns:
        dict: Discharge strategy with power_kw and reasoning
    """
    # SOC-Aware Strategy Parameters (Conservative)
    MIN_SOC_THRESHOLD = 20.0  # Keep 20% minimum charge for safety
    EXCESS_DISCHARGE_RATE = 0.6  # Use 60% of excess for conservative approach
    
    # Safety check: Don't discharge below minimum SOC
    if current_soc_percent <= MIN_SOC_THRESHOLD:
        return {
            'power_kw': 0,
            'reasoning': f"SOC too low ({current_soc_percent:.1f}%) - maintaining {MIN_SOC_THRESHOLD}% minimum reserve"
        }
    
    # Conservative discharge: Only use 60% of demand excess
    conservative_demand = demand_excess_kw * EXCESS_DISCHARGE_RATE
    
    # Limit to battery capacity
    discharge_power = min(conservative_demand, battery_power_kw)
    
    # Additional SOC-based power limiting for very low SOC
    if current_soc_percent < 30.0:
        # Further reduce power when approaching minimum SOC
        soc_limiter = (current_soc_percent - MIN_SOC_THRESHOLD) / (30.0 - MIN_SOC_THRESHOLD)
        discharge_power *= soc_limiter
    
    return {
        'power_kw': discharge_power,
        'reasoning': f"SOC-Aware: {discharge_power:.1f}kW (60% of {demand_excess_kw:.1f}kW excess, SOC: {current_soc_percent:.1f}%)"
    }


def _get_tariff_aware_discharge_strategy(current_soc_percent, demand_excess_kw, battery_power_kw):
    """
    Default tariff-aware aggressive discharge strategy for battery management.
    
    Args:
        current_soc_percent (float): Current state of charge as percentage
        demand_excess_kw (float): Excess demand above target that needs shaving
        battery_power_kw (float): Maximum battery discharge power
        
    Returns:
        dict: Discharge strategy with power_kw and reasoning
    """
    # Default Strategy Parameters (Aggressive)
    MIN_SOC_THRESHOLD = 5.0   # Allow discharge to 5% for maximum shaving
    EXCESS_DISCHARGE_RATE = 0.8  # Use 80% of excess for aggressive approach
    
    # Safety check: Don't discharge below minimum SOC
    if current_soc_percent <= MIN_SOC_THRESHOLD:
        return {
            'power_kw': 0,
            'reasoning': f"SOC critical ({current_soc_percent:.1f}%) - maintaining {MIN_SOC_THRESHOLD}% emergency reserve"
        }
    
    # Aggressive discharge: Use 80% of demand excess
    aggressive_demand = demand_excess_kw * EXCESS_DISCHARGE_RATE
    
    # Limit to battery capacity
    discharge_power = min(aggressive_demand, battery_power_kw)
    
    return {
        'power_kw': discharge_power,
        'reasoning': f"Default: {discharge_power:.1f}kW (80% of {demand_excess_kw:.1f}kW excess, SOC: {current_soc_percent:.1f}%)"
    }


def _get_strategy_aware_discharge(strategy_mode, current_soc_percent, demand_excess_kw, battery_power_kw):
    """
    Router function to select appropriate discharge strategy based on user selection.
    
    Args:
        strategy_mode (str): Either "Default Shaving" or "SOC-Aware"
        current_soc_percent (float): Current state of charge as percentage
        demand_excess_kw (float): Excess demand above target that needs shaving
        battery_power_kw (float): Maximum battery discharge power
        
    Returns:
        dict: Discharge strategy with power_kw, reasoning, and strategy_type
    """
    if strategy_mode == "SOC-Aware":
        result = _get_soc_aware_discharge_strategy(current_soc_percent, demand_excess_kw, battery_power_kw)
        result['strategy_type'] = 'SOC-Aware (Conservative)'
    else:  # Default Shaving
        result = _get_tariff_aware_discharge_strategy(current_soc_percent, demand_excess_kw, battery_power_kw)
        result['strategy_type'] = 'Default (Aggressive)'
    
    return result


# =============================================================================
# V2 BATTERY SIMULATION FUNCTIONS - MOVED TO RESOLVE FUNCTION ORDER ISSUES
# =============================================================================



def _create_v2_conditional_demand_line_with_dynamic_targets(fig, df, power_col, target_series, selected_tariff=None, holidays=None, trace_name="Original Demand"):
    """
    V2 ENHANCEMENT: Enhanced conditional coloring logic for Original Demand line with DYNAMIC monthly targets.
    Creates continuous line segments with different colors based on monthly target conditions.
    
    Key V2 Innovation: Uses dynamic monthly targets instead of static averaging for color decisions.
    
    Color Logic:
    - Red: Above monthly target during Peak Periods (based on selected tariff) - Direct MD cost impact
    - Green: Above monthly target during Off-Peak Periods - No MD cost impact  
    - Blue: Below monthly target (any time) - Within acceptable limits
    
    Args:
        fig: Plotly figure to add traces to
        df: Simulation dataframe
        power_col: Power column name
        target_series: V2's dynamic monthly target series (same index as df)
        selected_tariff: Tariff configuration for period classification
        holidays: Set of holiday dates
        trace_name: Name for the trace
        
    Returns:
        Modified plotly figure with colored demand line segments
    """
    from tariffs.peak_logic import is_peak_rp4, get_period_classification
    
    # Validate inputs
    if target_series is None or len(target_series) == 0:
        st.warning("⚠️ V2 Dynamic Coloring: target_series is empty, falling back to single average")
        # Fallback to V1 approach with average target
        avg_target = df[power_col].quantile(0.9)
        return create_conditional_demand_line_with_peak_logic(fig, df, power_col, avg_target, selected_tariff, holidays, trace_name)
    
    # FIXED: Handle datetime index conversion properly
    df_copy = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_copy.index):
        try:
            df_copy.index = pd.to_datetime(df_copy.index)
        except Exception as e:
            st.error(f"Error converting index to datetime: {str(e)}")
            return fig
    
    # Create a series with color classifications using DYNAMIC monthly targets
    df_copy['color_class'] = ''
    
    for i in range(len(df_copy)):
        timestamp = df_copy.index[i]
        # FIXED: Ensure demand_value is extracted as scalar from the start
        demand_value = float(df_copy.iloc[i][power_col])
        
        # V2 ENHANCEMENT: Get DYNAMIC monthly target for this specific timestamp
        if timestamp in target_series.index:
            current_target = float(target_series.loc[timestamp])  # FIXED: Explicit scalar conversion
        else:
            # Fallback to closest available target
            try:
                month_period = timestamp.to_period('M')
                # FIXED: Use .index instead of creating list to avoid Series ambiguity
                if len(target_series) > 0:
                    # Find closest timestamp by distance - FIXED: Ensure scalar result
                    time_diffs = (target_series.index - timestamp).abs()
                    closest_idx = time_diffs.idxmin()
                    # FIXED: Ensure we get a single scalar value, not a Series
                    if hasattr(closest_idx, '__iter__') and not isinstance(closest_idx, (str, bytes)):
                        closest_idx = closest_idx[0] if len(closest_idx) > 0 else target_series.index[0]
                    current_target = float(target_series.loc[closest_idx])  # FIXED: Explicit scalar conversion
                else:
                    current_target = float(df_copy[power_col].quantile(0.9))  # FIXED: Use df_copy instead of df
            except Exception:
                current_target = 1000.0  # FIXED: Direct float assignment
        
        # Get MD window classification using RP4 2-period system
        is_md = is_peak_rp4(timestamp, holidays if holidays else set())
        period_type = 'Peak' if is_md else 'Off-Peak'
        
        # V2 LOGIC: Color classification using dynamic monthly target - FIXED: All values are now scalars
        try:
            # Both values are already floats, perform comparison safely
            if demand_value > current_target:
                if period_type == 'Peak':
                    df_copy.iloc[i, df_copy.columns.get_loc('color_class')] = 'red'
                else:
                    df_copy.iloc[i, df_copy.columns.get_loc('color_class')] = 'green'
            else:
                df_copy.iloc[i, df_copy.columns.get_loc('color_class')] = 'blue'
                
        except Exception as e:
            # Defensive fallback for any unexpected comparison issues
            st.error(f"❌ Series conversion error at {timestamp}: demand_value={type(demand_value)}, current_target={type(current_target)}, error={str(e)}")
            df_copy.iloc[i, df_copy.columns.get_loc('color_class')] = 'blue'  # Safe fallback
    
    # Create continuous line segments with color-coded segments
    x_data = df_copy.index
    y_data = df_copy[power_col]
    colors = df_copy['color_class']
    
    # Track legend status
    legend_added = {'red': False, 'green': False, 'blue': False}
    
    # Create continuous line segments by color groups with bridge points
    i = 0
    while i < len(df_copy):
        current_color = colors.iloc[i]
        
        # Find the end of current color segment
        j = i
        while j < len(colors) and colors.iloc[j] == current_color:
            j += 1
        
        # Extract segment data
        segment_x = list(x_data[i:j])
        segment_y = list(y_data[i:j])
        
        # Add bridge points for better continuity (connect to adjacent segments)
        if i > 0:  # Add connection point from previous segment
            segment_x.insert(0, x_data[i-1])
            segment_y.insert(0, y_data[i-1])
        
        if j < len(colors):  # Add connection point to next segment
            segment_x.append(x_data[j])
            segment_y.append(y_data[j])
        
        # Determine trace name based on color and tariff type
        tariff_description = _get_tariff_description(selected_tariff) if selected_tariff else "RP4 Peak Period"
        
        # Check if it's a TOU tariff for enhanced hover info
        is_tou = False
        if selected_tariff:
            tariff_type = selected_tariff.get('Type', '').lower()
            tariff_name = selected_tariff.get('Tariff', '').lower()
            is_tou = tariff_type == 'tou' or 'tou' in tariff_name
        
        if current_color == 'red':
            segment_name = f'{trace_name} (Above Target - {tariff_description})'
            if is_tou:
                hover_info = f'<b>Above Monthly Target - TOU Peak Rate Period</b><br><i>High Energy Cost + MD Cost Impact</i><br><i>Using V2 Dynamic Monthly Targets</i>'
            else:
                hover_info = f'<b>Above Monthly Target - General Tariff</b><br><i>MD Cost Impact Only (Flat Energy Rate)</i><br><i>Using V2 Dynamic Monthly Targets</i>'
        elif current_color == 'green':
            segment_name = f'{trace_name} (Above Target - Off-Peak)'
            if is_tou:
                hover_info = '<b>Above Monthly Target - TOU Off-Peak</b><br><i>Low Energy Cost, No MD Impact</i><br><i>Using V2 Dynamic Monthly Targets</i>'
            else:
                hover_info = '<b>Above Monthly Target - General Tariff</b><br><i>This should not appear for General tariffs</i><br><i>Using V2 Dynamic Monthly Targets</i>'
        else:  # blue
            segment_name = f'{trace_name} (Below Target)'
            hover_info = '<b>Below Monthly Target</b><br><i>Within Acceptable Limits</i><br><i>Using V2 Dynamic Monthly Targets</i>'
        
        # Only show legend for the first occurrence of each color
        show_legend = not legend_added[current_color]
        legend_added[current_color] = True
        
        # Add line segment
        fig.add_trace(go.Scatter(
            x=segment_x,
            y=segment_y,
            mode='lines',
            line=dict(color=current_color, width=2),
            name=segment_name,
            hovertemplate=f'{trace_name}: %{{y:.2f}} kW<br>%{{x}}<br>{hover_info}<extra></extra>',
            showlegend=show_legend,
            legendgroup=current_color,
            connectgaps=True  # Connect gaps within segments
        ))
        
        i = j
    
    return fig


def _render_v2_peak_events_timeline(df, power_col, selected_tariff, holidays, target_method, shave_percent, target_percent, target_manual_kw, target_description):
    """Render the V2 Peak Events Timeline visualization with dynamic monthly-based targets and enhanced color logic."""
    
    try:
        # Calculate tariff-specific monthly targets using V2 functions
        monthly_targets, reference_peaks, tariff_type, enhanced_target_description = _calculate_monthly_targets_v2(
            df, power_col, selected_tariff, holidays, 
            target_method, shave_percent, target_percent, target_manual_kw
        )
        
        # Create visualization with enhanced color logic
        fig = go.Figure()
        
        # Create a target series that maps monthly targets to the full dataframe index
        target_series = pd.Series(index=df.index, dtype=float)
        
        if not monthly_targets.empty:
            for month_period, target_value in monthly_targets.items():
                month_start = pd.Timestamp(month_period.start_time)
                month_end = pd.Timestamp(month_period.end_time)
                month_mask = (df.index >= month_start) & (df.index <= month_end)
                target_series.loc[month_mask] = target_value
        else:
            # Fallback to a single target
            avg_target = df[power_col].quantile(0.9)
            target_series[:] = avg_target
        
        # Use V2 enhanced conditional demand line with dynamic monthly targets and color logic
        try:
            fig = _create_v2_conditional_demand_line_with_dynamic_targets(
                fig, df, power_col, target_series, selected_tariff, holidays, "Power Consumption"
            )
            
        except Exception as e:
            st.error(f"❌ Error in timeline visualization: {str(e)}")
            # Continue without the enhanced coloring
            st.warning("⚠️ Falling back to basic visualization without enhanced coloring")
            return None
        
        # Add monthly targets as horizontal lines for reference
        if not monthly_targets.empty:
            for month_period, target_value in monthly_targets.items():
                month_start = pd.Timestamp(month_period.start_time)
                month_end = pd.Timestamp(month_period.end_time)
                month_mask = (df.index >= month_start) & (df.index <= month_end)
                month_data = df[month_mask]
                
                if not month_data.empty:
                    fig.add_trace(go.Scatter(
                        x=[month_data.index[0], month_data.index[-1]],
                        y=[target_value, target_value],
                        mode='lines',
                        name=f'Target {month_period}',
                        line=dict(color='orange', width=2, dash='dash'),
                        opacity=0.7,
                        showlegend=False
                    ))
        
        # Update layout
        fig.update_layout(
            title=f"V2 Peak Events Timeline with Color Logic - {tariff_type} ({enhanced_target_description})",
            xaxis_title="Time",
            yaxis_title="Power (kW)",
            height=600,
            showlegend=True,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error in _render_v2_peak_events_timeline: {str(e)}")
        return None
        
        return fig
        
    except Exception as e:
        st.error(f"Error in V2 timeline rendering: {str(e)}")
        return None


def _create_v2_dynamic_target_series(simulation_index, monthly_targets):
    """
    Create a dynamic target series that matches the simulation dataframe index
    with stepped monthly targets from V2's monthly_targets.
    
    Args:
        simulation_index: DatetimeIndex from the simulation dataframe
        monthly_targets: V2's monthly targets (Series with Period index)
        
    Returns:
        Series with same index as simulation_index, containing monthly target values
    """
    target_series = pd.Series(index=simulation_index, dtype=float)
    
    for timestamp in simulation_index:
        # FIXED: Ensure timestamp is a pandas Timestamp object
        if not isinstance(timestamp, pd.Timestamp):
            try:
                timestamp = pd.to_datetime(timestamp)
            except Exception as e:
                try:
                    import streamlit as st
                    st.warning(f"Could not convert timestamp {timestamp} to datetime: {e}")
                except ImportError:
                    print(f"Could not convert timestamp {timestamp} to datetime: {e}")
                continue
        
        # Get the month period for this timestamp
        try:
            month_period = timestamp.to_period('M')
        except Exception as e:
            try:
                import streamlit as st
                st.warning(f"Could not convert timestamp {timestamp} to period: {e}")
            except ImportError:
                print(f"Could not convert timestamp {timestamp} to period: {e}")
            continue
        
        # Find the corresponding monthly target - FIXED: Ensure scalar float value
        if month_period in monthly_targets.index:
            target_value = monthly_targets.loc[month_period]
            target_series.loc[timestamp] = float(target_value)
        else:
            # Fallback: use the closest available monthly target
            available_months = list(monthly_targets.index)
            if available_months:
                # Find the closest month
                closest_month = min(available_months, 
                                  key=lambda m: abs((month_period - m).n))
                target_value = monthly_targets.loc[closest_month]
                target_series.loc[timestamp] = float(target_value)
            else:
                # Ultimate fallback
                target_series.loc[timestamp] = 1000.0  # Safe default
    
    return target_series


def _calculate_c_rate_limited_power_simple(current_soc_percent, max_power_rating_kw, battery_capacity_kwh, c_rate=1.0):
    """
    Simple C-rate power limitation for charging/discharging.
    
    Args:
        current_soc_percent: Current state of charge percentage
        max_power_rating_kw: Battery's rated power
        battery_capacity_kwh: Battery's energy capacity
        c_rate: Battery's C-rate (default 1.0C)
        
    Returns:
        Dictionary with power limits
    """
    # Calculate C-rate based power limits
    c_rate_power_limit = battery_capacity_kwh * c_rate
    
    # SOC-based derating (power reduces at extreme SOC levels)
    if current_soc_percent > 90:
        soc_factor = 0.8  # Reduce power at high SOC
    elif current_soc_percent < 20:
        soc_factor = 0.7  # Reduce power at low SOC
    else:
        soc_factor = 1.0  # Full power in normal SOC range
    
    # Final power limit is minimum of C-rate limit and rated power
    effective_max_discharge_kw = min(c_rate_power_limit, max_power_rating_kw) * soc_factor
    effective_max_charge_kw = min(c_rate_power_limit, max_power_rating_kw) * soc_factor * 0.8  # Charging typically slower
    
    return {
        'max_discharge_power_kw': effective_max_discharge_kw,
        'max_charge_power_kw': effective_max_charge_kw,
        'c_rate_power_limit_kw': c_rate_power_limit,
        'soc_derating_factor': soc_factor,
        'limiting_factor': 'C-rate' if c_rate_power_limit < max_power_rating_kw else 'Power Rating'
    }


def _get_tou_charging_urgency(current_timestamp, soc_percent, holidays=None):
    """
    Determine TOU charging urgency based on time until MD window and current SOC.
    
    TOU charging windows:
    - Primary: 10 PM - 2 PM next day (overnight charging)
    - Target: 95% SOC by 2 PM on weekdays for MD readiness
    
    Args:
        current_timestamp: Current datetime
        soc_percent: Current battery SOC percentage
        holidays: Set of holiday dates
    
    Returns:
        dict: Charging urgency information
    """
    from datetime import datetime, timedelta
    
    # Check if it's a charging window (10 PM - 2 PM next day)
    hour = current_timestamp.hour
    is_weekday = current_timestamp.weekday() < 5
    is_holiday = holidays and current_timestamp.date() in holidays
    
    # TOU charging window: 10 PM to 2 PM next day
    is_charging_window = hour >= 22 or hour < 14
    
    # Calculate time until next MD window (2 PM)
    current_date = current_timestamp.date()
    if hour < 14:
        # Same day 2 PM
        next_md_start = datetime.combine(current_date, datetime.min.time().replace(hour=14))
    else:
        # Next day 2 PM
        next_day = current_date + timedelta(days=1)
        next_md_start = datetime.combine(next_day, datetime.min.time().replace(hour=14))
    
    # Only consider weekday MD windows
    while next_md_start.weekday() >= 5 or (holidays and next_md_start.date() in holidays):
        next_md_start += timedelta(days=1)
        next_md_start = next_md_start.replace(hour=14, minute=0, second=0, microsecond=0)
    
    time_until_md = next_md_start - current_timestamp
    hours_until_md = time_until_md.total_seconds() / 3600
    
    # Determine charging urgency
    urgency_level = 'normal'
    charge_rate_multiplier = 1.0
    
    if hours_until_md <= 4:  # Less than 4 hours until MD window
        if soc_percent < 95:
            urgency_level = 'critical'
            charge_rate_multiplier = 1.5  # Aggressive charging
    elif hours_until_md <= 8:  # 4-8 hours until MD window
        if soc_percent < 90:
            urgency_level = 'high'
            charge_rate_multiplier = 1.2  # Enhanced charging
    elif hours_until_md <= 16:  # 8-16 hours until MD window
        if soc_percent < 80:
            urgency_level = 'normal'
            charge_rate_multiplier = 1.0  # Standard charging
    
    return {
        'urgency_level': urgency_level,
        'charge_rate_multiplier': charge_rate_multiplier,
        'hours_until_md': hours_until_md,
        'is_charging_window': is_charging_window,
        'is_weekday': is_weekday and not is_holiday,
        'next_md_window': next_md_start
    }


def is_md_window(timestamp, holidays=None):
    """
    Determine if timestamp is in MD recording window based on RP4 tariff rules.
    
    Args:
        timestamp: datetime object
        holidays: set of holiday dates
        
    Returns:
        bool: True if in MD recording period
    """
    # MD recording: 2 PM - 10 PM on weekdays (non-holidays)
    if holidays and timestamp.date() in holidays:
        return False
    
    if timestamp.weekday() >= 5:  # Weekend
        return False
    
    hour = timestamp.hour
    return 14 <= hour < 22  # 2 PM to 10 PM


def _get_enhanced_shaving_success(row, holidays=None):
    """
    Enhanced shaving success classification using comprehensive battery status evaluation.
    Uses simplified 4-category system for better user understanding.
    
    Args:
        row: DataFrame row with simulation data
        holidays: Set of holiday dates
        
    Returns:
        str: Success classification with emoji and description
    """
    try:
        timestamp = row.name
        is_md = is_md_window(timestamp, holidays)
        
        # Get key metrics
        original_demand = row.get('Original_Demand', 0)
        net_demand = row.get('Net_Demand_kW', original_demand)
        monthly_target = row.get('Monthly_Target', 0)
        battery_power = row.get('Battery_Power_kW', 0)
        soc_percent = row.get('Battery_SOC_Percent', 50)
        
        # Calculate demand reduction
        demand_reduction = original_demand - net_demand
        excess_above_target = max(0, original_demand - monthly_target)
        
        # During MD periods
        if is_md:
            if net_demand <= monthly_target * 1.05:  # Within 5% tolerance
                if soc_percent > 20:
                    return "✅ Success - Target Met"
                else:
                    return "🟡 Partial - Target Met (Low SOC)"
            elif demand_reduction > 0:  # Some battery help
                if demand_reduction >= excess_above_target * 0.5:  # Reduced 50%+ of excess
                    return "🟡 Partial - Significant Reduction"
                else:
                    return "🟡 Partial - Limited Reduction"
            else:
                return "🔴 Failed - No Battery Response"
        
        # Outside MD periods
        else:
            if battery_power < 0:  # Charging
                return "✅ Success - Charging"
            elif battery_power == 0:  # Idle
                return "✅ Success - Idle"
            else:  # Discharging outside MD (unusual)
                return "🟡 Partial - Non-MD Discharge"
                
    except Exception as e:
        return "❓ Unknown"


def _simulate_battery_operation_v2(df, power_col, monthly_targets, battery_sizing, battery_params, interval_hours, selected_tariff=None, holidays=None):
    """
    V2-specific battery simulation with monthly target floor constraints.
    """
    import numpy as np
    
    # Create simulation dataframe
    df_sim = df[[power_col]].copy()
    df_sim['Original_Demand'] = df_sim[power_col]
    
    # V2 ENHANCEMENT: Create dynamic monthly target series
    target_series = _create_v2_dynamic_target_series(df_sim.index, monthly_targets)
    df_sim['Monthly_Target'] = target_series
    df_sim['Excess_Demand'] = (df_sim[power_col] - df_sim['Monthly_Target']).clip(lower=0)
    
    # Battery parameters - FIXED: Ensure all values are scalar floats
    battery_capacity = float(battery_sizing['capacity_kwh'])
    usable_capacity = float(battery_capacity * (battery_params['depth_of_discharge'] / 100))
    max_power = float(battery_sizing['power_rating_kw'])
    efficiency = float(battery_params['round_trip_efficiency'] / 100)
    
    # Initialize arrays
    soc = np.zeros(len(df_sim))
    soc_percent = np.zeros(len(df_sim))
    battery_power = np.zeros(len(df_sim))
    net_demand = df_sim[power_col].copy()
    
    # Main simulation loop
    for i in range(len(df_sim)):
        # FIXED: Convert all pandas Series values to scalar floats
        current_demand = float(df_sim[power_col].iloc[i])
        monthly_target = float(df_sim['Monthly_Target'].iloc[i])
        excess = max(0, current_demand - monthly_target)
        current_timestamp = df_sim.index[i]
        
        # Get current SOC - FIXED: Ensure scalar values
        current_soc_kwh = float(soc[i-1]) if i > 0 else float(usable_capacity * 0.80)
        current_soc_percent = float((current_soc_kwh / usable_capacity) * 100)
        
        # Discharge logic
        if excess > 0 and is_md_window(current_timestamp, holidays):
            # Calculate discharge power (limited by monthly target floor)
            max_allowable_discharge = current_demand - monthly_target
            
            # C-rate and SOC constraints
            power_limits = _calculate_c_rate_limited_power_simple(
                current_soc_percent, max_power, battery_capacity, 1.0
            )
            max_discharge_power = power_limits['max_discharge_power_kw']
            
            # Apply all constraints - FIXED: Ensure all values are scalar
            required_discharge = min(
                float(max_allowable_discharge),
                float(max_power),
                float(max_discharge_power)
            )
            
            # Check energy availability (5% minimum SOC)
            min_soc_energy = float(usable_capacity * 0.05)
            max_discharge_energy = max(0, float(current_soc_kwh - min_soc_energy))
            max_discharge_from_energy = float(max_discharge_energy / interval_hours)
            
            actual_discharge = min(float(required_discharge), float(max_discharge_from_energy))
            actual_discharge = max(0, float(actual_discharge))
            
            battery_power[i] = actual_discharge
            soc[i] = current_soc_kwh - actual_discharge * interval_hours
            net_demand.iloc[i] = max(current_demand - actual_discharge, monthly_target)
            
        else:
            # Charging logic (simplified)
            soc[i] = current_soc_kwh
            
            # Basic charging if demand is low and SOC < 95%
            if current_soc_percent < 95 and current_demand < monthly_target * 0.8:
                # FIXED: Ensure all calculations use scalar values
                remaining_capacity = float(usable_capacity * 0.95 - current_soc_kwh)
                charge_power = min(
                    float(max_power * 0.3), 
                    float(remaining_capacity / interval_hours)
                )
                
                if charge_power > 0:
                    battery_power[i] = -float(charge_power)
                    soc[i] = float(current_soc_kwh + charge_power * interval_hours * efficiency)
                    net_demand.iloc[i] = current_demand + charge_power
                else:
                    net_demand.iloc[i] = current_demand
            else:
                net_demand.iloc[i] = current_demand
        
        # Ensure SOC limits - FIXED: All operations use scalar values
        soc[i] = max(float(usable_capacity * 0.05), min(float(soc[i]), float(usable_capacity * 0.95)))
        soc_percent[i] = float((soc[i] / usable_capacity) * 100)
    
    # Add results to dataframe
    df_sim['Battery_Power_kW'] = battery_power
    df_sim['Battery_SOC_kWh'] = soc
    df_sim['Battery_SOC_Percent'] = soc_percent
    df_sim['Net_Demand_kW'] = net_demand
    
    # Add shaving success classification for chart compatibility
    df_sim['Shaving_Success'] = df_sim.apply(lambda row: _get_enhanced_shaving_success(row, holidays), axis=1)
    
    # Calculate metrics - FIXED: Ensure scalar calculations
    total_discharge = float(sum([p * interval_hours for p in battery_power if p > 0]))
    total_charge = float(sum([abs(p) * interval_hours for p in battery_power if p < 0]))
    
    return {
        'df_sim': df_sim,
        'total_discharge_kwh': total_discharge,
        'total_charge_kwh': total_charge,
        'peak_reduction_kw': float(df_sim['Original_Demand'].max() - df_sim['Net_Demand_kW'].max()),
        'avg_soc_percent': float(df_sim['Battery_SOC_Percent'].mean())
    }


def _handle_battery_simulation_workflow(simulation_params):
    """
    Handle the complete battery simulation workflow.
    This function is called after _simulate_battery_operation_v2 is defined to avoid function order issues.
    """
    # Extract parameters
    simulation_data = simulation_params['simulation_data']
    power_col = simulation_params['power_col']
    monthly_targets = simulation_params['monthly_targets']
    battery_sizing = simulation_params['battery_sizing']
    battery_params = simulation_params['battery_params']
    interval_hours = simulation_params['interval_hours']
    selected_tariff = simulation_params['selected_tariff']
    holidays = simulation_params['holidays']
    enable_forecasting = simulation_params.get('enable_forecasting', False)
    battery_config = simulation_params.get('battery_config', {})
    
    # Run the battery simulation
    st.markdown("### 🔄 Running Battery Operation Simulation...")
    with st.spinner("Simulating battery operation..."):
        simulation_results = _simulate_battery_operation_v2(
            df=simulation_data,
            power_col=power_col,
            monthly_targets=monthly_targets,
            battery_sizing=battery_sizing,
            battery_params=battery_params,
            interval_hours=interval_hours,
            selected_tariff=selected_tariff,
            holidays=holidays
        )
    
    if simulation_results and 'df_sim' in simulation_results:
        st.success("✅ **Battery simulation completed successfully!**")
        
        # Display simulation summary
        df_sim = simulation_results['df_sim']
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            total_discharge = df_sim[df_sim['Battery_Power_kW'] > 0]['Battery_Power_kW'].sum() * interval_hours
            st.metric("Total Discharge", f"{total_discharge:.1f} kWh")
            
        with col2:
            total_charge = abs(df_sim[df_sim['Battery_Power_kW'] < 0]['Battery_Power_kW'].sum()) * interval_hours
            st.metric("Total Charge", f"{total_charge:.1f} kWh")
            
        with col3:
            peak_reduction = df_sim['Original_Demand'].max() - df_sim['Net_Demand_kW'].max()
            st.metric("Peak Reduction", f"{peak_reduction:.1f} kW")
            
        with col4:
            avg_soc = df_sim['Battery_SOC_Percent'].mean()
            st.metric("Avg SOC", f"{avg_soc:.1f}%")
        
        # Display the comprehensive battery simulation chart
        st.markdown("### 📊 Interactive Battery Operation Analysis")
        
        # Prepare proper sizing dictionary for chart display
        if hasattr(st.session_state, 'tabled_analysis_selected_battery'):
            selected_battery = st.session_state.tabled_analysis_selected_battery
            battery_spec = selected_battery['spec']
            quantity = getattr(st.session_state, 'tabled_analysis_battery_quantity', 1)
            
            sizing_for_chart = {
                'power_rating_kw': battery_spec.get('power_kW', 100) * quantity,
                'capacity_kwh': battery_spec.get('energy_kWh', 100) * quantity
            }
        else:
            # Fallback sizing from battery_sizing
            sizing_for_chart = {
                'power_rating_kw': battery_sizing.get('power_rating_kw', 100),
                'capacity_kwh': battery_sizing.get('capacity_kwh', 100)
            }
        
        # Display battery simulation chart
        st.markdown("### 📊 V2 Battery Operation Analysis")
        
        _display_v2_battery_simulation_chart(
            df_sim=df_sim,
            monthly_targets=monthly_targets,
            sizing=sizing_for_chart,
            selected_tariff=selected_tariff,
            holidays=holidays
        )
        
        # Store simulation results for further analysis
        st.session_state['v2_simulation_results'] = simulation_results
        
    else:
        st.error("❌ Battery simulation failed. Please check your configuration.")

def _display_v2_battery_simulation_chart(df_sim, monthly_targets=None, sizing=None, selected_tariff=None, holidays=None):

    """
    V2-specific battery operation simulation chart with DYNAMIC monthly targets.
    
    Key V2 Enhancement: Replaces static target line with stepped monthly target line.
    
    Args:
        df_sim: Simulation dataframe with battery operation data
        monthly_targets: V2's dynamic monthly targets (Series with Period index)
        sizing: Battery sizing dictionary from V2 analysis
        selected_tariff: Tariff configuration for MD period detection
        holidays: Set of holiday dates
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    # Handle None parameters with safe defaults
    if monthly_targets is None:
        st.error("❌ V2 Chart Error: monthly_targets is required for dynamic target visualization")
        return
        
    if sizing is None:
        sizing = {'power_rating_kw': 100, 'capacity_kwh': 100}
    
    # ===== V2 TWO-LEVEL CASCADING FILTERING =====
    st.markdown("##### 🎯 V2 Two-Level Cascading Filters")
    
    # Success/Failure dropdown filter instead of timestamp filter
    if len(df_sim) > 0:
        # Calculate shaving success for each point if not already available
        if 'Shaving_Success' not in df_sim.columns:
            # Use the comprehensive battery status if Success_Status exists
            if 'Success_Status' in df_sim.columns:
                df_sim['Shaving_Success'] = df_sim['Success_Status']
            else:
                df_sim['Shaving_Success'] = df_sim.apply(lambda row: _get_enhanced_shaving_success(row, holidays), axis=1)
        
        # ===== LEVEL 1: DAY TYPE FILTER =====
        col1, col2 = st.columns([4, 1])
        with col1:
            filter_options = [
                "All Days",
                "All Success Days", 
                "All Partial Days",
                "All Failed Days"
            ]
            
            selected_filter = st.selectbox(
                "🎯 Level 1: Filter by Day Type:",
                options=filter_options,
                index=0,
                key="chart_success_filter",
                help="First level: Filter chart data to show complete days that contain specific event types"
            )
            
        with col2:
            if st.button("🔄 Reset All Filters", key="reset_chart_success_filter"):
                st.session_state.chart_success_filter = "All Days"
                if 'specific_day_filter' in st.session_state:
                    del st.session_state.specific_day_filter
                st.rerun()
        
        # ===== LEVEL 2: SPECIFIC DAY FILTER =====
        level2_days = []
        
        # Get available days based on Level 1 filter
        if selected_filter == "All Success Days":
            success_patterns = '✅ Success'
            success_days = df_sim[df_sim['Shaving_Success'].str.contains(success_patterns, na=False)].index.date
            level2_days = sorted(set(success_days))
        elif selected_filter == "All Partial Days":
            partial_patterns = '🟡 Partial'
            partial_days = df_sim[df_sim['Shaving_Success'].str.contains(partial_patterns, na=False)].index.date
            level2_days = sorted(set(partial_days))
        elif selected_filter == "All Failed Days":
            failed_patterns = '🔴 Failed'
            failed_days = df_sim[df_sim['Shaving_Success'].str.contains(failed_patterns, na=False)].index.date
            level2_days = sorted(set(failed_days))
        else:
            all_days = sorted(set(df_sim.index.date))
            level2_days = all_days
        
        # Always show Level 2 filter interface
        st.markdown("**Level 2: Select Specific Day for Detailed Analysis**")
        col3, col4 = st.columns([5, 1])
        
        with col3:
            # Create options for specific day selection
            if selected_filter == "All Days":
                day_options = ["All Days"]
            else:
                day_options = ["All " + selected_filter.split()[-2] + " " + selected_filter.split()[-1]]
            
            # Add individual days if available
            if level2_days:
                day_options.extend([str(day) for day in level2_days])
            
            selected_specific_day = st.selectbox(
                "🎯 Select Specific Day:",
                options=day_options,
                index=0,
                key="specific_day_filter",
                help="Second level: Choose a specific date for detailed analysis"
            )
        
        with col4:
            if st.button("🔄 Reset Day", key="reset_specific_day_filter"):
                if 'specific_day_filter' in st.session_state:
                    del st.session_state.specific_day_filter
                st.rerun()
                
        # ===== APPLY FILTERS TO DATA =====
        df_filtered = df_sim.copy()
        
        # Apply Level 1 filter
        if selected_filter == "All Success Days":
            success_patterns = '✅ Success'
            success_days = set(df_sim[df_sim['Shaving_Success'].str.contains(success_patterns, na=False)].index.date)
            df_filtered = df_sim[pd.Series(df_sim.index.date).isin(success_days).values]
        elif selected_filter == "All Partial Days":
            partial_patterns = '🟡 Partial'
            partial_days = set(df_sim[df_sim['Shaving_Success'].str.contains(partial_patterns, na=False)].index.date)
            df_filtered = df_sim[pd.Series(df_sim.index.date).isin(partial_days).values]
        elif selected_filter == "All Failed Days":
            failed_patterns = '🔴 Failed'
            failed_days = set(df_sim[df_sim['Shaving_Success'].str.contains(failed_patterns, na=False)].index.date)
            df_filtered = df_sim[pd.Series(df_sim.index.date).isin(failed_days).values]
        
        # Apply Level 2 filter (specific day selection)
        if selected_specific_day not in ["All Days", "All Success Days", "All Partial Days", "All Failed Days"]:
            try:
                from datetime import datetime
                selected_date = datetime.strptime(selected_specific_day, "%Y-%m-%d").date()
                df_filtered = df_filtered[df_filtered.index.date == selected_date]
            except:
                st.error(f"Invalid date format: {selected_specific_day}")
                return
        
        if len(df_filtered) == 0:
            st.warning(f"No data matches the selected filters: {selected_filter} + {selected_specific_day}")
            return
    else:
        st.error("❌ No simulation data available")
        return
    
    # Resolve Net Demand column name flexibly
    net_candidates = ['Net_Demand_kW', 'Net_Demand_KW', 'Net_Demand']
    net_col = next((c for c in net_candidates if c in df_filtered.columns), None)
    
    # Validate required columns exist
    required_base = ['Original_Demand', 'Battery_Power_kW', 'Battery_SOC_Percent']
    missing_columns = [col for col in required_base if col not in df_filtered.columns]
    if net_col is None:
        missing_columns.append('Net_Demand_kW')
    
    if missing_columns:
        st.error(f"❌ Missing required columns in V2 simulation data: {missing_columns}")
        st.info("Available columns: " + ", ".join(df_filtered.columns.tolist()))
        return
    
    # Create V2 dynamic target series (stepped monthly targets) - filtered to match chart data
    target_series = _create_v2_dynamic_target_series(df_filtered.index, monthly_targets)
    
    # Display filtered event range info
    if selected_filter != "All Days" and len(df_filtered) > 0:
        filter_start = df_filtered.index.min()
        filter_end = df_filtered.index.max()
        st.info(f"📅 **Filtered Event Range**: {filter_start.strftime('%Y-%m-%d %H:%M')} to {filter_end.strftime('%Y-%m-%d %H:%M')}")
    
    # Panel 1: V2 Enhanced MD Shaving Effectiveness with Dynamic Monthly Targets
    st.markdown("##### 1️⃣ V2 MD Shaving Effectiveness: Demand vs Battery vs Dynamic Monthly Targets")
    
    # Display filtering status info (updated for always-visible Level 2)
    level2_active = ('specific_day_filter' in st.session_state and 
                    st.session_state.get('specific_day_filter', '').strip() and 
                    not st.session_state.get('specific_day_filter', '').startswith("All "))
    
    if level2_active:
        specific_day = st.session_state.get('specific_day_filter', '')
        st.info(f"🆕 **V2 Enhancement with Two-Level Filtering**: Target line changes monthly based on V2 configuration, showing **{selected_filter}** filtered to **{specific_day}**")
    elif selected_filter != "All Days":
        st.info(f"🆕 **V2 Enhancement with Level 1 Filtering**: Target line changes monthly based on V2 configuration, showing only **{selected_filter.lower()}**")
    else:
        st.info("🆕 **V2 Enhancement**: Target line changes monthly based on your V2 target configuration")
    
    fig = go.Figure()
    
    # Add demand lines
    fig.add_trace(
        go.Scatter(x=df_filtered.index, y=df_filtered[net_col], 
                  name='Net Demand (with Battery)', line=dict(color='#00BFFF', width=2),
                  hovertemplate='Net: %{y:.1f} kW<br>%{x}<extra></extra>')
    )
    
    # V2 ENHANCEMENT: Add stepped monthly target line instead of static line
    fig.add_trace(
        go.Scatter(x=df_filtered.index, y=target_series, 
                  name='Monthly Target (V2 Dynamic)', 
                  line=dict(color='green', dash='dash', width=3),
                  hovertemplate='Monthly Target: %{y:.1f} kW<br>%{x}<extra></extra>')
    )
    
    # Replace area fills with bar charts for battery discharge/charge
    discharge_series = df_filtered['Battery_Power_kW'].where(df_filtered['Battery_Power_kW'] > 0, other=0)
    charge_series = df_filtered['Battery_Power_kW'].where(df_filtered['Battery_Power_kW'] < 0, other=0)
    
    # Discharge bars
    fig.add_trace(go.Bar(
        x=df_filtered.index,
        y=discharge_series,
        name='Battery Discharge (kW)',
        marker=dict(color='orange'),
        opacity=0.6,
        hovertemplate='Discharge: %{y:.1f} kW<br>%{x}<extra></extra>',
        yaxis='y2'
    ))
    
    # Charge bars (negative values)
    fig.add_trace(go.Bar(
        x=df_filtered.index,
        y=charge_series,
        name='Battery Charge (kW)',
        marker=dict(color='green'),
        opacity=0.6,
        hovertemplate='Charge: %{y:.1f} kW<br>%{x}<extra></extra>',
        yaxis='y2'
    ))
    
    # V2 ENHANCEMENT: Add dynamic conditional coloring using monthly targets instead of static average
    # This replaces the V1 averaging approach with dynamic monthly target-based coloring
    try:
        fig = _create_v2_conditional_demand_line_with_dynamic_targets(
            fig, df_filtered, 'Original_Demand', target_series, selected_tariff, holidays, "Original Demand"
        )
        
    except Exception as e:
        st.error(f"❌ Error in chart conditional coloring: {str(e)}")
        # Continue without the enhanced coloring
        st.warning("⚠️ Continuing chart without enhanced coloring due to error")
    
    # Compute symmetric range for y2 to show positive/negative bars
    try:
        max_abs_power = float(df_filtered['Battery_Power_kW'].abs().max())
    except Exception:
        max_abs_power = float(sizing.get('power_rating_kw', 100))
    y2_limit = max(max_abs_power * 1.1, sizing.get('power_rating_kw', 100) * 0.5)
    
    fig.update_layout(
        title='🎯 V2 MD Shaving Effectiveness: Demand vs Battery vs Dynamic Monthly Targets',
        xaxis_title='Time',
        yaxis_title='Power Demand (kW)',
        yaxis2=dict(
            title='Battery Power (kW) [+ discharge | - charge]',
            overlaying='y',
            side='right',
            range=[-y2_limit, y2_limit],
            zeroline=True,
            zerolinecolor='gray'
        ),
        height=500,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="top", 
            y=-0.15,
            xanchor="center", 
            x=0.5
        ),
        margin=dict(b=100),
        barmode='overlay',
        template="none",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # V2 ENHANCEMENT INFO: Add explanation about dynamic color coding
    st.info("""
    🆕 **V2 Color Coding Enhancement**: The colored line segments now use **dynamic monthly targets** instead of a static average target.
    - **Blue segments**: Below monthly target (acceptable levels)
    - **Green segments**: Above monthly target during off-peak periods (energy cost only)
    - **Red segments**: Above monthly target during peak periods (energy + MD cost impact)
    
    This provides more accurate visual feedback about when intervention is needed based on realistic monthly billing patterns.
    """)
    
    # ===== V2 SUMMARY METRICS =====
    # Get dynamic interval hours for energy calculations
    interval_hours = _get_dynamic_interval_hours(df_filtered)
    
    # Calculate basic metrics
    total_energy_discharged = df_filtered['Battery_Power_kW'].where(df_filtered['Battery_Power_kW'] > 0, 0).sum() * interval_hours
    total_energy_charged = abs(df_filtered['Battery_Power_kW'].where(df_filtered['Battery_Power_kW'] < 0, 0).sum()) * interval_hours
    success_rate = len([s for s in df_filtered['Shaving_Success'] if '✅' in s]) / len(df_filtered) * 100
    
    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Energy Discharged", f"{total_energy_discharged:.1f} kWh")
    col2.metric("Energy Charged", f"{total_energy_charged:.1f} kWh")
    col3.metric("Success Rate", f"{success_rate:.1f}%")
    col4.metric("Avg SOC", f"{df_filtered['Battery_SOC_Percent'].mean():.1f}%")
    
    # Panel 2: Combined SOC and Battery Power Chart (same as V1)
    st.markdown("##### 2️⃣ Combined SOC and Battery Power Chart")
    
    fig2 = make_subplots(specs=[[{"secondary_y": True}]])
    
    # SOC line (left y-axis)
    fig2.add_trace(
        go.Scatter(x=df_filtered.index, y=df_filtered['Battery_SOC_Percent'],
                  name='SOC (%)', line=dict(color='purple', width=2),
                  hovertemplate='SOC: %{y:.1f}%<br>%{x}<extra></extra>'),
        secondary_y=False
    )
    
    # Battery power line (right y-axis) 
    fig2.add_trace(
        go.Scatter(x=df_filtered.index, y=df_filtered['Battery_Power_kW'],
                  name='Battery Power', line=dict(color='orange', width=2),
                  hovertemplate='Power: %{y:.1f} kW<br>%{x}<extra></extra>'),
        secondary_y=True
    )
    
    # Add horizontal line for minimum SOC warning (updated to 10% based on 5% safety limit)
    fig2.add_hline(y=10, line_dash="dot", line_color="red", 
                   annotation_text="Low SOC Warning (10% - 5% Safety Limit)", secondary_y=False)
    
    # Update axes
    fig2.update_xaxes(title_text="Time")
    fig2.update_yaxes(title_text="State of Charge (%)", secondary_y=False, range=[0, 100])
    fig2.update_yaxes(title_text="Battery Discharge Power (kW)", secondary_y=True)
    
    fig2.update_layout(
        title='⚡ SOC vs Battery Power: Timing Analysis',
        height=400,
        hovermode='x unified',
        template="none",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig2, use_container_width=True)
    
    # Panel 3: Battery Power Utilization Heatmap (same as V1)
    st.markdown("##### 3️⃣ Battery Power Utilization Heatmap")
    
    # Prepare data for heatmap
    df_heatmap = df_filtered.copy()
    df_heatmap['Date'] = df_heatmap.index.date
    df_heatmap['Hour'] = df_heatmap.index.hour
    df_heatmap['Battery_Utilization_%'] = (df_heatmap['Battery_Power_kW'] / sizing['power_rating_kw'] * 100).clip(0, 100)
    
    # Create pivot table for heatmap
    heatmap_data = df_heatmap.pivot_table(
        values='Battery_Utilization_%', 
        index='Hour', 
        columns='Date', 
        aggfunc='mean',
        fill_value=0
    )
    
    # Create heatmap
    fig3 = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=[str(d) for d in heatmap_data.columns],
        y=heatmap_data.index,
        colorscale='Viridis',
        hoverongaps=False,
        hovertemplate='Date: %{x}<br>Hour: %{y}<br>Utilization: %{z:.1f}%<extra></extra>',
        colorbar=dict(title="Battery Utilization (%)")
    ))
    
    fig3.update_layout(
        title='🔥 Battery Power Utilization Heatmap (% of Rated Power)',
        xaxis_title='Date',
        yaxis_title='Hour of Day',
        height=400,
        template="none",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig3, use_container_width=True)
    
    # V2 Key insights with monthly target context
    st.markdown("##### 🔍 V2 Key Insights from Enhanced Monthly Target Analysis")
    
    insights = []
    
    # Use V2 energy efficiency calculation
    energy_efficiency = (total_energy_discharged / max(total_energy_charged, 1) * 100)
        
    if energy_efficiency < 80:
        insights.append("⚠️ **V2 MD Energy Shortfall**: Battery capacity may be insufficient for complete monthly target-based MD peak shaving")
    elif energy_efficiency >= 95:
        insights.append("✅ **Excellent V2 MD Coverage**: Battery effectively handles all monthly target energy requirements")
    
    # Check V2 success rate
    if success_rate > 90:
        insights.append("✅ **High V2 Success Rate**: Battery effectively manages most peak events against dynamic monthly targets")
    elif success_rate < 60:
        insights.append("❌ **Low V2 Success Rate**: Consider increasing battery power rating or capacity for better monthly target management")
    
    # Check battery utilization if heatmap data is available
    if len(df_heatmap) > 0:
        avg_utilization = df_heatmap['Battery_Utilization_%'].mean()
        if avg_utilization < 30:
            insights.append("📊 **Under-utilized**: Battery power rating may be oversized for V2 monthly targets")
        elif avg_utilization > 80:
            insights.append("🔥 **High Utilization**: Battery operating near maximum capacity for V2 monthly targets")
    
    # Check for low SOC events (updated to 10% warning threshold based on 5% safety limit)
    low_soc_events = len(df_filtered[df_filtered['Battery_SOC_Percent'] < 10])
    if low_soc_events > 0:
        insights.append(f"🔋 **Low SOC Warning**: {low_soc_events} intervals with SOC below 10% during V2 operation (5% safety limit)")
    
    # Add insight about V2 methodology
    if len(monthly_targets) > 0:
        insights.append(f"📊 **V2 Innovation**: Analysis uses {len(monthly_targets)} dynamic monthly targets vs traditional static targets for superior accuracy")
        insights.append(f"🎨 **V2 Color Enhancement**: Line color coding now reflects dynamic monthly targets instead of static averaging - providing month-specific intervention guidance")
    
    if not insights:
        insights.append("✅ **Optimal V2 Performance**: Battery system operating within acceptable parameters with monthly targets")
    
    for insight in insights:

        st.info(insight)             

def render_md_shaving_v2():
    """
    Main function to display the MD Shaving Solution V2 interface.
    Enhanced implementation with file upload functionality.
    """
    st.title("🔋 MD Shaving Solution V2")
    st.markdown("""
    **Next-generation Maximum Demand (MD) shaving analysis** with enhanced features and advanced optimization algorithms.
    
    🆕 **V2 Enhancements:**
    - 🔧 **Advanced Battery Sizing**: Multi-parameter optimization algorithms
    - 📊 **Multi-Scenario Analysis**: Compare different battery configurations
    - 💰 **Enhanced Cost Analysis**: ROI calculations and payback period analysis
    - 📈 **Improved Visualizations**: Interactive charts and detailed reporting
    - 🎯 **Smart Recommendations**: AI-powered optimization suggestions
    """)
    
    # File upload section
    st.subheader("📁 Data Upload")
    st.markdown("*Upload your energy data file to begin MD shaving analysis*")
    
    uploaded_file = st.file_uploader(
        "Upload your energy data file", 
        type=["csv", "xls", "xlsx"], 
        key="md_shaving_v2_file_uploader",
        help="Upload your load profile data (CSV, Excel formats supported)"
    )
    
    if uploaded_file:
        try:
            # Read the uploaded file
            with st.spinner("Reading uploaded file..."):
                df = read_uploaded_file(uploaded_file)
            
            if df is None or df.empty:
                st.error("❌ The uploaded file appears to be empty or invalid.")
                return
            
            if len(df.columns) < 2:
                st.error("❌ The uploaded file doesn't have enough columns. Need at least timestamp and power columns.")
                return
            
            st.success(f"✅ File uploaded successfully! Shape: {df.shape[0]} rows × {df.shape[1]} columns")
            
            # Configure data inputs (reusing V1 logic)
            try:
                timestamp_col, power_col, holidays = _configure_data_inputs(df)
                
                if timestamp_col and power_col:
                    # Process the dataframe
                    with st.spinner("Processing data..."):
                        df_processed = _process_dataframe(df, timestamp_col)
                        # Store processed dataframe and column names in session state for later use
                        st.session_state['df_processed'] = df_processed
                        st.session_state['v2_power_col'] = power_col
                        st.session_state['v2_timestamp_col'] = timestamp_col
                    
                    st.success(f"✅ Data processed successfully! Final shape: {df_processed.shape[0]} rows")
                    
                    # Display basic statistics
                    st.subheader("📊 Data Summary")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Records", f"{len(df_processed):,}")
                        
                    with col2:
                        date_range = df_processed.index.max() - df_processed.index.min()
                        st.metric("Date Range", f"{date_range.days} days")
                        
                    with col3:
                        avg_power = df_processed[power_col].mean()
                        st.metric("Average Power", f"{avg_power:.2f} kW")
                    
                    # Data preview
                    st.subheader("📋 Data Preview")
                    st.dataframe(df_processed[[power_col]].head(10), use_container_width=True)
                    
                    # V2 Tariff Selection
                    st.subheader("⚡ V2 Tariff Configuration")
                    try:
                        selected_tariff = _configure_tariff_selection()
                        
                        if selected_tariff:
                            st.success("✅ Tariff configuration completed!")
                            
                            # V2 Monthly Peak Analysis
                            st.subheader("📊 V2 Monthly Peak Analysis")
                            try:
                                with st.spinner("Calculating tariff-specific monthly peaks..."):
                                    monthly_general_peaks, monthly_tou_peaks, tariff_type = _calculate_tariff_specific_monthly_peaks(
                                        df_processed, power_col, selected_tariff, holidays
                                    )
                                
                                st.success(f"✅ Monthly peaks calculated for {tariff_type} tariff")
                                
                                # Display peak analysis results
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.subheader("📈 General Peaks (24/7)")
                                    if not monthly_general_peaks.empty:
                                        st.dataframe(monthly_general_peaks, use_container_width=True)
                                    else:
                                        st.info("No general peaks calculated")
                                
                                with col2:
                                    st.subheader("🕐 TOU Peaks (2PM-10PM Weekdays)")
                                    if not monthly_tou_peaks.empty:
                                        st.dataframe(monthly_tou_peaks, use_container_width=True)
                                    else:
                                        st.info("No TOU peaks calculated")
                                
                                # V2 Monthly Targets Calculation
                                st.subheader("🎯 V2 Monthly Targets Calculation")
                                
                                # Target method selection for V2
                                target_method = st.selectbox(
                                    "Select target calculation method:",
                                    ["Percentage to Shave", "Percentage of Current Max", "Manual Target (kW)"],
                                    help="Choose how to calculate monthly demand targets"
                                )
                                
                                # Method-specific parameters
                                shave_percent = None
                                target_percent = None
                                target_manual_kw = None
                                
                                if target_method == "Percentage to Shave":
                                    shave_percent = st.slider("Shave Percentage (%)", 1, 50, 10)
                                elif target_method == "Percentage of Current Max":
                                    target_percent = st.slider("Target Percentage of Peak (%)", 50, 95, 85)
                                elif target_method == "Manual Target (kW)":
                                    target_manual_kw = st.number_input("Manual Target (kW)", min_value=0.0, value=1000.0, step=10.0)
                                
                                # Calculate monthly targets (Automatic)
                                try:
                                    # Validate inputs before calculation
                                    if target_method == "Percentage to Shave" and shave_percent is None:
                                        st.error("❌ Please set the shave percentage")
                                    elif target_method == "Percentage of Current Max" and target_percent is None:
                                        st.error("❌ Please set the target percentage")
                                    elif target_method == "Manual Target (kW)" and target_manual_kw is None:
                                        st.error("❌ Please set the manual target value")
                                    else:
                                        with st.spinner("Calculating V2 monthly targets..."):
                                            monthly_targets, reference_peaks, calc_tariff_type, target_description = _calculate_monthly_targets_v2(
                                                df_processed, power_col, selected_tariff, holidays,
                                                target_method, shave_percent, target_percent, target_manual_kw
                                            )
                                        
                                        st.success("✅ V2 Monthly targets calculated successfully!")
                                        
                                        # Store in session state for use in other functions
                                        st.session_state['v2_monthly_targets'] = monthly_targets
                                        st.session_state['v2_reference_peaks'] = reference_peaks
                                        st.session_state['v2_tariff_type'] = calc_tariff_type
                                        st.session_state['v2_target_description'] = target_description
                                        
                                        # Generate Monthly Target Calculation Summary Table
                                        if not reference_peaks.empty and not monthly_targets.empty:
                                            # Get the general and TOU peaks for comparison
                                            monthly_general_peaks, monthly_tou_peaks, _ = _calculate_tariff_specific_monthly_peaks(
                                                df_processed, power_col, selected_tariff, holidays
                                            )
                                            
                                            comparison_data = []
                                            
                                            for month_period in reference_peaks.index:
                                                general_peak = monthly_general_peaks[month_period] if month_period in monthly_general_peaks.index else 0
                                                tou_peak = monthly_tou_peaks[month_period] if month_period in monthly_tou_peaks.index else 0
                                                reference_peak = reference_peaks[month_period]
                                                target = monthly_targets[month_period]
                                                shaving_amount = reference_peak - target
                                                
                                                comparison_data.append({
                                                    'Month': str(month_period),
                                                    'General Peak (24/7)': f"{general_peak:.1f} kW",
                                                    'TOU Peak (2PM-10PM)': f"{tou_peak:.1f} kW",
                                                    'Reference Peak': f"{reference_peak:.1f} kW",
                                                    'Target MD': f"{target:.1f} kW",
                                                    'Shaving Amount': f"{shaving_amount:.1f} kW",
                                                    'Tariff Type': calc_tariff_type
                                                })
                                            
                                            df_comparison = pd.DataFrame(comparison_data)
                                            
                                            st.markdown("#### 6.1 📋 Monthly Target Calculation Summary")
                                            
                                            # Highlight the reference column based on tariff type
                                            def highlight_reference_peak(row):
                                                colors = []
                                                for col in row.index:
                                                    if col == 'Reference Peak':
                                                        colors.append('background-color: rgba(0, 255, 0, 0.3)')  # Green highlight
                                                    elif col == 'TOU Peak (2PM-10PM)' and calc_tariff_type == 'TOU':
                                                        colors.append('background-color: rgba(255, 255, 0, 0.2)')  # Yellow highlight
                                                    elif col == 'General Peak (24/7)' and calc_tariff_type == 'General':
                                                        colors.append('background-color: rgba(255, 255, 0, 0.2)')  # Yellow highlight
                                                    else:
                                                        colors.append('')
                                                return colors
                                            
                                            styled_comparison = df_comparison.style.apply(highlight_reference_peak, axis=1)
                                            st.dataframe(styled_comparison, use_container_width=True, hide_index=True)
                                            
                                            st.info(f"""
                                            **📊 Target Calculation Explanation:**
                                            - **General Peak**: Highest demand anytime (24/7) 
                                            - **TOU Peak**: Highest demand during peak period (2PM-10PM weekdays only)
                                            - **Reference Peak**: Used for target calculation based on {calc_tariff_type} tariff
                                            - **Target MD**: {target_description}
                                            - 🟢 **Green**: Reference peak used for calculations
                                            - 🟡 **Yellow**: Peak type matching selected tariff
                                            """)
                                    
                                except Exception as e:
                                    st.error(f"❌ Error calculating V2 monthly targets: {str(e)}")
                                
                            except Exception as e:
                                st.error(f"❌ Error calculating monthly peaks: {str(e)}")
                        
                        else:
                            st.warning("⚠️ Please configure tariff settings to proceed with V2 analysis")
                            
                    except Exception as e:
                        st.error(f"❌ Error configuring tariff: {str(e)}")
                    
                    # V2 Peak Events Analysis
                    st.subheader("📊 V2 Peak Events Analysis")
                    try:
                        # Detect data interval using V2 function
                        detected_interval_hours = _infer_interval_hours(df_processed.index)
                        st.success(f"✅ Detected sampling interval: {int(round(detected_interval_hours * 60))} minutes")
                        
                        # Peak Events Detection (Automatic)
                        peak_events = []
                        try:
                            with st.spinner("Detecting peak events..."):
                                # Use the monthly targets from session state - MONTH BY MONTH APPROACH
                                if 'v2_monthly_targets' in st.session_state:
                                    monthly_targets = st.session_state['v2_monthly_targets']
                                    all_monthly_events = []
                                    
                                    # Get MD rate from selected tariff
                                    total_md_rate = 0
                                    if selected_tariff and isinstance(selected_tariff, dict):
                                        rates = selected_tariff.get('Rates', {})
                                        total_md_rate = rates.get('Capacity Rate', 0) + rates.get('Network Rate', 0)
                                    
                                    # Process each month separately (like copy file)
                                    for month_period, target_value in monthly_targets.items():
                                        month_start = month_period.start_time
                                        month_end = month_period.end_time
                                        month_mask = (df_processed.index >= month_start) & (df_processed.index <= month_end)
                                        month_data = df_processed[month_mask]
                                        
                                        if not month_data.empty:
                                            # Find peak events for this month using TOU-aware detection
                                            month_peak_events = _detect_peak_events_tou_aware(
                                                month_data, power_col, target_value, total_md_rate, detected_interval_hours, selected_tariff, holidays
                                            )
                                            
                                            # Add month info to each event
                                            for event in month_peak_events:
                                                event['Month'] = str(month_period)
                                                event['Monthly_Target'] = target_value
                                                all_monthly_events.append(event)
                                    
                                    peak_events = all_monthly_events
                                    st.write(f"- Peak events found: {len(peak_events) if peak_events else 0}")
                                else:
                                    st.error("❌ Monthly targets not calculated. Please calculate targets first.")
                                    peak_events = []
                        
                            if peak_events and len(peak_events) > 0:
                                st.success(f"✅ Detected {len(peak_events)} peak events")
                                
                                # Display peak events summary with original detailed columns
                                events_summary = []
                                for i, event in enumerate(peak_events):
                                    events_summary.append({
                                        "Start Date": event.get('Start Date', 'N/A'),
                                        "Start Time": event.get('Start Time', 'N/A'),
                                        "End Date": event.get('End Date', 'N/A'),
                                        "End Time": event.get('End Time', 'N/A'),
                                        "General Peak Load (kW)": f"{event.get('General Peak Load (kW)', 0):.2f}",
                                        "General Excess (kW)": f"{event.get('General Excess (kW)', 0):.2f}",
                                        "TOU Peak Load (kW)": f"{event.get('TOU Peak Load (kW)', 0):.2f}",
                                        "TOU Excess (kW)": f"{event.get('TOU Excess (kW)', 0):.2f}",
                                        "TOU Peak Time": event.get('TOU Peak Time', 'N/A'),
                                        "Duration (min)": f"{event.get('Duration (min)', 0):.1f}",
                                        "General Required Energy (kWh)": f"{event.get('General Required Energy (kWh)', 0):.2f}",
                                        "TOU Required Energy (kWh)": f"{event.get('TOU Required Energy (kWh)', 0):.2f}",
                                        "MD Cost Impact (RM)": f"{event.get('MD Cost Impact (RM)', 0):.2f}",
                                        "Tariff Type": event.get('Tariff Type', 'N/A')
                                    })
                                
                                st.dataframe(pd.DataFrame(events_summary), use_container_width=True)
                            else:
                                st.info("ℹ️ No peak events detected above the targets")
                                
                        except Exception as e:
                            st.error(f"❌ Error detecting peak events: {str(e)}")
                        
                        # V2 Peak Events Timeline Visualization (Automatic)
                        try:
                            with st.spinner("Generating peak events timeline..."):
                                # Create visualization using V2 timeline function
                                fig = _render_v2_peak_events_timeline(
                                    df_processed, 
                                    power_col, 
                                    selected_tariff, 
                                    holidays,
                                    target_method, 
                                    shave_percent, 
                                    target_percent, 
                                    target_manual_kw, 
                                    target_description
                                )
                                
                                if fig:
                                    st.plotly_chart(fig, use_container_width=True)
                                    st.success("✅ Peak events timeline generated successfully!")
                                else:
                                    st.warning("⚠️ Timeline visualization not available")
                                    
                        except Exception as e:
                            st.error(f"❌ Error generating timeline: {str(e)}")
                            # Fallback: Use conditional demand line function
                            try:
                                st.info("🔄 Using alternative visualization...")
                                fig = px.line()
                                # Get targets from session state for fallback
                                avg_target = 1000  # Default value
                                if 'v2_monthly_targets' in st.session_state:
                                    monthly_targets = st.session_state['v2_monthly_targets']
                                    avg_target = monthly_targets.mean() if not monthly_targets.empty else 1000
                                
                                fig = create_conditional_demand_line_with_peak_logic(
                                    fig, 
                                    df_processed, 
                                    power_col, 
                                    avg_target,
                                    selected_tariff, 
                                    holidays, 
                                    "Demand with Peak Logic"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                st.success("✅ Alternative demand visualization generated!")
                            except Exception as fallback_e:
                                st.error(f"❌ Fallback visualization failed: {str(fallback_e)}")
                        
                    except Exception as e:
                        st.error(f"❌ Error in peak events analysis: {str(e)}")
                    
                    # Daily Clustering Summary Table (after peak events detection)
                    try:
                        if 'peak_events' in locals() and peak_events:
                            # Generate daily clustering summary table
                            clustering_summary_df = _generate_clustering_summary_table(
                                peak_events, selected_tariff, holidays
                            )
                            
                            if not clustering_summary_df.empty:
                                st.markdown("#### 6.3.1 📊 Daily Clustering Summary")
                                st.caption("Summary of peak events grouped by date with MD cost impact analysis")
                                
                                # Display the daily clustering summary table
                                st.dataframe(clustering_summary_df, use_container_width=True, hide_index=True)
                    
                    except Exception as e:
                        st.error(f"❌ Error generating daily clustering summary: {str(e)}")
                    
                    # Monthly Summary Table (after peak events detection)
                    try:
                        if 'peak_events' in locals() and peak_events:
                            # Generate monthly summary table (uses daily clustering as intermediary)
                            monthly_summary_df = _generate_monthly_summary_table(
                                peak_events, selected_tariff, holidays
                            )
                            
                            if not monthly_summary_df.empty:
                                st.markdown("#### 6.3.2 📅 Monthly Summary")
                                st.caption("Maximum MD excess and energy requirements aggregated by month")
                                
                                # Display the monthly summary table
                                st.dataframe(monthly_summary_df, use_container_width=True, hide_index=True)
                                
                                # Add summary metrics below the monthly summary table
                                col1, col2, col3 = st.columns(3)
                                
                                # Calculate totals from the monthly summary
                                total_months = len(monthly_summary_df)
                                
                                # Find the column names dynamically (could be "TOU" or "General")
                                md_excess_col = [col for col in monthly_summary_df.columns if 'MD Excess' in col]
                                energy_col = [col for col in monthly_summary_df.columns if 'Required Energy' in col]
                                
                                if md_excess_col and energy_col:
                                    max_monthly_md_excess = monthly_summary_df[md_excess_col[0]].max()
                                    max_monthly_energy = monthly_summary_df[energy_col[0]].max()
                                    
                                    col1.metric("Total Months", total_months)
                                    col2.metric("Max Monthly MD Excess", f"{max_monthly_md_excess:.2f} kW")
                                    col3.metric("Max Monthly Required Energy", f"{max_monthly_energy:.2f} kWh")
                                else:
                                    st.warning("Could not extract summary metrics from monthly data")
                                    
                            else:
                                st.info("No monthly summary data available.")
                                
                    except Exception as e:
                        st.error(f"❌ Error generating monthly summary table: {str(e)}")
                    
                    # Battery Sizing Analysis (using Monthly Summary data)
                    try:
                        if 'peak_events' in locals() and peak_events and 'monthly_summary_df' in locals() and not monthly_summary_df.empty:
                            # Use values from Monthly Summary table for accurate battery sizing
                            md_excess_col = [col for col in monthly_summary_df.columns if 'MD Excess' in col]
                            energy_col = [col for col in monthly_summary_df.columns if 'Required Energy' in col]
                            
                            if md_excess_col and energy_col:
                                max_power_shaving_required = monthly_summary_df[md_excess_col[0]].max()
                                max_required_energy = monthly_summary_df[energy_col[0]].max()
                                total_months = len(monthly_summary_df)
                                total_md_cost = sum([event.get('MD Cost Impact (RM)', 0) for event in peak_events]) if peak_events else 0
                                
                                if max_power_shaving_required > 0:
                                    st.markdown("### 6.5 🔋 Battery Sizing Analysis")
                                    
                                    # Display key metrics using Monthly Summary data
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Total Months", total_months)
                                    with col2:
                                        st.metric("Max Monthly MD Excess", f"{max_power_shaving_required:.2f} kW")
                                    with col3:
                                        st.metric("Max Monthly Required Energy", f"{max_required_energy:.2f} kWh")
                                
                                    # Call the battery sizing analysis function
                                    _render_battery_sizing_analysis(max_power_shaving_required, max_required_energy, total_md_cost)
                            else:
                                st.warning("Could not extract MD excess and energy columns from Monthly Summary data")
                        elif 'peak_events' in locals() and peak_events:
                            # Fallback to individual peak events calculation if Monthly Summary is not available
                            max_power_shaving_required = max([event.get('General Excess (kW)', 0) for event in peak_events]) if peak_events else 0
                            max_required_energy = max([event.get('General Required Energy (kWh)', 0) for event in peak_events]) if peak_events else 0
                            total_md_cost = sum([event.get('MD Cost Impact (RM)', 0) for event in peak_events]) if peak_events else 0
                            
                            if max_power_shaving_required > 0:
                                st.markdown("### 6.5 🔋 Battery Sizing Analysis")
                                st.warning("⚠️ Using individual peak events data (Monthly Summary not available)")
                                
                                # Display key metrics using fallback calculation
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Total Months", len(set([event.get('Start Date', '') for event in peak_events])))
                                with col2:
                                    st.metric("Max Monthly MD Excess", f"{max_power_shaving_required:.2f} kW")
                                with col3:
                                    st.metric("Max Monthly Required Energy", f"{max_required_energy:.2f} kWh")
                                
                                # Call the battery sizing analysis function
                                _render_battery_sizing_analysis(max_power_shaving_required, max_required_energy, total_md_cost)
                                
                    except Exception as e:
                        st.error(f"❌ Error in battery sizing analysis: {str(e)}")
                    
                    # V2 Battery Configuration
                    st.subheader("🔋 V2 Battery Configuration")
                    try:
                        # Pass calculated values if available, otherwise use None for default behavior
                        power_shaving_val = max_power_shaving_required if 'max_power_shaving_required' in locals() else None
                        energy_requirement_val = max_required_energy if 'max_required_energy' in locals() else None
                        
                        battery_config = _render_v2_battery_controls(power_shaving_val, energy_requirement_val)
                        
                        if battery_config:
                            st.success("✅ Battery configuration completed!")
                            
                            # Display configuration summary
                            with st.expander("📋 Battery Configuration Summary"):
                                st.json(battery_config)
                            
                            # Additional V2 Analysis Features
                            st.info("🔄 **Additional V2 analysis features integrated and ready for testing.**")
                            
                            # Forecasting Section
                            st.markdown("---")  # Separator line
                            st.subheader("📈 Forecasting")
                            
                            # Forecasting enable/disable checkbox
                            enable_forecasting = st.checkbox(
                                "Enable Forecasting",
                                value=False,
                                key="v2_enable_forecasting",
                                help="Enable advanced forecasting capabilities for demand prediction and optimization"
                            )
                            
                            if enable_forecasting:
                                st.success("🔮 **Forecasting Mode:** Advanced prediction capabilities activated")
                                
                                # Forecasting Method Selection
                                st.markdown("#### 🔧 Forecasting Method Selection")
                                
                                # Define available forecasting methods
                                forecasting_methods = {
                                    "Rate of Change (ROC)": {
                                        "description": "Analyzes demand patterns based on historical rate of change trends",
                                        "status": "Available",
                                        "complexity": "Medium",
                                        "accuracy": "Good for short-term predictions"
                                    },
                                    "Linear Regression": {
                                        "description": "Statistical method using linear relationships in historical data",
                                        "status": "Coming Soon",
                                        "complexity": "Low",
                                        "accuracy": "Moderate for trend-based data"
                                    },
                                    "ARIMA (AutoRegressive Integrated Moving Average)": {
                                        "description": "Time series forecasting using autoregressive and moving average components",
                                        "status": "Planned",
                                        "complexity": "High",
                                        "accuracy": "High for seasonal data"
                                    },
                                    "Prophet": {
                                        "description": "Facebook's forecasting tool designed for business time series data",
                                        "status": "Planned",
                                        "complexity": "Medium",
                                        "accuracy": "Excellent for seasonal patterns"
                                    },
                                    "LSTM Neural Networks": {
                                        "description": "Deep learning approach for complex pattern recognition in time series",
                                        "status": "Future Release",
                                        "complexity": "Very High",
                                        "accuracy": "Excellent for complex patterns"
                                    }
                                }
                                
                                # Method selection dropdown
                                method_names = list(forecasting_methods.keys())
                                selected_method = st.selectbox(
                                    "Select Forecasting Method:",
                                    options=method_names,
                                    index=0,  # Default to ROC
                                    key="forecasting_method_selection",
                                    help="Choose the forecasting algorithm for demand prediction"
                                )
                                
                                # Display method details in a table
                                if selected_method:
                                    method_details = forecasting_methods[selected_method]
                                    
                                    st.markdown("##### � Method Details")
                                    
                                    # Create method details table
                                    method_info = [
                                        ["Method", selected_method],
                                        ["Description", method_details["description"]],
                                        ["Status", method_details["status"]],
                                        ["Complexity", method_details["complexity"]],
                                        ["Accuracy", method_details["accuracy"]]
                                    ]
                                    
                                    method_df = pd.DataFrame(method_info, columns=["Parameter", "Details"])
                                    st.table(method_df)
                                    
                                    # Status-based messaging
                                    if method_details["status"] == "Available":
                                        st.success(f"✅ **{selected_method}** is ready for use")
                                        
                                        # ROC Method Implementation
                                        if selected_method == "Rate of Change (ROC)":
                                            st.markdown("#### 🔧 ROC Forecasting Configuration")
                                            
                                            # Validate data quality first
                                            try:
                                                # Check for NaN values in the power data
                                                nan_count = df_processed[power_col].isna().sum()
                                                total_points = len(df_processed)
                                                nan_percentage = (nan_count / total_points) * 100 if total_points > 0 else 0
                                                
                                                if nan_count > 0:
                                                    st.warning(f"""
                                                    ⚠️ **Data Quality Notice:** 
                                                    Found {nan_count} missing values ({nan_percentage:.1f}% of data).
                                                    Data will be automatically cleaned using interpolation before forecasting.
                                                    """)
                                                    
                                                    if nan_percentage > 20:
                                                        st.error(f"""
                                                        🚨 **High Missing Data Warning:** 
                                                        {nan_percentage:.1f}% missing values may impact forecast accuracy.
                                                        Consider reviewing your data source for quality issues.
                                                        """)
                                                
                                            except Exception as e:
                                                st.warning(f"Could not validate data quality: {str(e)}")
                                            
                                            # Infer adaptive horizons based on data interval
                                            try:
                                                base_interval, adaptive_horizons, horizons_minutes = get_adaptive_forecast_horizons(df_processed)
                                                
                                                st.info(f"""
                                                📊 **Auto-detected Data Characteristics:**
                                                - Base Interval: {base_interval} 
                                                - Recommended Horizons: {[f'{h}' for h in adaptive_horizons]}
                                                - Available Horizons (minutes): {horizons_minutes}
                                                """)
                                                
                                            except Exception as e:
                                                st.error(f"❌ Error detecting data interval: {str(e)}")
                                                # Fallback to default horizons
                                                horizons_minutes = [1, 5, 10]
                                                st.warning("Using fallback horizons: [1, 5, 10] minutes")
                                            
                                            # Configuration controls
                                            col1, col2 = st.columns(2)
                                            
                                            with col1:
                                                # Use adaptive horizons or fallback
                                                available_options = horizons_minutes if 'horizons_minutes' in locals() else [1, 5, 10, 15, 20, 30]
                                                default_selection = horizons_minutes[:2] if 'horizons_minutes' in locals() else [1, 10]
                                                
                                                horizons = st.multiselect(
                                                    "Forecast Horizons (minutes)",
                                                    options=available_options,
                                                    default=default_selection,
                                                    help="Horizons auto-generated based on your data's native interval"
                                                )
                                            
                                            with col2:
                                                enable_backtesting = st.checkbox(
                                                    "Enable Historical Backtesting",
                                                    value=True,
                                                    help="Generate forecasts for all historical data points"
                                                )
                                            
                                            if enable_backtesting and horizons:
                                                with st.spinner("🔄 Generating ROC forecasts for historical backtesting..."):
                                                    try:
                                                        # Generate time series forecasts for each horizon
                                                        forecast_series = {}
                                                        validation_metrics = {}
                                                        
                                                        for horizon in horizons:
                                                            # Use our ROC forecast function with validation
                                                            validated_df, metrics = roc_forecast_with_validation(
                                                                df_processed[power_col], 
                                                                horizon=horizon, 
                                                                return_metrics=True
                                                            )
                                                            
                                                            # Create clean time series aligned to actual timestamps
                                                            # Each forecast[t] predicts the value at t+horizon
                                                            forecast_ts = pd.Series(
                                                                validated_df['Power_Forecast (kW)'].values,
                                                                index=validated_df['Forecast_Timestamp'],
                                                                name=f'Forecast_{horizon}min'
                                                            )
                                                            
                                                            # Store aligned forecast series and metrics
                                                            forecast_series[f'forecast_{horizon}min'] = forecast_ts
                                                            validation_metrics[horizon] = metrics
                                                        
                                                        # Store in session state for future use
                                                        st.session_state['roc_forecast_series'] = forecast_series
                                                        st.session_state['roc_validation_metrics'] = validation_metrics
                                                        st.session_state['roc_actual_series'] = df_processed[power_col]
                                                        
                                                        # Display summary
                                                        st.success("✅ ROC forecasting completed successfully!")
                                                        
                                                        # Data Quality Report
                                                        try:
                                                            # Get validation report from the first horizon (they all use same data)
                                                            first_horizon = list(validation_metrics.keys())[0] if validation_metrics else None
                                                            if first_horizon:
                                                                # Try to get validation report from ROC calculation
                                                                roc_df_sample = _calculate_roc_from_series(df_processed[power_col])
                                                                if hasattr(roc_df_sample, '_validation_report'):
                                                                    validation_report = roc_df_sample._validation_report
                                                                    
                                                                    with st.expander("📊 Data Quality Report", expanded=False):
                                                                        st.markdown("**Original Data Analysis:**")
                                                                        
                                                                        qual_col1, qual_col2, qual_col3 = st.columns(3)
                                                                        
                                                                        with qual_col1:
                                                                            st.metric("Total Data Points", validation_report['total_points'])
                                                                            
                                                                        with qual_col2:
                                                                            st.metric("Missing Values", validation_report['nan_count'])
                                                                            
                                                                        with qual_col3:
                                                                            quality_color = {
                                                                                'good': '🟢',
                                                                                'fair': '🟡', 
                                                                                'poor': '🔴'
                                                                            }.get(validation_report['data_quality'], '⚪')
                                                                            st.metric("Data Quality", f"{quality_color} {validation_report['data_quality'].title()}")
                                                                        
                                                                        # Additional details
                                                                        if validation_report['nan_count'] > 0:
                                                                            st.markdown(f"""
                                                                            **🔧 Data Cleaning Applied:**
                                                                            - Missing Values: {validation_report['nan_count']} ({validation_report['nan_percentage']:.1f}%)
                                                                            - Fill Method: {validation_report['fill_method_used']}
                                                                            - Remaining Issues: {validation_report.get('remaining_nans', 0)} NaNs
                                                                            """)
                                                                        else:
                                                                            st.info("✅ No missing values detected - data is clean and ready for analysis")
                                                        except Exception as e:
                                                            st.warning(f"Could not generate data quality report: {str(e)}")
                                                        
                                                        # Summary metrics
                                                        col1, col2, col3 = st.columns(3)
                                                        
                                                        with col1:
                                                            st.metric("Horizons Generated", len(horizons))
                                                        
                                                        with col2:
                                                            total_forecasts = sum(
                                                                metrics['forecasts_made'] 
                                                                for metrics in validation_metrics.values()
                                                            )
                                                            st.metric("Total Forecasts", total_forecasts)
                                                        
                                                        with col3:
                                                            avg_validation_rate = np.mean([
                                                                metrics['validation_rate'] 
                                                                for metrics in validation_metrics.values()
                                                            ])
                                                            st.metric("Avg Validation Rate", f"{avg_validation_rate:.1f}%")
                                                        
                                                        # Performance summary table
                                                        st.markdown("#### 📊 Forecast Performance Summary")
                                                        
                                                        perf_data = []
                                                        for horizon in sorted(horizons):
                                                            metrics = validation_metrics[horizon]
                                                            perf_data.append({
                                                                'Horizon (min)': horizon,
                                                                'Forecasts Made': metrics['forecasts_made'],
                                                                'Validations Available': metrics['validations_available'],
                                                                'Validation Rate (%)': f"{metrics['validation_rate']:.1f}",
                                                                'MAE (kW)': f"{metrics['mae_kw']:.2f}" if not pd.isna(metrics['mae_kw']) else "N/A",
                                                                'RMSE (kW)': f"{metrics['rmse_kw']:.2f}" if not pd.isna(metrics['rmse_kw']) else "N/A",
                                                                'Mean % Error': f"{metrics['mean_pct_error']:.2f}%" if not pd.isna(metrics['mean_pct_error']) else "N/A",
                                                                'Bias (kW)': f"{metrics['bias_kw']:.2f}" if not pd.isna(metrics['bias_kw']) else "N/A"
                                                            })
                                                        
                                                        perf_df = pd.DataFrame(perf_data)
                                                        st.dataframe(perf_df, use_container_width=True)
                                                        
                                                        # Data structure info
                                                        with st.expander("📋 Generated Time Series Info"):
                                                            st.markdown("**Time Series Structure:**")
                                                            for horizon in horizons:
                                                                series_name = f'forecast_{horizon}min'
                                                                series = forecast_series[series_name]
                                                                st.write(f"- `{series_name}[t]`: {len(series)} points, forecasts value at t+{horizon} minutes")
                                                                st.write(f"  - Time range: {series.index.min()} to {series.index.max()}")
                                                                st.write(f"  - Available forecasts: {series.notna().sum()}")
                                                            
                                                            st.markdown("**Actual Series:**")
                                                            actual = df_processed[power_col]
                                                            st.write(f"- `actual[t]`: {len(actual)} points")
                                                            st.write(f"  - Time range: {actual.index.min()} to {actual.index.max()}")
                                                            
                                                            st.markdown("**Direct Comparison Ready:**")
                                                            st.write("Each forecast series is aligned to target timestamps for direct comparison:")
                                                            st.code("""
# Example comparison at any timestamp t:
actual_value = actual[t]
forecast_1min = forecast_1min[t]  # Predicted 1 minute ago  
forecast_10min = forecast_10min[t]  # Predicted 10 minutes ago

# Error calculation:
error_1min = forecast_1min - actual_value
error_10min = forecast_10min - actual_value
                                                            """)
                                                        
                                                        # P90 Forecasting Section
                                                        st.markdown("#### 🎯 P90 Forecast Generation")
                                                        
                                                        with st.spinner("🔄 Generating P90 forecast bands from historical residuals..."):
                                                            try:
                                                                # Convert ROC backtest results to long format
                                                                forecast_dict = {}
                                                                actual_dict = {}
                                                                
                                                                for horizon in horizons:
                                                                    series_name = f'forecast_{horizon}min'
                                                                    forecast_dict[horizon] = forecast_series[series_name]
                                                                    actual_dict[horizon] = df_processed[power_col]
                                                                
                                                                # Convert to long format
                                                                df_long = convert_roc_backtest_to_long_format(
                                                                    forecast_dict, actual_dict, horizons
                                                                )
                                                                
                                                                if not df_long.empty:
                                                                    # Compute residual quantiles by horizon
                                                                    residual_quantiles = compute_residual_quantiles_by_horizon(df_long)
                                                                    
                                                                    if not residual_quantiles.empty:
                                                                        # Generate P90 forecast bands
                                                                        df_long_with_bands = generate_p90_forecast_bands(df_long, residual_quantiles)
                                                                        
                                                                        # Store results in session state
                                                                        st.session_state['roc_long_format'] = df_long_with_bands
                                                                        st.session_state['residual_quantiles'] = residual_quantiles
                                                                        
                                                                        # Display results
                                                                        col1, col2 = st.columns(2)
                                                                        
                                                                        with col1:
                                                                            st.success("✅ P90 bands generated successfully!")
                                                                            st.metric("Long Format Records", len(df_long_with_bands))
                                                                            st.metric("Horizons with Quantiles", len(residual_quantiles))
                                                                        
                                                                        with col2:
                                                                            # Show residual quantiles summary
                                                                            st.markdown("**Residual Quantiles by Horizon:**")
                                                                            for _, row in residual_quantiles.iterrows():
                                                                                horizon = row['horizon_min']
                                                                                p10 = row.get('residual_p10', np.nan)
                                                                                p90 = row.get('residual_p90', np.nan)
                                                                                st.write(f"• {horizon}min: P10={p10:.1f}kW, P90={p90:.1f}kW")
                                                                        
                                                                        # Interactive P90 Forecast Visualization
                                                                        st.markdown("#### 🎯 Interactive P90 Forecast Analysis")
                                                                        
                                                                        # Interactive controls
                                                                        col_controls1, col_controls2 = st.columns(2)
                                                                        
                                                                        with col_controls1:
                                                                            # Horizon selection
                                                                            available_horizons = sorted(df_long_with_bands['horizon_min'].unique())
                                                                            default_horizon = 5 if 5 in available_horizons else available_horizons[0]
                                                                            selected_horizon = st.radio(
                                                                                "🕒 Select Forecast Horizon",
                                                                                options=available_horizons,
                                                                                index=available_horizons.index(default_horizon),
                                                                                format_func=lambda x: f"{int(x)} minutes",
                                                                                help="Choose a single horizon for detailed analysis"
                                                                            )
                                                                        
                                                                        with col_controls2:
                                                                            # Date range selection
                                                                            if not df_long_with_bands.empty:
                                                                                min_date = df_long_with_bands['t'].min().date()
                                                                                max_date = df_long_with_bands['t'].max().date()
                                                                                
                                                                                st.markdown("📅 **Date Range Selection**")
                                                                                date_range = st.date_input(
                                                                                    "Select date range",
                                                                                    value=(min_date, min_date),
                                                                                    min_value=min_date,
                                                                                    max_value=max_date,
                                                                                    help="Select start and end dates for analysis (defaults to single day)"
                                                                                )
                                                                        
                                                                        # Filter data based on selections
                                                                        if not df_long_with_bands.empty and date_range:
                                                                            # Handle single date or date range
                                                                            if isinstance(date_range, tuple) and len(date_range) == 2:
                                                                                start_date, end_date = date_range
                                                                            else:
                                                                                start_date = end_date = date_range
                                                                            
                                                                            # Filter by horizon and date range
                                                                            filtered_df = df_long_with_bands[
                                                                                (df_long_with_bands['horizon_min'] == selected_horizon) &
                                                                                (df_long_with_bands['t'].dt.date >= start_date) &
                                                                                (df_long_with_bands['t'].dt.date <= end_date)
                                                                            ].copy().sort_values('t')
                                                                            
                                                                            if not filtered_df.empty:
                                                                                # Create focused forecast table
                                                                                st.markdown(f"##### 📊 {int(selected_horizon)}-Minute Forecast Table")
                                                                                
                                                                                # Prepare display DataFrame
                                                                                display_cols = ['t', 'actual', 'forecast_p10', 'forecast_p50', 'forecast_p90']
                                                                                available_cols = [col for col in display_cols if col in filtered_df.columns]
                                                                                
                                                                                display_df = filtered_df[available_cols].copy()
                                                                                if 't' in display_df.columns:
                                                                                    display_df['Time'] = display_df['t'].dt.strftime('%Y-%m-%d %H:%M:%S')
                                                                                    display_df = display_df[['Time'] + [col for col in available_cols if col != 't']]
                                                                                
                                                                                # Show filtered table
                                                                                st.dataframe(display_df.head(20), use_container_width=True, height=300)
                                                                                st.caption(f"Showing first 20 of {len(filtered_df)} records for {int(selected_horizon)}-minute horizon")
                                                                                
                                                                                # ROC Calculation Validation Table
                                                                                with st.expander("🔍 ROC Calculation Validation (10 sample entries)", expanded=False):
                                                                                    st.markdown(f"**Validate ROC calculations for {int(selected_horizon)}-minute horizon**")
                                                                                    
                                                                                    # Create ROC validation table with step-by-step calculations
                                                                                    if len(filtered_df) > 0:
                                                                                        validation_df = filtered_df.head(10).copy()
                                                                                        
                                                                                        # Calculate ROC step by step for validation display
                                                                                        if 't' in validation_df.columns and 'actual' in validation_df.columns:
                                                                                            validation_display = []
                                                                                            
                                                                                            # Sort by timestamp to ensure proper order for ROC calculation
                                                                                            validation_df = validation_df.sort_values('t')
                                                                                            
                                                                                            for i, (idx, row) in enumerate(validation_df.iterrows()):
                                                                                                timestamp = row['t']
                                                                                                power = row.get('actual', 0)
                                                                                                forecast = row.get('forecast_p50', 0)
                                                                                                
                                                                                                # Calculate Power Diff and ROC for current row
                                                                                                if i == 0:
                                                                                                    # First row - no previous data for diff calculation
                                                                                                    power_diff = 0  # or NaN
                                                                                                    roc_value = 0   # or NaN
                                                                                                else:
                                                                                                    # Calculate difference from previous row
                                                                                                    prev_row = validation_df.iloc[i-1]
                                                                                                    prev_power = prev_row.get('actual', 0)
                                                                                                    prev_time = prev_row['t']
                                                                                                    
                                                                                                    # Power difference
                                                                                                    power_diff = power - prev_power
                                                                                                    
                                                                                                    # Time difference in minutes
                                                                                                    time_diff_min = (timestamp - prev_time).total_seconds() / 60
                                                                                                    
                                                                                                    # ROC calculation
                                                                                                    roc_value = power_diff / time_diff_min if time_diff_min > 0 else 0
                                                                                                
                                                                                                validation_display.append({
                                                                                                    'Timestamp': timestamp.strftime('%H:%M:%S'),
                                                                                                    'Power (kW)': f"{power:.2f}",
                                                                                                    'Power Diff (kW)': f"{power_diff:.2f}" if i > 0 else "N/A",
                                                                                                    'ROC (kW/min)': f"{roc_value:.4f}" if i > 0 else "N/A",
                                                                                                    'Forecast (kW)': f"{forecast:.2f}"
                                                                                                })
                                                                                            
                                                                                            validation_table = pd.DataFrame(validation_display)
                                                                                            st.dataframe(validation_table, use_container_width=True, height=350)
                                                                                            
                                                                                            # Add calculation methodology
                                                                                            st.markdown("**📊 ROC Calculation Formula:**")
                                                                                            st.markdown(f"""
                                                                                            - **Power Diff**: Current Power - Previous Power
                                                                                            - **ROC**: Power Diff ÷ Time Interval (minutes)
                                                                                            - **Forecast**: Current Power + (ROC × {int(selected_horizon)} minutes)
                                                                                            - **Formula**: P_forecast = P_current + ROC × horizon
                                                                                            """)
                                                                                            
                                                                                        else:
                                                                                            st.info("Validation data not available for selected time period")
                                                                                    else:
                                                                                        st.info("No data available for ROC validation")
                                                                                
                                                                                # Generate P90 visualization for selected horizon
                                                                                st.markdown(f"##### 📈 {int(selected_horizon)}-Minute Forecast Visualization with P10-P90 Bands")
                                                                                
                                                                                # Create the plot
                                                                                fig_p90 = go.Figure()
                                                                                
                                                                                # Add P10-P90 shaded band
                                                                                if 'forecast_p10' in filtered_df.columns and 'forecast_p90' in filtered_df.columns:
                                                                                    fig_p90.add_trace(go.Scatter(
                                                                                        x=filtered_df['t'],
                                                                                        y=filtered_df['forecast_p90'],
                                                                                        mode='lines',
                                                                                        line=dict(color='rgba(0,0,0,0)'),
                                                                                        showlegend=False,
                                                                                        hoverinfo='skip'
                                                                                    ))
                                                                                    
                                                                                    fig_p90.add_trace(go.Scatter(
                                                                                        x=filtered_df['t'],
                                                                                        y=filtered_df['forecast_p10'],
                                                                                        mode='lines',
                                                                                        line=dict(color='rgba(0,0,0,0)'),
                                                                                        fill='tonexty',
                                                                                        fillcolor='rgba(128, 128, 128, 0.3)',
                                                                                        name='P10-P90 Uncertainty Band',
                                                                                        hovertemplate='P10-P90 Band<br>Time: %{x}<br>P10: %{y:.2f} kW<extra></extra>'
                                                                                    ))
                                                                                
                                                                                # Add actual values
                                                                                if 'actual' in filtered_df.columns:
                                                                                    fig_p90.add_trace(go.Scatter(
                                                                                        x=filtered_df['t'],
                                                                                        y=filtered_df['actual'],
                                                                                        mode='lines',
                                                                                        line=dict(color='blue', width=2),
                                                                                        name='Actual Load',
                                                                                        hovertemplate='Actual: %{y:.2f} kW<br>Time: %{x}<extra></extra>'
                                                                                    ))
                                                                                
                                                                                # Add P50 forecast
                                                                                if 'forecast_p50' in filtered_df.columns:
                                                                                    fig_p90.add_trace(go.Scatter(
                                                                                        x=filtered_df['t'],
                                                                                        y=filtered_df['forecast_p50'],
                                                                                        mode='lines',
                                                                                        line=dict(color='red', width=2, dash='dash'),
                                                                                        name='P50 Forecast',
                                                                                        hovertemplate='P50 Forecast: %{y:.2f} kW<br>Time: %{x}<extra></extra>'
                                                                                    ))
                                                                                
                                                                                # Update layout
                                                                                fig_p90.update_layout(
                                                                                    title=f"{int(selected_horizon)}-Minute Forecast with P10-P90 Uncertainty Bands",
                                                                                    xaxis_title="Time",
                                                                                    yaxis_title="Power (kW)",
                                                                                    height=500,
                                                                                    showlegend=True,
                                                                                    hovermode='x unified',
                                                                                    legend=dict(
                                                                                        orientation="h",
                                                                                        yanchor="bottom",
                                                                                        y=1.02,
                                                                                        xanchor="right",
                                                                                        x=1
                                                                                    )
                                                                                )
                                                                                
                                                                                # Display the plot
                                                                                st.plotly_chart(fig_p90, use_container_width=True)
                                                                                
                                                                                # Show filtered metrics
                                                                                col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
                                                                                
                                                                                with col_metrics1:
                                                                                    st.metric("Data Points", len(filtered_df))
                                                                                    
                                                                                with col_metrics2:
                                                                                    if 'forecast_p10' in filtered_df.columns and 'forecast_p90' in filtered_df.columns:
                                                                                        band_width = (filtered_df['forecast_p90'] - filtered_df['forecast_p10']).mean()
                                                                                        st.metric("Avg Uncertainty Band", f"{band_width:.1f} kW")
                                                                                    else:
                                                                                        st.metric("Avg Uncertainty Band", "N/A")
                                                                                
                                                                                with col_metrics3:
                                                                                    if 'actual' in filtered_df.columns and 'forecast_p50' in filtered_df.columns:
                                                                                        mae = abs(filtered_df['forecast_p50'] - filtered_df['actual']).mean()
                                                                                        st.metric("MAE", f"{mae:.1f} kW")
                                                                                    else:
                                                                                        st.metric("MAE", "N/A")
                                                                            
                                                                            else:
                                                                                st.warning(f"⚠️ No data available for {int(selected_horizon)}-minute horizon in the selected date range.")
                                                                        
                                                                        # Export options for filtered data
                                                                        with st.expander("� Export Filtered Forecast Data"):
                                                                            if 'filtered_df' in locals() and not filtered_df.empty:
                                                                                export_df = filtered_df[available_cols].copy()
                                                                                if 't' in export_df.columns:
                                                                                    export_df['timestamp'] = export_df['t'].dt.strftime('%Y-%m-%d %H:%M:%S')
                                                                                    export_df = export_df.drop('t', axis=1)
                                                                                
                                                                                st.download_button(
                                                                                    label=f"📥 Download {int(selected_horizon)}-Min Filtered Forecast",
                                                                                    data=export_df.to_csv(index=False),
                                                                                    file_name=f"p90_forecast_{int(selected_horizon)}min_{start_date}_to_{end_date}_{datetime.now().strftime('%H%M')}.csv",
                                                                                    mime="text/csv",
                                                                                    help=f"Export {int(selected_horizon)}-minute P90 forecast data for selected date range"
                                                                                )
                                                                                
                                                                                # Summary of exported data
                                                                                st.info(f"📊 Export contains {len(export_df)} records for {int(selected_horizon)}-minute horizon from {start_date} to {end_date}")
                                                                            else:
                                                                                st.warning("No filtered data available for export")
                                                                        
                                                                        # Overall horizon comparison  
                                                                        if len(df_long_with_bands['horizon_min'].unique()) > 1:
                                                                            st.markdown("#### 📈 All Horizons Overview")
                                                                            
                                                                            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                                                                            
                                                                            with metrics_col1:
                                                                                # Count data points for each horizon
                                                                                horizon_counts = df_long_with_bands.groupby('horizon_min').size()
                                                                                st.markdown("**📊 Total Data Points**")
                                                                                for horizon, count in horizon_counts.items():
                                                                                    st.write(f"• {int(horizon)} min: {count:,} points")
                                                                            
                                                                            with metrics_col2:
                                                                                # Calculate uncertainty band widths
                                                                                if 'forecast_p10' in df_long_with_bands.columns and 'forecast_p90' in df_long_with_bands.columns:
                                                                                    st.markdown("**🎯 Avg Uncertainty Band**")
                                                                                    band_widths = df_long_with_bands.groupby('horizon_min').apply(
                                                                                        lambda x: (x['forecast_p90'] - x['forecast_p10']).mean()
                                                                                    )
                                                                                    for horizon, width in band_widths.items():
                                                                                        st.write(f"• {int(horizon)} min: {width:.1f} kW")
                                                                            
                                                                            with metrics_col3:
                                                                                # Calculate forecast accuracy (MAE) if actual values available
                                                                                if 'actual' in df_long_with_bands.columns and 'forecast_p50' in df_long_with_bands.columns:
                                                                                    st.markdown("**🔍 Overall MAE**")
                                                                                    mae_by_horizon = df_long_with_bands.groupby('horizon_min').apply(
                                                                                        lambda x: abs(x['forecast_p50'] - x['actual']).mean()
                                                                                    )
                                                                                    for horizon, mae in mae_by_horizon.items():
                                                                                        st.write(f"• {int(horizon)} min: {mae:.1f} kW")
                                                                        
                                                                        # Complete dataset export options
                                                                        with st.expander("💾 Export Complete P90 Forecast Dataset"):
                                                                            col_export1, col_export2 = st.columns(2)
                                                                            
                                                                            with col_export1:
                                                                                # CSV export
                                                                                csv_data = df_long_with_bands.to_csv(index=False)
                                                                                st.download_button(
                                                                                    label="📄 Download Complete Dataset (CSV)",
                                                                                    data=csv_data,
                                                                                    file_name=f"p90_forecast_complete_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                                                                    mime="text/csv",
                                                                                    help="Export all P90 forecast data for all horizons"
                                                                                )
                                                                            
                                                                            with col_export2:
                                                                                # Parquet export (more efficient for large datasets)
                                                                                import io
                                                                                parquet_buffer = io.BytesIO()
                                                                                df_long_with_bands.to_parquet(parquet_buffer, index=False)
                                                                                st.download_button(
                                                                                    label="📦 Download Complete Dataset (Parquet)",
                                                                                    data=parquet_buffer.getvalue(),
                                                                                    file_name=f"p90_forecast_complete_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet",
                                                                                    mime="application/octet-stream",
                                                                                    help="Export all P90 forecast data in efficient Parquet format"
                                                                                )
                                                                    else:
                                                                        st.warning("⚠️ Could not compute residual quantiles - insufficient data")
                                                                else:
                                                                    st.warning("⚠️ Could not convert to long format - no valid forecast/actual pairs found")
                                                                    
                                                            except Exception as e:
                                                                st.error(f"❌ Error generating P90 forecasts: {str(e)}")
                                                                import traceback
                                                                st.error(f"Traceback: {traceback.format_exc()}")
                                                        
                                                        st.info("💾 **Data Ready:** P90 forecast bands are generated and stored for visualization and analysis")
                                                        
                                                    except Exception as e:
                                                        st.error(f"❌ Error generating ROC forecasts: {str(e)}")
                                            
                                            elif enable_backtesting and not horizons:
                                                st.warning("⚠️ Please select at least one forecast horizon")
                                            
                                            elif not enable_backtesting:
                                                st.info("📊 **Configuration Mode:** Enable backtesting to generate historical forecasts")
                                        
                                        else:
                                            st.info("🔧 Method configuration and execution will be implemented here")
                                    elif method_details["status"] == "Coming Soon":
                                        st.warning(f"⏳ **{selected_method}** implementation in progress")
                                        st.info("📅 Expected availability in next release")
                                    elif method_details["status"] == "Planned":
                                        st.info(f"📋 **{selected_method}** is planned for future development")
                                        st.info("🗓️ Scheduled for upcoming releases")
                                    else:  # Future Release
                                        st.info(f"🚀 **{selected_method}** scheduled for future release")
                                        st.info("💡 Advanced feature under research and development")
                                
                            else:
                                st.info("📈 **Standard Mode:** Enable forecasting to access prediction features")
                            
                            # =============================================================================
                            # SHAVING STRATEGY DECISION TREE SCAFFOLDING
                            # =============================================================================
                            st.markdown("---")
                            st.markdown("## 🎯 Demand Shaving Strategy Selection")
                            
                            # Decision tree logic based on forecasting enable/disable
                            if enable_forecasting:
                                st.success("🔮 **Forecast-Based Shaving:** Using P10/P50/P90 forecast data for optimization")
                                
                                # Retrieve stored forecast data
                                forecast_data_available = False
                                p10_data = None
                                p50_data = None 
                                p90_data = None
                                
                                # Check if forecast data exists in session state (with Series ambiguity protection)
                                if 'roc_long_format' in st.session_state:
                                    forecast_df = st.session_state['roc_long_format']
                                    # Safe column checking to prevent Series ambiguity
                                    required_cols = ['forecast_p10', 'forecast_p50', 'forecast_p90']
                                    has_all_cols = all(col in forecast_df.columns for col in required_cols)
                                    if has_all_cols and len(forecast_df) > 0:
                                        forecast_data_available = True
                                        p10_data = forecast_df[['t', 'actual', 'forecast_p10']].copy()
                                        p50_data = forecast_df[['t', 'actual', 'forecast_p50']].copy()
                                        p90_data = forecast_df[['t', 'actual', 'forecast_p90']].copy()
                                        
                                        st.info(f"📊 Forecast data loaded: {len(forecast_df):,} records with P10/P50/P90 bands")
                                    else:
                                        st.warning("⚠️ Forecast data exists but P10/P50/P90 columns not found")
                                else:
                                    st.warning("⚠️ No forecast data available - please generate forecasts first")
                                
                                # Store forecast data for later use
                                if forecast_data_available:
                                    st.session_state['shaving_forecast_data'] = {
                                        'p10': p10_data,
                                        'p50': p50_data,
                                        'p90': p90_data,
                                        'full_data': forecast_df
                                    }
                                
                                # Define forecast-enabled shaving strategies (includes all 6 strategies)
                                shaving_strategies_forecast = [
                                    "Default Shaving",
                                    "SOC-Aware",
                                    "Hybrid",
                                    "Policy A (Forecast Only)",
                                    "Policy B (Forecast Only)", 
                                    "Policy C (Forecast Only)"
                                ]
                                
                            else:
                                st.info("📈 **Historical Data Shaving:** Using uploaded historical data for optimization")
                                
                                # Use uploaded historical data
                                historical_data_available = False
                                historical_power_data = None
                                
                                # Check if uploaded data exists
                                if uploaded_file is not None and 'df_processed' in st.session_state:
                                    historical_df = st.session_state['df_processed']
                                    # Get the power column name from session state or use a default
                                    if 'v2_power_col' in st.session_state:
                                        power_column = st.session_state['v2_power_col']
                                    else:
                                        # Try to find power column from available columns
                                        power_cols = [col for col in historical_df.columns if any(kw in col.lower() for kw in ['power', 'kw', 'demand', 'load'])]
                                        power_column = power_cols[0] if power_cols else historical_df.columns[0]
                                    
                                    if power_column in historical_df.columns:
                                        historical_data_available = True
                                        # Use the DataFrame with datetime index (no need for separate timestamp column)
                                        historical_power_data = historical_df[[power_column]].copy()
                                        st.info(f"📊 Historical data loaded: {len(historical_df):,} records")
                                    else:
                                        st.warning("⚠️ Historical data exists but power column not found")
                                else:
                                    st.warning("⚠️ No historical data available - please upload data first")
                                
                                # Store historical data for later use
                                if historical_data_available:
                                    st.session_state['shaving_historical_data'] = historical_power_data
                                
                                # Define historical-only shaving strategies (excludes forecast-only policies)
                                shaving_strategies_forecast = [
                                    "Default Shaving",
                                    "SOC-Aware", 
                                    "Hybrid"
                                ]
                            
                            # Shaving Strategy Selection Dropdown
                            st.markdown("#### 🔧 Strategy Configuration")
                            
                            selected_strategy = st.selectbox(
                                "Select Shaving Strategy:",
                                options=shaving_strategies_forecast,
                                key="v2_shaving_strategy",
                                help="Choose the demand shaving strategy based on your optimization goals"
                            )
                            
                            # Display strategy information (no actions implemented yet)
                            strategy_descriptions = {
                                "Default Shaving": {
                                    "description": "Standard peak shaving using fixed thresholds and basic battery operation",
                                    "data_source": "Historical/Forecast",
                                    "complexity": "Low",
                                    "optimization": "Basic peak reduction"
                                },
                                "SOC-Aware": {
                                    "description": "State-of-charge aware shaving that considers battery capacity and health",
                                    "data_source": "Historical/Forecast", 
                                    "complexity": "Medium",
                                    "optimization": "Battery longevity + peak reduction"
                                },
                                "Hybrid": {
                                    "description": "Combination of multiple strategies with dynamic switching based on conditions",
                                    "data_source": "Historical/Forecast",
                                    "complexity": "High", 
                                    "optimization": "Multi-objective optimization"
                                },
                                "Policy A (Forecast Only)": {
                                    "description": "Advanced forecast-based policy using predictive uncertainty bands",
                                    "data_source": "Forecast Only",
                                    "complexity": "High",
                                    "optimization": "Uncertainty-aware peak reduction"
                                },
                                "Policy B (Forecast Only)": {
                                    "description": "Risk-averse strategy using P90 forecast bounds for conservative shaving",
                                    "data_source": "Forecast Only", 
                                    "complexity": "High",
                                    "optimization": "Risk-minimized peak management"
                                },
                                "Policy C (Forecast Only)": {
                                    "description": "Aggressive strategy leveraging P10 forecasts for maximum peak reduction",
                                    "data_source": "Forecast Only",
                                    "complexity": "High", 
                                    "optimization": "Maximum peak reduction with forecast confidence"
                                }
                            }
                            
                            if selected_strategy in strategy_descriptions:
                                strategy_info = strategy_descriptions[selected_strategy]
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.markdown(f"**Description:** {strategy_info['description']}")
                                    st.markdown(f"**Data Source:** {strategy_info['data_source']}")
                                    
                                with col2:
                                    st.markdown(f"**Complexity:** {strategy_info['complexity']}")
                                    st.markdown(f"**Optimization:** {strategy_info['optimization']}")
                            
                            # Strategy status and next steps
                            st.markdown("#### 📋 Implementation Status")
                            
                            if enable_forecasting:
                                if forecast_data_available:
                                    st.success("✅ Forecast data ready for strategy implementation")
                                    if selected_strategy in ["Policy A (Forecast Only)", "Policy B (Forecast Only)", "Policy C (Forecast Only)"]:
                                        st.info(f"🎯 **{selected_strategy}** selected - Advanced forecast-based optimization")
                                    else:
                                        st.info(f"🎯 **{selected_strategy}** selected - Standard optimization with forecast enhancement")
                                else:
                                    st.warning("⚠️ Generate forecast data first to enable strategy implementation")
                            else:
                                if 'shaving_historical_data' in st.session_state:
                                    st.success("✅ Historical data ready for strategy implementation") 
                                    st.info(f"🎯 **{selected_strategy}** selected - Historical data optimization")
                                else:
                                    st.warning("⚠️ Upload historical data first to enable strategy implementation")
                            
                    except Exception as e:
                        st.error(f"❌ Error in V2 battery configuration: {str(e)}")
                        st.info("Some V2 features may not be available in this environment.")
                        
                    # =============================================================================
                    # V2 BATTERY SIMULATION AND RESULTS DISPLAY SECTION
                    # =============================================================================
                    st.markdown("---")
                    st.markdown("## 🔋 Battery Simulation Results")
                    
                    # Check if we have all necessary data and configuration
                    if battery_config and battery_config.get('run_analysis', False):
                        data_ready = False
                        simulation_data = None
                        power_column = None
                        
                        # Determine data source based on forecast mode
                        if enable_forecasting and forecast_data_available:
                            # Get forecast data from session state
                            original_forecast_df = st.session_state.get('roc_long_format', None)
                            
                            if original_forecast_df is not None:
                                # 🔮 SIMPLIFIED FORECAST DATA CONVERSION - Use P50 only
                                st.markdown("### 🔄 **Converting P50 Forecast Data for MD Shaving**")
                                
                                # Check if we have the required forecast columns - FIXED: Convert to list to avoid Series ambiguity
                                if 't' in original_forecast_df.columns.tolist() and 'forecast_p50' in original_forecast_df.columns.tolist():
                                    # Create simulation data exactly like historical data format
                                    simulation_data = pd.DataFrame(index=pd.to_datetime(original_forecast_df['t']))
                                    simulation_data['Active Power Demand (kW)'] = original_forecast_df['forecast_p50'].values
                                    simulation_data['Original_Demand'] = original_forecast_df['forecast_p50'].values
                                    
                                    # Set the power column name to match what MD shaving expects
                                    power_col = 'Active Power Demand (kW)'
                                    
                                    st.success("🔮 **Using P50 forecast data for MD shaving simulation**")
                                    data_ready = True
                                    
                                    # Display conversion summary
                                    st.info(f"""
                                    **📊 P50 Forecast Data Conversion Summary:**
                                    - **Original forecast data**: {original_forecast_df.shape[0]} rows × {original_forecast_df.shape[1]} columns
                                    - **Converted simulation data**: {simulation_data.shape[0]} rows × {simulation_data.shape[1]} columns
                                    - **Power column**: `{power_col}` (using forecast_p50 values)
                                    - **Index**: Datetime index from forecast timestamps
                                    - **Approach**: Treating P50 forecast exactly like historical demand data
                                    """)
                                    
                                else:
                                    st.error("❌ Forecast data missing required columns ('t' and 'forecast_p50')")
                                    data_ready = False
                                    simulation_data = None
                            else:
                                st.error("❌ No forecast data available for conversion")
                                data_ready = False
                                simulation_data = None
                                
                        elif not enable_forecasting and 'shaving_historical_data' in st.session_state:
                            # Use historical data
                            simulation_data = st.session_state['df_processed']
                            power_col = st.session_state.get('v2_power_col', simulation_data.columns[0])
                            st.success("📊 **Running simulation with historical data**")
                            data_ready = True
                            
                        else:
                            st.warning("⚠️ No data available for simulation. Please generate forecast data or upload historical data.")
                            data_ready = False
                        
                        if data_ready and simulation_data is not None:
                            try:
                                # Get monthly targets from V2 calculation - FIXED: Safe retrieval to avoid Series ambiguity
                                if 'v2_monthly_targets' in st.session_state:
                                    try:
                                        monthly_targets_raw = st.session_state['v2_monthly_targets']
                                        
                                        # FIXED: Ensure we have a proper Series, not a boolean result
                                        if hasattr(monthly_targets_raw, 'index') and hasattr(monthly_targets_raw, 'values'):
                                            monthly_targets = pd.Series(
                                                data=monthly_targets_raw.values, 
                                                index=monthly_targets_raw.index,
                                                name='monthly_targets'
                                            )
                                        else:
                                            # Fallback if it's not a Series
                                            monthly_targets = monthly_targets_raw
                                            
                                    except Exception as e:
                                        st.error(f"❌ Error retrieving monthly targets: {str(e)}")
                                        monthly_targets = None
                                else:
                                    st.error("❌ Monthly targets not calculated. Please configure battery settings first.")
                                    monthly_targets = None
                                
                                if monthly_targets is not None:
                                    # Extract battery parameters from battery_config with corrected key names - FIXED: Explicit scalar conversion
                                    try:
                                        # FIXED: Convert to float to avoid Series ambiguity
                                        capacity_raw = battery_config.get('total_energy_capacity', 100)
                                        power_raw = battery_config.get('total_power_capacity', 100)
                                        
                                        # Ensure scalar values
                                        capacity_kwh = float(capacity_raw.iloc[0]) if hasattr(capacity_raw, 'iloc') else float(capacity_raw)
                                        power_rating_kw = float(power_raw.iloc[0]) if hasattr(power_raw, 'iloc') else float(power_raw)
                                        
                                        battery_sizing = {
                                            'capacity_kwh': capacity_kwh,
                                            'power_rating_kw': power_rating_kw
                                        }
                                        
                                    except Exception as e:
                                        st.error(f"❌ Error extracting battery parameters: {str(e)}")
                                        # Fallback values
                                        battery_sizing = {
                                            'capacity_kwh': 100.0,
                                            'power_rating_kw': 100.0
                                        }
                                    
                                    battery_params = {
                                        'round_trip_efficiency': 95,  # Default efficiency
                                        'depth_of_discharge': 85      # Default DoD
                                    }
                                    
                                    # Set up simulation parameters - ensure all variables are defined for both modes - FIXED: Safe access
                                    interval_hours = 0.25  # 15-minute intervals
                                    
                                    # FIXED: Safe session state access to avoid Series ambiguity
                                    try:
                                        selected_tariff = st.session_state.get('selected_tariff_dict')
                                        holidays = st.session_state.get('holidays', set())
                                        
                                        # Ensure holidays is a proper set, not a Series
                                        if hasattr(holidays, 'tolist'):
                                            holidays = set(holidays.tolist())
                                        elif not isinstance(holidays, set):
                                            holidays = set(holidays) if holidays else set()
                                            
                                    except Exception as e:
                                        st.warning(f"⚠️ Session state access issue: {str(e)}")
                                        selected_tariff = None
                                        holidays = set()

                                    
                                    # Ensure session state variables are set for forecast mode compatibility - FIXED: Safe operations
                                    if enable_forecasting:
                                        try:
                                            if not hasattr(st.session_state, 'tabled_analysis_selected_battery'):
                                                if battery_config and battery_config.get('active_battery_spec'):
                                                    # FIXED: Safe access to battery_config
                                                    battery_spec = battery_config['active_battery_spec']
                                                    st.session_state.tabled_analysis_selected_battery = {
                                                        'spec': battery_spec,
                                                        'label': 'Selected Battery'
                                                    }
                                            
                                            if not hasattr(st.session_state, 'tabled_analysis_battery_quantity'):
                                                if battery_config:
                                                    # FIXED: Safe scalar extraction
                                                    quantity_raw = battery_config.get('selected_quantity', 1)
                                                    quantity = int(quantity_raw.iloc[0]) if hasattr(quantity_raw, 'iloc') else int(quantity_raw)
                                                    st.session_state.tabled_analysis_battery_quantity = quantity
                                                    
                                        except Exception as e:
                                            st.warning(f"⚠️ Forecasting mode session state setup issue: {str(e)}")
                                            # Continue with defaults
                                    
                                    # Prepare simulation parameters for the new handler function
                                    simulation_params = {
                                        'simulation_data': simulation_data,
                                        'power_col': power_col,
                                        'monthly_targets': monthly_targets,
                                        'battery_sizing': battery_sizing,
                                        'battery_params': battery_params,
                                        'interval_hours': interval_hours,
                                        'selected_tariff': selected_tariff,
                                        'holidays': holidays,
                                        'enable_forecasting': enable_forecasting,
                                        'battery_config': battery_config
                                    }
                                    
                                    # Call the handler function (defined after _simulate_battery_operation_v2)
                                    _handle_battery_simulation_workflow(simulation_params)
                                        
                            except Exception as e:
                                st.error(f"❌ Error during battery simulation: {str(e)}")
                                st.info("Please check your data and configuration settings.")
                    
                    else:
                        st.info("🔋 **Enable 'Run V2 Analysis' in Battery Configuration to see simulation results**")
                        
                    # Strategy Implementation and Testing
                    st.markdown("#### 🚀 Strategy Execution")
                    
                    # Get battery configuration for strategy testing
                    if battery_config and battery_config.get('run_analysis', False):
                        battery_power_kw = battery_config.get('power_kw', 100)  # Default 100kW if not specified
                        
                        # Strategy Testing Interface
                        st.markdown("**🔬 Test Strategy Parameters:**")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            test_soc = st.slider("Current SOC (%)", min_value=5, max_value=100, value=50, step=5, 
                                                help="State of charge to test strategy behavior")
                        with col2:
                            test_excess = st.slider("Demand Excess (kW)", min_value=0, max_value=200, value=80, step=10,
                                                   help="Excess demand above target that needs shaving")
                        with col3:
                            st.metric("Battery Power", f"{battery_power_kw:.0f} kW", help="Maximum discharge power from battery config")
                        
                        # Strategy Comparison
                        if st.button("🔍 Compare Strategies", help="Compare Default vs SOC-Aware strategies with current parameters"):
                            
                            # Test both strategies
                            default_result = _get_strategy_aware_discharge("Default Shaving", test_soc, test_excess, battery_power_kw)
                            soc_aware_result = _get_strategy_aware_discharge("SOC-Aware", test_soc, test_excess, battery_power_kw)
                            
                            st.markdown("**📊 Strategy Comparison Results:**")
                            
                            # Create comparison table
                            comparison_data = {
                                "Strategy": ["Default (Aggressive)", "SOC-Aware (Conservative)"],
                                "Discharge Power (kW)": [f"{default_result['power_kw']:.1f}", f"{soc_aware_result['power_kw']:.1f}"],
                                "Power Difference": ["Baseline", f"{soc_aware_result['power_kw'] - default_result['power_kw']:+.1f} kW"],
                                "Reasoning": [default_result['reasoning'], soc_aware_result['reasoning']]
                    }
                    
                            comparison_df = pd.DataFrame(comparison_data)
                            st.dataframe(comparison_df, use_container_width=True)
                            
                            # Highlight selected strategy
                            if selected_strategy == "Default Shaving":
                                st.success(f"✅ **Selected Strategy:** {default_result['strategy_type']} - {default_result['power_kw']:.1f} kW discharge")
                            elif selected_strategy == "SOC-Aware":
                                st.success(f"✅ **Selected Strategy:** {soc_aware_result['strategy_type']} - {soc_aware_result['power_kw']:.1f} kW discharge")
                            
                            # Strategy recommendations
                            power_diff = abs(soc_aware_result['power_kw'] - default_result['power_kw'])
                            if power_diff > 5:  # Significant difference
                                if test_soc < 25:
                                    st.info("💡 **Recommendation:** SOC-Aware strategy is safer for low battery levels")
                                elif test_excess > 100:
                                    st.info("💡 **Recommendation:** Default strategy provides maximum peak shaving for high demand")
                                else:
                                    st.info("💡 **Recommendation:** Both strategies viable - choose based on priority: battery life vs. peak reduction")
                            else:
                                st.info("💡 **Note:** Strategies produce similar results for these parameters")
                
                # Current Strategy Status (outside battery config scope)
                st.markdown("**🎯 Current Strategy Selection:**")
                if selected_strategy == "Default Shaving":
                    st.success("**Active:** Default (Aggressive)")
                    st.info("**Strategy:** Uses 80% excess discharge, 5% min SOC - prioritizes maximum MD cost savings")
                elif selected_strategy == "SOC-Aware":
                    st.success("**Active:** SOC-Aware (Conservative)")
                    st.info("**Strategy:** Uses 60% excess discharge, 20% min SOC - prioritizes battery longevity")
                else:
                    st.warning(f"**{selected_strategy}** - Advanced implementation pending")
                    
                    # Store selected strategy for future use
                    st.session_state['selected_shaving_strategy'] = selected_strategy
                    st.session_state['strategy_config'] = {
                        'strategy': selected_strategy,
                        'forecasting_enabled': enable_forecasting,
                        'data_available': forecast_data_available if enable_forecasting else ('shaving_historical_data' in st.session_state)
                    }
            
            except Exception as e:
                st.error(f"❌ Error configuring data inputs: {str(e)}")
                
        except Exception as e:
            st.error(f"❌ Error processing uploaded file: {str(e)}")
            
    else:
        # Show information when no file is uploaded
        st.info("👆 **Upload your energy data file to begin V2 analysis**")
        
        # Information about current development status
        with st.expander("ℹ️ Development Status & Available Functions"):
            st.markdown("""
            **Available V2 Functions:**
            - ✅ `_render_v2_battery_controls()` - Enhanced battery configuration interface
            - ✅ `load_vendor_battery_database()` - Battery database integration
            - ✅ `_calculate_monthly_targets_v2()` - Monthly-based target calculation
            - ✅ `_generate_clustering_summary_table()` - Peak events clustering analysis
            - ✅ `build_daily_simulator_structure()` - Advanced daily simulation structure
            - ✅ Multiple utility functions for enhanced analysis
            
            **Ready for Use:**
            - 🔄 File upload and data processing
            - 🔄 Battery configuration interface
            - 🔄 Data validation and preprocessing
            
            **In Development:**
            - 🔄 Complete V2 user interface
            - 🔄 Advanced battery optimization algorithms
            - 🔄 Multi-scenario comparison engine
            - 🔄 Enhanced cost analysis and ROI calculations
            - 🔄 Advanced visualization suite
            """)
        
        # Information section
        st.subheader("📖 About MD Shaving V2")
        st.markdown("""
        The MD Shaving Solution V2 represents the next evolution of maximum demand optimization technology,
        building upon the proven foundation of V1 with significant enhancements:
        
        **Key Improvements over V1:**
        - **File Upload Integration**: Seamless data import with validation
        - **Monthly Target Calculation**: More sophisticated target setting algorithms
        - **Battery Database Integration**: Access to vendor specifications and real-world performance data
        - **Enhanced Analytics**: Advanced clustering and event analysis capabilities
        - **Improved User Experience**: Streamlined interface with better visualization
        
        **Technical Architecture:**
        - Modular design with reusable components
        - Backward compatibility with V1 functionality
        - Extensible framework for future enhancements
        - Integration-ready for external systems
        
        **File Format Requirements:**
        - **CSV, Excel**: .csv, .xls, .xlsx formats supported
        - **Columns**: Timestamp (date/time) and Power (kW) columns required
        - **Data Quality**: Clean, continuous time series data recommended
        """)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    render_md_shaving_v2()