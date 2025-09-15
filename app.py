# Fresh start - Load Forecasting MVP
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# Helper function to read different file formats
def read_uploaded_file(file):
    """Read uploaded file based on its extension"""
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    elif file.name.endswith(('.xls', '.xlsx')):
        return pd.read_excel(file)
    else:
        raise ValueError("Unsupported file format. Please upload CSV, XLS, or XLSX files.")

def _auto_detect_columns(df):
    """
    Auto-detect timestamp and power columns based on common patterns.
    Returns tuple of (timestamp_col, power_col)
    """
    timestamp_col = None
    power_col = None
    
    # Auto-detect timestamp column
    for col in df.columns:
        col_lower = col.lower()
        
        # Check for common timestamp column names
        timestamp_keywords = ['date', 'time', 'timestamp', 'datetime', 'dt', 'period']
        if any(keyword in col_lower for keyword in timestamp_keywords):
            timestamp_col = col
            break
        
        # If no keyword match, check if column contains datetime-like values
        if timestamp_col is None:
            try:
                # Try to parse first few non-null values as datetime
                sample_values = df[col].dropna().head(5)
                if len(sample_values) > 0:
                    pd.to_datetime(sample_values.iloc[0])
                    timestamp_col = col
                    break
            except:
                continue
    
    # Auto-detect power column
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in numeric_cols:
        col_lower = col.lower()
        
        # Check for common power/demand column names
        power_keywords = ['power', 'kw', 'kilowatt', 'demand', 'load', 'consumption', 'kwh']
        if any(keyword in col_lower for keyword in power_keywords):
            power_col = col
            break
            
    # If no keyword match, use first numeric column as fallback
    if power_col is None and numeric_cols:
        power_col = numeric_cols[0]
    
    return timestamp_col, power_col

def _configure_data_inputs(df):
    """Configure data inputs including column selection."""
    st.subheader("Data Configuration")
    
    # Auto-detect columns
    auto_timestamp_col, auto_power_col = _auto_detect_columns(df)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Column Selection**")
        
        # Auto-selected timestamp column with option to override
        timestamp_options = list(df.columns)
        
        if auto_timestamp_col:
            try:
                timestamp_index = timestamp_options.index(auto_timestamp_col)
            except ValueError:
                timestamp_index = 0
        else:
            timestamp_index = 0
        
        timestamp_col = st.selectbox(
            "Timestamp column (auto-detected):", 
            timestamp_options, 
            index=timestamp_index,
            key="timestamp_col",
            help="Auto-detected based on datetime patterns. Change if incorrect."
        )
        
        # Auto-selected power column with option to override
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if auto_power_col and auto_power_col in numeric_cols:
            try:
                power_index = numeric_cols.index(auto_power_col)
            except ValueError:
                power_index = 0
        else:
            power_index = 0
        
        power_col = st.selectbox(
            "Power (kW) column (auto-detected):", 
            numeric_cols, 
            index=power_index,
            key="power_col",
            help="Auto-detected based on column names containing 'power', 'kw', 'demand', etc."
        )
    
    with col2:
        st.markdown("**Data Preview**")
        if timestamp_col and power_col:
            preview_df = df[[timestamp_col, power_col]].head(10)
            st.dataframe(preview_df, use_container_width=True)
    
    return timestamp_col, power_col

def _calculate_roc(df_processed, power_col):
    """Calculate Rate of Change (ROC) in kW per minute."""
    df_roc = df_processed.copy()
    
    # Calculate time differences in minutes
    df_roc['time_diff_min'] = df_roc.index.to_series().diff().dt.total_seconds() / 60
    
    # Calculate power differences
    df_roc['power_diff_kw'] = df_roc[power_col].diff()
    
    # Calculate ROC (kW per minute)
    df_roc['roc_kw_per_min'] = df_roc['power_diff_kw'] / df_roc['time_diff_min']
    
    # Create clean output dataframe
    roc_df = pd.DataFrame({
        'Timestamp': df_roc.index,
        'Power (kW)': df_roc[power_col],
        'ROC (kW/min)': df_roc['roc_kw_per_min']
    })
    
    return roc_df

def _detect_data_interval(df_processed):
    """Detect the data interval from timestamps."""
    if len(df_processed) > 1:
        # Get time differences
        time_diffs = df_processed.index.to_series().diff().dropna()
        
        # Find the most common interval
        mode_interval = time_diffs.mode()
        if len(mode_interval) > 0:
            interval_minutes = mode_interval.iloc[0].total_seconds() / 60
            return interval_minutes
    
    return None

def _process_dataframe(df, timestamp_col):
    """Process the dataframe with timestamp parsing, sorting validation, and indexing."""
    df_processed = df.copy()
    
    # Parse timestamp column
    df_processed[timestamp_col] = pd.to_datetime(df_processed[timestamp_col], errors='coerce')
    
    # Remove rows with invalid timestamps
    initial_rows = len(df_processed)
    df_processed = df_processed.dropna(subset=[timestamp_col])
    final_rows = len(df_processed)
    
    if final_rows < initial_rows:
        st.warning(f"Removed {initial_rows - final_rows} rows with invalid timestamps")
    
    # Sort by timestamp
    df_processed = df_processed.sort_values(timestamp_col)
    
    # Set timestamp as index
    df_processed.set_index(timestamp_col, inplace=True)
    
    return df_processed

# Main app
st.title("🔋 Load Forecasting MVP")
st.markdown("""
Upload your load profile data to begin analysis.
Supports CSV, XLS, and XLSX file formats.
""")

# File upload
uploaded_file = st.file_uploader(
    "Upload your data file", 
    type=["csv", "xls", "xlsx"], 
    key="file_uploader"
)

if uploaded_file:
    try:
        # Read the uploaded file
        with st.spinner("Reading uploaded file..."):
            df = read_uploaded_file(uploaded_file)
        
        st.success(f"✅ File uploaded successfully! Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        
        # Configure data inputs
        timestamp_col, power_col = _configure_data_inputs(df)
        
        if timestamp_col and power_col:
            # Process the dataframe
            with st.spinner("Processing data..."):
                df_processed = _process_dataframe(df, timestamp_col)
            
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
            
            # Show processed data preview
            st.subheader("📋 Processed Data Preview")
            st.dataframe(df_processed[[power_col]].head(20), use_container_width=True)
            
            # Detect data interval
            interval_minutes = _detect_data_interval(df_processed)
            if interval_minutes:
                st.info(f"📊 Detected data interval: {interval_minutes:.1f} minutes")
            
            # Rate of Change (ROC) Analysis
            st.subheader("📈 Rate of Change (ROC)")
            st.markdown("*Rate of change in power consumption (kW per minute)*")
            
            # Calculate ROC
            roc_df = _calculate_roc(df_processed, power_col)
            
            # Display ROC statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                avg_roc = roc_df['ROC (kW/min)'].mean()
                st.metric("Average ROC", f"{avg_roc:.3f} kW/min")
                
            with col2:
                max_roc = roc_df['ROC (kW/min)'].max()
                st.metric("Max ROC", f"{max_roc:.3f} kW/min")
                
            with col3:
                min_roc = roc_df['ROC (kW/min)'].min()
                st.metric("Min ROC", f"{min_roc:.3f} kW/min")
            
            # Display ROC table
            st.markdown("**ROC Data Table** (showing first 20 rows)")
            
            # Format the ROC values for display
            roc_display = roc_df.head(20).copy()
            roc_display['ROC (kW/min)'] = roc_display['ROC (kW/min)'].apply(
                lambda x: "" if pd.isna(x) else f"{x:.3f}"
            )
            roc_display['Power (kW)'] = roc_display['Power (kW)'].apply(
                lambda x: f"{x:.2f}"
            )
            
            st.dataframe(roc_display, use_container_width=True)
            
            # ROC insights
            with st.expander("💡 ROC Analysis Insights"):
                st.markdown(f"""
                **Understanding Rate of Change (ROC):**
                - **Positive ROC**: Power consumption is increasing
                - **Negative ROC**: Power consumption is decreasing  
                - **Zero ROC**: Power consumption is stable
                
                **Your Data:**
                - **Data Interval**: {interval_minutes:.1f} minutes (auto-detected)
                - **ROC Range**: {min_roc:.3f} to {max_roc:.3f} kW/min
                - **Average ROC**: {avg_roc:.3f} kW/min
                
                **Note:** First row ROC is blank as it requires a previous data point for calculation.
                """)
            
            # Basic power statistics
            st.subheader("⚡ Power Statistics")
            power_stats = df_processed[power_col].describe()
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Minimum", f"{power_stats['min']:.2f} kW")
            col2.metric("Maximum", f"{power_stats['max']:.2f} kW")
            col3.metric("Mean", f"{power_stats['mean']:.2f} kW")
            col4.metric("Std Dev", f"{power_stats['std']:.2f} kW")
            
        else:
            st.error("Please select both timestamp and power columns to proceed.")
            
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.info("Please ensure your file contains proper timestamp and numeric power data.")
else:
    st.info("👆 Please upload a data file to begin analysis.")
    
    # Instructions
    with st.expander("📖 File Format Instructions"):
        st.markdown("""
        **Supported file formats:**
        - CSV (.csv)
        - Excel (.xls, .xlsx)
        
        **Required columns:**
        - **Timestamp column**: Contains date/time information
        - **Power column**: Contains numeric power values in kW
        
        **Example formats:**
        ```
        Timestamp,Power_kW
        2024-01-01 00:00:00,150.5
        2024-01-01 00:30:00,145.2
        ```
        
        The app will automatically detect your columns based on common naming patterns.
        """)
