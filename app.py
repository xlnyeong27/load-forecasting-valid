# Fresh start - Load Forecasting MVP
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# Chart functions
def _create_demand_chart(df_processed, power_col):
    """Create line chart of actual kW over time."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df_processed.index,
        y=df_processed[power_col],
        mode='lines',
        name='Power Demand',
        line=dict(color='blue', width=2),
        hovertemplate='Time: %{x}<br>Power: %{y:.2f} kW<extra></extra>'
    ))
    
    fig.update_layout(
        title='Power Demand Over Time',
        xaxis_title='Time',
        yaxis_title='Power (kW)',
        height=400,
        hovermode='x unified'
    )
    
    return fig

def _create_roc_chart(roc_df, threshold_kw_per_min):
    """Create ROC chart with threshold guides."""
    fig = go.Figure()
    
    # ROC line
    fig.add_trace(go.Scatter(
        x=roc_df['Timestamp'],
        y=roc_df['ROC (kW/min)'],
        mode='lines',
        name='Rate of Change',
        line=dict(color='green', width=2),
        hovertemplate='Time: %{x}<br>ROC: %{y:.3f} kW/min<extra></extra>'
    ))
    
    # Positive threshold line
    fig.add_hline(
        y=threshold_kw_per_min,
        line_dash="dash",
        line_color="red",
        annotation_text=f"+{threshold_kw_per_min} kW/min threshold",
        annotation_position="top left"
    )
    
    # Negative threshold line
    fig.add_hline(
        y=-threshold_kw_per_min,
        line_dash="dash",
        line_color="red",
        annotation_text=f"-{threshold_kw_per_min} kW/min threshold",
        annotation_position="bottom left"
    )
    
    # Zero line
    fig.add_hline(
        y=0,
        line_dash="dot",
        line_color="gray",
        opacity=0.5
    )
    
    fig.update_layout(
        title=f'Rate of Change (ROC) with ±{threshold_kw_per_min} kW/min Thresholds',
        xaxis_title='Time',
        yaxis_title='ROC (kW/min)',
        height=400,
        hovermode='x unified'
    )
    
    return fig

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
            
            # ROC Threshold Control
            roc_threshold = st.slider(
                "ROC Threshold (kW/min)",
                min_value=0.1,
                max_value=10.0,
                value=3.0,
                step=0.1,
                help="Threshold for ROC analysis. Values above +T or below -T will be highlighted."
            )
            
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
            
            # Charts Section
            st.subheader("📈 Demand & ROC Charts")
            st.markdown(f"*Using ROC threshold: ±{roc_threshold} kW/min*")
            
            # Chart 1: Power Demand Over Time
            st.markdown("#### 1️⃣ Power Demand Over Time")
            demand_chart = _create_demand_chart(df_processed, power_col)
            st.plotly_chart(demand_chart, use_container_width=True)
            
            # Chart 2: ROC Over Time with Thresholds
            st.markdown("#### 2️⃣ Rate of Change (ROC) with Thresholds")
            roc_chart = _create_roc_chart(roc_df, roc_threshold)
            st.plotly_chart(roc_chart, use_container_width=True)
            
            # ROC threshold analysis
            st.markdown("#### 🎯 ROC Threshold Analysis")
            
            # Count values above/below thresholds
            above_positive = len(roc_df[roc_df['ROC (kW/min)'] > roc_threshold])
            below_negative = len(roc_df[roc_df['ROC (kW/min)'] < -roc_threshold])
            within_threshold = len(roc_df) - above_positive - below_negative - 1  # -1 for NaN first row
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Above +T", f"{above_positive}", delta=f"{above_positive/len(roc_df)*100:.1f}%")
            col2.metric("Below -T", f"{below_negative}", delta=f"{below_negative/len(roc_df)*100:.1f}%")
            col3.metric("Within ±T", f"{within_threshold}", delta=f"{within_threshold/len(roc_df)*100:.1f}%")
            

            # Chart insights
            with st.expander("💡 Chart Insights"):
                st.markdown(f"""
                **Demand Chart Analysis:**
                - Shows actual power consumption over time
                - Identify patterns, peaks, and trends
                - Look for daily/weekly cycles
                
                **ROC Chart Analysis:**
                - Red dashed lines: ±{roc_threshold} kW/min thresholds
                - **Above +{roc_threshold}**: Rapid power increase ({above_positive} points)
                - **Below -{roc_threshold}**: Rapid power decrease ({below_negative} points)
                - **Within ±{roc_threshold}**: Stable/gradual changes ({within_threshold} points)
                
                **Use the slider to adjust the threshold and see how it affects the analysis!**
                """)

            # --- Power Usage Forecast Section ---
            st.subheader("🔮 Power Usage Forecast Table")
            st.markdown("Forecast future power using anchor points and ROC.")

            # Controls
            colA, colB, colC = st.columns(3)
            with colA:
                n_anchors = st.number_input("Number of anchors", min_value=3, max_value=100, value=10, step=1)
            with colB:
                anchor_method = st.selectbox("Anchor sampling method", ["Random", "Grid (every 15 min)", "Ramp starts"])
            with colC:
                horizon_options = [1, 2, 5, 10, 20]
                horizons = st.multiselect("Forecast horizons (min)", horizon_options, default=[1, 5, 10, 20])

            # Optional controls for headroom/MD risk
            st.markdown("**(Optional)**: Enter MD target and margin for headroom/risk analysis.")
            colD, colE = st.columns(2)
            with colD:
                md_target = st.number_input("MD target (kW)", min_value=0.0, value=200.0, step=1.0)
            with colE:
                md_margin = st.number_input("MD margin (kW)", min_value=0.0, value=10.0, step=1.0)

            # Prepare anchor candidates (exclude first row, drop NA ROC)
            anchor_df = roc_df.dropna(subset=["ROC (kW/min)"]).copy()
            anchor_df = anchor_df.reset_index(drop=True)

            # Anchor sampling with deterministic approach
            import numpy as np
            import random
            
            # Create a deterministic seed based on data characteristics to ensure consistency
            data_seed = hash(str(len(anchor_df)) + str(anchor_df['Timestamp'].min()) + str(anchor_df['Timestamp'].max())) % 2147483647
            random.seed(data_seed)
            np.random.seed(data_seed)
            
            anchor_indices = []
            if anchor_method == "Random":
                if len(anchor_df) <= n_anchors:
                    anchor_indices = list(range(len(anchor_df)))
                else:
                    anchor_indices = sorted(random.sample(range(len(anchor_df)), n_anchors))
            elif anchor_method == "Grid (every 15 min)":
                # Find indices spaced by at least 15 min
                anchor_indices = [0]
                last_ts = anchor_df.loc[0, "Timestamp"]
                for i, row in anchor_df.iterrows():
                    if (row["Timestamp"] - last_ts).total_seconds() >= 15*60:
                        anchor_indices.append(i)
                        last_ts = row["Timestamp"]
                    if len(anchor_indices) >= n_anchors:
                        break
            elif anchor_method == "Ramp starts":
                # Anchor at points where ROC crosses threshold
                ramp_mask = (anchor_df["ROC (kW/min)"] > roc_threshold) | (anchor_df["ROC (kW/min)"] < -roc_threshold)
                ramp_indices = list(np.where(ramp_mask)[0])
                if len(ramp_indices) > n_anchors:
                    anchor_indices = sorted(random.sample(ramp_indices, n_anchors))
                else:
                    anchor_indices = ramp_indices
            
            # Reset random state to avoid affecting other parts
            random.seed()
            np.random.seed()
            
            # Fallback if not enough anchors
            if len(anchor_indices) == 0:
                st.warning("No anchor points found for the selected method. Try another method or lower the threshold.")
            else:
                # Build forecast table
                results = []
                for idx in anchor_indices:
                    anchor_row = anchor_df.iloc[idx]
                    anchor_ts = anchor_row["Timestamp"]
                    P_now = anchor_row["Power (kW)"]
                    ROC_now = anchor_row["ROC (kW/min)"]
                    for h in sorted(horizons):
                        # Find actual future value
                        target_ts = anchor_ts + pd.Timedelta(minutes=h)
                        # Find closest row in df_processed (timestamp is the index)
                        try:
                            # Try exact match first
                            P_actual = df_processed.loc[target_ts, power_col]
                        except KeyError:
                            # Try nearest time (if exact not found)
                            nearest_idx = (df_processed.index - target_ts).abs().idxmin()
                            P_actual = df_processed.loc[nearest_idx, power_col]
                        # Forecast
                        P_hat = P_now + ROC_now * h
                        error = P_hat - P_actual
                        abs_error = abs(error)
                        headroom = md_target - P_hat
                        md_risk = headroom <= md_margin
                        results.append({
                            "anchor_ts": anchor_ts,
                            "dt_min": 1,
                            "horizon_min": h,
                            "P_now_kW": P_now,
                            "ROC_now_kW_per_min": ROC_now,
                            "P_hat_kW": P_hat,
                            "P_actual_kW": P_actual,
                            "error_kW": error,
                            "abs_error_kW": abs_error,
                            "headroom_kW": headroom,
                            "md_risk": md_risk
                        })
                forecast_df = pd.DataFrame(results)
                # Formatting
                forecast_df["anchor_ts"] = forecast_df["anchor_ts"].dt.strftime("%Y-%m-%d %H:%M")
                st.dataframe(forecast_df, use_container_width=True)
                
                # --- Interactive Anchor Graphs ---
                st.markdown("#### 📈 Anchor Point Analysis")
                st.markdown("*View prediction vs actual for individual anchor points*")
                
                # Get unique anchor timestamps for pill selector
                unique_anchors = forecast_df["anchor_ts"].unique()
                
                # Debug: Show what anchors were generated
                st.write("🔍 **Debug Info:**")
                st.write(f"Generated {len(unique_anchors)} unique anchors: {list(unique_anchors)}")
                
                if len(unique_anchors) > 0:
                    # Create selectbox for anchor selection (no rerun needed)
                    st.markdown("**Select Anchor Point:**")
                    
                    # Use selectbox instead of buttons to avoid rerun
                    selected_anchor = st.selectbox(
                        "Choose anchor timestamp:",
                        options=unique_anchors,
                        index=0 if len(unique_anchors) > 0 else 0,
                        key="anchor_selectbox",
                        help="Select an anchor point to view its prediction vs actual graph"
                    )
                    
                    # Debug: Show what was selected
                    st.write(f"Selected anchor: **{selected_anchor}**")
                    
                    # Filter data for selected anchor - ensure exact match
                    selected_anchor_data = forecast_df[forecast_df["anchor_ts"] == selected_anchor].copy()
                    st.write(f"Found {len(selected_anchor_data)} rows for selected anchor")
                    
                    if not selected_anchor_data.empty:
                        # Sort by horizon for proper line plotting
                        selected_anchor_data = selected_anchor_data.sort_values("horizon_min")
                        
                        # Create the graph for selected anchor
                        fig_anchor = go.Figure()
                        
                        # Plot Prediction line
                        fig_anchor.add_trace(go.Scatter(
                            x=selected_anchor_data["horizon_min"],
                            y=selected_anchor_data["P_hat_kW"],
                            mode='lines+markers',
                            name='Prediction',
                            line=dict(color='#FF6B6B', width=3),
                            marker=dict(color='#FF6B6B', size=8),
                            hovertemplate='Horizon: %{x} min<br>Prediction: %{y:.2f} kW<extra></extra>'
                        ))
                        
                        # Plot Actual line
                        fig_anchor.add_trace(go.Scatter(
                            x=selected_anchor_data["horizon_min"],
                            y=selected_anchor_data["P_actual_kW"],
                            mode='lines+markers',
                            name='Actual',
                            line=dict(color='#4ECDC4', width=3),
                            marker=dict(color='#4ECDC4', size=8),
                            hovertemplate='Horizon: %{x} min<br>Actual: %{y:.2f} kW<extra></extra>'
                        ))
                        
                        # Update layout
                        fig_anchor.update_layout(
                            title=f'Prediction vs Actual for Anchor: {selected_anchor}',
                            xaxis_title='Horizon (minutes)',
                            yaxis_title='Power (kW)',
                            height=400,
                            hovermode='x unified',
                            showlegend=True,
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1
                            ),
                            xaxis=dict(
                                tickmode='array',
                                tickvals=selected_anchor_data["horizon_min"].tolist()
                            )
                        )
                        
                        # Display the chart
                        st.plotly_chart(fig_anchor, use_container_width=True)
                        
                        # Display summary statistics for selected anchor
                        with st.expander(f"📊 Summary for Anchor {selected_anchor}"):
                            col1, col2, col3, col4 = st.columns(4)
                            
                            mean_error = selected_anchor_data["error_kW"].mean()
                            mean_abs_error = selected_anchor_data["abs_error_kW"].mean()
                            max_error = selected_anchor_data["error_kW"].abs().max()
                            anchor_power = selected_anchor_data["P_now_kW"].iloc[0]
                            
                            col1.metric("Anchor Power", f"{anchor_power:.2f} kW")
                            col2.metric("Mean Error", f"{mean_error:+.2f} kW")
                            col3.metric("Mean Abs Error", f"{mean_abs_error:.2f} kW")
                            col4.metric("Max Error", f"{max_error:.2f} kW")
                    else:
                        st.error(f"No data available for selected anchor point: {selected_anchor}")
                        st.write("Available data preview:")
                        st.write(forecast_df[["anchor_ts", "horizon_min", "P_hat_kW", "P_actual_kW"]].head())
                else:
                    st.info("No anchor points available. Please generate forecasts first.")

                # --- Error Percentage Table ---
                st.markdown("#### 📊 Forecast Error Percentage Table")
                st.markdown("*Rows: Anchors | Columns: Forecast Minutes | Values: Percentage Error (%)*")
                
                if len(unique_anchors) > 0 and len(forecast_df) > 0:
                    # Calculate percentage errors
                    forecast_df_copy = forecast_df.copy()
                    forecast_df_copy["percent_error"] = (forecast_df_copy["abs_error_kW"] / forecast_df_copy["P_actual_kW"]) * 100
                    
                    # Create pivot table with anchors as rows and forecast minutes as columns
                    error_table = forecast_df_copy.pivot_table(
                        index="anchor_ts", 
                        columns="horizon_min", 
                        values="percent_error", 
                        aggfunc="first"
                    )
                    
                    # Sort by anchor timestamp and horizon
                    error_table = error_table.sort_index()
                    error_table = error_table.reindex(columns=sorted(error_table.columns))
                    
                    # Format column headers to include "min"
                    error_table.columns = [f"{col} min" for col in error_table.columns]
                    
                    # Display table dimensions
                    st.info(f"**Table Dimensions:** {len(error_table)} anchors × {len(error_table.columns)} forecast horizons")
                    
                    # Format the table for better display
                    styled_table = error_table.style.format("{:.2f}%").background_gradient(
                        cmap='RdYlGn_r',  # Red-Yellow-Green reversed (red for high errors)
                        subset=error_table.columns,
                        vmin=0,
                        vmax=30  # Cap gradient at 30% for better color distribution
                    ).set_table_styles([
                        {'selector': 'thead th', 'props': [('background-color', '#40466e'), ('color', 'white'), ('font-weight', 'bold')]},
                        {'selector': 'tbody td', 'props': [('text-align', 'center')]},
                        {'selector': 'th.row_heading', 'props': [('background-color', '#f0f2f6'), ('font-weight', 'bold')]}
                    ])
                    
                    # Display the styled table
                    st.dataframe(styled_table, use_container_width=True)
                    
                    # Summary statistics for the table
                    with st.expander("📈 Table Summary Statistics"):
                        col1, col2, col3, col4 = st.columns(4)
                        
                        # Calculate overall statistics
                        all_errors = error_table.values.flatten()
                        all_errors = all_errors[~pd.isna(all_errors)]  # Remove NaN values
                        
                        if len(all_errors) > 0:
                            col1.metric("Mean Error", f"{np.mean(all_errors):.2f}%")
                            col2.metric("Median Error", f"{np.median(all_errors):.2f}%")
                            col3.metric("Max Error", f"{np.max(all_errors):.2f}%")
                            col4.metric("Min Error", f"{np.min(all_errors):.2f}%")
                            
                            # Best and worst performing combinations
                            st.markdown("**Performance Analysis:**")
                            
                            # Find best and worst anchor-horizon combinations
                            flat_data = error_table.stack().reset_index()
                            flat_data.columns = ['Anchor', 'Horizon', 'Error_%']
                            
                            best_combo = flat_data.loc[flat_data['Error_%'].idxmin()]
                            worst_combo = flat_data.loc[flat_data['Error_%'].idxmax()]
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.success(f"**🏆 Best Performance**  \n{best_combo['Anchor']} @ {best_combo['Horizon']}  \n**{best_combo['Error_%']:.2f}%** error")
                            with col2:
                                st.error(f"**⚠️ Worst Performance**  \n{worst_combo['Anchor']} @ {worst_combo['Horizon']}  \n**{worst_combo['Error_%']:.2f}%** error")
                            
                            # Error distribution by horizon
                            st.markdown("**Error by Forecast Horizon:**")
                            horizon_stats = error_table.describe().T
                            st.dataframe(horizon_stats.round(2), use_container_width=True)
                        else:
                            st.warning("No valid error data available for analysis.")
                else:
                    st.info("Generate forecasts first to view error percentage table.")

                # --- Enhanced Evaluation Metrics ---
                st.markdown("#### 📈 Comprehensive Error Metrics by Horizon")
                st.markdown("*Advanced performance metrics including MAE, RMSE, sMAPE, WAPE, and percentiles*")
                
                if len(unique_anchors) > 0 and len(forecast_df) > 0:
                    # Configuration for MAPE threshold and filters
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        mape_threshold = st.number_input(
                            "MAPE Threshold (kW)",
                            min_value=0.0,
                            value=200.0,
                            step=10.0,
                            help="Exclude rows where actual power < threshold from MAPE calculation"
                        )
                    
                    with col2:
                        analysis_type = st.selectbox(
                            "Analysis Type",
                            ["Overall", "Time-of-Day", "ROC Regime", "Load Bands"],
                            help="Choose how to segment the error analysis"
                        )
                    
                    with col3:
                        if analysis_type == "Load Bands":
                            num_load_bands = st.slider(
                                "Number of Load Bands",
                                min_value=3,
                                max_value=8,
                                value=5,
                                help="Number of equal-sized load bands to create"
                            )
                    
                    # Prepare forecast data with additional features for segmentation
                    enhanced_forecast_df = forecast_df.copy()
                    enhanced_forecast_df["ape"] = (enhanced_forecast_df["abs_error_kW"] / enhanced_forecast_df["P_actual_kW"]) * 100
                    
                    # Add time-of-day information if datetime column exists
                    if "datetime" in enhanced_forecast_df.columns:
                        enhanced_forecast_df["hour"] = pd.to_datetime(enhanced_forecast_df["datetime"]).dt.hour
                        enhanced_forecast_df["time_bucket"] = enhanced_forecast_df["hour"].apply(
                            lambda h: "Night (0-6)" if 0 <= h < 6
                            else "Morning (6-12)" if 6 <= h < 12
                            else "Afternoon (12-18)" if 12 <= h < 18
                            else "Evening (18-24)"
                        )
                    # Add time-of-day information if datetime column exists
                    if "datetime" in enhanced_forecast_df.columns:
                        enhanced_forecast_df["hour"] = pd.to_datetime(enhanced_forecast_df["datetime"]).dt.hour
                        enhanced_forecast_df["time_bucket"] = enhanced_forecast_df["hour"].apply(
                            lambda h: "Night (0-6)" if 0 <= h < 6
                            else "Morning (6-12)" if 6 <= h < 12
                            else "Afternoon (12-18)" if 12 <= h < 18
                            else "Evening (18-24)"
                        )
                    else:
                        enhanced_forecast_df["time_bucket"] = "Unknown"
                    
                    # Add ROC regime information (simplified approach)
                    enhanced_forecast_df["roc_regime"] = "Calm"  # Default to Calm regime
                    
                    # Add load bands
                    if analysis_type == "Load Bands":
                        load_min = enhanced_forecast_df["P_actual_kW"].min()
                        load_max = enhanced_forecast_df["P_actual_kW"].max()
                        load_range = load_max - load_min
                        band_size = load_range / num_load_bands
                        
                        def get_load_band(power):
                            band_num = min(int((power - load_min) / band_size) + 1, num_load_bands)
                            band_start = load_min + (band_num - 1) * band_size
                            band_end = load_min + band_num * band_size
                            return f"Band {band_num}: {band_start:.0f}-{band_end:.0f} kW"
                        
                        enhanced_forecast_df["load_band"] = enhanced_forecast_df["P_actual_kW"].apply(get_load_band)
                    
                    # Define segmentation based on analysis type
                    if analysis_type == "Overall":
                        segments = [("Overall", enhanced_forecast_df)]
                    elif analysis_type == "Time-of-Day":
                        segments = [(bucket, group) for bucket, group in enhanced_forecast_df.groupby("time_bucket")]
                    elif analysis_type == "ROC Regime":
                        segments = [(regime, group) for regime, group in enhanced_forecast_df.groupby("roc_regime")]
                    elif analysis_type == "Load Bands":
                        segments = [(band, group) for band, group in enhanced_forecast_df.groupby("load_band")]
                    
                    # Calculate enhanced metrics per segment and horizon
                    def calculate_segment_metrics(segment_name, segment_data):
                        segment_metrics = []
                        available_horizons = sorted(segment_data["horizon_min"].unique())
                        
                        for horizon in available_horizons:
                            horizon_data = segment_data[segment_data["horizon_min"] == horizon].copy()
                            
                            if len(horizon_data) > 0:
                                # Filter for MAPE calculation (exclude low power values)
                                mape_eligible = horizon_data[horizon_data["P_actual_kW"] >= mape_threshold]
                                
                                # MAE and RMSE in kW
                                mae_kw = horizon_data["abs_error_kW"].mean()
                                rmse_kw = np.sqrt((horizon_data["error_kW"] ** 2).mean())
                                
                                # MAPE (only for eligible data points)
                                mape = mape_eligible["ape"].mean() if len(mape_eligible) > 0 else np.nan
                                mape_count = len(mape_eligible)
                                
                                # sMAPE (Symmetric MAPE)
                                smape_values = []
                                for _, row in horizon_data.iterrows():
                                    actual = row["P_actual_kW"]
                                    forecast = row["P_hat_kW"]
                                    if actual != 0 or forecast != 0:
                                        smape_val = (abs(forecast - actual) / ((abs(actual) + abs(forecast)) / 2)) * 100
                                        smape_values.append(smape_val)
                                smape = np.mean(smape_values) if smape_values else np.nan
                                
                                # WAPE (Weighted Absolute Percentage Error)
                                total_abs_error = horizon_data["abs_error_kW"].sum()
                                total_actual = horizon_data["P_actual_kW"].sum()
                                wape = (total_abs_error / total_actual) * 100 if total_actual > 0 else np.nan
                                
                                # Percentiles of APE
                                p50_ape = horizon_data["ape"].median()
                                p90_ape = horizon_data["ape"].quantile(0.9)
                                
                                segment_metrics.append({
                                    'Segment': segment_name,
                                    'Horizon (min)': horizon,
                                    'Count': len(horizon_data),
                                    'MAE (kW)': f"{mae_kw:.1f}",
                                    'RMSE (kW)': f"{rmse_kw:.1f}",
                                    'MAPE (%)': f"{mape:.1f}" if not np.isnan(mape) else "N/A",
                                    'sMAPE (%)': f"{smape:.1f}" if not np.isnan(smape) else "N/A",
                                    'WAPE (%)': f"{wape:.1f}" if not np.isnan(wape) else "N/A",
                                    'P50 APE (%)': f"{p50_ape:.1f}",
                                    'P90 APE (%)': f"{p90_ape:.1f}"
                                })
                        
                        return segment_metrics
                    
                    # Generate metrics for all segments
                    all_segment_metrics = []
                    for segment_name, segment_data in segments:
                        segment_metrics = calculate_segment_metrics(segment_name, segment_data)
                        all_segment_metrics.extend(segment_metrics)
                    
                    if all_segment_metrics:
                        segment_metrics_df = pd.DataFrame(all_segment_metrics)
                        
                        # Display segmented metrics
                        st.markdown(f"**📊 Error Metrics by {analysis_type}:**")
                        
                        # Show summary statistics
                        if analysis_type != "Overall":
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                total_segments = len(segments)
                                st.metric("Total Segments", total_segments)
                            with col2:
                                avg_count_per_segment = segment_metrics_df.groupby('Segment')['Count'].sum().mean()
                                st.metric("Avg Points/Segment", f"{avg_count_per_segment:.0f}")
                            with col3:
                                total_points = segment_metrics_df['Count'].astype(int).sum()
                                st.metric("Total Data Points", total_points)
                        
                        # Display compact tables per segment
                        for segment_name, segment_data in segments:
                            segment_subset = segment_metrics_df[segment_metrics_df['Segment'] == segment_name]
                            if len(segment_subset) > 0:
                                with st.expander(f"📋 {segment_name} ({len(segment_data)} points)", expanded=(analysis_type == "Overall")):
                                    # Remove the Segment column for cleaner display
                                    display_df = segment_subset.drop('Segment', axis=1)
                                    st.dataframe(display_df, use_container_width=True)
                                    
                                    # Mini summary for this segment
                                    if len(display_df) > 1:
                                        avg_mae = pd.to_numeric(display_df['MAE (kW)'], errors='coerce').mean()
                                        avg_rmse = pd.to_numeric(display_df['RMSE (kW)'], errors='coerce').mean()
                                        st.caption(f"Segment Average: MAE {avg_mae:.1f} kW, RMSE {avg_rmse:.1f} kW")
                        
                        # Visualization for segmented analysis
                        if analysis_type != "Overall" and len(segments) > 1:
                            st.markdown("**📈 Segment Comparison Visualization:**")
                            
                            # Create comparison chart
                            fig_segments = go.Figure()
                            
                            # Group by segment and calculate average metrics
                            segment_summary = segment_metrics_df.groupby('Segment').agg({
                                'MAE (kW)': lambda x: pd.to_numeric(x, errors='coerce').mean(),
                                'RMSE (kW)': lambda x: pd.to_numeric(x, errors='coerce').mean(),
                                'Count': lambda x: pd.to_numeric(x, errors='coerce').sum()
                            }).reset_index()
                            
                            fig_segments.add_trace(go.Bar(
                                x=segment_summary['Segment'],
                                y=segment_summary['MAE (kW)'],
                                name='Average MAE (kW)',
                                marker_color='lightblue',
                                yaxis='y'
                            ))
                            
                            fig_segments.add_trace(go.Bar(
                                x=segment_summary['Segment'],
                                y=segment_summary['RMSE (kW)'],
                                name='Average RMSE (kW)',
                                marker_color='darkblue',
                                yaxis='y'
                            ))
                            
                            # Add count as secondary axis
                            fig_segments.add_trace(go.Scatter(
                                x=segment_summary['Segment'],
                                y=segment_summary['Count'],
                                name='Data Points',
                                mode='lines+markers',
                                marker_color='red',
                                yaxis='y2'
                            ))
                            
                            fig_segments.update_layout(
                                title=f"Error Metrics Comparison by {analysis_type}",
                                xaxis_title="Segment",
                                yaxis=dict(title="Error (kW)", side="left"),
                                yaxis2=dict(title="Data Points", side="right", overlaying="y"),
                                barmode='group'
                            )
                            
                            st.plotly_chart(fig_segments, use_container_width=True)
                        
                        # Download segmented metrics
                        segment_csv = segment_metrics_df.to_csv(index=False)
                        st.download_button(
                            label=f"📥 Download {analysis_type} Metrics (CSV)",
                            data=segment_csv,
                            file_name=f"segmented_forecast_metrics_{analysis_type.lower().replace('-', '_').replace(' ', '_')}.csv",
                            mime="text/csv"
                        )
                        
                        # Add metric definitions
                        with st.expander("� Metric Definitions"):
                            st.markdown(f"""
                            - **MAE (kW)**: Mean Absolute Error in kilowatts
                            - **RMSE (kW)**: Root Mean Square Error in kilowatts  
                            - **MAPE (%)**: Mean Absolute Percentage Error (excludes actual < {mape_threshold:.0f} kW)
                            - **sMAPE (%)**: Symmetric Mean Absolute Percentage Error
                            - **WAPE (%)**: Weighted Absolute Percentage Error
                            - **P50/P90 APE (%)**: 50th/90th percentile of Absolute Percentage Error
                            """)
                    
                    else:
                        st.warning("No segment data available for analysis.")
                
                else:
                    st.info("Generate forecasts first to view enhanced metrics.")

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
