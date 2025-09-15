import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from io import BytesIO
from datetime import timedelta

# --- Helper Functions ---
def load_and_validate(file):
    try:
        df = pd.read_excel(file)
        if df.shape[1] < 2:
            st.error("File must contain at least 2 columns (timestamp and kW).")
            return None
        # Use first column as timestamp, second column as kW
        df = df.iloc[:, :2]  # Take only first two columns
        df.columns = ['timestamp', 'kw']  # Rename to standard names
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp', 'kw'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

def interpolate_and_roc(df, freq_min):
    df = df.set_index('timestamp')
    freq = f'{freq_min}T'
    df_interp = df.resample(freq).mean().interpolate('linear')
    df_interp['roc'] = df_interp['kw'].diff() / freq_min
    df_interp = df_interp.reset_index()
    return df_interp

def detect_ramps(df_interp, roc_enter, roc_exit):
    ramps = []
    in_ramp = False
    ramp_start = None
    for i, row in df_interp.iterrows():
        roc = abs(row['roc'])
        if not in_ramp and roc > roc_enter:
            in_ramp = True
            ramp_start = i
        elif in_ramp and roc < roc_exit:
            in_ramp = False
            ramp_end = i
            ramp_df = df_interp.iloc[ramp_start:ramp_end+1]
            ramps.append({
                'start_time': ramp_df['timestamp'].iloc[0],
                'end_time': ramp_df['timestamp'].iloc[-1],
                'duration_min': (ramp_df['timestamp'].iloc[-1] - ramp_df['timestamp'].iloc[0]).total_seconds() / 60,
                'peak_roc': ramp_df['roc'].abs().max(),
                'delta_kw': ramp_df['kw'].iloc[-1] - ramp_df['kw'].iloc[0]
            })
    return pd.DataFrame(ramps)

def seasonal_baseline(df_interp):
    df = df_interp.copy()
    df['weekday'] = df['timestamp'].dt.weekday
    df['time'] = df['timestamp'].dt.time
    baseline = df.groupby(['weekday', 'time'])['kw'].median().reset_index()
    return baseline

def blended_forecast(df_interp, baseline, horizons, alpha=0.7):
    last = df_interp.iloc[-1]
    forecasts = []
    for h in horizons:
        # ROC extrapolation
        roc_pred = last['kw'] + last['roc'] * h
        # Seasonal baseline
        future_time = last['timestamp'] + timedelta(minutes=h)
        weekday = future_time.weekday()
        time = future_time.time()
        base_row = baseline[(baseline['weekday'] == weekday) & (baseline['time'] == time)]
        base_pred = base_row['kw'].values[0] if not base_row.empty else last['kw']
        # Blend
        w = max(0, 1 - h/300) * alpha
        pred_kw = w * roc_pred + (1-w) * base_pred
        forecasts.append({'horizon_min': h, 'pred_kw': pred_kw, 'pred_kwh': pred_kw * h / 60})
    return pd.DataFrame(forecasts)

def md_risk_table(forecasts, md_target, md_margin):
    forecasts['headroom'] = md_target - forecasts['pred_kw']
    forecasts['md_flag'] = forecasts['headroom'] < md_margin
    return forecasts

def plot_interpolation(df_raw, df_interp):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_raw['timestamp'], y=df_raw['kw'], mode='markers', name='Actual', marker=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df_interp['timestamp'], y=df_interp['kw'], mode='lines+markers', name='Interpolated', line=dict(color='orange')))
    fig.update_layout(title='Actual vs Interpolated', xaxis_title='Timestamp', yaxis_title='kW')
    return fig

def plot_roc(df_interp, roc_enter, roc_exit, ramps):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_interp['timestamp'], y=df_interp['roc'], mode='lines+markers', name='ROC', line=dict(color='green')))
    fig.add_hline(y=roc_enter, line_dash='dash', line_color='red', annotation_text='ROC Enter', annotation_position='top left')
    fig.add_hline(y=-roc_enter, line_dash='dash', line_color='red')
    fig.add_hline(y=roc_exit, line_dash='dot', line_color='orange', annotation_text='ROC Exit', annotation_position='bottom left')
    fig.add_hline(y=-roc_exit, line_dash='dot', line_color='orange')
    # Highlight ramps
    for _, ramp in ramps.iterrows():
        fig.add_vrect(x0=ramp['start_time'], x1=ramp['end_time'], fillcolor='pink', opacity=0.2, line_width=0)
    fig.update_layout(title='Rate of Change (ROC)', xaxis_title='Timestamp', yaxis_title='kW/min')
    return fig

def plot_forecast_fan(df_interp, forecasts, md_target):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_interp['timestamp'], y=df_interp['kw'], mode='lines', name='Actual', line=dict(color='blue')))
    last_time = df_interp['timestamp'].iloc[-1]
    for _, row in forecasts.iterrows():
        future_time = last_time + timedelta(minutes=row['horizon_min'])
        fig.add_trace(go.Scatter(x=[last_time, future_time], y=[df_interp['kw'].iloc[-1], row['pred_kw']], mode='lines', name=f"{row['horizon_min']} min forecast", line=dict(width=1)))
    fig.add_hline(y=md_target, line_dash='dash', line_color='red', annotation_text='MD Target', annotation_position='top right')
    fig.update_layout(title='Forecast Fan', xaxis_title='Timestamp', yaxis_title='kW')
    return fig

def plot_headroom_gauge(current_headroom, md_margin):
    color = 'green' if current_headroom > md_margin else ('orange' if current_headroom > 0 else 'red')
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = current_headroom,
        title = {'text': "Current Headroom (kW)"},
        gauge = {'axis': {'range': [None, max(1, current_headroom+md_margin)]},
                 'bar': {'color': color},
                 'steps': [
                     {'range': [0, md_margin], 'color': 'orange'},
                     {'range': [md_margin, max(1, current_headroom+md_margin)], 'color': 'green'}],
                 'threshold': {'line': {'color': 'red', 'width': 4}, 'thickness': 0.75, 'value': 0}}
    ))
    return fig

# --- Streamlit App ---
st.set_page_config(page_title="Load Forecasting MVP", layout="wide")
st.title("Load Forecasting MVP")

with st.sidebar:
    st.header("Controls")
    freq_min = st.slider("Interpolation Frequency (min)", 5, 30, 30, step=5)
    md_target = st.number_input("MD Target (kW)", min_value=0.0, value=1000.0)
    md_margin = st.number_input("MD Margin (kW)", min_value=0.0, value=50.0)
    roc_enter = st.number_input("ROC Enter Threshold (kW/min)", min_value=0.0, value=10.0)
    roc_exit = st.number_input("ROC Exit Threshold (kW/min)", min_value=0.0, value=5.0)

uploaded_file = st.file_uploader("Upload .xlsx file (1st column: timestamps, 2nd column: kW values)", type=['xlsx'])

if uploaded_file:
    df_raw = load_and_validate(uploaded_file)
    if df_raw is not None:
        tabs = st.tabs(["Upload", "Interpolation + ROC", "Ramps", "Forecast", "MD Risk"])
        with tabs[0]:
            st.subheader("Raw Data Preview")
            st.dataframe(df_raw.head(20))
        with tabs[1]:
            st.subheader("Interpolation & ROC")
            df_interp = interpolate_and_roc(df_raw, freq_min)
            st.dataframe(df_interp.head(20))
            st.plotly_chart(plot_interpolation(df_raw, df_interp), use_container_width=True)
            st.plotly_chart(plot_roc(df_interp, roc_enter, roc_exit, pd.DataFrame()), use_container_width=True)
        with tabs[2]:
            st.subheader("Ramp Events")
            df_interp = interpolate_and_roc(df_raw, freq_min)
            ramps = detect_ramps(df_interp, roc_enter, roc_exit)
            st.dataframe(ramps)
            st.plotly_chart(plot_roc(df_interp, roc_enter, roc_exit, ramps), use_container_width=True)
        with tabs[3]:
            st.subheader("Forecasts")
            df_interp = interpolate_and_roc(df_raw, freq_min)
            baseline = seasonal_baseline(df_interp)
            horizons = [30, 60, 120, 180, 300]
            forecasts = blended_forecast(df_interp, baseline, horizons)
            forecasts = md_risk_table(forecasts, md_target, md_margin)
            st.dataframe(forecasts)
            st.plotly_chart(plot_forecast_fan(df_interp, forecasts, md_target), use_container_width=True)
        with tabs[4]:
            st.subheader("MD Risk")
            df_interp = interpolate_and_roc(df_raw, freq_min)
            baseline = seasonal_baseline(df_interp)
            horizons = [30, 60, 120, 180, 300]
            forecasts = blended_forecast(df_interp, baseline, horizons)
            forecasts = md_risk_table(forecasts, md_target, md_margin)
            current_headroom = forecasts['headroom'].iloc[0]
            st.plotly_chart(plot_headroom_gauge(current_headroom, md_margin), use_container_width=True)
            st.write(f"Current headroom to MD: {current_headroom:.2f} kW")
else:
    st.info("Please upload a valid .xlsx file to begin.")
