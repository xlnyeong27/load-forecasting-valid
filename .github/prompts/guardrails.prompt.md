---
mode: agent
---

Copilot: Repository-Wide Instructions (MD Shaving App)

Purpose. Keep the application’s logical flow intact, make additive, reviewable changes, and avoid regressions when inserting or modifying code between stages (e.g., between Forecast and Strategy).

0) Canonical Pipeline (Do Not Reorder)

Order is strictly:

Data Upload → 2) Data Validation & Summary → 3) Peak Events Analysis & Target Calculation → 4) Battery Sizing & Financial Analysis → 5) Forecasting (Toggle ON/OFF) → 6) Shaving Strategy Selection → 7) Battery Simulation Results

Do not reorder, merge, or split these stages without an explicit instruction that says “reorder allowed”.

When adding new logic, place it only in the Pre-Strategy Hook (or in a new, clearly named hook subsection) and wire it using anchors (see §3).

1) Non-Negotiable Invariants

Public APIs & state keys are stable. Do not rename or remove public functions, Streamlit routes, callbacks, or state keys.

Additive by default. Prefer adding small, scoped functions over editing large ones. Avoid wide refactors unless asked.

No silent schema drift. If any dataframe columns or units change, update the schema mapping + validators in the same patch (§4).

Idempotent insertions. Re-running the same change should be a no-op. Use anchors and existence checks.

Reproducible diff. Always output a unified diff across files, listing added/modified lines.

2) Canonical Identifiers & Variable Linking

Use canonical names and link to existing variables—do not invent near-duplicates. 

{
  "dataframes": ["df","df_filtered","filtered_df","df_processed","df_sim","forecast_df","original_forecast_df","validated_forecast_df","df_long","df_long_with_bands","downsampled_df","monthly_summary_df","df_monthly","validation_df","export_df"],
  "columns": ["power_col","timestamp_col","net_demand","holidays"],
  "forecast": ["forecast_series","forecast_series_dict","forecast_timestamps","forecast_ts","forecast_ts_series","horizon","horizons","available_horizons","power_forecast","forecast_available","forecast_data_available","valid_forecasts","original_points","aligned_forecast","forecast_lookup","p10","p50","p90"],
  "battery_soc": ["battery_power_kw","battery_energy_kwh","soc","soc_percent","current_soc_percent","current_soc_kwh","min_soc_energy","soc_limiter","soc_factor","low_soc_events","battery_spec","active_battery_spec","battery_config","battery_db","battery_capacity","battery_capacity_kwh","usable_capacity","remaining_capacity","battery_params","battery_list","selected_battery","selected_battery_data","selected_battery_label","battery_id","battery_model","battery_company"],
  "tariff_md": ["selected_tariff","tariff_type","tariff_name","tariff_config","tariff_type_field","tariff_description","is_md_window","next_md_start","hours_until_md","time_until_md","tou_windows","recharge_windows","charge_allowed_windows","windows","window_start","window_end","md_excess_col","md_excess_values","max_md_excess","max_monthly_md_excess","actual_md_shaved","md_shaved_kw","md_shaving_percentage","md_shaving_coverage","md_rate_rm_per_kw","total_md_rate","total_md_cost","monthly_targets","monthly_tou_peaks","peak_events","peak_data","peak_classifications"],
  "strategy": ["selected_strategy","strategy_info","strategy_descriptions","selected_filter","selected_capacity","base_interval"],
  "constants": ["MIN_SOC_THRESHOLD","EXCESS_DISCHARGE_RATE"],
  "session_state_keys": ["v2_monthly_targets","df_processed","v2_power_col","roc_long_format","v2_simulation_results","v2_timestamp_col","v2_reference_peaks","v2_tariff_type","v2_target_description","roc_forecast_series","roc_validation_metrics","roc_actual_series","residual_quantiles","shaving_forecast_data","shaving_historical_data","selected_shaving_strategy","strategy_config"]
}

Variable naming rule:
When introducing new logic, always check whether the variable conceptually maps to an existing feature or data source.

If the new code interacts with or extends an existing feature (e.g., reads user-uploaded historical data, accesses battery specs, or continues an existing processing chain), it must reuse the canonical variable names defined in the protected names dictionary (e.g., df_processed, forecast_df, battery_spec).

If the new code implements a genuinely new feature or metric (e.g., calculating a weightage from uncertainty bands, a new KPI, or diagnostic statistics) and does not correspond to an existing variable, it may create new, clearly named variables, provided they do not shadow or overload existing ones.
Always prefer semantic clarity: reuse only when semantically linked; create new when logically independent.

3) Anchors (Stable Insertion Points)

Before creating insertions, create anchors in this format:

==ANCHOR_NAME==
# Anchor description (e.g., "Before Strategy Hook")
# Do not remove or reorder anchors without explicit permission.
# code ...
==ANCHOR_NAME_END==

Rules:

New imports for the hook must be localized (lazy or function-scoped) when possible to protect cold-start time.

Do not write outside anchored blocks unless explicitly instructed.

4) Allowed vs Prohibited Operations

Allowed (preferred):

Add a small function/class in a new module under hooks/ or an existing *_service.py.

Thread existing values through the pipeline by referencing canonical variables.

Insert a middleware that reads forecast.* and writes a minimal, documented field into state (e.g., strategy_inputs.uncertainty_band_pct).

Prohibited (unless explicitly requested):

Renaming/removing public functions, Streamlit page routes, or state keys.

Reordering the pipeline stages.

Introducing new, near-duplicate variables (e.g., soc_percent, SOCpercentage) when SOC_pct exists.

Changing default behavior without a feature flag (see §6).

6) Feature Flags & Backward Compatibility

Any new behavior must be gated by a feature flag with a safe default (OFF).
Example: features.pre_strategy_hook = True/False.

When OFF, runtime behavior must match pre-change behavior bit-for-bit.

Create flags by default unless user requests otherwise. 


7) Uncertainty & SOC Propagation Rules

When adding logic that consumes or produces uncertainty/SOC signals:

Consume from forecast.q10/q50/q90; compute uncertainty band as (q90 - q10) / max(q50, ε).

Do not rename these; store any derived value under strategy_inputs.*.

Do not mutate forecast.*; treat it as read-only.

Maintain SOC floors via soc_policy without hard-coding constants; read them from config/policies.py.

8) Error Handling, Interlocks, and Safe Mode

If inputs are missing/stale (e.g., telemetry gap > X min), do not proceed—call safe_mode() and log the reason.

Rate-limit setpoint changes using ramp_kW_per_min.

Never emit a setpoint that violates P_rated_kW or inverter apparent power limits (P^2 + Q^2 ≤ S_rated^2).


10) When In Doubt

Prefer no change over speculative edits.

Ask for the specific file and function to touch if scope is unclear.

Never infer a new variable when a canonical one is available—link, don’t duplicate.

End of Instructions