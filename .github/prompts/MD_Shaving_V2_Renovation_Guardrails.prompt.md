# MD Shaving V2 Renovation Instructions
## Surgical Function Replacement Guidelines

### 🎯 **RENOVATION SCOPE**
**REPLACE ONLY**: Core battery simulation and calculation functions  
**PRESERVE**: All UI, visualization, data processing, and interface infrastructure

---

## 🚨 **CRITICAL GUARDRAILS**

### **1. FUNCTION SIGNATURE PRESERVATION**
```python
# ✅ CORRECT: Keep exact same inputs/outputs
def _simulate_battery_operation_v2(df, power_col, monthly_targets, battery_power_kw, 
                                 battery_energy_kwh, initial_soc_percent, 
                                 selected_tariff, holidays, soc_threshold):
    # REPLACE: Internal logic only
    return results_dict  # SAME structure as original

# ❌ WRONG: Changing function signature
def _simulate_battery_operation_v2(df, power_col, new_param):  # DON'T DO THIS
```

### **2. RETURN VALUE STRUCTURE PRESERVATION**
```python
# ✅ MAINTAIN: Exact same dictionary keys and data types
results = {
    'peak_reduction_kw': float,
    'success_rate_percent': float,
    'total_discharge_kwh': float,
    'avg_soc_percent': float,
    'conservation_periods': int,
    'conservation_rate_percent': float,
    # ... all existing keys must remain
}
```

### **3. IMPORT STATEMENT PROTECTION**
```python
# ✅ DO NOT MODIFY: Any import statements
from datetime import datetime, timedelta
from md_shaving_solution import (...)  # KEEP UNCHANGED
from tariffs.peak_logic import (...)   # KEEP UNCHANGED

# ❌ DO NOT ADD: New imports unless absolutely necessary
```

---

## 🔧 **RENOVATION TARGETS**

### **Phase 1: Core Shaving Logic Replacement**
```python
# FUNCTIONS TO REPLACE (copy from md_shaving_solution_v2_copy.py):
_simulate_battery_operation_v2()           # Lines ~3300-3500
_get_soc_aware_discharge_strategy()        # Strategy functions
_get_tariff_aware_discharge_strategy()     # Strategy functions
_get_strategy_aware_discharge()            # Strategy dispatcher
```

### **Phase 2: Supporting Calculation Replacement**
```python
# FUNCTIONS TO REPLACE:
_generate_monthly_summary_table()          # Lines ~320-400
_render_battery_quantity_recommendation()  # Quantity calculations
_calculate_success_rate_from_shaving_status() # Lines ~4344-4450
```

---

## 🛡️ **MANDATORY BACKUP PROTOCOL**

### **Before Each Function Replacement:**
```python
# 1. CREATE BACKUP WITH TIMESTAMP
_function_name_backup_YYYYMMDD = original_function

# 2. COMMENT OUT ORIGINAL (DON'T DELETE)
# def original_function(...):
#     # Original implementation preserved
#     pass

# 3. INSERT REPLACEMENT WITH CLEAR MARKER
def original_function(...):
    """
    RENOVATED: [DATE] - Replaced with copy file logic
    Original backed up as: _function_name_backup_YYYYMMDD
    """
    # New implementation from copy file
```

---

## 🚫 **ABSOLUTE DON'Ts**

### **DO NOT MODIFY:**
- ✋ Any UI rendering functions (`render_*`, `_display_*`)
- ✋ Data upload/validation functions (`read_uploaded_file`, `_process_dataframe`)
- ✋ Visualization functions (`_create_*_chart`, `_render_*_timeline`)
- ✋ Database functions (`load_vendor_battery_database`)
- ✋ Forecasting functions (`roc_forecast*`, `generate_forecast_*`)
- ✋ Session state variable names
- ✋ Function parameter names or orders

### **DO NOT DELETE:**
- ✋ Any working function (backup instead)
- ✋ Any import statements
- ✋ Any class definitions
- ✋ Documentation strings (replace content, keep structure)

---

## ✅ **VALIDATION CHECKLIST**

### **After Each Function Replacement:**
- [ ] Function signature identical to original
- [ ] Return dictionary has same keys and data types
- [ ] No new import statements required
- [ ] Original function backed up with timestamp
- [ ] Clear renovation comment added
- [ ] Test with same input data produces reasonable output

### **Before Moving to Next Function:**
- [ ] UI still loads without errors
- [ ] Charts still render
- [ ] No broken function calls in console
- [ ] Session state variables still accessible

---

## 📋 **REPLACEMENT SEQUENCE**

### **Recommended Order:**
1. **Strategy Functions First** (smallest impact)
   - `_get_soc_aware_discharge_strategy()`
   - `_get_tariff_aware_discharge_strategy()`
   - `_get_strategy_aware_discharge()`

2. **Core Simulation Engine** (main target)
   - `_simulate_battery_operation_v2()`

3. **Supporting Calculations** (dependent functions)
   - `_generate_monthly_summary_table()`
   - `_render_battery_quantity_recommendation()`
   - `_calculate_success_rate_from_shaving_status()`

### **Test After Each Step:**
```python
# Minimal test to ensure no breakage:
# 1. Load sample data
# 2. Run simulation
# 3. Check results dictionary structure
# 4. Verify UI still renders
```

---

## 🚨 **EMERGENCY ROLLBACK**

### **If Something Breaks:**
```python
# 1. IMMEDIATE RESTORATION
original_function = _function_name_backup_YYYYMMDD

# 2. IDENTIFY ISSUE
# Check console errors, compare function signatures

# 3. DOCUMENT PROBLEM
# Add comment about what broke and why

# 4. TRY AGAIN WITH SMALLER CHANGES
# Replace internal logic in smaller chunks
```

---

**Remember: We're replacing windows, not demolishing the building. Every change should be reversible and isolated.**