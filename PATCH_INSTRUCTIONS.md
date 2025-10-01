"""
Simple patch for Series boolean ambiguity debugging.
Add this code around the problematic boolean condition in _simulate_battery_operation_v2.

FIND THIS CODE:
    if excess > 0 and is_md_window(current_timestamp, holidays):

REPLACE WITH THIS CODE:
"""

# Around line 9803 in the original function, replace the boolean condition:

# OLD CODE:
# should_discharge_condition = has_excess and in_md_window

# NEW CODE WITH ERROR TRACKING:
st.info(f"🔍 CRITICAL DEBUG i={i}: has_excess={has_excess} (type: {type(has_excess)}), in_md_window={in_md_window} (type: {type(in_md_window)})")

# Test the boolean operation that might fail
try:
    should_discharge_condition = has_excess and in_md_window
    st.info(f"✅ Boolean operation successful: result={should_discharge_condition} (type: {type(should_discharge_condition)})")
except Exception as bool_error:
    st.error(f"❌ SERIES BOOLEAN AMBIGUITY ERROR CAUGHT at iteration {i}!")
    st.error(f"Boolean error: {bool_error}")
    st.error(f"has_excess: {has_excess} (type: {type(has_excess)})")  
    st.error(f"in_md_window: {in_md_window} (type: {type(in_md_window)})")
    if hasattr(has_excess, 'item'):
        st.error(f"has_excess.item(): {has_excess.item()}")
    if hasattr(in_md_window, 'item'):
        st.error(f"in_md_window.item(): {in_md_window.item()}")
    raise

# This will help us identify exactly what types are causing the Series boolean ambiguity error