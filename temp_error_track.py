# Minimal error tracking patch for Series boolean debugging

def add_simple_error_tracking():
    """Simple wrapper to add error tracking around the problematic boolean condition"""
    
    # Original problematic code:
    # if excess > 0 and is_md_window(current_timestamp, holidays):
    
    # Replace with error-tracked version:
    try:
        # Test each part separately
        excess_test = excess > 0
        md_window_test = is_md_window(current_timestamp, holidays)
        
        # Log the types for debugging
        st.info(f"🔍 DEBUG: excess type: {type(excess)}, value: {excess}")
        st.info(f"🔍 DEBUG: excess_test type: {type(excess_test)}, value: {excess_test}")
        st.info(f"🔍 DEBUG: md_window_test type: {type(md_window_test)}, value: {md_window_test}")
        
        # This is the line that might fail
        combined_condition = excess_test and md_window_test
        
        st.info(f"🔍 DEBUG: combined_condition type: {type(combined_condition)}, value: {combined_condition}")
        
        return combined_condition
        
    except Exception as e:
        st.error(f"❌ SERIES BOOLEAN ERROR CAUGHT!")
        st.error(f"Error: {e}")
        st.error(f"excess type: {type(excess)}")
        st.error(f"excess value: {excess}")
        if hasattr(excess, 'item'):
            st.error(f"excess.item(): {excess.item()}")
        raise

# Minimal patch instructions:
# 1. Find line: if excess > 0 and is_md_window(current_timestamp, holidays):
# 2. Replace with:
#    excess_scalar = float(excess) if hasattr(excess, 'item') else excess
#    has_excess = excess_scalar > 0
#    in_md_window = is_md_window(current_timestamp, holidays)
#    st.info(f"🔍 DEBUG i={i}: excess_scalar={excess_scalar} ({type(excess_scalar)}), has_excess={has_excess} ({type(has_excess)}), in_md_window={in_md_window} ({type(in_md_window)})")
#    if has_excess and in_md_window: