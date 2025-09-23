# Main Energy Analytics Platform
import streamlit as st
import warnings
import sys
import os

warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="Energy Analytics Platform",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="auto"
)

def run_load_forecasting():
    """Run the load forecasting module by executing the file content."""
    try:
        # Read and execute the load forecasting file
        with open('load_forecasting.py', 'r') as f:
            load_forecasting_code = f.read()
        
        # Create a new namespace for execution to avoid conflicts
        namespace = {
            '__name__': '__main__',
            'st': st,
            'warnings': warnings
        }
        
        # Execute the load forecasting code
        exec(load_forecasting_code, namespace)
        
    except FileNotFoundError:
        st.error("❌ load_forecasting.py file not found")
        return False
    except Exception as e:
        st.error(f"❌ Error running Load Forecasting: {str(e)}")
        with st.expander("🔧 Debug Information"):
            st.code(f"Error details: {str(e)}")
        return False
    
    return True

# Import MD Shaving V2 with error handling
md_shaving_v2_available = True
md_shaving_v2_error = None

try:
    from md_shaving_solution_v2 import render_md_shaving_v2
except ImportError as e:
    md_shaving_v2_available = False
    md_shaving_v2_error = str(e)

# Main app header
st.title("🔋 Energy Analytics Platform")
st.markdown("""
**Comprehensive energy analysis toolkit** with advanced forecasting and optimization capabilities.
""")

# Status indicators
col1, col2, col3 = st.columns(3)

with col1:
    if os.path.exists('load_forecasting.py'):
        st.success("📊 Load Forecasting: Ready")
    else:
        st.error("📊 Load Forecasting: File Missing")

with col2:
    if md_shaving_v2_available:
        st.success("🔋 MD Shaving V2: Ready")
    else:
        st.warning("🔋 MD Shaving V2: Dependencies Missing")

with col3:
    available_modules = []
    if os.path.exists('load_forecasting.py'):
        available_modules.append("Load Forecasting")
    if md_shaving_v2_available:
        available_modules.append("MD Shaving V2")
    
    st.info(f"✅ {len(available_modules)}/2 Modules Available")

# Create tabs
tab1, tab2 = st.tabs(["📊 Load Forecasting MVP", "🔋 MD Shaving Solution V2"])

# Tab 1: Load Forecasting
with tab1:
    st.header("📊 Load Forecasting MVP")
    
    if os.path.exists('load_forecasting.py'):
        run_load_forecasting()
    else:
        st.error("❌ Load Forecasting module not available: load_forecasting.py file not found")
        st.markdown("""
        **Load Forecasting Requirements:**
        - Ensure `load_forecasting.py` exists in the project directory
        - The file should contain the complete Load Forecasting MVP implementation
        """)

# Tab 2: MD Shaving Solution V2
with tab2:
    st.header("🔋 MD Shaving Solution V2")
    
    if md_shaving_v2_available:
        try:
            render_md_shaving_v2()
        except Exception as e:
            st.error(f"❌ Error running MD Shaving V2: {str(e)}")
            
            with st.expander("🔧 Debug Information"):
                st.code(f"Error details: {str(e)}")
                st.markdown("""
                **Possible Solutions:**
                - Check that required modules are available
                - Install missing dependencies
                - Verify database files are present
                """)
    else:
        st.error(f"❌ MD Shaving V2 module not available")
        
        if md_shaving_v2_error:
            st.code(f"Import error: {md_shaving_v2_error}")
        
        st.markdown("""
        **MD Shaving V2 Requirements:**
        - Ensure `md_shaving_solution_v2.py` exists in the project directory
        - Install missing dependencies (check import statements in the file)
        - Common missing modules: `tariffs`, `utils`, `battery_algorithms`
        
        **Quick Fix Options:**
        1. Comment out problematic imports in `md_shaving_solution_v2.py`
        2. Create minimal fallback modules for missing dependencies
        3. Use only the available functions from MD Shaving V2
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.8em; margin-top: 2em;'>
🔋 Energy Analytics Platform | Load Forecasting & MD Shaving Solutions
</div>
""", unsafe_allow_html=True)
