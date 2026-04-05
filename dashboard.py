import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Digital Twin Rehab", layout="wide")

st.title("🦿 Virtual X-Ray: Human Digital Twin Monitor")
st.sidebar.header("Patient Data Settings")

# Use the path we just verified
file_path = r'C:\OpenSim 4.5\Resources\Models\Gait2354_Simbody\subject01_walk1_grf.mot'
data = pd.read_csv(file_path, sep='\t', skiprows=6)

# Sidebar - Safety Threshold
threshold = st.sidebar.slider("Safety Force Threshold (N)", 500, 1000, 800)

# Main Dashboard
col1, col2 = st.columns(2)

with col1:
    st.subheader("Real-Time Ground Reaction Force")
    fig, ax = plt.subplots()
    ax.plot(data['time'], data['ground_force_vy'], color='blue')
    ax.axhline(y=threshold, color='red', linestyle='--', label='Safety Limit')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Vertical Force (N)")
    st.pyplot(fig)

with col2:
    st.subheader("System Status")
    current_force = data['ground_force_vy'].max()
    if current_force > threshold:
        st.error(f"⚠️ ALERT: High Joint Loading Detected! ({current_force:.2f} N)")
        st.write("Exoskeleton: Increasing Support Torque...")
    else:
        st.success("✅ Gait Pattern: Stable & Safe")
        st.write("Exoskeleton: Assistance-as-Needed Mode Active")

st.dataframe(data.head(20))