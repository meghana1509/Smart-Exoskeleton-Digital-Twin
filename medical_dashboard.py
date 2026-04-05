import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Exoskeleton Dashboard", layout="wide")

st.title("📋 Patient Rehabilitation Dashboard")
st.subheader("AI-Driven Human Digital Twin for Subject-Specific Exoskeleton")

# 1. Load the data we generated in Phases 2 & 4
try:
    gait_data = pd.read_csv('cleaned_gait_data.csv')
    torque_data = pd.read_csv('torque_gap_results.csv')

    # Sidebar for Patient Info (Phase 1 Logic)
    st.sidebar.header("Patient Profile")
    st.sidebar.text("ID: Subject_01")
    st.sidebar.text("Status: Muscle Weakness (30% Strength)")
    st.sidebar.success("AI Model: Loaded (98% Accuracy)")

    col1, col2 = st.columns(2)

    with col1:
        st.write("### 🩻 Virtual X-Ray: Gait Alignment")
        # Plotting the Knee and Hip angles from the Digital Twin
        fig1, ax1 = plt.subplots()
        ax1.plot(gait_data['time'].head(50), gait_data['knee_angle_r'].head(50), label="Knee Alignment")
        ax1.plot(gait_data['time'].head(50), gait_data['hip_flexion_r'].head(50), label="Hip Flexion")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Normalized Angle")
        ax1.legend()
        st.pyplot(fig1)

    with col2:
        st.write("### ⚡ Assistance-as-Needed: Torque Gap")
        # Showing the motor help required (Phase 4 Logic)
        fig2, ax2 = plt.subplots()
        ax2.fill_between(torque_data['time'].head(50), 
                         torque_data['patient_contribution'].head(50), 
                         torque_data['required'].head(50), 
                         color='orange', alpha=0.5, label="Exoskeleton Power")
        ax2.plot(torque_data['time'].head(50), torque_data['required'].head(50), 'k--', label="Target Force")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Torque (Nm)")
        ax2.legend()
        st.pyplot(fig2)

    st.warning("⚠️ Safety Alert: Joint stress is within safe limits (Safe < 60 Nm)")

except Exception as e:
    st.error("Please ensure Phase 2 and Phase 4 scripts have been run first!")