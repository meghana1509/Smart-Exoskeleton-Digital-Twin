import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the Inverse Dynamics file (Required Force)
# Look for a file ending in _id.sto or _inverse_dynamics.sto
id_path = r'C:\OpenSim 4.5\Resources\Models\Gait2354_Simbody\OutputReference\ResultsInverseDynamics\inverse_dynamics.sto'

# Smart Loader for OpenSim files
with open(id_path, 'r') as f:
    lines = f.readlines()
    skip = next(i for i, line in enumerate(lines) if 'endheader' in line) + 1

id_data = pd.read_csv(id_path, sep='\t', skiprows=skip)

# 2. Extract the Knee Torque (Right Leg)
knee_torque_required = id_data['knee_angle_r_moment']

# 3. Simulate a Patient with 30% muscle strength
patient_strength = knee_torque_required * 0.3
torque_gap = knee_torque_required - patient_strength

# 4. Save the Gap data for the Dashboard
gap_df = pd.DataFrame({
    'time': id_data['time'],
    'required': knee_torque_required,
    'patient_contribution': patient_strength,
    'exoskeleton_assist': torque_gap
})
gap_df.to_csv('torque_gap_results.csv', index=False)

print("✅ Phase 4 Complete: Torque Gap calculated!")