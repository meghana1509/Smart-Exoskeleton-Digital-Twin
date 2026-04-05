import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 1. Path to the IK file you just found
file_path = r'C:\OpenSim 4.5\Resources\Models\Gait2354_Simbody\OutputReference\subject01_walk1_ik.mot'

# 2. Smart Header Finder
# This looks for the line after 'endheader' automatically
with open(file_path, 'r') as f:
    lines = f.readlines()
    skip = next(i for i, line in enumerate(lines) if 'endheader' in line) + 1

# 3. Load the data
data = pd.read_csv(file_path, sep='\t', skiprows=skip)
print("--- File Loaded Successfully! ---")
print("Columns available:", data.columns.tolist())

# 4. Feature Selection (Tracking the Right Leg for the Exoskeleton)
features = ['time', 'hip_flexion_r', 'knee_angle_r', 'ankle_angle_r']

# Check if these exact names exist
if all(f in data.columns for f in features):
    df_filtered = data[features].copy()
    
    # 5. Normalization (Scaling 0 to 1 for the LSTM AI)
    scaler = MinMaxScaler()
    cols_to_scale = ['hip_flexion_r', 'knee_angle_r', 'ankle_angle_r']
    df_filtered[cols_to_scale] = scaler.fit_transform(df_filtered[cols_to_scale])
    
    # 6. Save the Cleaned Dataset
    df_filtered.to_csv('cleaned_gait_data.csv', index=False)
    print("✅ SUCCESS: 'cleaned_gait_data.csv' is ready for Phase 3!")
    print(df_filtered.head())
else:
    print("❌ Error: Column names don't match. Check the printout above for correct names.")