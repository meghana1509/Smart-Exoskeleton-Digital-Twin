import pandas as pd
import matplotlib.pyplot as plt

# Using the raw string path you confirmed
file_path = r'C:\OpenSim 4.5\Resources\Models\Gait2354_Simbody\subject01_walk1_grf.mot'

# We will try a smaller skip to ensure we catch the column names
# Most .mot files have the column names right after the 'endheader' line
data = pd.read_csv(file_path, sep='\t', skiprows=6)

# --- DATA EXPLORER ---
# This prints the first 5 rows and column names so we can see the 'Real' names
print("--- File successfully loaded! ---")
print("Columns found in your file:", data.columns.tolist())
print("\nFirst few rows of data:")
print(data.head())

# --- DYNAMIC PLOTTING ---
# Instead of guessing the name 'time', we use the first column (usually time)
# and the second column (usually force)
try:
    plt.figure(figsize=(10, 5))
    plt.plot(data.iloc[:, 0], data.iloc[:, 1], color='blue', label='Ground Force')
    plt.title('Digital Twin: Ground Reaction Force')
    plt.xlabel('Time')
    plt.ylabel('Force Value')
    plt.legend()
    plt.grid(True)
    plt.show()
except Exception as e:
    print(f"Plotting error: {e}")