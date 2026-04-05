import pandas as pd
import matplotlib.pyplot as plt

# Load the results we just generated
df = pd.read_csv('torque_gap_results.csv')

plt.figure(figsize=(10, 6))
plt.plot(df['time'], df['required'], label='Healthy Torque (Goal)', color='black', linestyle='--')
plt.fill_between(df['time'], df['patient_contribution'], df['required'], color='orange', alpha=0.3, label='Exoskeleton Assistance (The Gap)')
plt.plot(df['time'], df['patient_contribution'], label='Patient Strength (30%)', color='red')

plt.title('Phase 4: Torque Gap Analysis')
plt.xlabel('Time (s)')
plt.ylabel('Torque (Nm)')
plt.legend()
plt.grid(True)
plt.show()