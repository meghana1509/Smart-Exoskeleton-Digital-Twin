import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. Load the Brain and the Data
data = pd.read_csv('cleaned_gait_data.csv')
features = data[['hip_flexion_r', 'knee_angle_r', 'ankle_angle_r']].values
model_state = torch.load('gait_lstm_model.pth')

# Re-initialize the model structure to load the weights
class GaitLSTM(torch.nn.Module):
    def __init__(self, input_size=3, hidden_size=50, num_layers=2):
        super(GaitLSTM, self).__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = torch.nn.Linear(hidden_size, input_size)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.linear(out[:, -1, :])

model = GaitLSTM()
model.load_state_dict(model_state)
model.eval()

# 2. Pick a window of 10 frames to test
test_seq = torch.from_numpy(features[50:60]).float().unsqueeze(0)
actual_next_step = features[60]

# 3. AI Prediction
with torch.no_grad():
    prediction = model(test_seq).numpy()[0]

# 4. Visualize the accuracy
labels = ['Hip', 'Knee', 'Ankle']
x = np.arange(len(labels))

plt.bar(x - 0.2, actual_next_step, 0.4, label='Real (OpenSim)', color='blue')
plt.bar(x + 0.2, prediction, 0.4, label='AI Prediction', color='orange')
plt.xticks(x, labels)
plt.title("AI Prediction vs. Gold Standard Data")
plt.legend()
plt.show()

print(f"Prediction Error: {np.mean(np.abs(prediction - actual_next_step)):.4f}")