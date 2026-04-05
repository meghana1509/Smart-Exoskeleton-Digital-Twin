import torch
import torch.nn as nn
import pandas as pd
import numpy as np

# 1. Load the cleaned data from Phase 2
data = pd.read_csv('cleaned_gait_data.csv')
# We exclude 'time' for the AI training, focusing only on the angles
features = data[['hip_flexion_r', 'knee_angle_r', 'ankle_angle_r']].values

# 2. Define the LSTM Architecture (The "Brain" Structure)
class GaitLSTM(nn.Module):
    def __init__(self, input_size=3, hidden_size=50, num_layers=2):
        super(GaitLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, input_size) # Output 3 joint angles
        
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.linear(out[:, -1, :])

# 3. Initialize Model
model = GaitLSTM()
criterion = nn.MSELoss() # Measures prediction error
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print("🧠 AI Brain initialized. !")
# 4. Prepare Sequences (Windowing)
def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 10  # AI looks at 10 previous frames to predict the next 1
X, y = create_sequences(features, seq_length)

# Convert to PyTorch Tensors
X_train = torch.from_numpy(X).float()
y_train = torch.from_numpy(y).float()

# 5. The Training Loop (The "Studying" Phase)
epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    # Forward pass: AI makes a "guess"
    y_pred = model(X_train)
    
    # Calculate error (Loss)
    loss = criterion(y_pred, y_train)
    
    # Backward pass: AI learns from its mistakes
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}')

# 6. Save the Brain
torch.save(model.state_dict(), 'gait_lstm_model.pth')
print("✅ Training Complete! 'gait_lstm_model.pth' is saved and ready for the Digital Twin.")