import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import torch
import torch.nn as nn
import numpy as np

# 1. Define the same LSTM Architecture you trained
class GaitLSTM(nn.Module):
    def __init__(self, input_size=6, hidden_size=64, num_layers=2, output_size=3):
        super(GaitLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

class GaitPredictorNode(Node):
    def __init__(self):
        super().__init__('gait_predictor_node')
        
        # 2. Setup Subscriber (Listening for sensor data)
        self.subscription = self.create_subscription(
            Float32MultiArray,
            'exoskeleton_sensors',
            self.listener_callback,
            10)
        
        # 3. Setup Publisher (Sending predictions to Digital Twin)
        self.publisher_ = self.create_publisher(Float32MultiArray, 'gait_predictions', 10)

        # 4. Load the "Brain"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = GaitLSTM().to(self.device)
        
        # Point to the file we pushed to GitHub earlier
        model_path = 'src/exoskeleton_ai/models/gait_lstm_model.pth'
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            self.get_logger().info(f'✅ Successfully loaded AI Brain from {model_path}')
        except Exception as e:
            self.get_logger().error(f'❌ Failed to load model: {str(e)}')

    def listener_callback(self, msg):
        # 5. Process incoming data
        input_data = torch.tensor([msg.data], dtype=torch.float32).to(self.device)
        input_data = input_data.unsqueeze(0) # Add batch dimension

        # 6. Make Prediction
        with torch.no_state():
            prediction = self.model(input_data)
        
        # 7. Publish Result
        output_msg = Float32MultiArray()
        output_msg.data = prediction.cpu().numpy().flatten().tolist()
        self.publisher_.publish(output_msg)
        self.get_logger().info('🤖 Prediction published to Digital Twin!')

def main(args=None):
    rclpy.init(args=args)
    node = GaitPredictorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
