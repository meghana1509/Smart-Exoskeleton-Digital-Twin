import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import torch
import torch.nn as nn
import os


# 1. Define the LSTM Architecture (Must match your trained model)
class GaitLSTM(nn.Module):
    def __init__(self, input_size=3, hidden_size=50, num_layers=2, output_size=3):
        super(GaitLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # Your model saved this as 'linear', not 'fc'
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.linear(out[:, -1, :])
        return out

class GaitPredictorNode(Node):
    def __init__(self):
        super().__init__('gait_predictor_node')
        
        # 2. Setup Publisher & Subscriber
        self.publisher_ = self.create_publisher(Float32MultiArray, 'gait_predictions', 10)
        self.subscription = self.create_subscription(
            Float32MultiArray,
            'exoskeleton_sensors',
            self.listener_callback,
            10)
        
        # 3. Load the Model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = GaitLSTM().to(self.device)
        
        # Absolute path to avoid "file not found" errors
        model_path = os.path.expanduser('~/ros2_ws/src/exoskeleton_ai/models/gait_lstm_model.pth')
        
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            self.get_logger().info(f'✅ AI Brain Loaded Successfully from: {model_path}')
        else:
            self.get_logger().error(f'❌ Model file NOT found at: {model_path}')

    def listener_callback(self, msg):
        # Convert incoming sensor data to Tensor
        input_tensor = torch.tensor([msg.data], dtype=torch.float32).to(self.device)
        input_tensor = input_tensor.unsqueeze(0) # Add batch dimension

        with torch.no_grad():
            prediction = self.model(input_tensor)
        
        # Publish the prediction
        output_msg = Float32MultiArray()
        output_msg.data = prediction.cpu().numpy().flatten().tolist()
        self.publisher_.publish(output_msg)
        self.get_logger().info('🤖 Prediction Sent!')

# 4. The CRITICAL Main Function (Don't skip this!)
def main(args=None):
    rclpy.init(args=args)
    node = GaitPredictorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
