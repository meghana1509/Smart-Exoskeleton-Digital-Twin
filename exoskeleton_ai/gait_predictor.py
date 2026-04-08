import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import torch
import torch.nn as nn
import numpy as np

class GaitLSTM(nn.Module):
    def __init__(self, input_size=3, hidden_size=50, num_layers=2, output_size=3):
        super(GaitLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.linear(out[:, -1, :])

class GaitPredictorNode(Node):
    def __init__(self):
        super().__init__('gait_predictor_node')
        
        # Load the Model
        self.device = torch.device('cpu')
        self.model = GaitLSTM()
        model_path = '/home/meghana/ros2_ws/src/exoskeleton_ai/models/gait_lstm_model.pth'
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.get_logger().info('✅ AI Brain Connected to Simulation')

        # Subscriber: Listen to the current joint angles
        self.subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.listener_callback,
            10)
        # This creates the topic that PlotJuggler will listen to
        self.prediction_pub = self.create_publisher(JointState, '/ai_prediction', 10)
        # Publisher: Send predicted movement
        self.publisher_ = self.create_publisher(JointState, '/joint_group_position_controller/commands', 10)
    def listener_callback(self, msg):
        # Extract the angles (assuming hip and knee are the first two)
        # We need 3 inputs for your model, so we'll use hip, knee, and a 0 padding
        current_angles = np.array(msg.position[:2])
        input_data = np.append(current_angles, [0.0]) # Making it size 3
         # AI Prediction
        input_tensor = torch.FloatTensor(input_data).view(1, 1, 3).to(self.device)
        with torch.no_grad():
            prediction = self.model(input_tensor).numpy()[0]

        self.get_logger().info(f'🤖 AI Predicted Next Angles: {prediction[:2]}')
        # Calculate simple Mean Squared Error (MSE) for this frame
        target_angles = np.array([0.3, 0.25]) # REPLACE THESE with values from your training dataset
        error = np.mean((prediction[:2] - target_angles)**2)

        self.get_logger().info(f'📊 Prediction Error (MSE): {error:.5f}')
        # Create the message
       # --- COMPLETE FROM HERE ---
        # Create the message for PlotJuggler
        pred_msg = JointState()
        pred_msg.header.stamp = self.get_clock().now().to_msg()
        pred_msg.name = ['hip_pred', 'knee_pred']
        # We use standard float() to ensure ROS can read the numpy values
        pred_msg.position = [float(prediction[0]), float(prediction[1])]
        # --- AFTER (Calibration) ---
# We adjust the AI output to match the ROS2 simulation range
        hip_calibrated = (float(prediction[0]) * 1.6) - 1.4  # Adjust these numbers!
        knee_calibrated = (float(prediction[1]) * 2.2) - 0.15      # Adjust these numbers!

        pred_msg.position = [hip_calibrated, knee_calibrated]
# Broadcast the prediction to the /ai_prediction topic
        self.prediction_pub.publish(pred_msg)
def main(args=None):
    rclpy.init(args=args)
    node = GaitPredictorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
