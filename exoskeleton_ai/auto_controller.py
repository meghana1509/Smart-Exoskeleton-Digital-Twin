import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import math

class AutoController(Node):
    def __init__(self):
        super().__init__('auto_controller')
        self.subscription = self.create_subscription(JointState, '/ai_prediction', self.control_callback, 10)
        self.publisher = self.create_publisher(JointState, '/joint_states', 10)
        self.timer = self.create_timer(1.0, self.kickstart_callback)
        self.received_first_prediction = False
        self.last_position = [0.0, 0.0]
        self.alpha = 0.2
        self.get_logger().info('🚀 Auto-Pilot Ready. Waiting for AI...')

    def kickstart_callback(self):
        if not self.received_first_prediction:
            self.get_logger().info('⚡ Wiggling the leg to wake up the LSTM...')
            t = self.get_clock().now().to_msg().sec % 10
            wiggle = 0.2 * math.sin(t)
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = ['hip_joint', 'knee_joint']
            msg.position = [float(wiggle), float(wiggle)]
            self.publisher.publish(msg)

    def control_callback(self, msg):
        self.received_first_prediction = True
        smoothed_hip = (self.alpha * msg.position[0]) + ((1 - self.alpha) * self.last_position[0])
        smoothed_knee = (self.alpha * msg.position[1]) + ((1 - self.alpha) * self.last_position[1])
        self.last_position = [smoothed_hip, smoothed_knee]
        command_msg = JointState()
        command_msg.header.stamp = self.get_clock().now().to_msg()
        command_msg.name = ['hip_joint', 'knee_joint']
        command_msg.position = msg.position
        self.publisher.publish(command_msg)

def main(args=None):
    rclpy.init(args=args)
    node = AutoController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
