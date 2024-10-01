import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class Sam2ExecutionNode(Node):
  def __init__(self):
      super().__init__('sam2_execution_node')
      self.publisher_ = self.create_publisher(String, 'sam2_command', 10)
      self.get_logger().info('Sam2 Execution Node has been started.')

  def send_command(self, command):
      msg = String()
      msg.data = command
      self.publisher_.publish(msg)
      self.get_logger().info(f'Sent command: {command}')

def main(args=None):
  rclpy.init(args=args)
  node = Sam2ExecutionNode()

  try:
      while rclpy.ok():
          action_id = input("Please enter actionID: ")
          if action_id == '1':
              node.send_command('run')
          else:
              print("Invalid actionID. Please enter '1'.")
  except KeyboardInterrupt:
      pass

  node.destroy_node()
  rclpy.shutdown()

if __name__ == '__main__':
  main()