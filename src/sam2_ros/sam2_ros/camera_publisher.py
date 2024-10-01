import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class CameraPublisher(Node):
    def __init__(self):
        super().__init__('camera_publisher')
        self.publisher_ = self.create_publisher(Image, 'camera_frames', 10)
        self.timer = self.create_timer(0.05, self.timer_callback)  # Timer to control publishing rate
        self.bridge = CvBridge()  # Used to convert between ROS and OpenCV image formats
        self.cap = cv2.VideoCapture(1)  # Open the default camera

    def timer_callback(self):
        
        ret, frame = self.cap.read()
        if ret:
            # Convert the OpenCV image to a ROS Image message and publish
            self.publisher_.publish(self.bridge.cv2_to_imgmsg(frame, "bgr8"))
            
            # Display the image locally (optional)
            cv2.imshow('Camera', frame)
            cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)  # Initialize the ROS client library
    camera_publisher = CameraPublisher()  # Create an instance of the camera publisher node
    rclpy.spin(camera_publisher)  # Keep the node alive
    camera_publisher.destroy_node()
    rclpy.shutdown()  # Shutdown the ROS client library

if __name__ == '__main__':
    main()
