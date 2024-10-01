import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger  # ROS2 Service for triggering processing
from std_msgs.msg import String
from cv_bridge import CvBridge
import numpy as np
import cv2

import sys
import os

# Add the parent directory of segment_anything_2 to the Python path
# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.dirname(current_dir)
# sys.path.append(parent_dir)

from ..segment_anything_2.sam2.sam2_image_predictor import SAM2ImagePredictor
from ..segment_anything_2.sam2.build_sam import build_sam2
import torch
import os
import sys
from hydra.core.global_hydra import GlobalHydra
import hydra

# Initialize the device (CUDA/CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SAM2ServiceNode(Node):
    def __init__(self, use_offline_image=False, offline_image_path=None):
        super().__init__('sam2_service_node')

        # Add the parent directory of segment_anything_2 to the Python path
        # current_dir = os.path.dirname(os.path.abspath(__file__))
        # parent_dir = os.path.dirname(current_dir)
        # sys.path.append(parent_dir)  # Ensure this path is correct

        # # Check if 'segment_anything_2' is directly under 'parent_dir'
        # sys.path.append(os.path.join(parent_dir, 'segment_anything_2'))


        # Publisher to send the top 3 masks
        self.mask_publisher = self.create_publisher(String, 'sam2_masks', 10)

        # print(os.path.join(current_dir, 'segment_anything_2', 'sam2_configs'), "()()()()()()()()")
        # hydra.initialize_config_dir(config_dir='/home/longuzi/deploy_sam2_ws/src/sam2_ros/sam2_ros/segment_anything_2/sam2_configs')

        # Create a ROS2 service for triggering SAM2 processing
        self.srv = self.create_service(Trigger, 'run_sam2_model', self.run_sam2_model_callback)
        
        
        config_dir = os.path.join(current_dir, 'segment_anything_2', 'sam2_configs')
        GlobalHydra.instance().clear()
        print(f"Config directory: {config_dir}")
        hydra.initialize_config_dir(config_dir=config_dir)

        # GlobalHydra.instance().clear()
        # hydra.initialize_config_dir(config_dir=os.path.join(current_dir, 'segment_anything_2', 'sam2_configs'))
        # hydra.initialize_config_dir(config_dir='/home/longuzi/sam2_model/segment-anything-2/sam2_configs')


        # Initialize SAM2 Model
        # config_path = os.path.join(current_dir, 'segment_anything_2', 'sam2_configs', 'sam2_hiera_s.yaml')
        # config_path = './segment_anything_2/sam2_configs/sam2_hiera_s.yaml'
        sam2_checkpoint = os.path.join(current_dir, 'segment_anything_2', 'checkpoints', "sam2_hiera_small.pt")
        self.sam2 = build_sam2(config_file="sam2_hiera_s.yaml", ckpt_path=sam2_checkpoint, device=device, apply_postprocessing=False)

        # config_path = './segment_anything_2/sam2_configs/sam2_hiera_s.yaml'
        sam2_checkpoint = "/home/longuzi/deploy_sam2_ws/src/sam2_ros/sam2_ros/segment_anything_2/checkpoints/sam2_hiera_small.pt"
        # self.sam2 = build_sam2(config_file="sam2_hiera_s.yaml", ckpt_path=sam2_checkpoint, device=device, apply_postprocessing=False)
        self.predictor = SAM2ImagePredictor(self.sam2)

        self.bridge = CvBridge()
        self.frame = None  # Variable to store the latest frame from the camera or offline image
        self.click_points = []
        self.click_labels = []

        # Determine mode (camera vs offline image)
        self.use_offline_image = use_offline_image
        self.offline_image_path = offline_image_path

        # If we're using the camera, set up a camera subscriber
        if not self.use_offline_image:
            self.subscription = self.create_subscription(
                Image,
                'camera_frames',
                self.listener_callback,
                10)
        else:
            # Load the offline image if no camera is available
            self.load_offline_image()

    def load_offline_image(self):
        """Loads an offline image from a given path."""
        if self.offline_image_path and os.path.exists(self.offline_image_path):
            self.frame = cv2.imread(self.offline_image_path)
            if self.frame is not None:
                self.get_logger().info(f"Offline image loaded from {self.offline_image_path}.")
            else:
                self.get_logger().error(f"Failed to load image from {self.offline_image_path}.")
        else:
            self.get_logger().error(f"Offline image path {self.offline_image_path} does not exist.")

    def listener_callback(self, msg):
        """Callback for camera frame subscriber (if using camera)"""
        self.frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def run_sam2_model_callback(self, request, response):
        """Service callback to trigger SAM2 processing"""

        if self.frame is None:
            response.success = False
            response.message = 'No camera frame or offline image available.'
            return response

        # Show the frame to the user for point selection
        cv2.imshow("Select Points on Frame", self.frame)
        cv2.setMouseCallback("Select Points on Frame", self.click_event)
        key = cv2.waitKey(0)

        # If user presses 's', process the frame
        if key == ord('s'):
            self.process_frame_with_sam2(self.frame)
            response.success = True
            response.message = 'SAM2 model processed successfully.'
        else:
            response.success = False
            response.message = 'No action taken.'

        return response

    def click_event(self, event, x, y, flags, params):
        """Callback for mouse clicks on the frame"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Add clicked points and labels (e.g., 1 for foreground)
            self.click_points.append([x, y])
            self.click_labels.append(1)
            self.get_logger().info(f"Point selected at: ({x}, {y})")

    def process_frame_with_sam2(self, frame):
        """Run SAM2 model with the selected points and publish top 3 masks"""
        input_image = np.array(frame)

        # Set the image for the SAM2 predictor
        self.predictor.set_image(input_image)

        input_points = np.array(self.click_points)
        input_labels = np.array(self.click_labels)

        # Run the SAM2 model to get masks
        masks, scores, _ = self.predictor.predict(
            point_coords=input_points, 
            point_labels=input_labels, 
            multimask_output=True)

        # Get the top 3 masks
        top_3_indices = np.argsort(scores)[-3:][::-1]
        top_3_masks = [masks[i] for i in top_3_indices]

        # Publish each mask to the 'sam2_masks' topic
        for i, mask in enumerate(top_3_masks):
            self.publish_mask(mask)

        # Clear click points after processing
        self.click_points.clear()
        self.click_labels.clear()

    def publish_mask(self, mask):
        """Convert mask to a string and publish it"""
        mask_msg = String()
        mask_msg.data = np.array_str(mask)
        self.mask_publisher.publish(mask_msg)
        self.get_logger().info("Published SAM2 mask.")

def main(args=None):
       

    rclpy.init(args=args)

    # Check if we are running in offline mode (loading an image instead of using the camera)
    use_offline_image = True  # Set this flag to True to use offline mode
    offline_image_path = "/home/longuzi/sam2_model/segment-anything-2/car.jpg"  # Provide the path to your offline image

    # Create the SAM2 service node in the appropriate mode
    node = SAM2ServiceNode(use_offline_image=use_offline_image, offline_image_path=offline_image_path)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
