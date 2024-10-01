import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import torch
import numpy as np
import cv2
from PIL import Image as PilImage
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import os
import hydra
from hydra.core.global_hydra import GlobalHydra
from functools import partial

from .sam2.build_sam import build_sam2
from .sam2.sam2_image_predictor import SAM2ImagePredictor

class Sam2SegmentationNode(Node):
    def __init__(self):
        super().__init__('sam2_segmentation_node')
        self.declare_parameter('mode', 'offline')  
        self.declare_parameter('image_dir', '/home/longuzi/deploy_sam2_ws/src/sam2_ros/car.jpg')  
        self.declare_parameter('camera_topic', 'camera_frames')  

        self.bridge = CvBridge()
        self.click_points = []
        self.click_labels = []
        self.top_3_masks = []
        self.selected_mask = None
        self.mode = self.get_parameter('mode').get_parameter_value().string_value

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        config_path = '/home/longuzi/deploy_sam2_ws/src/sam2_ros/sam2_ros/sam2_configs/sam2_hiera_s.yaml'
        sam2_checkpoint = "/home/longuzi/deploy_sam2_ws/src/sam2_ros/sam2_ros/checkpoints/sam2_hiera_small.pt"

        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        print("Current working directory:", os.getcwd())

        config_dir = '/home/longuzi/deploy_sam2_ws/src/sam2_ros/sam2_ros/sam2_configs'
        GlobalHydra.instance().clear()
        hydra.initialize_config_dir(config_dir=config_dir)
        
        self.sam2 = build_sam2(config_file='sam2_hiera_s.yaml', ckpt_path=sam2_checkpoint, device=self.device, apply_postprocessing=False)
        self.predictor = SAM2ImagePredictor(self.sam2)
        self.get_logger().info(f"SAM2 Model loaded on {self.device}")

        if self.mode == 'online':
            self.subscription = self.create_subscription(
                Image,
                self.get_parameter('camera_topic').get_parameter_value().string_value,
                self.image_callback,
                10)
            self.frame = None

        self.mask_publisher = self.create_publisher(Image, 'sam2/top_masks', 10)

        self.image_dir = self.get_parameter('image_dir').get_parameter_value().string_value
        if self.mode == 'offline':
            if os.path.isfile(self.image_dir):
                image = PilImage.open(self.image_dir)
                self.image_np = np.array(image.convert("RGB"))
                self.predictor.set_image(self.image_np)
                self.show_image()
            else:
                raise ValueError(f"Provided path '{self.image_dir}' is not a valid image file.")

        self.get_logger().info(f"Node initialized in {self.mode} mode.")
    
    def image_callback(self, msg):
        if self.mode == 'online':
            self.frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.show_image()

    def show_image(self):
        self.fig, self.axs = plt.subplots(1, 2, figsize=(15, 8))  # Increased figsize for bigger masks
        self.ax_input = self.axs[0]

        # Create a list to store the axes for the masks
        self.ax_masks = []
        self.buttons = []  # List to store button references

        # Adjust vertical spacing for the larger masks
        vertical_spacing = 0.35  # More space between the masks

        # Create mask axes independently
        for i in range(3):
            # Increase the height and width of the masks
            bottom = 1 - (i + 1) * vertical_spacing
            height = vertical_spacing * 0.85  # Larger masks but within the boundary

            # Create the mask axes
            ax_mask = self.axs[1].inset_axes([0, bottom, 1.0, height])  # Adjust width to fill the box
            self.ax_masks.append(ax_mask)

            # Remove any axis coordinates for the mask region
            ax_mask.axis('off')  # Ensure axis numbers are removed

        # Create 'Run Model' button
        ax_run = plt.axes([0.05, 0.05, 0.1, 0.075])
        btn_run = Button(ax_run, 'Run Model')
        btn_run.on_clicked(self.run_model)

        # Create 'Clear Points' button
        ax_clear = plt.axes([0.2, 0.05, 0.1, 0.075])
        btn_clear = Button(ax_clear, 'Clear Points')
        btn_clear.on_clicked(self.clear_points)

        # Display the input image
        self.ax_input.imshow(self.image_np)
        self.ax_input.set_title('Click to add points')
        self.ax_input.axis('off')  # Remove axis and ticks for the main input image

        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

        # Create 'Choose Mask' buttons outside the loop
        for i, ax_mask in enumerate(self.ax_masks):
            mask_button_offset = 0.02  # Slight shift to the right

            # Get the position of the mask axes using get_position()
            mask_position = ax_mask.get_position()

            # Add the 'Choose Mask' button (scaled down and re-positioned)
            mask_button_ax = self.fig.add_axes([mask_position.x0 + mask_position.width + mask_button_offset - 0.01,
                                                mask_position.y0 + 0.05,  # Slight vertical adjustment to avoid overlap
                                                0.08,  # Adjusted width
                                                0.05])  # Adjusted height

            # Create button and store reference
            btn_choose = Button(mask_button_ax, f'Choose Mask {i + 1}')
            btn_choose.on_clicked(partial(self.publish_mask, i))  # Use partial to correctly pass the index
            self.buttons.append(btn_choose)

            # Set font size for the button text
            plt.setp(btn_choose.label, fontsize=8)  # Adjust font size

        plt.show()

    def on_click(self, event):
        x, y = int(event.xdata), int(event.ydata)
        self.click_points.append([x, y])
        self.click_labels.append(1)
        self.ax_input.plot(x, y, 'ro')  # Display clicked points
        self.fig.canvas.draw()

    def run_model(self, event):
        input_points = np.array(self.click_points)
        input_labels = np.array(self.click_labels)
        masks, scores, _ = self.predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=True,
        )

        top_3_indices = np.argsort(scores)[-3:][::-1]
        self.top_3_masks = [masks[i] for i in top_3_indices]

        for i in range(3):
            mask_8bit = cv2.normalize(self.top_3_masks[i], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            mask_img = self.bridge.cv2_to_imgmsg(mask_8bit, encoding='mono8')
            self.mask_publisher.publish(mask_img)
            self.show_mask(self.top_3_masks[i], self.ax_masks[i], random_color=True)

        self.fig.canvas.draw()

    def show_mask(self, mask, ax, random_color=False):
        ax.clear()
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0) if random_color else np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        
        ax.imshow(mask_image, extent=[0, w, 0, h])  # Ensure mask fits within the bounding box
        ax.axis('off')  # Remove axes for mask display

    def publish_mask(self, mask_index, event):
        if self.top_3_masks:
            mask_8bit = cv2.normalize(self.top_3_masks[mask_index], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            mask_img = self.bridge.cv2_to_imgmsg(mask_8bit, encoding='mono8')
            self.mask_publisher.publish(mask_img)
            self.get_logger().info(f"Published Mask {mask_index + 1}.")

    def clear_points(self, event):
        self.click_points = []
        self.click_labels = []
        self.ax_input.clear()
        self.ax_input.imshow(self.image_np)
        self.ax_input.set_title('Click to add points')
        self.ax_input.axis('off')
        self.fig.canvas.draw()

def main(args=None):
    rclpy.init(args=args)
    node = Sam2SegmentationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
