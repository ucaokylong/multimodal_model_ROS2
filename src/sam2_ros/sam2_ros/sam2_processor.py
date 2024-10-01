# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import Image
# from std_msgs.msg import String
# from cv_bridge import CvBridge
# import torch
# import numpy as np
# import cv2
# from PIL import Image as PilImage
# import matplotlib.pyplot as plt
# from matplotlib.widgets import Button
# import os
# import hydra
# from hydra.core.global_hydra import GlobalHydra
# from functools import partial

# from .sam2.build_sam import build_sam2
# from .sam2.sam2_image_predictor import SAM2ImagePredictor

# class Sam2SegmentationNode(Node):
#   def __init__(self):
#       super().__init__('sam2_segmentation_node')
#       self.declare_parameter('mode', 'online')  
#       self.declare_parameter('image_dir', '/home/longuzi/deploy_sam2_ws/src/sam2_ros/car.jpg')  
#       self.declare_parameter('camera_topic', 'camera_frames')  

#       self.bridge = CvBridge()
#       self.click_points = []
#       self.click_labels = []
#       self.top_3_masks = []
#       self.selected_mask = None
#       self.stored_frame = None
#       self.frame = None  # Initialize frame
#       self.waiting_for_new_frame = False  # Flag to indicate waiting for a new frame
#       self.mode = self.get_parameter('mode').get_parameter_value().string_value

#       self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#       config_path = '/home/longuzi/deploy_sam2_ws/src/sam2_ros/sam2_ros/sam2_configs/sam2_hiera_s.yaml'
#       sam2_checkpoint = "/home/longuzi/deploy_sam2_ws/src/sam2_ros/sam2_ros/checkpoints/sam2_hiera_small.pt"

#       if not os.path.isfile(config_path):
#           raise FileNotFoundError(f"Config file not found: {config_path}")

#       print("Current working directory:", os.getcwd())

#       config_dir = '/home/longuzi/deploy_sam2_ws/src/sam2_ros/sam2_ros/sam2_configs'
#       GlobalHydra.instance().clear()
#       hydra.initialize_config_dir(config_dir=config_dir)
      
#       self.sam2 = build_sam2(config_file='sam2_hiera_s.yaml', ckpt_path=sam2_checkpoint, device=self.device, apply_postprocessing=False)
#       self.predictor = SAM2ImagePredictor(self.sam2)
#       self.get_logger().info(f"SAM2 Model loaded on {self.device}")

#       if self.mode == 'online':
#           self.subscription = self.create_subscription(
#               Image,
#               self.get_parameter('camera_topic').get_parameter_value().string_value,
#               self.image_callback,
#               10)

#       self.command_subscription = self.create_subscription(
#           String,
#           'sam2_command',
#           self.command_callback,
#           10
#       )

#       self.mask_publisher = self.create_publisher(Image, 'sam2/top_masks', 10)

#       self.image_dir = self.get_parameter('image_dir').get_parameter_value().string_value
#       if self.mode == 'offline':
#           if os.path.isfile(self.image_dir):
#               image = PilImage.open(self.image_dir)
#               self.image_np = np.array(image.convert("RGB"))
#               self.predictor.set_image(self.image_np)
#               self.stored_frame = self.image_np.copy()  # Use this as the frame in offline mode
#           else:
#               raise ValueError(f"Provided path '{self.image_dir}' is not a valid image file.")

#       self.get_logger().info(f"Node initialized in {self.mode} mode.")
#       self.init_figure()  # Initialize the figure and buttons

#   def init_figure(self):
#       # Initialize the figure and axes
#       plt.ion()  # Enable interactive mode
#       self.fig, self.axs = plt.subplots(1, 2, figsize=(15, 8))
#       self.ax_input = self.axs[0]

#       self.ax_masks = []

#       # Create mask axes independently
#       for i in range(3):
#           ax_mask = self.axs[1].inset_axes([0, 1 - (i + 1) * 0.35, 1.0, 0.3])  # Adjust height
#           self.ax_masks.append(ax_mask)
#           ax_mask.axis('off')

#       # Create 'Run Model' button
#       ax_run = plt.axes([0.05, 0.05, 0.1, 0.075])
#       btn_run = Button(ax_run, 'Run Model')
#       btn_run.on_clicked(self.run_model)

#       # Create 'Clear Points' button
#       ax_clear = plt.axes([0.2, 0.05, 0.1, 0.075])
#       btn_clear = Button(ax_clear, 'Clear Points')
#       btn_clear.on_clicked(self.clear_points)

#       # Create 'Get New Frame' button
#       ax_get_new_frame = plt.axes([0.35, 0.05, 0.1, 0.075])
#       btn_get_new_frame = Button(ax_get_new_frame, 'Get New Frame')
#       btn_get_new_frame.on_clicked(self.get_new_frame)

#       # Create 'Choose Mask' buttons
#       self.buttons = []
#       for i, ax_mask in enumerate(self.ax_masks):
#           mask_button_ax = self.fig.add_axes([0.85, 0.05 + i * 0.1, 0.1, 0.075])
#           btn_choose = Button(mask_button_ax, f'Choose Mask {i + 1}')
#           btn_choose.on_clicked(partial(self.publish_mask, i))
#           self.buttons.append(btn_choose)

#       self.ax_input.set_title('Click to add points')
#       self.ax_input.axis('off')  # Remove axis and ticks for the main input image

#       self.fig.canvas.mpl_connect('button_press_event', self.on_click)

#       # Show the figure
#       plt.show(block=False)

#   def command_callback(self, msg):
#       if msg.data == 'run':
#           self.get_logger().info('Received run command.')
#           self.get_new_frame(None)  # Trigger the get new frame process

#   def image_callback(self, msg):
#       if self.mode == 'online' and self.waiting_for_new_frame:
#           self.frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
#           self.show_image()
#           self.waiting_for_new_frame = False

#   def show_image(self):
#       self.ax_input.clear()
#       if self.frame is not None:
#           self.ax_input.imshow(cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB))
#           self.ax_input.set_title('Click to add points')
#           self.ax_input.axis('off')
#           self.fig.canvas.draw_idle()
#       else:
#           self.get_logger().warn('No frame to display.')

#   def on_click(self, event):
#       if event.inaxes == self.ax_input:
#           if event.xdata is not None and event.ydata is not None:
#               x, y = int(event.xdata), int(event.ydata)
#               self.click_points.append([x, y])
#               self.click_labels.append(1)
#               self.ax_input.plot(x, y, 'ro')  # Display clicked points
#               self.fig.canvas.draw_idle()

#   def run_model(self, event):
#       if self.frame is None:
#           self.get_logger().warn('No frame available to run the model. Please click "Get New Frame" first.')
#           return

#       self.stored_frame = self.frame.copy()  # Store the frame when "Run Model" is clicked

#       input_points = np.array(self.click_points)
#       input_labels = np.array(self.click_labels)
#       self.predictor.set_image(self.stored_frame)  # Set the stored frame or image as the input
      
#       if input_points.shape[0] == 0:
#           self.get_logger().warn('No points have been selected.')
#           return

#       masks, scores, _ = self.predictor.predict(
#           point_coords=input_points,
#           point_labels=input_labels,
#           multimask_output=True,
#       )

#       top_3_indices = np.argsort(scores)[-3:][::-1]
#       self.top_3_masks = [masks[i] for i in top_3_indices]

#       for i in range(3):
#           mask_8bit = cv2.normalize(self.top_3_masks[i], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
#           mask_img = self.bridge.cv2_to_imgmsg(mask_8bit, encoding='mono8')
#           self.mask_publisher.publish(mask_img)
#           self.show_mask(self.top_3_masks[i], self.ax_masks[i], random_color=True)

#       self.fig.canvas.draw_idle()

#   def show_mask(self, mask, ax, random_color=False):
#       ax.clear()
#       color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0) if random_color else np.array([30/255, 144/255, 255/255, 0.6])
#       h, w = mask.shape[-2:]
#       mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
#       ax.imshow(mask_image)
#       ax.axis('off')

#   def clear_points(self, event):
#       self.click_points = []
#       self.click_labels = []
#       self.ax_input.clear()
#       if self.frame is not None:
#           self.ax_input.imshow(cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB))
#       self.ax_input.set_title('Click to add points')
#       self.ax_input.axis('off')
#       self.fig.canvas.draw_idle()

#   def get_new_frame(self, event):
#       self.click_points = []
#       self.click_labels = []
#       self.waiting_for_new_frame = True  # Indicate we are waiting for a new frame
#       self.get_logger().info('Waiting for new frame...')
#       self.ax_input.clear()
#       self.ax_input.text(0.5, 0.5, 'Waiting for new frame...', 
#                          horizontalalignment='center', verticalalignment='center', fontsize=12)
#       self.ax_input.set_title('Click to add points')
#       self.ax_input.axis('off')
#       self.fig.canvas.draw_idle()

#   def publish_mask(self, mask_idx, event):
#       if mask_idx < len(self.top_3_masks):
#           self.selected_mask = self.top_3_masks[mask_idx]
#           mask_8bit = cv2.normalize(self.selected_mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
#           mask_img = self.bridge.cv2_to_imgmsg(mask_8bit, encoding='mono8')
#           self.mask_publisher.publish(mask_img)
#           self.get_logger().info(f"Published mask {mask_idx + 1}")
#       else:
#           self.get_logger().warn(f"No mask available at index {mask_idx}")

# def main(args=None):
#   rclpy.init(args=args)
#   node = Sam2SegmentationNode()

#   try:
#       while rclpy.ok():
#           rclpy.spin_once(node, timeout_sec=0.1)
#           plt.pause(0.01)  # Allow Matplotlib to process events
#   except KeyboardInterrupt:
#       pass

#   node.destroy_node()
#   rclpy.shutdown()

# if __name__ == '__main__':
#   main()



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
      self.declare_parameter('mode', 'online')  
      self.declare_parameter('image_dir', '/home/longuzi/deploy_sam2_ws/src/sam2_ros/car.jpg')  
      self.declare_parameter('camera_topic', 'camera_frames')  

      self.bridge = CvBridge()
      self.click_points = []
      self.click_labels = []
      self.top_3_masks = []
      self.selected_mask = None
      self.stored_frame = None
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
          self.image_np = None  # Initialize to None for online mode

      self.mask_publisher = self.create_publisher(Image, 'sam2/top_masks', 10)

      self.image_dir = self.get_parameter('image_dir').get_parameter_value().string_value
      if self.mode == 'offline':
          if os.path.isfile(self.image_dir):
              image = PilImage.open(self.image_dir)
              self.image_np = np.array(image.convert("RGB"))
              self.predictor.set_image(self.image_np)
          else:
              raise ValueError(f"Provided path '{self.image_dir}' is not a valid image file.")

      self.get_logger().info(f"Node initialized in {self.mode} mode.")
      self.show_image()  # Call show_image at the end of initialization

  def image_callback(self, msg):
      self.frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
      if not hasattr(self, 'fig'):
          self.show_image()
      else:
          self.update_image()

  def show_image(self):
      self.fig, self.axs = plt.subplots(1, 2, figsize=(15, 8))
      self.ax_input = self.axs[0]

      self.ax_masks = []
      self.buttons = []

      vertical_spacing = 0.35

      for i in range(3):
          bottom = 1 - (i + 1) * vertical_spacing
          height = vertical_spacing * 0.85

          ax_mask = self.axs[1].inset_axes([0, bottom, 1.0, height])
          self.ax_masks.append(ax_mask)

          ax_mask.axis('off')

      ax_run = plt.axes([0.05, 0.05, 0.1, 0.075])
      btn_run = Button(ax_run, 'Run Model')
      btn_run.on_clicked(self.run_model)

      ax_clear = plt.axes([0.2, 0.05, 0.1, 0.075])
      btn_clear = Button(ax_clear, 'Clear Points')
      btn_clear.on_clicked(self.clear_points)

      ax_get_new_frame = plt.axes([0.35, 0.05, 0.1, 0.075])
      btn_get_new_frame = Button(ax_get_new_frame, 'Get New Frame')
      btn_get_new_frame.on_clicked(self.get_new_frame)

      self.fig.canvas.mpl_connect('button_press_event', self.on_click)

      for i, ax_mask in enumerate(self.ax_masks):
          mask_button_offset = 0.02
          mask_position = ax_mask.get_position()

          mask_button_ax = self.fig.add_axes([mask_position.x0 + mask_position.width + mask_button_offset - 0.01,
                                              mask_position.y0 + 0.05,
                                              0.08,
                                              0.05])

          btn_choose = Button(mask_button_ax, f'Choose Mask {i + 1}')
          btn_choose.on_clicked(partial(self.publish_mask, i))
          self.buttons.append(btn_choose)

          plt.setp(btn_choose.label, fontsize=8)

      self.update_image()
      plt.show()

  def update_image(self):
      self.ax_input.clear()
      if self.mode == 'online':
          if self.frame is not None:
              self.ax_input.imshow(cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB))
          else:
              self.ax_input.text(0.5, 0.5, 'Waiting for frame...', ha='center', va='center')
      else:
          if self.image_np is not None:
              self.ax_input.imshow(self.image_np)
          else:
              self.ax_input.text(0.5, 0.5, 'No image loaded', ha='center', va='center')
      
      self.ax_input.set_title('Click to add points')
      self.ax_input.axis('off')
      for point in self.click_points:
          self.ax_input.plot(point[0], point[1], 'ro')
      self.fig.canvas.draw_idle()

  def on_click(self, event):
      if event.inaxes == self.ax_input:
          x, y = int(event.xdata), int(event.ydata)
          self.click_points.append([x, y])
          self.click_labels.append(1)
          self.ax_input.plot(x, y, 'ro')
          self.fig.canvas.draw_idle()

  def run_model(self, event):
      if self.mode == 'online':
          if self.frame is not None:
              self.stored_frame = self.frame.copy()
      elif self.mode == 'offline':
          if self.image_np is not None:
              self.stored_frame = self.image_np.copy()

      if self.stored_frame is not None:
          input_points = np.array(self.click_points)
          input_labels = np.array(self.click_labels)
          self.predictor.set_image(self.stored_frame)
          
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

          self.fig.canvas.draw_idle()

  def show_mask(self, mask, ax, random_color=False):
      ax.clear()
      color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0) if random_color else np.array([30/255, 144/255, 255/255, 0.6])
      h, w = mask.shape[-2:]
      mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
      ax.imshow(mask_image)
      ax.axis('off')

  def clear_points(self, event):
      self.click_points = []
      self.click_labels = []
      self.update_image()

  def get_new_frame(self, event):
      self.click_points = []
      self.click_labels = []
      self.top_3_masks = []
      self.selected_mask = None
      self.stored_frame = None

      if self.mode == 'online':
          self.frame = None
          while self.frame is None:
              rclpy.spin_once(self, timeout_sec=0.1)
      
      self.update_image()

  def publish_mask(self, mask_idx, event):
      if mask_idx < len(self.top_3_masks):
          self.selected_mask = self.top_3_masks[mask_idx]
          mask_8bit = cv2.normalize(self.selected_mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
          mask_img = self.bridge.cv2_to_imgmsg(mask_8bit, encoding='mono8')
          self.mask_publisher.publish(mask_img)
          self.get_logger().info(f"Published mask {mask_idx + 1}")
      else:
          self.get_logger().warn(f"Mask {mask_idx + 1} not available")

def main(args=None):
  rclpy.init(args=args)
  node = Sam2SegmentationNode()
  
  while rclpy.ok():
      rclpy.spin_once(node, timeout_sec=0.1)
      if plt.get_fignums():
          plt.pause(0.1)
      else:
          break

  node.destroy_node()
  rclpy.shutdown()

if __name__ == '__main__':
  main()
















