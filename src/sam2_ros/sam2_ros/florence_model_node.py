import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from transformers import AutoModelForCausalLM, AutoProcessor
import torch
from PIL import Image as PILImage
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

class FlorenceModelNode(Node):
    def __init__(self, mode='online', image_path=None):
        super().__init__('florence_model_node')
        self.mode = mode  # 'online' or 'offline'
        self.image_path = image_path
        self.bridge = CvBridge()
        self.task_prompt = None
        self.text_input = None
        self.task_received = False
        self.text_received = False

        # Florence-2 model setup
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base", torch_dtype=self.torch_dtype, trust_remote_code=True).to(self.device)
        self.processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)

        # Subscribe to task_prompt and text_input messages from FlorenceInputNode
        self.subscription_task_prompt = self.create_subscription(
            String,
            'task_prompt',
            self.task_prompt_callback,
            10
        )
        self.subscription_text_input = self.create_subscription(
            String,
            'text_input',
            self.text_input_callback,
            10
        )

        # Online mode
        if self.mode == 'online':
            self.subscription = self.create_subscription(
                Image,
                'camera_frames',
                self.image_callback,
                10
            )

    def task_prompt_callback(self, msg):
        self.task_prompt = msg.data
        self.task_received = True
        self.get_logger().info(f"Received task_prompt: {self.task_prompt}")
        # self.try_process_offline_image()

    def text_input_callback(self, msg):
        self.text_input = msg.data
        self.text_received = True
        self.get_logger().info(f"Received text_input: {self.text_input}")
        self.try_process_offline_image()

    def try_process_offline_image(self):
        # Ensure both task_prompt and text_input have been received
        if self.task_received and self.text_received and self.mode == 'offline':
            self.get_logger().info("Both inputs received, processing offline image...")
            self.process_offline_image()

    def image_callback(self, msg):
        if self.task_prompt is None or self.text_input is None:
            self.get_logger().info("Waiting for task_prompt and text_input.")
            return

        # Convert ROS Image to OpenCV format
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        # Convert the frame to PIL format
        pil_image = PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Run the Florence-2 model on the frame
        result = self.run_florence_model(self.task_prompt, self.text_input, image=pil_image)

        # Process the result (e.g., plot bounding boxes)
        self.plot_bbox(pil_image, result[self.task_prompt])

    def process_offline_image(self):
        if self.image_path is None:
            self.get_logger().error("No image path provided for offline mode.")
            return
        
        # Load the image from file
        if not os.path.isfile(self.image_path):
            self.get_logger().error(f"Image file not found: {self.image_path}")
            return
        
        frame = cv2.imread(self.image_path)
        pil_image = PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if self.task_prompt is None or self.text_input is None:
            self.get_logger().info("Waiting for task_prompt and text_input.")
            return
        
        # Run the Florence-2 model on the image
        result = self.run_florence_model(self.task_prompt, self.text_input, image=pil_image)

        # Process the result (e.g., plot bounding boxes)
        self.plot_bbox(pil_image, result[self.task_prompt])


    def run_florence_model(self, task_prompt, text_input, image):
        if task_prompt == "<CAPTION_TO_PHRASE_GROUNDING>":
            prompt = task_prompt + text_input
        else:
            prompt = task_prompt

        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device, torch.float16)

        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"].to(self.device),
            pixel_values=inputs["pixel_values"].to(self.device),
            max_new_tokens=1024,
            early_stopping=False,
            do_sample=False,
            num_beams=3
        )

        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = self.processor.post_process_generation(
            generated_text, 
            task=task_prompt, 
            image_size=(image.width, image.height)
        )
        print(parsed_answer)
        return parsed_answer
    
    def plot_bbox(self, image, data):
        # Create a figure and axes  
        fig, ax = plt.subplots()  
        ax.imshow(image)  # Display the image

        # Plot each bounding box  
        for bbox, label in zip(data['bboxes'], data['labels']):
            x1, y1, x2, y2 = bbox
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            plt.text(x1, y1, label, color='white', fontsize=8, bbox=dict(facecolor='red', alpha=0.5))

        ax.axis('off')  # Remove axis ticks and labels
        plt.show()

    
def main(args=None):
    rclpy.init(args=args)

    # Choose mode ('online' or 'offline') and set the image directory for offline mode
    mode = 'offline'  # Change to 'offline' to process images from a directory
    image_dir = '/home/longuzi/deploy_sam2_ws/src/sam2_ros/dinner_table.jpg'  # Set the directory for offline mode

    florence_model_node = FlorenceModelNode(mode=mode, image_path=image_dir)

    if mode == 'online':
        rclpy.spin(florence_model_node)
    else:
        florence_model_node.process_offline_image()
        rclpy.spin(florence_model_node)

    florence_model_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
        
#         # Offline mode
#         if self.mode == 'offline' and self.image_path:
#             self.process_offline_image()

#     def image_callback(self, msg):
#         # Convert ROS Image to OpenCV format
#         frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")

#         # Convert the frame to PIL format
#         pil_image = PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

#         # Run the Florence-2 model on the frame
#         task_prompt = '<CAPTION_TO_PHRASE_GROUNDING>'
#         text_input = "find out vase and fruits"
#         # if text_input is None:
#         #     prompt = task_prompt
#         # else:
#         #     prompt = task_prompt + text_input
#         result = self.run_florence_model(task_prompt, text_input, image=pil_image)

#         # Process the result (e.g., plot bounding boxes)
#         self.plot_bbox(pil_image, result[task_prompt])

#     def run_florence_model(self, task_prompt, text_input, image):
#         if task_prompt == "<CAPTION_TO_PHRASE_GROUNDING>":
#             prompt = task_prompt + text_input
#         else:
#             prompt = task_prompt 
#         # text_input = "find out vase and fruits" if task_prompt == '<CAPTION_TO_PHRASE_GROUNDING>' else None
#         inputs = self.processor(text= prompt, images=image, return_tensors="pt").to(self.device, torch.float16)

#         generated_ids = self.model.generate(
#             input_ids=inputs["input_ids"].to(self.device),
#             pixel_values=inputs["pixel_values"].to(self.device),
#             max_new_tokens=1024,
#             early_stopping=False,
#             do_sample=False,
#             num_beams=3
#         )
        
#         generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
#         parsed_answer = self.processor.post_process_generation(
#             generated_text, 
#             task=task_prompt, 
#             image_size=(image.width, image.height)
#         )
#         print(parsed_answer)
#         return parsed_answer

#     def process_offline_image(self):
#         # Process only one image
#         if self.image_path.endswith(('.png', '.jpg', '.jpeg')):
#             image = PILImage.open(self.image_path)

#             task_prompt = '<CAPTION_TO_PHRASE_GROUNDING>'
#             text_input = "find out food and vase"
#             result = self.run_florence_model(task_prompt, text_input, image=image)

#             # Plot result (bounding boxes)
#             self.plot_bbox(image, result[task_prompt])

#     def plot_bbox(self, image, data):
#         # Create a figure and axes  
#         fig, ax = plt.subplots()  
#         ax.imshow(image)  # Display the image

#         # Plot each bounding box  
#         for bbox, label in zip(data['bboxes'], data['labels']):
#             x1, y1, x2, y2 = bbox
#             rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')
#             ax.add_patch(rect)
#             plt.text(x1, y1, label, color='white', fontsize=8, bbox=dict(facecolor='red', alpha=0.5))

#         ax.axis('off')  # Remove axis ticks and labels
#         plt.show()

# def main(args=None):
#     rclpy.init(args=args)

#     # Choose mode ('online' or 'offline') and set the image directory for offline mode
#     mode = 'offline'  # Change to 'offline' to process images from a directory
#     image_dir = '/home/longuzi/deploy_sam2_ws/src/sam2_ros/dinner_table.jpg'  # Set the directory for offline mode

#     florence_model_node = FlorenceModelNode(mode=mode, image_path=image_dir)

#     if mode == 'online':
#         rclpy.spin(florence_model_node)
#     else:
#         florence_model_node.process_offline_image()
        
#     florence_model_node.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()
