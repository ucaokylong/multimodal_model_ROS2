# test_hydra_sam2_node_custom_config.py

import rclpy
from rclpy.node import Node
import hydra
from omegaconf import DictConfig
from hydra.core.global_hydra import GlobalHydra


# Assuming you have the build_sam2 and SAM2ImagePredictor available
from .sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Define a simple ROS2 Node class
class HydraTestSAM2Node(Node):

    def __init__(self, config: DictConfig):
        super().__init__('hydra_test_sam2_node')
        
        # Retrieve the device from the Hydra config
        self.device = config.device
        self.get_logger().info(f"Running SAM2 model on device: {self.device}")

        # Load the SAM2 model with Hydra-managed paths
        config_path = config.sam2.config_path
        sam2_checkpoint = config.sam2.ckpt_path
        
        self.sam2 = build_sam2(config_file=config_path, ckpt_path=sam2_checkpoint, device=self.device, apply_postprocessing=False)
        self.predictor = SAM2ImagePredictor(self.sam2)
        
        # Log that the model is loaded successfully
        self.get_logger().info(f"SAM2 Model loaded on {self.device}")

    def start(self):
        # Any additional start behavior for your node
        self.get_logger().info("HydraTestSAM2Node started successfully.")


def main(args=None):
    rclpy.init(args=args)
    
    # Specify the directory where the configuration files are located
    config_dir = '/home/longuzi/deploy_sam2_ws/src/sam2_ros/sam2_ros/sam2_configs'
    
    # Clear any existing Hydra instance before initializing a new one
    GlobalHydra.instance().clear()
    
    # Initialize Hydra with the specified configuration directory
    hydra.initialize_config_dir(config_dir=config_dir)
    
    # Compose the configuration from the 'sam2_hiera_s.yaml' file
    cfg = hydra.compose(config_name="sam2_hiera_s.yaml")
    
    # Create the ROS2 node with the loaded configuration
    node = HydraTestSAM2Node(cfg)
    node.start()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Hydra Test SAM2 Node')
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()

