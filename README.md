# Multimodal_model_ROS2

## Deploy Multimodal Model into ROS2 System

This project involves deploying a multimodal model into a ROS2 system, integrating several advanced models and functionalities, including segmentation, object detection, and speech recognition. The project is structured into multiple ROS2 nodes, each responsible for a specific task.

## Table of Contents

- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Nodes Description](#nodes-description)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

This project integrates the following components:

- **Segmentation Anything 2 Model**: Utilizes the SAM2 model for image segmentation.
- **Florence2 Multimodal Model**: Processes images and text inputs for various tasks.
- **Whisper Model**: Performs speech recognition to convert audio input into text.
- **Finetune Florence2 with OD Task**: Customizes the Florence2 model for object detection tasks specific to certain objects.
- **Build UI**: Provides a user interface for interacting with the system.
- **ROS2 Workflow**: Implements a workflow using ROS2 nodes for seamless integration.
- **Camera Publisher Node**: Captures and publishes frames from a camera.

## Installation

### Prerequisites

- ROS2 Humble or later
- Python 3.8 or later
- OpenCV
- PyTorch
- Transformers library
- SoundDevice
- Matplotlib
- Hydra

### Setup

1. Clone the repository
2. Install the required Python packages
3. Ensure ROS2 is sourced:
```
bash
source /opt/ros/humble/setup.bash
```
4.Build the ROS2 workspace:
```
bash
colcon build
```

## Usage

### Running the Nodes

1. **Camera Publisher Node**: Publishes frames from the camera.
```
bash
ros2 run your_package camera_publisher
```

2. **Florence Input Node**: Handles user input for task selection and text input.
```
bash
ros2 run your_package florence_input_node
```

3. **Florence Model Node**: Processes images using the Florence2 model.
```
bash
ros2 run your_package florence_model_node
```

4. **Sam2 Execution Node**: Sends execution commands.
```
bash
ros2 run your_package sam2_execution_node
```

5. **Sam2 Segmentation Node**: Performs image segmentation using the SAM2 model.
```
bash
ros2 run your_package sam2_processor
```

### Configuration

- **Mode**: Set the mode to 'online' or 'offline' in the `florence_model_node.py` and `sam2_processor.py` files.
- **Image Path**: Specify the path to the image for offline processing in the respective node files.

## Nodes Description

- **CameraPublisher**: Captures and publishes images from a camera.
- **FlorenceInputNode**: Collects user input for task selection and text input, including speech recognition.
- **FlorenceModelNode**: Uses the Florence2 model to process images based on task prompts and text input.
- **Sam2ExecutionNode**: Publishes commands for execution.
- **Sam2SegmentationNode**: Utilizes the SAM2 model to perform segmentation on images. It supports both online (real-time camera feed) and offline (pre-loaded images) modes. Users can interactively select points on the image to guide the segmentation process.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any improvements or bug fixes.
