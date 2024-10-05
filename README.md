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
