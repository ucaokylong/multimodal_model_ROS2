# multimodal_model_ROS2
## Deploy multimodal model into Ros2 system
- Segmantation_anything_2_model
- Florence2 multimodel
- Whisper model
- Finetune Florence2 with OD task for specific object
- Build UI
- Ros2 workflow
- Camera Publisher node


# CV Project with ROS2 and Florence-2

This project is a computer vision application using ROS2, OpenCV, and the Florence-2 model. It includes nodes for capturing camera frames, processing audio input, executing commands, and performing image segmentation.

## Table of Contents

- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Nodes Description](#nodes-description)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

This project integrates several components to perform tasks such as capturing images from a camera, processing audio input for speech recognition, and using the Florence-2 model for image processing tasks. The project is structured into multiple ROS2 nodes, each responsible for a specific task.

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
