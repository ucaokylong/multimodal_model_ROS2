from setuptools import find_packages, setup
import os

package_name = 'sam2_ros'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name] + find_packages(include=[package_name, f'{package_name}.*']),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=[
        'setuptools',
        'torch',  # PyTorch for model
        'opencv-python',  # OpenCV for image handling
        'cv_bridge',  # ROS2 CV bridge
        'hydra',
        'numpy',
        'matplotlib',
        'Pillow',
        'transformers',
        'sounddevice'
    ],
    zip_safe=True,
    maintainer='longuzi',
    maintainer_email='longuzi@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'camera_publisher = sam2_ros.camera_publisher:main',
            'sam2_segmentation_node = sam2_ros.sam2_processor:main',
            'florence_model_node = sam2_ros.florence_model_node:main',
            'florence_input_node = sam2_ros.florence_input_node:main',
            'sam2_execution_node = sam2_ros.sam2_execution_node:main'

            # 'hydra_test_sam2_node = sam2_ros.test_hydra_package.test_hydra_sam2_node:main',
        ],
    },
)