import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/longuzi/deploy_sam2_ws/install/sam2_ros'
