"""
Depth estimation using the MiDaS model (intel-isl/MiDaS_small).
Loads the model once as a module-level singleton.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def _load_depth_model():
    model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    model.eval()
    return model


# Singleton: loaded once when this module is first imported
depth_model = _load_depth_model()


def estimate_depth(image):
    """Uses AI-based depth estimation to improve circumference calculations.

    Args:
        image: BGR numpy array (as returned by cv2.imdecode).

    Returns:
        2-D numpy array representing the depth map (384x384 resolution).
    """
    input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
    input_tensor = torch.tensor(input_image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

    # Resize input to match MiDaS model input size
    input_tensor = F.interpolate(input_tensor, size=(384, 384), mode="bilinear", align_corners=False)

    with torch.no_grad():
        depth_map = depth_model(input_tensor)

    return depth_map.squeeze().numpy()
