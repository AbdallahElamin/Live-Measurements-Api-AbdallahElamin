"""
Flask Blueprint exposing the /upload_images endpoint.
"""

import cv2
import numpy as np
from flask import Blueprint, request, jsonify
import mediapipe as mp

from config import FOCAL_LENGTH, DEFAULT_HEIGHT_CM
from measurements.validator import validate_front_image
from measurements.calibration import calculate_distance_using_height, detect_reference_object
from measurements.depth import estimate_depth
from measurements.calculator import calculate_measurements

bp = Blueprint("measurements", __name__)

mp_holistic = mp.solutions.holistic
_holistic = mp_holistic.Holistic()


@bp.route("/upload_images", methods=["POST"])
def upload_images():
    if "front" not in request.files:
        return jsonify({"error": "Missing front image for reference."}), 400

    front_image_file = request.files["front"]
    front_image_np = np.frombuffer(front_image_file.read(), np.uint8)
    front_image_file.seek(0)  # Reset so we can re-read below

    is_valid, error_msg = validate_front_image(cv2.imdecode(front_image_np, cv2.IMREAD_COLOR))
    if not is_valid:
        return jsonify({"error": error_msg, "pose": "front", "code": "INVALID_POSE"}), 400

    # Resolve user height
    user_height_cm = request.form.get("height_cm")
    print(user_height_cm)
    if user_height_cm:
        try:
            user_height_cm = float(user_height_cm)
        except ValueError:
            user_height_cm = DEFAULT_HEIGHT_CM
    else:
        user_height_cm = DEFAULT_HEIGHT_CM

    received_images = {
        pose_name: request.files[pose_name]
        for pose_name in ["front", "left_side"]
        if pose_name in request.files
    }

    measurements = {}
    scale_factor = None
    focal_length = FOCAL_LENGTH
    results = {}
    frames = {}

    for pose_name, image_file in received_images.items():
        image_np = np.frombuffer(image_file.read(), np.uint8)
        frame = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        frames[pose_name] = frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results[pose_name] = _holistic.process(rgb_frame)
        image_height, image_width, _ = frame.shape

        if pose_name == "front":
            if results[pose_name].pose_landmarks:
                _, scale_factor = calculate_distance_using_height(
                    results[pose_name].pose_landmarks.landmark,
                    image_height,
                    user_height_cm,
                )
            else:
                scale_factor, focal_length = detect_reference_object(frame)

        depth_map = estimate_depth(frame) if pose_name in ("front", "left_side") else None

        if results[pose_name].pose_landmarks and pose_name == "front":
            measurements.update(
                calculate_measurements(
                    results[pose_name],
                    scale_factor,
                    image_width,
                    image_height,
                    depth_map,
                    frames[pose_name],
                    user_height_cm,
                )
            )

    debug_info = {
        "scale_factor": float(scale_factor) if scale_factor else None,
        "focal_length": float(focal_length),
        "user_height_cm": float(user_height_cm),
    }

    print(measurements)
    return jsonify({"measurements": measurements, "debug_info": debug_info})
