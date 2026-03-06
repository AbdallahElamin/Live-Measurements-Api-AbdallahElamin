"""
Body measurement calculations from MediaPipe pose landmarks.
"""

import numpy as np
import mediapipe as mp

from measurements.calibration import calculate_distance_using_height
from measurements.vision import get_body_width_at_height

mp_pose = mp.solutions.pose

# Depth map resolution used by MiDaS
_DEPTH_RES = 384


def calculate_measurements(results, scale_factor, image_width, image_height,
                            depth_map, frame=None, user_height_cm=None):
    """Compute all body measurements from MediaPipe pose results.

    Args:
        results:          MediaPipe holistic/pose results object.
        scale_factor:     Initial pixel-to-cm scale factor.
        image_width:      Image width in pixels.
        image_height:     Image height in pixels.
        depth_map:        2-D numpy array from MiDaS (or None).
        frame:            Original BGR frame for contour detection (or None).
        user_height_cm:   Known user height in cm for improved scale accuracy.

    Returns:
        dict mapping measurement name -> value in cm.
    """
    landmarks = results.pose_landmarks.landmark

    # Refine scale factor from the user's known height when available
    if user_height_cm:
        _, scale_factor = calculate_distance_using_height(landmarks, image_height, user_height_cm)

    # Pre-compute depth-map axis scales (relative to 384x384 MiDaS output)
    scale_y = _DEPTH_RES / image_height
    scale_x = _DEPTH_RES / image_width

    def pixel_to_cm(value):
        return round(value * scale_factor, 2)

    def calculate_circumference(width_px, depth_ratio=1.0):
        """Estimate circumference via elliptical approximation."""
        width_cm = width_px * scale_factor
        estimated_depth_cm = width_cm * depth_ratio * 0.7
        half_width = width_cm / 2
        half_depth = estimated_depth_cm / 2
        return round(2 * np.pi * np.sqrt((half_width ** 2 + half_depth ** 2) / 2), 2)

    def _depth_ratio_at(x_px, y_px):
        """Sample the depth map and return a depth ratio (1.0 if unavailable)."""
        if depth_map is None:
            return 1.0
        ys = int(y_px * scale_y)
        xs = int(x_px * scale_x)
        if 0 <= ys < _DEPTH_RES and 0 <= xs < _DEPTH_RES:
            depth = depth_map[ys, xs]
            max_depth = np.max(depth_map)
            return 1.0 + 0.5 * (1.0 - depth / max_depth)
        return 1.0

    measurements = {}

    # ── Shoulder Width ───────────────────────────────────────────────────────
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    shoulder_width_px = abs(left_shoulder.x * image_width - right_shoulder.x * image_width)
    shoulder_width_px *= 1.1  # 10% correction
    measurements["shoulder_width"] = pixel_to_cm(shoulder_width_px)

    # ── Chest / Bust ─────────────────────────────────────────────────────────
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

    chest_y_ratio = 0.15
    chest_y = left_shoulder.y + (left_hip.y - left_shoulder.y) * chest_y_ratio
    chest_width_px = abs((right_shoulder.x - left_shoulder.x) * image_width) * 1.15

    if frame is not None:
        chest_y_px = int(chest_y * image_height)
        center_x = (left_shoulder.x + right_shoulder.x) / 2
        detected = get_body_width_at_height(frame, chest_y_px, center_x)
        if detected > 0:
            chest_width_px = max(chest_width_px, detected)

    chest_center_x = ((left_shoulder.x + right_shoulder.x) / 2) * image_width
    chest_center_y = chest_y * image_height
    chest_depth_ratio = _depth_ratio_at(chest_center_x, chest_center_y)

    measurements["chest_width"] = pixel_to_cm(chest_width_px)
    measurements["chest_circumference"] = calculate_circumference(chest_width_px, chest_depth_ratio)

    # ── Waist ────────────────────────────────────────────────────────────────
    waist_y_ratio = 0.35
    waist_y = left_shoulder.y + (left_hip.y - left_shoulder.y) * waist_y_ratio

    if frame is not None:
        waist_y_px = int(waist_y * image_height)
        center_x = (left_hip.x + right_hip.x) / 2
        detected = get_body_width_at_height(frame, waist_y_px, center_x)
        waist_width_px = detected if detected > 0 else abs(right_hip.x - left_hip.x) * image_width * 0.9
    else:
        waist_width_px = abs(right_hip.x - left_hip.x) * image_width * 0.9

    waist_width_px *= 1.16  # correction factor

    waist_center_x = ((left_hip.x + right_hip.x) / 2) * image_width
    waist_center_y = waist_y * image_height
    waist_depth_ratio = _depth_ratio_at(waist_center_x, waist_center_y)

    measurements["waist_width"] = pixel_to_cm(waist_width_px)
    measurements["waist"] = calculate_circumference(waist_width_px, waist_depth_ratio)

    # ── Hip ──────────────────────────────────────────────────────────────────
    hip_width_px = abs(left_hip.x * image_width - right_hip.x * image_width) * 1.35

    if frame is not None:
        hip_y_offset = 0.1
        left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
        hip_y = left_hip.y + (left_knee.y - left_hip.y) * hip_y_offset
        hip_y_px = int(hip_y * image_height)
        center_x = (left_hip.x + right_hip.x) / 2
        detected = get_body_width_at_height(frame, hip_y_px, center_x)
        if detected > 0:
            hip_width_px = max(hip_width_px, detected)

    hip_center_x = ((left_hip.x + right_hip.x) / 2) * image_width
    hip_center_y = left_hip.y * image_height
    hip_depth_ratio = _depth_ratio_at(hip_center_x, hip_center_y)

    measurements["hip_width"] = pixel_to_cm(hip_width_px)
    measurements["hip"] = calculate_circumference(hip_width_px, hip_depth_ratio)

    # ── Neck ─────────────────────────────────────────────────────────────────
    neck = landmarks[mp_pose.PoseLandmark.NOSE.value]
    left_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR.value]
    neck_width_px = abs(neck.x * image_width - left_ear.x * image_width) * 2.0
    measurements["neck"] = calculate_circumference(neck_width_px, 1.0)
    measurements["neck_width"] = pixel_to_cm(neck_width_px)

    # ── Arm Length ───────────────────────────────────────────────────────────
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    sleeve_length_px = abs(left_shoulder.y * image_height - left_wrist.y * image_height)
    measurements["arm_length"] = pixel_to_cm(sleeve_length_px)

    # ── Shirt Length ─────────────────────────────────────────────────────────
    shirt_length_px = abs(left_shoulder.y * image_height - left_hip.y * image_height) * 1.2
    measurements["shirt_length"] = pixel_to_cm(shirt_length_px)

    # ── Thigh ────────────────────────────────────────────────────────────────
    left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
    thigh_y_ratio = 0.2
    thigh_y = left_hip.y + (left_knee.y - left_hip.y) * thigh_y_ratio
    thigh_width_px = hip_width_px * 0.5 * 1.2  # base estimate with correction

    if frame is not None:
        thigh_y_px = int(thigh_y * image_height)
        thigh_x = left_hip.x * 0.9
        detected = get_body_width_at_height(frame, thigh_y_px, thigh_x)
        if detected > 0 and detected < hip_width_px:
            thigh_width_px = detected

    thigh_center_x = left_hip.x * image_width
    thigh_center_y = thigh_y * image_height
    thigh_depth_ratio = _depth_ratio_at(thigh_center_x, thigh_center_y)

    measurements["thigh"] = pixel_to_cm(thigh_width_px)
    measurements["thigh_circumference"] = calculate_circumference(thigh_width_px, thigh_depth_ratio)

    # ── Trouser Length ───────────────────────────────────────────────────────
    left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
    trouser_length_px = abs(left_hip.y * image_height - left_ankle.y * image_height)
    measurements["trouser_length"] = pixel_to_cm(trouser_length_px)

    return measurements
