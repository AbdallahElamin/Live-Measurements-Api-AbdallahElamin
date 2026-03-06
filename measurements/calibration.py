"""
Camera calibration and scale-factor computation helpers.
"""

import cv2
import mediapipe as mp

from config import KNOWN_OBJECT_WIDTH_CM, FOCAL_LENGTH

mp_pose = mp.solutions.pose


def calibrate_focal_length(detected_width_px):
    """Dynamically calibrates focal length using a known reference object.

    Args:
        detected_width_px: Detected width of the reference object in pixels.

    Returns:
        Estimated focal length (falls back to the default if width is zero).
    """
    if detected_width_px:
        return (detected_width_px * FOCAL_LENGTH) / KNOWN_OBJECT_WIDTH_CM
    return FOCAL_LENGTH


def detect_reference_object(image):
    """Detect the largest contour in the image and use it as a reference object.

    Returns:
        Tuple of (scale_factor, focal_length).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        focal_length = calibrate_focal_length(w)
        scale_factor = KNOWN_OBJECT_WIDTH_CM / w
        return scale_factor, focal_length

    return 0.05, FOCAL_LENGTH


def calculate_distance_using_height(landmarks, image_height, user_height_cm):
    """Calculate camera distance and pixel-to-cm scale factor from known height.

    Args:
        landmarks:       MediaPipe pose landmark list.
        image_height:    Image height in pixels.
        user_height_cm:  Actual user height in centimetres.

    Returns:
        Tuple of (distance_cm, scale_factor).
    """
    top_head = landmarks[mp_pose.PoseLandmark.NOSE.value].y * image_height
    bottom_foot = max(
        landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y,
        landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y,
    ) * image_height

    person_height_px = abs(bottom_foot - top_head)

    # distance = (actual_height_cm * focal_length) / height_in_pixels
    distance = (user_height_cm * FOCAL_LENGTH) / person_height_px
    scale_factor = user_height_cm / person_height_px

    return distance, scale_factor
