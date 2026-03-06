"""
Front-image validation using MediaPipe Holistic.
"""

import cv2
import mediapipe as mp

mp_holistic = mp.solutions.holistic


def validate_front_image(image_np):
    """Validate a front-view image before attempting measurements.

    Checks that:
    - A person is detected in the frame.
    - The full body (not just a face/selfie) is visible.
    - Key upper- and lower-body landmarks are detected with sufficient confidence.

    Args:
        image_np: BGR numpy array representing the front image.

    Returns:
        Tuple of (is_valid: bool, message: str).
    """
    try:
        rgb_frame = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        image_height, image_width = image_np.shape[:2]

        with mp_holistic.Holistic(
            static_image_mode=True,
            model_complexity=1,
            enable_segmentation=False,
            refine_face_landmarks=False,
        ) as holistic:
            results = holistic.process(rgb_frame)

        if not hasattr(results, "pose_landmarks") or not results.pose_landmarks:
            return False, "No person detected. Please make sure you're clearly visible in the frame."

        MINIMUM_LANDMARKS = [
            mp_holistic.PoseLandmark.NOSE,
            mp_holistic.PoseLandmark.LEFT_SHOULDER,
            mp_holistic.PoseLandmark.RIGHT_SHOULDER,
            mp_holistic.PoseLandmark.LEFT_ELBOW,
            mp_holistic.PoseLandmark.RIGHT_ELBOW,
            mp_holistic.PoseLandmark.RIGHT_KNEE,
            mp_holistic.PoseLandmark.LEFT_KNEE,
        ]

        missing_upper = []
        for landmark in MINIMUM_LANDMARKS:
            lm = results.pose_landmarks.landmark[landmark]
            if lm.visibility < 0.5 or not (0 <= lm.x <= 1) or not (0 <= lm.y <= 1):
                missing_upper.append(landmark.name.replace("_", " "))

        if missing_upper:
            return False, "Couldn't detect full body. Please make sure your full body is visible."

        # Guard against selfies / close-up shots (no visible torso)
        nose = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE]
        left_shoulder = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER]

        shoulder_width = abs(left_shoulder.x - right_shoulder.x) * image_width
        head_to_shoulder = abs(left_shoulder.y - nose.y) * image_height

        if shoulder_width < head_to_shoulder * 1.2:
            return False, "Please step back to show more of your upper body, not just your face."

        return True, "Validation passed - proceeding with measurements"

    except Exception as e:
        print(f"Error validating body image: {e}")
        return False, "You aren't providing images correctly. Please try again."
