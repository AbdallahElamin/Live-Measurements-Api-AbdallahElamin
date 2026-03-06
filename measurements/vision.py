"""
Contour-based body width detection.
"""

import cv2


def get_body_width_at_height(frame, height_px, center_x):
    """Scan horizontally at a specific height to find body edges.

    Args:
        frame:     BGR numpy array of the image.
        height_px: Row index (pixels from top) to scan.
        center_x:  Normalised x position [0, 1] to start scanning from.

    Returns:
        Width in pixels between the detected left and right body edges.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY)

    # Clamp height_px to image bounds
    if height_px >= frame.shape[0]:
        height_px = frame.shape[0] - 1

    horizontal_line = thresh[height_px, :]
    center_x = int(center_x * frame.shape[1])
    left_edge, right_edge = center_x, center_x

    # Scan from centre to left
    for i in range(center_x, 0, -1):
        if horizontal_line[i] == 0:
            left_edge = i
            break

    # Scan from centre to right
    for i in range(center_x, len(horizontal_line)):
        if horizontal_line[i] == 0:
            right_edge = i
            break

    width_px = right_edge - left_edge

    # Apply a minimum width of 10% of the image width to avoid degenerate results
    min_width = 0.1 * frame.shape[1]
    if width_px < min_width:
        width_px = min_width

    return width_px
