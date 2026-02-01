"""
Mini Project 2: Real-Time Image Warping & Transformations
Course: CS5330 - Pattern Recognition and Computer Vision
"""

import cv2
import numpy as np
import time
import os

# Get script directory for saving outputs
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Transformation state
class TransformState:
    def __init__(self):
        self.reset()

    def reset(self):
        self.tx = 0  # translation x
        self.ty = 0  # translation y
        self.angle = 0  # rotation angle
        self.scale = 1.0  # scale factor
        self.flip_h = False  # horizontal flip
        self.flip_v = False  # vertical flip
        self.perspective = False  # perspective warp toggle
        self.mode = "NONE"  # current mode name


def apply_transforms(frame, state):
    """Apply all active transformations to the frame."""
    h, w = frame.shape[:2]
    result = frame.copy()

    # Apply flips
    if state.flip_h:
        result = cv2.flip(result, 1)
    if state.flip_v:
        result = cv2.flip(result, 0)

    # Apply rotation and scaling
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, state.angle, state.scale)
    result = cv2.warpAffine(result, rotation_matrix, (w, h))

    # Apply translation
    if state.tx != 0 or state.ty != 0:
        translation_matrix = np.float32([[1, 0, state.tx], [0, 1, state.ty]])
        result = cv2.warpAffine(result, translation_matrix, (w, h))

    # Apply perspective transform
    if state.perspective:
        # Define source points (corners of the image)
        src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        # Define destination points (create a perspective effect)
        margin = int(w * 0.1)
        dst_pts = np.float32([
            [margin, margin],
            [w - margin, 0],
            [w, h],
            [0, h - margin]
        ])
        perspective_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        result = cv2.warpPerspective(result, perspective_matrix, (w, h))

    return result


def get_mode_text(state):
    """Generate status text showing current transformation state."""
    parts = []
    if state.flip_h:
        parts.append("FLIP_H")
    if state.flip_v:
        parts.append("FLIP_V")
    if state.angle != 0:
        parts.append(f"rot={state.angle}")
    if state.scale != 1.0:
        parts.append(f"scale={state.scale:.2f}")
    if state.tx != 0 or state.ty != 0:
        parts.append(f"trans=({state.tx},{state.ty})")
    if state.perspective:
        parts.append("PERSPECTIVE")

    if not parts:
        return "Mode: NONE"
    return "Mode: " + " | ".join(parts)


def save_screenshot(combined, name):
    """Save a screenshot to the outputs folder."""
    output_path = os.path.join(SCRIPT_DIR, 'outputs', f'{name}.png')
    cv2.imwrite(output_path, combined)
    print(f"Screenshot saved: {output_path}")


def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    # Set resolution (optional, for better performance)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    state = TransformState()
    prev_time = time.time()
    fps = 0

    print("\n=== Real-Time Image Warping ===")
    print("Controls:")
    print("  0     - Reset all transforms")
    print("  1     - Toggle horizontal flip")
    print("  2     - Toggle vertical flip")
    print("  r/t   - Rotate left/right (±5°)")
    print("  +/-   - Scale up/down (±5%)")
    print("  Arrows- Translate image")
    print("  p     - Toggle perspective warp")
    print("  s     - Save screenshot")
    print("  q     - Quit")
    print("================================\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to read frame.")
            break

        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
        prev_time = current_time

        # Apply transformations
        transformed = apply_transforms(frame, state)

        # Create side-by-side display
        combined = np.hstack((frame, transformed))

        # Add text overlays
        mode_text = get_mode_text(state)
        cv2.putText(combined, mode_text, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(combined, f"FPS: {fps:.1f}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(combined, "Original", (10, combined.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(combined, "Transformed", (frame.shape[1] + 10, combined.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Display
        cv2.imshow("Image Warp - Press 'q' to quit", combined)

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('0'):
            state.reset()
            print("Reset all transforms")
        elif key == ord('1'):
            state.flip_h = not state.flip_h
            print(f"Horizontal flip: {state.flip_h}")
        elif key == ord('2'):
            state.flip_v = not state.flip_v
            print(f"Vertical flip: {state.flip_v}")
        elif key == ord('r'):
            state.angle += 5
            print(f"Rotation: {state.angle}°")
        elif key == ord('t'):
            state.angle -= 5
            print(f"Rotation: {state.angle}°")
        elif key == ord('+') or key == ord('='):
            state.scale += 0.05
            print(f"Scale: {state.scale:.2f}")
        elif key == ord('-') or key == ord('_'):
            state.scale = max(0.1, state.scale - 0.05)
            print(f"Scale: {state.scale:.2f}")
        elif key == ord('p'):
            state.perspective = not state.perspective
            print(f"Perspective: {state.perspective}")
        elif key == ord('s'):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            save_screenshot(combined, f"screenshot_{timestamp}")
        # Arrow keys (special keys have different codes)
        elif key == 81 or key == 2:  # Left arrow
            state.tx -= 10
            print(f"Translation: ({state.tx}, {state.ty})")
        elif key == 83 or key == 3:  # Right arrow
            state.tx += 10
            print(f"Translation: ({state.tx}, {state.ty})")
        elif key == 82 or key == 0:  # Up arrow
            state.ty -= 10
            print(f"Translation: ({state.tx}, {state.ty})")
        elif key == 84 or key == 1:  # Down arrow
            state.ty += 10
            print(f"Translation: ({state.tx}, {state.ty})")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
