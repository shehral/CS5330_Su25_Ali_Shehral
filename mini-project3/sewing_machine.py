"""
Mini Project 3: Sewing Machine - Real-Time Feature Detection & Matching
Course: CS5330 - Pattern Recognition and Computer Vision

Demonstrates SIFT feature detection with FLANN-based matching between
two video sources (2-camera) or a webcam and a static image (1-camera simulation).
"""

import cv2
import numpy as np
import time
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    # Allow camera index override: python sewing_machine.py [cam_index]
    cam_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    # --- 1. Initialize Video Sources ---
    cap_l = cv2.VideoCapture(cam_index)
    if not cap_l.isOpened():
        print("ERROR: Could not open webcam.")
        return

    cap_l.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap_l.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    cap_r = cv2.VideoCapture(1)
    simulation_mode = False

    if not cap_r.isOpened():
        print("Second camera not found. Switching to 1-Camera Simulation Mode...")
        simulation_mode = True
        # Look for a static image to match against
        static_path = os.path.join(SCRIPT_DIR, 'self.jpg')
        if not os.path.exists(static_path):
            # Try common fallback names
            for name in ['self.png', 'logo.jpg', 'logo.png', 'target.jpg', 'target.png']:
                alt = os.path.join(SCRIPT_DIR, name)
                if os.path.exists(alt):
                    static_path = alt
                    break
        static_img = cv2.imread(static_path)
        if static_img is None:
            print(f"ERROR: Static image not found at {static_path}")
            print("Place self.jpg (or logo.jpg) in the mini-project3 folder.")
            cap_l.release()
            return

    # --- 2. Configure SIFT Detector ---
    sift = cv2.SIFT_create()

    # --- 3. Setup FLANN Matcher (KD-Tree for SIFT) ---
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    prev_time = time.time()
    fps = 0
    ratio_threshold = 0.7

    print("\n=== Sewing Machine: Feature Matching ===")
    print("Controls:")
    print("  s     - Save screenshot")
    print("  +/-   - Adjust ratio threshold (±0.05)")
    print("  q     - Quit")
    print("=========================================\n")

    while True:
        # --- 4a. Capture ---
        ret_l, frame_l = cap_l.read()
        if not ret_l:
            print("ERROR: Failed to read from webcam.")
            break

        if simulation_mode:
            h, w = frame_l.shape[:2]
            frame_r = cv2.resize(static_img, (w, h))
        else:
            ret_r, frame_r = cap_r.read()
            if not ret_r:
                print("ERROR: Failed to read from second camera.")
                break

        # --- 4b. Convert to Grayscale & Detect/Compute ---
        gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)

        kp_l, des_l = sift.detectAndCompute(gray_l, None)
        kp_r, des_r = sift.detectAndCompute(gray_r, None)

        # --- 4c. Match & Filter ---
        good_matches = []
        if des_l is not None and des_r is not None and len(des_l) >= 2 and len(des_r) >= 2:
            matches = flann.knnMatch(des_l, des_r, k=2)

            # Lowe's Ratio Test
            for pair in matches:
                if len(pair) == 2:
                    m, n = pair
                    if m.distance < ratio_threshold * n.distance:
                        good_matches.append(m)

        # --- 5. Visualization ---
        match_img = cv2.drawMatches(
            frame_l, kp_l, frame_r, kp_r, good_matches, None,
            matchColor=(0, 255, 0),
            singlePointColor=(255, 0, 0),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        # FPS calculation
        current_time = time.time()
        dt = current_time - prev_time
        fps = 1 / dt if dt > 0 else 0
        prev_time = current_time

        # Overlay text
        mode_label = "1-CAM SIMULATION" if simulation_mode else "2-CAM LIVE"
        cv2.putText(match_img, f"Mode: {mode_label}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(match_img, f"Matches: {len(good_matches)} | Ratio: {ratio_threshold:.2f}",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(match_img, f"FPS: {fps:.1f}", (10, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(match_img, f"Keypoints: L={len(kp_l)} R={len(kp_r)}",
                    (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Sewing Machine - Press 'q' to quit", match_img)

        # --- Handle Input ---
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('s'):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            out_path = os.path.join(SCRIPT_DIR, 'outputs', f'screenshot_{timestamp}.png')
            cv2.imwrite(out_path, match_img)
            print(f"Screenshot saved: {out_path}")
        elif key == ord('+') or key == ord('='):
            ratio_threshold = min(0.95, ratio_threshold + 0.05)
            print(f"Ratio threshold: {ratio_threshold:.2f}")
        elif key == ord('-') or key == ord('_'):
            ratio_threshold = max(0.1, ratio_threshold - 0.05)
            print(f"Ratio threshold: {ratio_threshold:.2f}")

    # Cleanup
    cap_l.release()
    if not simulation_mode:
        cap_r.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
