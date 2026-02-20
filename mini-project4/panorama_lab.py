"""
Mini Project 4: Real-Time Panorama Stitching
Course: CS5330 - Pattern Recognition and Computer Vision

Builds a panoramic image from manually captured frames using a live camera feed.
Pipeline: ORB Detection → BF Matching → RANSAC Homography → Perspective Warp → Composition
"""

import cv2
import numpy as np
import time
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)
MIN_MATCH_COUNT = 15


def detect_and_match(img1, img2):
    """Detect ORB features and match between two images.

    Returns matched keypoints (src_pts, dst_pts) or (None, None) on failure.
    src_pts are from img2, dst_pts are from img1 — so the homography maps img2 → img1.
    """
    orb = cv2.ORB_create(nfeatures=3000)

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    if des1 is None or des2 is None:
        print("  [WARN] No descriptors found in one of the images.")
        return None, None

    # BFMatcher with Hamming distance (appropriate for ORB binary descriptors)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des2, des1, k=2)

    # Lowe's ratio test to filter ambiguous matches
    good = []
    for pair in matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good.append(m)

    print(f"  Matches found: {len(good)} (need {MIN_MATCH_COUNT})")

    if len(good) < MIN_MATCH_COUNT:
        return None, None

    src_pts = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    return src_pts, dst_pts


def compute_homography(src_pts, dst_pts):
    """Estimate homography with RANSAC.

    Returns the 3x3 homography matrix or None on failure.
    """
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None:
        print("  [WARN] Homography estimation failed.")
        return None

    inliers = mask.ravel().sum()
    print(f"  RANSAC inliers: {inliers}/{len(mask)}")
    return H


def stitch_pair(base, new_img):
    """Stitch new_img onto base using feature matching + homography + warping.

    Returns the stitched panorama or None on failure.
    """
    src_pts, dst_pts = detect_and_match(base, new_img)
    if src_pts is None:
        return None

    H = compute_homography(src_pts, dst_pts)
    if H is None:
        return None

    h_base, w_base = base.shape[:2]
    h_new, w_new = new_img.shape[:2]

    # Project corners of new_img through the homography to find canvas bounds
    corners_new = np.float32([
        [0, 0], [w_new, 0], [w_new, h_new], [0, h_new]
    ]).reshape(-1, 1, 2)
    warped_corners = cv2.perspectiveTransform(corners_new, H)

    # Combine with base image corners
    corners_base = np.float32([
        [0, 0], [w_base, 0], [w_base, h_base], [0, h_base]
    ]).reshape(-1, 1, 2)
    all_corners = np.concatenate((corners_base, warped_corners), axis=0)

    # Compute bounding box of the full panorama
    [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel())
    [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel())

    # Translation matrix to shift everything into positive coordinates
    translation = np.array([
        [1, 0, -x_min],
        [0, 1, -y_min],
        [0, 0, 1]
    ], dtype=np.float64)

    canvas_w = x_max - x_min
    canvas_h = y_max - y_min

    # Warp new_img into canvas space (translation @ H)
    warped = cv2.warpPerspective(new_img, translation @ H, (canvas_w, canvas_h))

    # Paste base image onto canvas at the translated position
    x_off = -x_min
    y_off = -y_min

    # Place base image onto canvas — base takes priority to avoid ghosting
    roi = warped[y_off:y_off + h_base, x_off:x_off + w_base]
    base_mask = (base > 0).any(axis=2)
    roi[base_mask] = base[base_mask]

    return crop_black_borders(warped)


def crop_black_borders(img):
    """Remove black borders around the stitched panorama."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img

    # Find bounding rect of the largest contour
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    return img[y:y + h, x:x + w]


def main():
    cam_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    captured_frames = []
    prev_time = time.time()
    fps = 0

    print("\n=== Real-Time Panorama Stitching ===")
    print("Controls:")
    print("  s  - Capture a frame")
    print("  a  - Stitch panorama from captured frames")
    print("  r  - Reset (clear all captured frames)")
    print("  q  - Quit")
    print("====================================\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to read from webcam.")
            break

        # FPS
        current_time = time.time()
        dt = current_time - prev_time
        fps = 1 / dt if dt > 0 else 0
        prev_time = current_time

        # Draw HUD on display copy (not on captured frames)
        display = frame.copy()
        cv2.putText(display, f"Captured: {len(captured_frames)} frames",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(display, f"FPS: {fps:.1f}",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(display, "s:capture  a:stitch  r:reset  q:quit",
                    (10, display.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Panorama Lab - Live Preview", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('s'):
            captured_frames.append(frame.copy())
            print(f"Frame {len(captured_frames)} captured.")

        elif key == ord('a'):
            if len(captured_frames) < 2:
                print("Need at least 2 frames to stitch. Keep capturing!")
                continue

            print(f"\nStitching {len(captured_frames)} frames...")
            panorama = captured_frames[0]

            success = True
            for i in range(1, len(captured_frames)):
                print(f"  Stitching frame {i + 1} onto panorama...")
                result = stitch_pair(panorama, captured_frames[i])
                if result is None:
                    print(f"  [FAIL] Could not stitch frame {i + 1}. "
                          "Ensure 60-70% overlap and textured scenes.")
                    success = False
                    break
                panorama = result

            if success:
                # Display result
                cv2.imshow("Panorama Result", panorama)
                # Save
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                out_path = os.path.join(OUTPUT_DIR, f'panorama_{timestamp}.png')
                cv2.imwrite(out_path, panorama)
                print(f"Panorama saved: {out_path}")

        elif key == ord('r'):
            captured_frames.clear()
            print("Cleared all captured frames.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
