"""
Workshop 2: SIFT Keypoint Detection
=====================================
Goal: Detect "blobs" and local patterns (keypoints) that are mathematically
described by a 128-dimensional vector (descriptor).

SIFT (Scale-Invariant Feature Transform) provides:
- Scale invariance: detects features at multiple scales
- Rotation invariance: assigns orientation to each keypoint
- Distinctive descriptors: 128-dim vectors for matching
"""

import cv2
import os

# Create outputs directory if it doesn't exist
os.makedirs('outputs', exist_ok=True)

# Load image
img = cv2.imread('image1.jpg')
if img is None:
    raise FileNotFoundError("Could not load image1.jpg - ensure it's in the same directory")

# Convert to grayscale (SIFT requires 8-bit grayscale, NOT float32)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Initialize SIFT detector
# SIFT_create() is available in opencv-contrib-python
sift = cv2.SIFT_create()

# Detect keypoints and compute descriptors
# keypoints: list of KeyPoint objects with location, scale, orientation
# descriptors: Nx128 array where N is number of keypoints
keypoints, descriptors = sift.detectAndCompute(gray, None)

# Draw rich keypoints (showing size and orientation)
# DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS draws circles indicating:
#   - Keypoint size (circle radius = scale)
#   - Keypoint orientation (line from center)
img_sift = cv2.drawKeypoints(
    gray,
    keypoints,
    img,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

# Save the output
cv2.imwrite('outputs/sift_keypoints.jpg', img_sift)

# Display the result
window_name = 'Workshop 2 - SIFT Keypoints'
cv2.imshow(window_name, img_sift)
print(f"SIFT Keypoint Detection complete!")
print(f"Number of keypoints detected: {len(keypoints)}")
if descriptors is not None:
    print(f"Descriptor shape: {descriptors.shape} (keypoints x 128-dim vectors)")
print(f"Output saved to: outputs/sift_keypoints.jpg")
print("Press any key or close window to exit...")

# Loop until window is closed or key is pressed (fixes macOS issue)
while True:
    key = cv2.waitKey(100) & 0xFF  # Check every 100ms
    if key != 255:  # Any key pressed
        break
    # Check if window was closed via X button
    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        break

cv2.destroyAllWindows()
