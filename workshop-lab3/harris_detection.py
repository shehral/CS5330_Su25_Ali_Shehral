"""
Workshop 1: Harris Corner Detection
====================================
Goal: Identify corners by calculating a "corner score" (R) based on
the local gradient of the image.

Harris Corner Detection works by:
1. Computing image gradients (Ix, Iy) using Sobel operators
2. Building a structure tensor M for each pixel neighborhood
3. Computing corner response R = det(M) - k * trace(M)^2
4. Corners have high R values; edges have negative R; flat regions ~0
"""

import cv2
import numpy as np
import os

# Create outputs directory if it doesn't exist
os.makedirs('outputs', exist_ok=True)

# Load image and convert to grayscale
img = cv2.imread('image1.jpg')
if img is None:
    raise FileNotFoundError("Could not load image1.jpg - ensure it's in the same directory")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Convert to float32 for Harris detection (Crucial Step)
# Harris uses gradient calculations that require floating point precision
gray = np.float32(gray)

# Apply Harris Corner Detection
# Parameters:
#   blockSize=2: neighborhood size for corner detection (2x2 window)
#   ksize=3: aperture parameter for Sobel derivative (3x3 kernel)
#   k=0.04: Harris detector free parameter (typically 0.04-0.06)
dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)

# Dilate to make corners more visible
# This enlarges the detected corner points for better visualization
dst = cv2.dilate(dst, None)

# Threshold for marking corners in red
# Only mark pixels where corner response > 1% of maximum response
# This filters out weak corners and keeps strong ones
threshold = 0.01 * dst.max()
img[dst > threshold] = [0, 0, 255]  # BGR format: Red color

# Save the output
cv2.imwrite('outputs/harris_corners.jpg', img)

# Display the result
window_name = 'Workshop 1 - Harris Corners'
cv2.imshow(window_name, img)
print(f"Harris Corner Detection complete!")
print(f"Threshold used: {threshold:.6f}")
print(f"Max corner response: {dst.max():.6f}")
print(f"Output saved to: outputs/harris_corners.jpg")
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
