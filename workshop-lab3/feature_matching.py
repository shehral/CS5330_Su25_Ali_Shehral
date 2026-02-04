"""
Workshop 3: Feature Matching (SIFT + FLANN)
============================================
Goal: Connect the keypoints found in image1 to the corresponding
keypoints in image2 using FLANN matcher and Lowe's Ratio Test.

FLANN (Fast Library for Approximate Nearest Neighbors):
- Uses KD-Tree for fast high-dimensional searches
- Much faster than brute-force matching for large feature sets

Lowe's Ratio Test:
- Compares best match distance to second-best match distance
- If best < 0.7 * second_best, it's likely a good match
- This eliminates ambiguous matches that could be noise
"""

import cv2
import sys
import os

# Create outputs directory if it doesn't exist
os.makedirs('outputs', exist_ok=True)

# 1. Load the original, clean images in grayscale
img1 = cv2.imread('image1.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('image2.jpg', cv2.IMREAD_GRAYSCALE)

if img1 is None:
    sys.exit("Could not find image1.jpg. Ensure it's in the same directory as this script.")
if img2 is None:
    sys.exit("Could not find image2.jpg. Ensure it's in the same directory as this script.")

print(f"Image 1 shape: {img1.shape}")
print(f"Image 2 shape: {img2.shape}")

# 2. SIFT Detection for both images
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

print(f"Keypoints in image 1: {len(kp1)}")
print(f"Keypoints in image 2: {len(kp2)}")

# 3. Initialize FLANN Matcher
# Index parameters for KD-Tree (algorithm=1 means FLANN_INDEX_KDTREE)
# trees=5: number of parallel KD-trees to use (more trees = more accuracy but slower)
index_params = dict(algorithm=1, trees=5)

# Search parameters
# checks=50: number of times the tree(s) should be recursively traversed
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

# 4. k-Nearest Neighbors matching
# k=2: find the 2 best matches for each descriptor
# This is needed for Lowe's Ratio Test
matches = flann.knnMatch(des1, des2, k=2)

# 5. Apply Lowe's Ratio Test
# Keep matches where the best match is significantly better than the second best
# ratio < 0.7 is a common threshold (some use 0.75 or 0.8)
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

print(f"Total matches found: {len(matches)}")
print(f"Good matches after ratio test: {len(good_matches)}")

# 6. Draw the matches
# This creates a visualization with both images side by side
# and lines connecting matched keypoints
result = cv2.drawMatches(
    img1, kp1,
    img2, kp2,
    good_matches,
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

# Save the output
cv2.imwrite('outputs/feature_matching.jpg', result)

# Display the result
window_name = 'Workshop 3 - Feature Matching'
cv2.imshow(window_name, result)
print("\nFeature Matching complete!")
print(f"Match rate: {len(good_matches)/len(matches)*100:.1f}% of matches passed ratio test")
print(f"Output saved to: outputs/feature_matching.jpg")
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
