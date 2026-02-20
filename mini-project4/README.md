# Mini Project 4: Real-Time Panorama Stitching

Real-time panorama creation from manually captured webcam frames using feature-based alignment and perspective warping.

## Team Members

| Name | NEU Email |
|------|-----------|
| Ali Shehral | shehral.m@northeastern.edu |

## Setup

```bash
pip install opencv-python numpy
python panorama_lab.py
```

## Keyboard Controls

| Key | Action |
|-----|--------|
| `s` | Capture a frame |
| `a` | Stitch all captured frames into a panorama |
| `r` | Reset (clear captured frames) |
| `q` | Quit |

## Capture Method

1. Face a textured scene (bookshelves, posters, desks — not blank walls)
2. Press `s` to capture the first frame
3. Slowly pan the camera sideways, maintaining **60–70% overlap** with the previous shot
4. Press `s` to capture each subsequent frame
5. Repeat for as many frames as desired
6. Press `a` to stitch

**Tips for best results:**
- Keep the camera at a consistent height and orientation
- Avoid fast motion, zoom changes, or large lighting shifts
- Textured scenes with distinct corners/edges produce the best matches

## Algorithm Pipeline

### 1. Feature Detection (ORB)
Each captured frame is converted to grayscale, then ORB (Oriented FAST and Rotated BRIEF) detects up to 3000 keypoints and computes binary descriptors. ORB is chosen for speed and because it is rotation-invariant, making it robust to slight camera tilts between captures.

### 2. Feature Matching (BFMatcher + Lowe's Ratio Test)
Binary descriptors are matched using a Brute-Force matcher with Hamming distance. Lowe's ratio test (threshold 0.75) filters ambiguous matches: a match is kept only if the best candidate is significantly closer than the second-best. This removes many false correspondences.

### 3. Homography Estimation (RANSAC)
From the filtered matches, `cv2.findHomography` with RANSAC computes a 3x3 projective transformation mapping the new image's coordinate system onto the base image. RANSAC iteratively:
- Samples minimal point subsets
- Fits a candidate homography
- Counts inliers (matches consistent with the transform)
- Keeps the best-supported model

This rejects outlier matches that survived the ratio test.

### 4. Image Warping & Canvas Projection
The four corners of the new image are projected through the homography to determine where they land in the base image's coordinate system. A bounding box of all corners (base + warped) defines the output canvas size. A translation matrix shifts coordinates so that nothing falls at negative pixel positions. The new image is then warped into canvas space using `cv2.warpPerspective(img, T @ H, canvas_size)`.

### 5. Multi-Image Composition
Stitching is sequential: the first captured frame serves as the base. Each subsequent frame is stitched onto the growing panorama, and the result becomes the new base. This builds a panorama in a single global coordinate frame. After composition, black borders are automatically cropped.

## Failure Cases

| Symptom | Likely Cause |
|---------|-------------|
| "Not enough matches" error | Insufficient overlap or textureless scene (blank walls) |
| Distorted / warped panorama | Bad homography from incorrect matches or repetitive patterns |
| Only first image visible | Canvas/warp error — check overlap between frames |
| Black regions in output | Camera moved too far between captures |
| Misaligned seams | Perspective inconsistency from large depth variation or zoom changes |

## Demo Video

[Watch the demonstration](https://youtu.be/hXpZ4xc4Bfw)

## Screenshots

See the `outputs/` folder for saved panoramas.
