# Mini Project 3: Sewing Machine - Feature Detection & Matching

Real-time SIFT feature detection and correspondence application that "sews" two viewpoints together by finding and visualizing matching keypoints.

## Team Members

| Name | NEU Email |
|------|-----------|
| Ali Shehral | shehral.m@northeastern.edu |

## Setup

```bash
pip install opencv-contrib-python numpy
python sewing_machine.py
```

**Note:** Place a static image named `self.jpg` in this folder for 1-camera simulation mode. The program auto-detects whether a second camera is available.

## Modes

- **2-Camera Mode**: Automatically activates when two webcams are detected. Matches features between live feeds.
- **1-Camera Simulation**: Uses webcam + a static image (`self.jpg`). Activates when only one camera is found.

## Keyboard Controls

| Key | Action |
|-----|--------|
| `+` | Increase ratio threshold (+0.05) |
| `-` | Decrease ratio threshold (-0.05) |
| `s` | Save screenshot |
| `q` | Quit |

## Features

- SIFT keypoint detection with 128-element descriptors
- FLANN-based matching with KD-Tree indexing
- Lowe's Ratio Test for filtering reliable matches
- Real-time FPS counter and match count overlay
- Adjustable ratio threshold for tuning match quality
- Side-by-side visualization with match lines

## Screenshots

See the `outputs/` folder for saved screenshots.
