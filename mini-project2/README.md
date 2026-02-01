# Mini Project 2: Real-Time Image Warping

Real-time image processing application using OpenCV that captures live webcam feed and applies geometric transformations.

## Team Members

| Name | NEU Email |
|------|-----------|
| Ali Shehral | shehral.m@northeastern.edu |

## Setup

```bash
pip install opencv-python numpy
python image_warp.py
```

## Keyboard Controls

| Key | Action |
|-----|--------|
| `0` | Reset all transforms |
| `1` | Toggle horizontal flip |
| `2` | Toggle vertical flip |
| `r` | Rotate left (+5°) |
| `t` | Rotate right (-5°) |
| `+` | Scale up (+5%) |
| `-` | Scale down (-5%) |
| Arrow keys | Translate image |
| `p` | Toggle perspective warp |
| `s` | Save screenshot |
| `q` | Quit |

## Features

- Live webcam capture with side-by-side display (original + transformed)
- Real-time FPS counter
- Multiple geometric transformations:
  - Horizontal/Vertical flip
  - Rotation (5° increments)
  - Scaling (5% increments)
  - Translation (arrow keys)
  - Perspective warp

## Demo Video

[Watch the demo on YouTube](https://youtu.be/iX5mEiQrG3s)

## Screenshots

See the `outputs/` folder for screenshots of each transformation mode.
