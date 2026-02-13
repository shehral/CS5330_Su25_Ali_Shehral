# Workshop Lab 5: Histogram Analysis

## Files
- `grayscale_histogram.py` - Computes and plots grayscale intensity histogram
- `color_histogram.py` - Computes and plots per-channel BGR color histograms
- `sample.jpg` - Sample image used for analysis

## How to Run
```
python grayscale_histogram.py
python color_histogram.py
```

## Observations
- The grayscale histogram shows two main peaks around intensity 90 and 220, indicating a bimodal image with both dark and bright regions.
- The color histogram reveals that the three channels peak at different intensities, showing the image has distinct color regions rather than being uniformly colored.
- A peak on the left side means the image is mostly dark in that region/channel.
- Two distinct peaks (bimodal) suggest the image has two dominant brightness levels, like a foreground and background with different intensities.
