"""
Workshop Lab 5: Color Histogram
Computes and plots per-channel (BGR) histograms of a color image.
"""

import cv2
import matplotlib.pyplot as plt
import os

# Load color image (BGR by default)
img = cv2.imread(os.path.join(os.path.dirname(__file__), 'sample.jpg'))

# Plot histogram for each channel
plt.figure(figsize=(10, 5))
plt.title('Color Histogram (BGR)')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')

for i, col in enumerate(('b', 'g', 'r')):
    hist = cv2.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(hist, color=col)

plt.xlim([0, 256])
plt.legend(['Blue', 'Green', 'Red'])
plt.tight_layout()

os.makedirs(os.path.join(os.path.dirname(__file__), 'outputs'), exist_ok=True)
plt.savefig(os.path.join(os.path.dirname(__file__), 'outputs', 'color_histogram.png'), dpi=150)
plt.show()
print('Saved outputs/color_histogram.png')
