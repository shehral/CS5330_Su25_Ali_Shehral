"""
Workshop Lab 5: Grayscale Histogram
Computes and plots the intensity histogram of a grayscale image.
"""

import cv2
import matplotlib.pyplot as plt
import os

# Load image in grayscale (flag 0)
img = cv2.imread(os.path.join(os.path.dirname(__file__), 'sample.jpg'), 0)

# Compute histogram: 256 bins covering intensity range [0, 256)
hist = cv2.calcHist([img], [0], None, [256], [0, 256])

# Plot
plt.figure(figsize=(10, 5))
plt.title('Grayscale Histogram')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.plot(hist, color='black')
plt.xlim([0, 256])
plt.tight_layout()

os.makedirs(os.path.join(os.path.dirname(__file__), 'outputs'), exist_ok=True)
plt.savefig(os.path.join(os.path.dirname(__file__), 'outputs', 'grayscale_histogram.png'), dpi=150)
plt.show()
print('Saved outputs/grayscale_histogram.png')
