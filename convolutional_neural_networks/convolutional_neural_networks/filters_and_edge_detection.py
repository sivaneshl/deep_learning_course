import cv2
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

K = np.array([[ -1, -1, -1, -1, -1],
              [ -1, -1, 4, -1, -1],
              [ -1, 4, 4, 4, -1],
              [ -1, -1, 4, -1, -1],
              [ -1, -1, -1, -1, -1]])

image = mpimg.imread('curved_lane.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
# plt.imshow(gray, cmap='gray')
filtered_image = cv2.filter2D(gray, -1, K)
plt.imshow(filtered_image, cmap='gray')
plt.show()