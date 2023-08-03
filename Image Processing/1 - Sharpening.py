import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('.\dataset\imgs\lady2.jpg',1)
filter = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
sharpen_img = cv2.filter2D(img,-1,filter)
plt.subplot(121)
plt.imshow(img)
plt.title("Original")
plt.xticks([])
plt.yticks([])
plt.subplot(122)
plt.imshow(sharpen_img)
plt.title("Sharpened")
plt.xticks([])
plt.yticks([])
plt.show()