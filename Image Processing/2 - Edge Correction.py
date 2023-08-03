import numpy as np
import cv2
from matplotlib import pyplot as plt

filter = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
img = cv2.imread("..\dataset\imgs\monkey.jpg",1)

sobel_vertical_img = cv2.filter2D(img,-1,filter)
sobel_hori_img = cv2.filter2D(img,-1,np.flip(filter.T,axis=0))

plt.subplot(121)
plt.imshow(sobel_vertical_img, cmap='gray')
plt.title("Vertical Edge")

plt.subplot(122)
plt.imshow(sobel_hori_img, cmap='gray')
plt.title("Horizontal Edge")

plt.show()