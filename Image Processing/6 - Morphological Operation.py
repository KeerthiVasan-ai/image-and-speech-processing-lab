import cv2
import numpy as np

import matplotlib.pyplot as plt

img = cv2.imread("..\dataset\imgs\letter1.png")

#Kernal 5x5 Matrix 

kernal = np.ones((3,3),np.uint8)

erosion = cv2.erode(img,kernel=kernal,iterations=1)
dilation = cv2.dilate(img,kernel=kernal,iterations=1)

plt.subplot(131)
plt.imshow(img)
plt.title("Original Image")

plt.subplot(132)
plt.imshow(erosion)
plt.title("Eroded Image")

plt.subplot(133)
plt.imshow(dilation)
plt.title("Dilated Image")