import numpy as np
from matplotlib import pyplot as plt

import cv2

image1 = cv2.imread("../dataset/imgs/obutterfly.jpg")

#Remove the below line if it shows Differnt color of the input Data
image = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

print(image.shape)
plt.subplot(121)
plt.imshow(image)
plt.title("Original Image")
plt.show()

#Segmenting

pixel_values = image.reshape((-1,3))
pixel_values = np.float32(pixel_values)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,100,0.85)
retral , Labels , centers = cv2.kmeans(pixel_values,3,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

centers = np.uint(centers)
segmented_data = centers[Labels.flatten()]
segmented_image = segmented_data.reshape(image.shape)
plt.subplot(122)
plt.imshow(segmented_image)
plt.title("Clustered Image")

plt.show()