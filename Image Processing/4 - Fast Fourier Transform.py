import cv2
import numpy as np
from matplotlib import pyplot as plt

img_c1 = cv2.imread("..\dataset\imgs\ship.jpg",0)
img_c2 = np.fft.fft2(img_c1)
img_c3 = np.fft.fftshift(img_c2)
img_c4 = np.fft.ifftshift(img_c3)
img_c5 = np.fft.ifft2(img_c4)

plt.figure(figsize=(6.4*5,4.8*5),constrained_layout=False)
plt.subplot(151)
plt.imshow(img_c1,"gray")
plt.title("Original Image")
#)

plt.subplot(152)
plt.imshow(np.log(1+np.abs(img_c2)),"gray")
plt.title("Spectrum")


plt.subplot(153)
plt.imshow(np.log(1+np.abs(img_c3)),"gray")
plt.title("Centered Spectrum")

plt.subplot(154)
plt.imshow(np.log(1+np.abs(img_c4)),"gray")
plt.title("Decentralized")


plt.subplot(155)
plt.imshow(np.abs(img_c5),"gray")
plt.title("Processed Image")
plt.show()