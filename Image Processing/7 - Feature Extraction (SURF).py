import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread("./dataset/imgs/ship.jpg",0)
plt.imshow(img)

surf = cv2.xfeatures2d.SIFT_create(200)
kp, des = surf.detectAndCompute(img,None)
# kp, des = surf.detectAndCompute(img,None)
print(len(kp))
# print(surf.getHessianThreshold())

# surf.setHessainThreshold(50000)
# kp,des = surf.detectAndCompute(img,None)
# print(len(kp))