import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os

img = cv.imread("..\dataset\imgs\obutterfly.jpg",0)
plt.subplot(2,2,1)
plt.title("Original Image")
plt.imshow(img)

hist,bins = np.histogram(img.flatten(),256,[0,256])
cdf = hist.cumsum()
cdf_normalized = cdf*float(hist.max()) / cdf.max()
plt.subplot(2,2,2)
plt.plot(cdf_normalized,color="b")
plt.hist(img.flatten(),256,[0,256],color="r")
plt.xlim([0,256])
plt.legend(("cdf","histogram"),loc="upper left")
plt.xlabel("Bins")

cdf_m = np.ma.masked_equal(cdf, 0)
cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf.min())
cdf = np.ma.filled(cdf_m,0).astype('uint8')
img2 = cdf[img]
plt.subplot(2,2,3)
plt.title("CDF Image")
plt.imshow(img2)


hist,bins = np.histogram(img2.flatten(),256,[0,256])
cdf = hist.cumsum()
cdf_normalized = cdf*float(hist.max()) / cdf.max()
plt.subplot(2,2,4)
plt.plot(cdf_normalized,color="b")
plt.hist(img2.flatten(),256,[0,256],color="r")
plt.xlim([0,256])
plt.legend(("cdf","Equi. Histogram"),loc="upper left")
plt.xlabel("Bins")
plt.tight_layout()



plt.show()


img = cv.imread("..\dataset\imgs\obutterfly.jpg",0)
equ = cv.equalizeHist(img)
res = np.hstack((img,equ))
cv.imwrite("result.png",res)
img = cv.imread("result.png",0)
plt.imshow(img)
plt.show()