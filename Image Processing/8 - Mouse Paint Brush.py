# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 12:06:05 2023

@author: GF LAB
"""

import cv2
import numpy as np


def draw_circle(event,x,y,flags,param):
    if(event == cv2.EVENT_LBUTTONDBLCLK):
        cv2.circle(img,(x,y),100,(255,255,0),-1)
        
img = np.zeros((512,512,3),np.uint8)
cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_circle)
while(1):
    cv2.imshow('image',img)
    keycode = cv2.waitKey(1)
    if cv2.getWindowProperty("image",cv2.WND_PROP_VISIBLE) <1:
        break
cv2.destroyAllWindows()