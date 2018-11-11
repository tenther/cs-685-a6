#!/usr/bin/env python
import cv2
import numpy as np
import sys
import pdb

# This is largely copied from this OpenCV tutorial page:
# https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_features_harris/py_features_harris.html

threshold = .2

filename = sys.argv[1]
img = cv2.imread(filename)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

gray = np.float32(gray)
dst = cv2.cornerHarris(gray,2,3,0.04)

dst = cv2.dilate(dst,None)

img[dst>threshold*dst.max()]=[0,0,255]

out_filename = filename.replace('.jpg', '-harris-{0:02.1f}.png'.format(threshold))

cv2.imwrite(out_filename, img)


