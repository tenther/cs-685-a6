#!/usr/bin/env python
import cv2
import numpy as np
import pdb
import sys

def draw_keypoints(img, keypoints, color=(0,255,255)):
    for kp in keypoints:
        x,y = kp.pt
        cv2.circle(img, (int(x), int(y)), int(kp.size), color)

def main(filenames, sift_threshold):
    sift = cv2.xfeatures2d.SIFT_create()
    images = [None, None]
    kps    = [None, None]
    descs  = [None, None]

    for idx, filename in enumerate(filenames):
        images[idx] = img = cv2.imread(filename)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        kp, desc = sift.detectAndCompute(gray,None)
        kps[idx] = kp
        descs[idx] = desc

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descs[0], descs[1])
    matches = sorted(matches, key = lambda x:x.distance)
    matches = [m for m in matches if m.distance <= sift_threshold]

    match_image = cv2.drawMatches(images[0], kps[0], images[1], kps[1], matches, None, flags=2)
    output_file_name = "{0}_{1}_sift_match_{2:03.1f}.png".format(filenames[0].replace('.jpg', ''), filenames[1].replace('.jpg', ''), sift_threshold)
    cv2.imwrite(output_file_name, match_image)

if __name__=='__main__':
    filename1 = sys.argv[1]
    filename2 = sys.argv[2]
    sift_threshold = 40.0
    if len(sys.argv) > 3:
        sift_threshold = float(sys.argv[3])
    main([filename1, filename2],sift_threshold)
    
