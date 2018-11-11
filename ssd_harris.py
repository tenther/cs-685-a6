#!/usr/bin/env python
import cv2
import numpy as np
import sys
import pdb

def main():
    ssd_threshold = 10000000
    ssd_width = 31

    image_filenames = ['house1.jpg', 'house2.jpg']
    images          = [None,None]
    gray_images     = [None,None]
    dsts            = [None,None]
    corners         = [None,None]
    
    for idx, filename in enumerate(image_filenames):
        img          = images[idx] = cv2.imread(filename)
        gray         = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        gray         = gray_images[idx] = np.float32(gray)
        dst          = cv2.cornerHarris(gray,2,3,0.04)
        dst          = dsts[idx] = cv2.dilate(dst,None)
        corners[idx] = np.argwhere(dst>0.2*dst.max())

    pad = int(ssd_width/2)

    padded_images = [cv2.copyMakeBorder(gray_images[0], pad, pad, pad, pad, cv2.BORDER_REFLECT),
                     cv2.copyMakeBorder(gray_images[1], pad, pad, pad, pad, cv2.BORDER_REFLECT)]

    patches = [np.array([padded_images[0][x:x+2*pad+1, y:y+2*pad+1] for x, y in corners[0]]),
               np.array([padded_images[1][x:x+2*pad+1, y:y+2*pad+1] for x, y in corners[1]])]

    all_mins = []
    matches = {}
    for idx1, p1 in enumerate(corners[0]):
        patch_diffs = patches[1] - patches[0][idx1]
        ssds = np.sum(patch_diffs * patch_diffs, axis=(1,2))
        idx2 = np.argmin(ssds)
        min_ssd = ssds[idx2]
        all_mins.append(min_ssd)
        if min_ssd < ssd_threshold:
            p2 = corners[1][idx2]
            matches[tuple([p1[1], p1[0]])] = tuple([p2[1], p2[0]])
    print("min_ssd={0} max_ssd={1}".format(min(all_mins), max(all_mins)))

    combined_image = np.concatenate(tuple(images), axis=1)
    for p1, p2 in matches.items():
        p2 = list(p2)
        p2[0] += images[0].shape[1]
        cv2.line(combined_image, p1, tuple(p2), (255, 0, 0), 1)

    cv2.imwrite('house-1-2.png', combined_image)
    return matches

if __name__=='__main__':
    main()
    
