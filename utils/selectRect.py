import cv2
import numpy as np
import os
p_dst = [(382, 219),(852, 219), (852, 537), (382, 537)]

def computeHomograpy(image, points):
    mask = np.ones(5, dtype=int)
    mask[2]=0
    img = cv2.imread(os.path.join(os.getcwd(), 'templates', 'original_rey22.png'))
    right_points = np.array(points)[np.ma.make_mask(mask)]
    hm, status = cv2.findHomography(np.array(right_points), np.array(p_dst))
    nH, nW, _ = img.shape
    im_dst =  cv2.warpPerspective(image, hm, (nW, nH))
    return im_dst

def computeHomographyRhomb(image, points):
    img = image.copy()
    point_rhomb = points[2] + (1,)
    mask = np.ones(5, dtype=int)
    mask[2] = 0
    right_points = np.array(points)[np.ma.make_mask(mask)]
    hm, status = cv2.findHomography(np.array(right_points), np.array(p_dst))
    new_point = np.dot(hm, point_rhomb)
    new_point = tuple(np.round(new_point/new_point[2]).astype(int))
    center = new_point[0:2]
    return center




