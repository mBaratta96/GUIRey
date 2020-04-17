import numpy as np
import cv2
from matplotlib import pyplot as plt


def maxDeviationThresh(hist):
    maximum = max(hist)
    index_max = list(hist).index(maximum)
    index_min = 0
    for i in range(index_max, -1, -1):
        if not hist[i]:
            index_min = i
            break
    distances = []
    x1 = index_min
    y1 = hist[index_min]
    x2 = index_max
    y2 = hist[index_max]
    for i in range(index_min + 1, index_max):
        x0 = i
        y0 = hist[i]
        distance = np.abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1) / np.sqrt(
            (y2 - y1) ** 2 + (x2 - x1) ** 2)
        distances.append(distance)

    T_index = distances.index(max(distances))
    return T_index + index_min


def extractDrawing(img):
    dst = img.copy()
    max_occ = np.bincount(dst[dst>0]).argmax()
    dst[dst == 0] = max_occ
    hist, n, _ = plt.hist(dst.ravel(), 256, [0, 256])
    plt.close('all')
    thresh_val = maxDeviationThresh(hist)
    threshed = np.ones(dst.shape, np.uint8)*255
    mask = dst < thresh_val
    threshed[mask] = 0
    return threshed

if __name__ == '__main__':
    img = cv2.imread('./images/img_homography.png')
    extractDrawing(img)