import cv2
import numpy as np


def unique_color(img):
    mask = np.all(img != [0,0,0], axis=-1)
    only_color = img[mask]
    colors, count = np.unique(only_color, axis=0, return_counts=True)
    max_color = colors[count.argmax()]
    return max_color


def removeScore(image, color=(255, 255, 255)):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 30, 10])
    upper_red = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    # Range for upper range
    lower_red = np.array([160, 30, 10])
    upper_red = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)

    # Generating the final mask to detect red color
    BW = (mask1 + mask2) > 0
    result = image.copy()
    # result = cv2.bitwise_and(result, result, mask=BW)

    result[BW] = color

    return result

