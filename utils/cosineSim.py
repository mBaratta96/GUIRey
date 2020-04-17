import scipy.spatial.distance as dist
from skimage import feature
from utils.edgeDetection import extractDrawing
import cv2


def compute_chamfer(template, match):
    dist = cv2.distanceTransform(template, cv2.DIST_L2, 3,  dstType=cv2.CV_32F)
    return dist[match.astype('bool')].sum()

def HOGsimilarity(img1, img2):
    fd1 = feature.hog(img1, orientations=9, pixels_per_cell=(8, 8),
                                 cells_per_block=(1, 1), feature_vector=True)
    fd2 = feature.hog(img2, orientations=9, pixels_per_cell=(8, 8),
                                 cells_per_block=(1, 1), feature_vector=True)
    distance = dist.cosine(fd1, fd2)
    return distance

def prepareImages(input1, input2):
    blur = cv2.GaussianBlur(input1, (5, 5), 0)
    ret3, im1 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    height, width = im1.shape
    im2 = extractDrawing(input2)
    im2 = cv2.resize(im2, (width, height))
    return im1, im2

def typeImages(input1,input2):
    im1 = input1.astype('float32')
    im1 /= 255
    im2 = input2.astype('float32')
    im2 /= 255
    return im1, im2

def chamfer(template, img_input):
    im1, im2 = prepareImages(template, img_input)
    edges1 = cv2.Canny(im1, 100, 200)
    edges2 = cv2.Canny(im2, 100, 200)
    result = 0.5 * (compute_chamfer(edges1, edges2) + compute_chamfer(edges2, edges1))
    return result

def cosine(template, img_input):
    im1, im2 = prepareImages(template, img_input)
    im1, im2 = typeImages(im1, im2)
    result = dist.cosine(im2.ravel(), im1.ravel())
    return result

def hausdorff(template, img_input):
    im1, im2 = prepareImages(template, img_input)
    im1, im2 = typeImages(im1, im2)
    result = dist.directed_hausdorff(im2, im1)
    return result[0]

def histGrad(template, img_input):
    im1, im2 = prepareImages(template, img_input)
    result = HOGsimilarity(im1, im2)
    return result
