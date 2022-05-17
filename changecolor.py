import cv2 as cv
import matplotlib.pyplot as plt

def image_Gray(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return gray

def image_HSV(img):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    return hsv

def select_roi(img):
    showCrosshair = False
    fromCenter = False
    r = cv.selectROI("image", img, showCrosshair, fromCenter)
    imgCrop = img[int(r[1]): int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
    cv.imshow("ROI_selection", imgCrop)
    cv.waitKey(0)

