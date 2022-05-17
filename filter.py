import cv2 as cv

def averaging(img):
    img_averaging = cv.blur(img, (11, 11))
    return img_averaging

def sobel(img):
    x = cv.Sobel(img, cv.CV_16S, 1, 0)
    y = cv.Sobel(img, cv.CV_16S, 0, 1)
    absX = cv.convertScaleAbs(x)
    absY = cv.convertScaleAbs(y)
    img_sobel = cv.addWeighted(absX, 0.5, absY, 0.5, 0)
    return img_sobel

def median(img):
    img_median = cv.medianBlur(img, 11)
    return img_median

def gaussian(img):
    img_gaussian = cv.GaussianBlur(img, (11, 11), -1)
    return img_gaussian

def bilateral(img):
    bilateral = cv.bilateralFilter(img, 9, 100,15)
    return bilateral

def laplacian(img):
    gar_lap = cv.Laplacian(img, cv.CV_16S, ksize=3)
    img_laplacian = cv.convertScaleAbs(gar_lap)
    return img_laplacian