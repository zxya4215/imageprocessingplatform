import cv2 as cv
import matplotlib.pyplot as plt

def histogram(img):
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        histr = cv.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(histr, color=col)
        plt.xlim([0, 256])
    plt.title('Histogram Image')
    plt.show()

def histogramequalization(img):
    equalize_image = cv.equalizeHist(img)
    plt.hist(equalize_image.ravel(), 256, [0, 256])
    plt.title('Equalize Image')
    plt.show()