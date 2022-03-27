import sys
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QDialog, QGridLayout, QLabel, QPushButton, QFileDialog, QMainWindow
from design import *

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.setup_picture()

    def setup_picture(self):
        self.actionOpen_File.triggered.connect(self.openFile)
        self.actionROI.triggered.connect(self.select_ROI)
        self.actionimageHistogram.triggered.connect(self.pictureHistogram)
        self.actionGray.triggered.connect(self.imageGary)
        self.actionHSV.triggered.connect(self.imageHSV)
        self.imageHistogramEqualization.clicked.connect(self.histogramEqualization)
        self.ROI.clicked.connect(self.select_ROI)
        self.imageThresholding.clicked.connect(self.pictureThresholding)
        self.thresholding_slider.valueChanged[int].connect(self.adjustThreshold)

    def openFile(self):
        filename, _ = QFileDialog.getOpenFileName(self, 'Open Image', 'Image', '*.jpg *.png *.bmp')
        self.img = filename
        self.showImage()

    def showImage(self):
        self.originalimg = cv.imread(self.img)
        height, width, channel = self.originalimg.shape
        bytesPerline = 3 * width
        self.qImg = QImage(self.originalimg, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        self.qpixmap = QPixmap.fromImage(self.qImg)
        self.qpixmap_height = self.qpixmap.height()
        self.originalImage.setPixmap(QPixmap.fromImage(self.qImg))

    def resize_image(self):
        scaled_pixmap = self.qpixmap.scaledToHeight(self.qpixmap_height)
        self.originalImage.setPixmap(scaled_pixmap)

    def select_ROI(self):
        img = self.originalimg
        showCrosshair = False
        fromCenter = False
        r = cv.selectROI("image", img, showCrosshair, fromCenter)
        imgCrop = img[int(r[1]): int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
        cv.imshow("ROI_selection", imgCrop)
        cv.waitKey(0)

    def pictureHistogram(self):
        gray = cv.cvtColor(self.originalimg, cv.COLOR_BGR2GRAY)
        plt.hist(gray.ravel(), 256, [0, 256])
        plt.title('Histogram Image')
        plt.show()

    def histogramEqualization(self):
        gray = cv.cvtColor(self.originalimg, cv.COLOR_BGR2GRAY)
        equalize_image = cv.equalizeHist(gray)
        plt.hist(equalize_image.ravel(), 256, [0, 256])
        plt.title('Equalize Image')
        plt.show()

    def imageGary(self):
        gray = cv.cvtColor(self.originalimg, cv.COLOR_BGR2GRAY)
        cv.imshow('Gary Image', gray)

    def imageHSV(self):
        hsv = cv.cvtColor(self.originalimg, cv.COLOR_BGR2HSV)
        cv.imshow('Gary Image', hsv)

    def pictureThresholding(self):
        gray = cv.cvtColor(self.originalimg, cv.COLOR_BGR2GRAY)
        ret, thresholding = cv.threshold(gray, self.thresholding_slider.value(), 255, cv.THRESH_BINARY)
        cv.imshow("Thresholding", thresholding)


    def adjustThreshold(self):
        self.thresholdinglabel.setText(str(self.thresholding_slider.value()))
        self.imageThresholding.clicked.connect(self.pictureThresholding)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
