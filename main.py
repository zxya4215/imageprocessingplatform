import sys, os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QFileDialog, QMainWindow
from PyQt5.QtCore import *
from design import *

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.setup_picture()

    def setup_picture(self):
        self.actionOpen_File.triggered.connect(self.openFile)
        self.actionSave_File.triggered.connect(self.saveFile)
        self.action_information.triggered.connect(self.information)
        self.actionROI.triggered.connect(self.select_ROI)
        self.actionimageHistogram.triggered.connect(self.pictureHistogram)
        self.actionGray_2.triggered.connect(self.imageGary)
        self.actionHSV_2.triggered.connect(self.imageHSV)
        self.imageHistogramEqualization.clicked.connect(self.histogramEqualization)
        self.ROI.clicked.connect(self.select_ROI)
        self.imageThresholding.clicked.connect(self.pictureThresholding)
        self.thresholding_slider.valueChanged[int].connect(self.adjustThreshold)
        self.angle_Slider.valueChanged[int].connect(self.adjustThreshold)
        self.action_averaging.triggered.connect(self.averaging)
        self.action_sobel.triggered.connect(self.sobel)
        self.action_median.triggered.connect(self.median)
        self.action_gaussian.triggered.connect(self.gaussian)
        self.action_bilateral.triggered.connect(self.bilateral)
        self.action_laplacian.triggered.connect(self.laplacian)
        self.action_translation.triggered.connect(self.translation)
        self.action_flip.triggered.connect(self.flip)
        self.action_affline.triggered.connect(self.affline)
        self.action_perspective.triggered.connect(self.label_mouse_Event)
        self.perspective_pst_dst = []
        self.perspective_counter = 4

    def label_mouse_Event(self):
        self.originalImage.mousePressEvent = self.mousepress
        self.originalImage.mouseMoveEvent = self.mousemove
        self.originalImage.mouseReleaseEvent = self.perspective_get_clicked_position

    def mousemove(self, event):
        self.x1, self.y1 = event.x(), event.y()

    def mousepress(self, event):
        self.x0, self.y0 = event.x(), event.y()

    def openFile(self):
        self.filename, _ = QFileDialog.getOpenFileName(self, 'Open Image', 'Image', '*.jpg *.png *.bmp')
        self.img = self.filename
        self.abspath = os.path.abspath(self.img)
        self.showImage()

    def saveFile(self):
        if self.filename == "":
            self.imageShape.setText('無檔案儲存')
            return 0
        else:
            savepath, file_type = QtWidgets.QFileDialog.getSaveFileName(self, "Image_Save", 'Image', '*.jpg *.png *.bmp')
            if savepath == "":
                self.imageShape.setText('取消檔案儲存')
                return 0
            else:
                cv.imwrite(savepath, self.change_img)
                self.imageShape.setText('檔案儲存')

    def information(self):
        height, width, channel = self.originalimg.shape
        self.imageShape.setText('影像大小: ' + str(width) + ' X ' + str(height))
        self.imagePath.setText(self.abspath)
        self.imageSize.setText('圖檔大小: ' + str(round(os.stat(self.abspath).st_size/1e+6, 3)) + 'MB')

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
        color = ('b', 'g', 'r')
        for i, col in enumerate(color):
            histr = cv.calcHist([self.originalimg], [i], None, [256], [0, 256])
            plt.plot(histr, color=col)
            plt.xlim([0, 256])
        plt.title('Histogram Image')
        plt.show()

    def histogramEqualization(self):
        gray = cv.cvtColor(self.originalimg, cv.COLOR_BGR2GRAY)
        equalize_image = cv.equalizeHist(gray)
        height, width = gray.shape
        bytesPerline = 1 * width
        self.qImg = QImage(equalize_image, width, height, bytesPerline, QImage.Format_Grayscale8).rgbSwapped()
        self.qpixmap = QPixmap.fromImage(self.qImg)
        self.changeImage.setPixmap(QPixmap.fromImage(self.qImg))
        self.change_img = equalize_image
        plt.hist(equalize_image.ravel(), 256, [0, 256])
        plt.title('Equalize Image')
        plt.show()

    def imageGary(self):
        gray = cv.cvtColor(self.originalimg, cv.COLOR_BGR2GRAY)
        height, width = gray.shape
        bytesPerline = 1 * width
        self.qImg = QImage(gray, width, height, bytesPerline, QImage.Format_Grayscale8).rgbSwapped()
        self.qpixmap = QPixmap.fromImage(self.qImg)
        self.changeImage.setPixmap(QPixmap.fromImage(self.qImg))
        self.change_img = gray

    def imageHSV(self):
        hsv = cv.cvtColor(self.originalimg, cv.COLOR_BGR2HSV)
        height, width, channel = self.originalimg.shape
        bytesPerline = 3 * width
        self.qImg = QImage(hsv, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        self.qpixmap = QPixmap.fromImage(self.qImg)
        self.changeImage.setPixmap(QPixmap.fromImage(self.qImg))
        self.change_img = hsv

    def pictureThresholding(self):
        gray = cv.cvtColor(self.originalimg, cv.COLOR_BGR2GRAY)
        ret, thresholding = cv.threshold(gray, self.thresholding_slider.value(), 255, cv.THRESH_BINARY)
        height, width = gray.shape
        bytesPerline = 1 * width
        self.qImg = QImage(thresholding, width, height, bytesPerline, QImage.Format_Grayscale8).rgbSwapped()
        self.qpixmap = QPixmap.fromImage(self.qImg)
        self.changeImage.setPixmap(QPixmap.fromImage(self.qImg))
        self.change_img = thresholding

    def adjustThreshold(self):
        self.thresholdinglabel.setText(str(self.thresholding_slider.value()))
        self.imageThresholding.clicked.connect(self.pictureThresholding)
        self.anglelabel.setText(str(self.angle_Slider.value()))
        self.action_rotate.triggered.connect(self.rotate)

    def averaging(self):
        img_averaging = cv.blur(self.originalimg, (11, 11))
        height, width, channel = self.originalimg.shape
        bytesPerline = 3 * width
        self.qImg = QImage(img_averaging, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        self.qpixmap = QPixmap.fromImage(self.qImg)
        self.changeImage.setPixmap(QPixmap.fromImage(self.qImg))
        self.change_img = img_averaging

    def sobel(self):
        x = cv.Sobel(self.originalimg, cv.CV_16S, 1, 0)
        y = cv.Sobel(self.originalimg, cv.CV_16S, 0, 1)
        absX = cv.convertScaleAbs(x)
        absY = cv.convertScaleAbs(y)
        img_sobel = cv.addWeighted(absX, 0.5, absY, 0.5, 0)
        height, width, channel = self.originalimg.shape
        bytesPerline = 3 * width
        self.qImg = QImage(img_sobel, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        self.qpixmap = QPixmap.fromImage(self.qImg)
        self.changeImage.setPixmap(QPixmap.fromImage(self.qImg))
        self.change_img = img_sobel

    def median(self):
        img_median = cv.medianBlur(self.originalimg, 11)
        height, width, channel = self.originalimg.shape
        bytesPerline = 3 * width
        self.qImg = QImage(img_median, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        self.qpixmap = QPixmap.fromImage(self.qImg)
        self.changeImage.setPixmap(QPixmap.fromImage(self.qImg))
        self.change_img = img_median

    def gaussian(self):
        img_gaussian = cv.GaussianBlur(self.originalimg, (11, 11), -1)
        height, width, channel = self.originalimg.shape
        bytesPerline = 3 * width
        self.qImg = QImage(img_gaussian, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        self.qpixmap = QPixmap.fromImage(self.qImg)
        self.changeImage.setPixmap(QPixmap.fromImage(self.qImg))
        self.change_img = img_gaussian

    def bilateral(self):
        img_bilateral = cv.bilateralFilter(self.originalimg, 9, 100, 15)
        height, width, channel = self.originalimg.shape
        bytesPerline = 3 * width
        self.qImg = QImage(img_bilateral, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        self.qpixmap = QPixmap.fromImage(self.qImg)
        self.changeImage.setPixmap(QPixmap.fromImage(self.qImg))
        self.change_img = img_bilateral

    def laplacian(self):
        gray_lap = cv.Laplacian(self.originalimg, cv.CV_16S, ksize=3)
        img_laplacian = cv.convertScaleAbs(gray_lap)
        height, width, channel = self.originalimg.shape
        bytesPerline = 3 * width
        self.qImg = QImage(img_laplacian, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        self.qpixmap = QPixmap.fromImage(self.qImg)
        self.changeImage.setPixmap(QPixmap.fromImage(self.qImg))
        self.change_img = img_laplacian

    def rotate(self):
        height, width, channel = self.originalimg.shape
        #cv.getRotationMatrix2D(中心點座標, 旋轉的角度, 圖片的縮放倍率)
        matrix = cv.getRotationMatrix2D((width / 2.0, height / 2.0), self.angle_Slider.value(), 1)
        img_rotate = cv.warpAffine(self.originalimg, matrix, (width, height))
        bytesPerline = 3 * width
        self.qImg = QImage(img_rotate, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        self.qpixmap = QPixmap.fromImage(self.qImg)
        self.changeImage.setPixmap(QPixmap.fromImage(self.qImg))
        self.change_img = img_rotate

    def translation(self):
        rightleft = int(self.rightleft_textEdit.toPlainText())
        toplower = int(self.toplowertextEdit.toPlainText())
        matrix = np.float32([[1, 0, rightleft], [0, 1, toplower]])
        height, width, channel = self.originalimg.shape
        img_shifted = cv.warpAffine(self.originalimg, matrix, (width, height))
        bytesPerline = 3 * width
        self.qImg = QImage(img_shifted, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        self.qpixmap = QPixmap.fromImage(self.qImg)
        self.changeImage.setPixmap(QPixmap.fromImage(self.qImg))
        self.change_img = img_shifted

    def flip(self):
        flip = int(self.flip_textEdit.toPlainText())
        flip_img = cv.flip(self.originalimg, flip)
        height, width, channel = self.originalimg.shape
        bytesPerline = 3 * width
        self.qImg = QImage(flip_img, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        self.qpixmap = QPixmap.fromImage(self.qImg)
        self.changeImage.setPixmap(QPixmap.fromImage(self.qImg))
        self.change_img = flip_img

    def affline(self):
        rightleft = int(self.rightleft_textEdit.toPlainText())
        toplower = int(self.toplowertextEdit.toPlainText())
        zoom_rightleft = float(self.zoom_rightleft_textEdit.toPlainText())
        zoom_toplower = float(self.zoom_toplower_textEdit.toPlainText())
        if rightleft == None:
            rightleft = 0
        if toplower == None:
            toplower = 0
        matrix = np.float32([[1, zoom_rightleft, rightleft], [zoom_toplower, 1, toplower]])
        height, width, channel = self.originalimg.shape
        img_affline = cv.warpAffine(self.originalimg, matrix, (width, height))
        bytesPerline = 3 * width
        self.qImg = QImage(img_affline, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        self.qpixmap = QPixmap.fromImage(self.qImg)
        self.changeImage.setPixmap(QPixmap.fromImage(self.qImg))
        self.change_img = img_affline

    def perspective(self):
        originalimg = self.originalimg
        self.perspective_pst_dst = np.array(self.perspective_pst_dst, dtype=np.float32)
        mapping_matrix, status = cv.findHomography(self.perspective_pst_src, self.perspective_pst_dst)
        img_transformed = cv.warpPerspective(originalimg, mapping_matrix, (originalimg.shape[1], originalimg.shape[0]))
        return img_transformed

    def perspective_get_clicked_position(self, event):
        height, width = self.originalimg.shape[:2]
        self.perspective_pst_src = np.array([
            [0, 0],
            [width - 1, 0],
            [0, height - 1],
            [width - 1, height - 1]
        ], dtype=np.float32)

        if self.perspective_counter > 0:
            self.imageShape.setText('')
            self.imagePath.setText('')
            self.x, self.y = event.x(), event.y()
            self.perspective_pst_dst.append([self.x, self.y])
            self.imageSize.setText(str(self.perspective_pst_dst[:]))
            self.perspective_counter -= 1
            if self.perspective_counter == 0:
                transform_image = self.perspective()
                transform_image = cv.cvtColor(transform_image, cv.COLOR_BGR2RGB)
                plt.imshow(transform_image)
                plt.axis('off')
                plt.show()
                self.perspective_pst_dst =[]
                self.perspective_counter = 4

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
