import sys, os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import changecolor
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QFileDialog, QMainWindow

import filter
import histogram
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
        self.action_averaging_2.triggered.connect(self.averaging)
        self.action_sobel_2.triggered.connect(self.sobel)
        self.action_median_2.triggered.connect(self.median)
        self.action_gaussian_2.triggered.connect(self.gaussian)
        self.action_bilateral_2.triggered.connect(self.bilateral)
        self.action_laplacian_2.triggered.connect(self.laplacian)
        self.action_translation_2.triggered.connect(self.translation)
        self.action_flip_2.triggered.connect(self.flip)
        self.action_affline_2.triggered.connect(self.affline)
        self.action_rotate_2.triggered.connect(self.rotate)
        self.action_perspective_2.triggered.connect(self.label_mouse_Event)
        self.harrisSlider.valueChanged[int].connect(self.harriscorner)
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

    def set_rgb_label(self, img):
        height, width, channel = img.shape
        bytesPerline = 3 * width
        self.qImg = QImage(img, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        self.qpixmap = QPixmap.fromImage(self.qImg)
        self.qpixmap_height = self.qpixmap.height()
        self.changeImage.setPixmap(QPixmap.fromImage(self.qImg))

    def set_gray_label(self, img):
        height, width = img.shape
        bytesPerline = 1 * width
        self.qImg = QImage(img, width, height, bytesPerline, QImage.Format_Grayscale8).rgbSwapped()
        self.qpixmap = QPixmap.fromImage(self.qImg)
        self.changeImage.setPixmap(QPixmap.fromImage(self.qImg))

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
        self.imagePath.setText('影像位置： ' + str(self.abspath))
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
        changecolor.select_roi(self.originalimg)

    def pictureHistogram(self):
        changecolor.histogram(self.originalimg)

    def histogramEqualization(self):
        gray = changecolor.image_Gray(self.originalimg)
        histogram.histogramequalization(gray)
        self.set_gray_label(gray)

    def imageGary(self):
        gray = changecolor.image_Gray(self.originalimg)
        self.set_gray_label(gray)

    def imageHSV(self):
        hsv = changecolor.image_HSV(self.originalimg)
        self.set_rgb_label(hsv)

    def pictureThresholding(self):
        gray = cv.cvtColor(self.originalimg, cv.COLOR_BGR2GRAY)
        ret, thresholding = cv.threshold(gray, self.thresholding_slider.value(), 255, cv.THRESH_BINARY)
        self.set_gray_label(thresholding)

    def adjustThreshold(self):
        self.thresholdinglabel.setText(str(self.thresholding_slider.value()))
        self.imageThresholding.clicked.connect(self.pictureThresholding)
        self.anglelabel.setText(str(self.angle_Slider.value()))
        self.action_rotate.triggered.connect(self.rotate)

    def averaging(self):
        img_averaging = filter.averaging(self.originalimg)
        self.set_rgb_label(img_averaging)

    def sobel(self):
        img_sobel = filter.sobel(self.originalimg)
        self.set_rgb_label(img_sobel)

    def median(self):
        img_median = filter.median(self.originalimg)
        self.set_rgb_label(img_median)

    def gaussian(self):
        img_gaussian = filter.gaussian(self.originalimg)
        self.set_rgb_label(img_gaussian)

    def bilateral(self):
        img_bilateral = filter.bilateral(self.originalimg)
        self.set_rgb_label(img_bilateral)

    def laplacian(self):
        img_laplacian = filter.laplacian(self.originalimg)
        self.set_rgb_label(img_laplacian)

    def rotate(self):
        height, width, channel = self.originalimg.shape
        #cv.getRotationMatrix2D(中心點座標, 旋轉的角度, 圖片的縮放倍率)
        matrix = cv.getRotationMatrix2D((width / 2.0, height / 2.0), self.angle_Slider.value(), 1)
        img_rotate = cv.warpAffine(self.originalimg, matrix, (width, height))
        self.set_rgb_label(img_rotate)

    def translation(self):
        rightleft = int(self.rightleft_textEdit.toPlainText())
        toplower = int(self.toplowertextEdit.toPlainText())
        matrix = np.float32([[1, 0, rightleft], [0, 1, toplower]])
        height, width, channel = self.originalimg.shape
        img_shifted = cv.warpAffine(self.originalimg, matrix, (width, height))
        self.set_rgb_label(img_shifted)

    def flip(self):
        flip = int(self.flip_textEdit.toPlainText())
        img_flip = cv.flip(self.originalimg, flip)
        self.set_rgb_label(img_flip)

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
        self.set_rgb_label(img_affline)

    def perspective_get_clicked_position(self, event):
        displayed_size = self.originalImage.size()
        width, height = displayed_size.width(), displayed_size.height()
        displayed_image = cv.resize(self.originalimg, (width, height), interpolation=cv.INTER_AREA)
        self.perspective_pst_src = np.float32([
            [0, 0],
            [width, 0],
            [0, height],
            [width, height]
        ])

        if self.perspective_counter > 0:
            self.imageShape.setText('')
            self.imagePath.setText('')
            self.x, self.y = event.x(), event.y()
            self.perspective_pst_dst.append([self.x, self.y])
            self.imageSize.setText(str(self.perspective_pst_dst[:]))
            self.perspective_counter -= 1
            if self.perspective_counter == 0:
                self.perspective_pst_dst = np.float32(self.perspective_pst_dst)
                mapping_matrix = cv.getPerspectiveTransform(self.perspective_pst_dst, self.perspective_pst_src)
                img_transformed = cv.warpPerspective(displayed_image, mapping_matrix, (width, height), cv.INTER_LINEAR)
                transform_image = cv.cvtColor(img_transformed, cv.COLOR_BGR2RGB)
                plt.imshow(transform_image)
                plt.axis('off')
                plt.show()
                self.perspective_pst_dst = []
                self.perspective_counter = 4

    def harriscorner(self):
        self.harrislabel.setText(str(self.harrisSlider.value()))
        gray = changecolor.image_Gray(self.originalimg)
        #設定參數
        blockSize = 2
        apertureSize = 3
        k = 0.04
        #檢測
        dst = cv.cornerHarris(gray, blockSize, apertureSize, k)
        #normalizing
        dst_norm = np.empty(dst.shape, dtype=np.float32)
        cv.normalize(dst, dst_norm, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
        dst_norm_scaled = cv.convertScaleAbs(dst_norm)

        for i in range(dst_norm.shape[0]):
            for j in range(dst_norm.shape[1]):
                if int(dst_norm[i, j]) > self.harrisSlider.value():
                    cv.circle(dst_norm_scaled, (j, i), 5, (0), 2)
        self.set_gray_label(dst_norm_scaled)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
