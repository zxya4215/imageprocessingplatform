# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'design.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1648, 1102)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.originalImage = QtWidgets.QLabel(self.centralwidget)
        self.originalImage.setGeometry(QtCore.QRect(40, 30, 1011, 971))
        self.originalImage.setText("")
        self.originalImage.setAlignment(QtCore.Qt.AlignCenter)
        self.originalImage.setObjectName("originalImage")
        self.imageHistogramEqualization = QtWidgets.QPushButton(self.centralwidget)
        self.imageHistogramEqualization.setGeometry(QtCore.QRect(1200, 200, 93, 28))
        self.imageHistogramEqualization.setObjectName("imageHistogramEqualization")
        self.thresholding_slider = QtWidgets.QSlider(self.centralwidget)
        self.thresholding_slider.setGeometry(QtCore.QRect(1270, 110, 251, 22))
        self.thresholding_slider.setMaximum(255)
        self.thresholding_slider.setOrientation(QtCore.Qt.Horizontal)
        self.thresholding_slider.setObjectName("thresholding_slider")
        self.ROI = QtWidgets.QPushButton(self.centralwidget)
        self.ROI.setGeometry(QtCore.QRect(1200, 160, 93, 28))
        self.ROI.setObjectName("ROI")
        self.red = QtWidgets.QLabel(self.centralwidget)
        self.red.setGeometry(QtCore.QRect(1147, 110, 91, 20))
        self.red.setObjectName("red")
        self.imageThresholding = QtWidgets.QPushButton(self.centralwidget)
        self.imageThresholding.setGeometry(QtCore.QRect(1200, 240, 93, 28))
        self.imageThresholding.setObjectName("imageThresholding")
        self.thresholdinglabel = QtWidgets.QLabel(self.centralwidget)
        self.thresholdinglabel.setGeometry(QtCore.QRect(1300, 80, 70, 20))
        self.thresholdinglabel.setObjectName("thresholdinglabel")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1648, 25))
        self.menubar.setObjectName("menubar")
        self.menu = QtWidgets.QMenu(self.menubar)
        self.menu.setObjectName("menu")
        self.menu_2 = QtWidgets.QMenu(self.menubar)
        self.menu_2.setObjectName("menu_2")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionOpen_File = QtWidgets.QAction(MainWindow)
        self.actionOpen_File.setObjectName("actionOpen_File")
        self.actionROI = QtWidgets.QAction(MainWindow)
        self.actionROI.setObjectName("actionROI")
        self.actionimageHistogram = QtWidgets.QAction(MainWindow)
        self.actionimageHistogram.setObjectName("actionimageHistogram")
        self.actionGray = QtWidgets.QAction(MainWindow)
        self.actionGray.setObjectName("actionGray")
        self.actionHSV = QtWidgets.QAction(MainWindow)
        self.actionHSV.setObjectName("actionHSV")
        self.menu.addAction(self.actionOpen_File)
        self.menu_2.addAction(self.actionROI)
        self.menu_2.addAction(self.actionimageHistogram)
        self.menu_2.addAction(self.actionGray)
        self.menu_2.addAction(self.actionHSV)
        self.menubar.addAction(self.menu.menuAction())
        self.menubar.addAction(self.menu_2.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.imageHistogramEqualization.setText(_translate("MainWindow", "???????????????"))
        self.ROI.setText(_translate("MainWindow", "ROI"))
        self.red.setText(_translate("MainWindow", "???????????????"))
        self.imageThresholding.setText(_translate("MainWindow", "???????????????"))
        self.thresholdinglabel.setText(_translate("MainWindow", "0"))
        self.menu.setTitle(_translate("MainWindow", "??????"))
        self.menu_2.setTitle(_translate("MainWindow", "??????"))
        self.actionOpen_File.setText(_translate("MainWindow", "Open File"))
        self.actionROI.setText(_translate("MainWindow", "ROI"))
        self.actionimageHistogram.setText(_translate("MainWindow", "imageHistogram"))
        self.actionGray.setText(_translate("MainWindow", "Gray"))
        self.actionHSV.setText(_translate("MainWindow", "HSV"))
