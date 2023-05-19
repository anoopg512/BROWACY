#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Guide: open conda tensorflow_cpu : cd to the folder save file.ui: comment: pyuic5 -x test.ui -o test.py
# convert ipynb to py ### comment: ipython nbconvert --to script "PyQt5_v10.ipynb"
# convert py to exe:
# # comment in conda with direction from file.py: pip install pyinstaller
                                                # pyinstaller PyQt5_v10.py


from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QTextEdit, QDialog
from PyQt5 import uic
import sys
import cv2

from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QImage
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QTimer
from PyQt5 import QtCore, QtGui, QtWidgets

from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time

from detector import detect_frame 

buffer = 32

greenLower = (29, 86, 6)
greenUpper = (64, 255, 255)
#greenUpper = (225, 100, 70)

pts = deque(maxlen=buffer)
counter = 0
(dX, dY) = (0, 0)
direction = ""

time.sleep(2.0)

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(900, 600)
        self.imgLabel_1 = QtWidgets.QLabel(Dialog)
        self.imgLabel_1.setGeometry(QtCore.QRect(150, 80, 471, 441))
        self.imgLabel_1.setAutoFillBackground(False)
        self.imgLabel_1.setFrameShape(QtWidgets.QFrame.Box)
        self.imgLabel_1.setFrameShadow(QtWidgets.QFrame.Raised)
        self.imgLabel_1.setLineWidth(2)
        self.imgLabel_1.setScaledContents(True)
        self.imgLabel_1.setObjectName("imgLabel_1")
        self.SHOW = QtWidgets.QPushButton(Dialog)
        self.SHOW.setGeometry(QtCore.QRect(10, 80, 71, 31))
        self.SHOW.setObjectName("SHOW")
        self.CAPTURE = QtWidgets.QPushButton(Dialog)
        self.CAPTURE.setGeometry(QtCore.QRect(10, 120, 131, 51))
        self.CAPTURE.setObjectName("CAPTURE")
        self.TEXT = QtWidgets.QTextBrowser(Dialog)
        self.TEXT.setGeometry(QtCore.QRect(10, 10, 256, 61))
        self.TEXT.setObjectName("TEXT")
        self.imgLabel_2 = QtWidgets.QLabel(Dialog)
        self.imgLabel_2.setGeometry(QtCore.QRect(630, 80, 220, 220))
        self.imgLabel_2.setFrameShape(QtWidgets.QFrame.Box)
        self.imgLabel_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.imgLabel_2.setLineWidth(2)
        self.imgLabel_2.setScaledContents(True)
        self.imgLabel_2.setObjectName("imgLabel_2")
        self.imgLabel_3 = QtWidgets.QLabel(Dialog)
        self.imgLabel_3.setGeometry(QtCore.QRect(630, 305, 220, 215))
        self.imgLabel_3.setFrameShape(QtWidgets.QFrame.Box)
        self.imgLabel_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.imgLabel_3.setLineWidth(2)
        self.imgLabel_3.setScaledContents(True)
        self.imgLabel_3.setObjectName("imgLabel_3")
        # self.imgLabel_4 = QtWidgets.QLabel(Dialog)
        # self.imgLabel_4.setGeometry(QtCore.QRect(630, 400, 151, 121))
        # self.imgLabel_4.setFrameShape(QtWidgets.QFrame.Box)
        # self.imgLabel_4.setFrameShadow(QtWidgets.QFrame.Raised)
        # self.imgLabel_4.setLineWidth(2)
        # self.imgLabel_4.setScaledContents(True)
        # self.imgLabel_4.setObjectName("imgLabel_4")
        self.TEXT_2 = QtWidgets.QTextBrowser(Dialog)
        self.TEXT_2.setGeometry(QtCore.QRect(10, 310, 131, 31))
        font = QtGui.QFont()
        font.setPointSize(9)
        # self.TEXT_2.setFont(font)
        # self.TEXT_2.setObjectName("TEXT_2")
        # self.TEXT_3 = QtWidgets.QTextBrowser(Dialog)
        # self.TEXT_3.setGeometry(QtCore.QRect(10, 270, 101, 31))
        # self.TEXT_3.setObjectName("TEXT_3")
        # self.TEXT_4 = QtWidgets.QTextBrowser(Dialog)
        # self.TEXT_4.setGeometry(QtCore.QRect(10, 310, 101, 31))
        # self.TEXT_4.setObjectName("TEXT_4")
        # self.TEXT_5 = QtWidgets.QTextBrowser(Dialog)
        # self.TEXT_5.setGeometry(QtCore.QRect(10, 350, 101, 31))
        # self.TEXT_5.setObjectName("TEXT_5")
        self.TEXT_6 = QtWidgets.QTextBrowser(Dialog)
        self.TEXT_6.setGeometry(QtCore.QRect(90, 80, 51, 31))
        self.TEXT_6.setObjectName("TEXT_6")
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(20, 289, 111, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label.setFont(font)
        self.label.setObjectName("label")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.imgLabel_1.setText(_translate("Dialog", "TextLabel"))
        self.SHOW.setText(_translate("Dialog", "Show"))
        self.CAPTURE.setText(_translate("Dialog", "Capture Screen Shot"))
        self.imgLabel_2.setText(_translate("Dialog", "TextLabel"))
        self.imgLabel_3.setText(_translate("Dialog", "TextLabel"))
        #self.imgLabel_4.setText(_translate("Dialog", "TextLabel"))
        self.label.setText(_translate("Dialog", "IOU SCORE"))


class MainWindow(QWidget):
    # class constructor
    def __init__(self):
        # call QWidget constructor
        super().__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)

        # create a timer
        self.timer = QTimer()
        # set timer timeout callback function
        self.timer.timeout.connect(self.viewCam)
        
        self.ui.SHOW.clicked.connect(self.controlTimer)
        
        self.logic = 0
        self.value = 1
        self.ui.CAPTURE.clicked.connect(self.CaptureClicked)
    
    def CaptureClicked(self):
        self.logic =2
        
    def viewCam(self):
        self.ui.TEXT.setText('Kindly Press "Capture Image" to capture image')
        
        # self.cap = cv2.VideoCapture(0)
        # start timer
        #self.timer.start(20)
        # read image in BGR format
        ret, image = self.cap.read()
        
        # convert image to RGB format
        image = imutils.resize(image, width=600)
        
        
        # convert image to RGB format 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        right_area, left_area, iou_score, img_both_eye, img_corrected = detect_frame(image)

        iou_score = "{:.4f}".format(iou_score)
        self.ui.TEXT_2.setText(iou_score)

        # get image infos
        height, width, channel = img_corrected.shape
        step = channel * width
        # create QImage from image
        qImg = QImage(img_corrected.data, width, height, step, QImage.Format_RGB888)
        # show image in img_label
        self.ui.imgLabel_1.setPixmap(QPixmap.fromImage(qImg))


        
        
        # get image blurred
        blurred = img_both_eye
        hsv = img_corrected 

        height_2, width_2, channel_2 = blurred.shape
        step_2 = channel_2 * width_2
        qImg_2 =  QImage(blurred.data, width_2, height_2, step_2, QImage.Format_RGB888)
        self.ui.imgLabel_2.setPixmap(QPixmap.fromImage(qImg_2))
        
        # get image hsv
        height_3, width_3, channel_3 = image.shape
        step_3 = channel_3 * width_3
        qImg_3 =  QImage(image.data, width_3, height_3, step_3, QImage.Format_RGB888)
        self.ui.imgLabel_3.setPixmap(QPixmap.fromImage(qImg_3))
        
        # # get image mask
        # height_4, width_4= mask.shape
        # step_4 = 3 * width_4                                                      # dont need step -> int bytesPerLine
        # qImg_4 =  QImage(mask.data, width_4, height_4, QImage.Format_Grayscale8)  # link ref https://qt.developpez.com/doc/4.7/qimage/ ; https://doc.qt.io/qt-5/qimage.html
        # self.ui.imgLabel_4.setPixmap(QPixmap.fromImage(qImg_4))
    
        if(self.logic==2):   
            self.value = self.value + 1
            self.ui.TEXT_3.setText("Capture")
            image = cv2.cvtColor(img_corrected, cv2.COLOR_BGR2RGB)
            cv2.imwrite('myout.png',img_corrected)
            self.logic=1       
    
    def controlTimer(self):
        # if timer is stopped
        if not self.timer.isActive():
            # create video capture
            self.cap = cv2.VideoCapture(0)
            # start timer
            self.timer.start(20)
            # update control_bt text
            self.ui.TEXT_6.setText("Running")
        # if timer is started
        else:
            # stop timer
            self.timer.stop()
            # release video capture
            self.cap.release()
            # update control_bt text
            self.ui.TEXT_6.setText("Stop")
        
if __name__ == "__main__":
    app = QApplication(sys.argv)

    # create and show mainWindow
    mainWindow = MainWindow()
    mainWindow.show()

    sys.exit(app.exec_())
    


# In[ ]:




