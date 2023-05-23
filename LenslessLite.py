# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'LenslessLite.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.

from main import process
from NewGUI import Ui_new_MainWindow
from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def NewGUI(self):
        self.Window2 = QtWidgets.QMainWindow()
        ui = Ui_new_MainWindow()
        ui.setupUi(self.Window2)
        self.Window2.show()
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setEnabled(True)
        MainWindow.resize(1016, 548)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.btnStartRec = QtWidgets.QPushButton(self.centralwidget)
        self.btnStartRec.setGeometry(QtCore.QRect(760, 270, 191, 31))
        self.btnStartRec.setObjectName("btnStartRec")
        self.btnStartCamera = QtWidgets.QPushButton(self.centralwidget)
        self.btnStartCamera.setGeometry(QtCore.QRect(760, 140, 91, 31))
        self.btnStartCamera.setObjectName("btnStartCamera")
        self.btnStopCamera = QtWidgets.QPushButton(self.centralwidget)
        self.btnStopCamera.setGeometry(QtCore.QRect(860, 140, 91, 31))
        self.btnStopCamera.setObjectName("btnStopCamera")
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(20, 10, 648, 486))
        self.widget.setObjectName("widget")
        self.RedLED = QtWidgets.QCheckBox(self.centralwidget)
        self.RedLED.setGeometry(QtCore.QRect(760, 40, 70, 17))
        self.RedLED.setObjectName("RedLED")
        self.GreenLED = QtWidgets.QCheckBox(self.centralwidget)
        self.GreenLED.setGeometry(QtCore.QRect(760, 60, 70, 17))
        self.GreenLED.setObjectName("GreenLED")
        self.BlueLED = QtWidgets.QCheckBox(self.centralwidget)
        self.BlueLED.setGeometry(QtCore.QRect(760, 80, 70, 17))
        self.BlueLED.setObjectName("BlueLED")
        self.btnLEDOn = QtWidgets.QPushButton(self.centralwidget)
        self.btnLEDOn.setGeometry(QtCore.QRect(860, 43, 75, 21))
        self.btnLEDOn.setObjectName("btnLEDOn")
        self.btnLEDOff = QtWidgets.QPushButton(self.centralwidget)
        self.btnLEDOff.setGeometry(QtCore.QRect(860, 73, 75, 21))
        self.btnLEDOff.setObjectName("btnLEDOff")
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setGeometry(QtCore.QRect(720, 320, 273, 31))
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(830, 20, 46, 13))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.line_2 = QtWidgets.QFrame(self.centralwidget)
        self.line_2.setGeometry(QtCore.QRect(720, 3, 273, 16))
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.line_3 = QtWidgets.QFrame(self.centralwidget)
        self.line_3.setGeometry(QtCore.QRect(705, 10, 31, 511))
        self.line_3.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_3.setObjectName("line_3")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(820, 110, 101, 16))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(760, 240, 51, 16))
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(910, 240, 46, 13))
        self.label_4.setObjectName("label_4")
        self.line_8 = QtWidgets.QFrame(self.centralwidget)
        self.line_8.setGeometry(QtCore.QRect(983, 10, 21, 511))
        self.line_8.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_8.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_8.setObjectName("line_8")
        self.recTime = QtWidgets.QTextEdit(self.centralwidget)
        self.recTime.setGeometry(QtCore.QRect(820, 230, 81, 31))
        self.recTime.setObjectName("recTime")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(730, 310, 46, 13))
        self.label_5.setObjectName("label_5")
        self.Statuslbl = QtWidgets.QLabel(self.centralwidget)
        self.Statuslbl.setGeometry(QtCore.QRect(780, 310, 191, 16))
        self.Statuslbl.setObjectName("Statuslbl")
        self.btnCaptured = QtWidgets.QPushButton(self.centralwidget)
        self.btnCaptured.setGeometry(QtCore.QRect(860, 180, 91, 31))
        self.btnCaptured.setObjectName("btnCaptured")
        self.line_4 = QtWidgets.QFrame(self.centralwidget)
        self.line_4.setGeometry(QtCore.QRect(720, 90, 273, 31))
        self.line_4.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_4.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_4.setObjectName("line_4")
        self.line_5 = QtWidgets.QFrame(self.centralwidget)
        self.line_5.setGeometry(QtCore.QRect(720, 507, 273, 31))
        self.line_5.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_5.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_5.setObjectName("line_5")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(730, 330, 261, 191))
        self.label_6.setText("")
        self.label_6.setPixmap(QtGui.QPixmap("E:/IHT - LENA/BaslerProject/Lensless app lite/triologo.jpg"))
        self.label_6.setObjectName("label_6")
        self.camNo = QtWidgets.QSpinBox(self.centralwidget)
        self.camNo.setGeometry(QtCore.QRect(805, 185, 42, 22))
        self.camNo.setMinimum(-1)
        self.camNo.setMaximum(9)
        self.camNo.setProperty("value", 1)
        self.camNo.setObjectName("camNo")
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(760, 190, 41, 16))
        self.label_7.setObjectName("label_7")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(750, 390, 93, 28))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(870, 390, 93, 28))
        self.pushButton_2.setObjectName("pushButton_2")
        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        self.label_8.setGeometry(QtCore.QRect(770, 350, 181, 20))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_8.setFont(font)
        self.label_8.setObjectName("label_8")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        def action(self):
            # print("Working")
            process(self, MainWindow)

        self.pushButton_2.clicked.connect(action)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.btnStartRec.setText(_translate("MainWindow", "Record"))
        self.btnStartCamera.setText(_translate("MainWindow", "Start Camera"))
        self.btnStopCamera.setText(_translate("MainWindow", "Stop Camera"))
        self.RedLED.setText(_translate("MainWindow", "Red"))
        self.GreenLED.setText(_translate("MainWindow", "Green"))
        self.BlueLED.setText(_translate("MainWindow", "Blue"))
        self.btnLEDOn.setText(_translate("MainWindow", "On"))
        self.btnLEDOff.setText(_translate("MainWindow", "Off"))
        self.label.setText(_translate("MainWindow", "LED"))
        self.label_2.setText(_translate("MainWindow", "Live View"))
        self.label_3.setText(_translate("MainWindow", "Record in"))
        self.label_4.setText(_translate("MainWindow", "minutes"))
        self.label_5.setText(_translate("MainWindow", "Status :"))
        self.Statuslbl.setText(_translate("MainWindow", "Idle..."))
        self.btnCaptured.setText(_translate("MainWindow", "Image Capture"))
        self.camNo.setSpecialValueText(_translate("MainWindow", "1"))
        self.label_7.setText(_translate("MainWindow", "Cam No"))
        self.pushButton.setText(_translate("MainWindow", "ImageJ"))
        self.pushButton_2.setText(_translate("MainWindow", "Advanced"))
        self.label_8.setText(_translate("MainWindow", "IMAGE PROCESSING"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
