# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'NewGUI.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_new_MainWindow(object):
    # def __init__(self):
    #     super(UI,self).__init__()



    def setupUi(self, new_MainWindow):
        new_MainWindow.setObjectName("new_MainWindow")
        new_MainWindow.resize(800, 444)
        self.centralwidget = QtWidgets.QWidget(new_MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(280, 10, 221, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(690, 190, 93, 28))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(530, 240, 93, 28))
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_4.setGeometry(QtCore.QRect(690, 290, 93, 28))
        self.pushButton_4.setObjectName("pushButton_4")
        self.pushButton_5 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_5.setGeometry(QtCore.QRect(690, 340, 93, 28))
        self.pushButton_5.setObjectName("pushButton_5")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(530, 140, 91, 28))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_6 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_6.setGeometry(QtCore.QRect(690, 140, 93, 28))
        self.pushButton_6.setObjectName("pushButton_6")
        self.pushButton_7 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_7.setGeometry(QtCore.QRect(530, 190, 93, 28))
        self.pushButton_7.setObjectName("pushButton_7")
        self.pushButton_8 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_8.setGeometry(QtCore.QRect(690, 240, 93, 28))
        self.pushButton_8.setObjectName("pushButton_8")
        self.pushButton_9 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_9.setGeometry(QtCore.QRect(530, 290, 93, 28))
        self.pushButton_9.setObjectName("pushButton_9")
        self.pushButton_10 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_10.setGeometry(QtCore.QRect(530, 340, 93, 28))
        self.pushButton_10.setObjectName("pushButton_10")
        self.pushButton_11 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_11.setGeometry(QtCore.QRect(530, 60, 93, 28))
        self.pushButton_11.setObjectName("pushButton_11")
        self.pushButton_12 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_12.setGeometry(QtCore.QRect(690, 60, 93, 28))
        self.pushButton_12.setObjectName("pushButton_12")
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(20, 40, 491, 331))
        self.widget.setObjectName("widget")
        new_MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(new_MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 18))
        self.menubar.setObjectName("menubar")
        new_MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(new_MainWindow)
        self.statusbar.setObjectName("statusbar")
        new_MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(new_MainWindow)
        QtCore.QMetaObject.connectSlotsByName(new_MainWindow)

    def retranslateUi(self, new_MainWindow):
        _translate = QtCore.QCoreApplication.translate
        new_MainWindow.setWindowTitle(_translate("new_MainWindow", "MainWindow"))
        self.label.setText(_translate("new_MainWindow", "ADVANCED OPERATIONS"))
        self.pushButton_2.setText(_translate("new_MainWindow", "4"))
        self.pushButton_3.setText(_translate("new_MainWindow", "5"))
        self.pushButton_4.setText(_translate("new_MainWindow", "8"))
        self.pushButton_5.setText(_translate("new_MainWindow", "10"))
        self.pushButton.setText(_translate("new_MainWindow", "B1"))
        self.pushButton_6.setText(_translate("new_MainWindow", "2"))
        self.pushButton_7.setText(_translate("new_MainWindow", "3"))
        self.pushButton_8.setText(_translate("new_MainWindow", "6"))
        self.pushButton_9.setText(_translate("new_MainWindow", "7"))
        self.pushButton_10.setText(_translate("new_MainWindow", "9"))
        self.pushButton_11.setText(_translate("new_MainWindow", "ON"))
        self.pushButton_12.setText(_translate("new_MainWindow", "OFF"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    new_MainWindow = QtWidgets.QMainWindow()
    ui = Ui_new_MainWindow()
    ui.setupUi(new_MainWindow)
    new_MainWindow.show()
    sys.exit(app.exec_())
