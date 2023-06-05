from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap, QPalette, QPainter
from PyQt5.QtPrintSupport import QPrintDialog, QPrinter
from PyQt5.QtWidgets import QLabel, QSizePolicy, QScrollArea, QMessageBox, QMainWindow, QMenu, QAction, \
    qApp, QFileDialog, QSlider, QRadioButton, QVBoxLayout
from PyQt5.QtWidgets import (QApplication, QCheckBox, QGridLayout, QGroupBox,
                             QMenu, QPushButton, QRadioButton, QVBoxLayout, QWidget, QSlider, QHBoxLayout)
import cv2

import numpy as np
# from PyQt5.QtGui import QImage


class QImageViewer(QMainWindow):
    

    def __init__(self):
        super().__init__()

        self.printer = QPrinter()
        self.scaleFactor = 0.0

        self.imageLabel = QLabel()
        self.imageLabel.setBackgroundRole(QPalette.Base)
        self.imageLabel.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.imageLabel.setScaledContents(True)

        self.scrollArea = QScrollArea()
        self.scrollArea.setBackgroundRole(QPalette.Dark)
        self.scrollArea.setWidget(self.imageLabel)
        self.scrollArea.setVisible(False)

        self.setCentralWidget(self.scrollArea)

        self.createActions()
        self.createMenus()

        # selfQHBoxLayout()
        self.slider = QSlider(Qt.Orientation.Horizontal, self)
        self.slider.setGeometry(5, 30, 200, 20)
        self.slider.setRange(1, 5)
        self.slider.setValue(1)
        self.slider.valueChanged.connect(self.smoothen)
        # self.slider.setStyleSheet()
        # label = QLabel("SMOOTHEN", self)
 
        # # setting geometry to the label
        # label.setGeometry(55,30,200,20)
 
        # # making label multi line
        # label.setWordWrap(True)

        

 
        self.setWindowTitle("Image Viewer")
        self.resize(800, 600)

    def stylesheet(self):
        return """
            QSlider::groove:horizontal {
                background: white;
                height: 40px;
            }

            # QSlider::sub-page:horizontal {
            #     background: qlineargradient(x1: 0, y1: 0,    x2: 0, y2: 1,
            #         stop: 0 #66e, stop: 1 #bbf);
            #     background: qlineargradient(x1: 0, y1: 0.2, x2: 1, y2: 1,
            #         stop: 0 #bbf, stop: 1 #55f);
            #     height: 40px;
            # }

            QSlider::add-page:horizontal {
                background: #fff;
                height: 40px;
            }

            QSlider::handle:horizontal {
                background: #bbf;
                border: 0px;
                width: 0px;
                margin-top: 0px;
                margin-bottom: 0px;
                border-radius: 0px;
            }
        """


    def common(self,img):
        self.imageLabel.setPixmap(QPixmap.fromImage(img))
        self.scaleFactor = 1.0

        self.scrollArea.setVisible(True)
        self.printAct.setEnabled(True)
        self.fitToWindowAct.setEnabled(True)
        self.negative_action.setEnabled(True)
        self.cannyedgeAct.setEnabled(True)
        self.normalizeAct.setEnabled(True)
        self.sobeledgeAct.setEnabled(True)
        self.prewittedgeAct.setEnabled(True)
        # self.smoothenAct.setEnabled(True)
        self.updateActions()

        if not self.fitToWindowAct.isChecked():
                self.imageLabel.adjustSize()
    def open(self):
        options = QFileDialog.Options()
        # fileName = QFileDialog.getOpenFileName(self, "Open File", QDir.currentPath())
        global fileName
        fileName, _ = QFileDialog.getOpenFileName(self, 'QFileDialog.getOpenFileName()', '',
                                                  'Images (*.png *.jpeg *.jpg *.bmp *.gif *.tiff)', options=options)
        if fileName:
            image = QImage(fileName)
            if image.isNull():
                QMessageBox.information(self, "Image Viewer", "Cannot load %s." % fileName)
                return

            self.common(image)
            global temp1,temp2
            temp1  = "temp1.jpg"
            temp2 = "temp2.jpg"
            
            cv2.imwrite(temp1,cv2.imread(fileName))
            cv2.imwrite(temp2,cv2.imread(fileName))
            

    def print_(self):
        dialog = QPrintDialog(self.printer, self)
        if dialog.exec_():
            painter = QPainter(self.printer)
            rect = painter.viewport()
            size = self.imageLabel.pixmap().size()
            size.scale(rect.size(), Qt.KeepAspectRatio)
            painter.setViewport(rect.x(), rect.y(), size.width(), size.height())
            painter.setWindow(self.imageLabel.pixmap().rect())
            painter.drawPixmap(0, 0, self.imageLabel.pixmap())
    
    def undo(self):
        cv2.imwrite(temp2,cv2.imread(temp1))
        img = QImage(temp2)
        self.common(img)
        
    def zoomIn(self):
        self.scaleImage(1.25)

    def zoomOut(self):
        self.scaleImage(0.8)

    def normalSize(self):
        self.imageLabel.adjustSize()
        self.scaleFactor = 1.0

    def fitToWindow(self):
        fitToWindow = self.fitToWindowAct.isChecked()
        self.scrollArea.setWidgetResizable(fitToWindow)
        if not fitToWindow:
            self.normalSize()

        self.updateActions()
    def cannyedge(self):
        self.cannyedgeAct.isChecked()
        img = cv2.imread(temp2)
        kernel2 = np.ones((8,8), np.float32)/25
        sharpen = cv2.filter2D(src=img, ddepth=-1, kernel=kernel2)
        img = cv2.Canny(sharpen,80,85)
        cv2.imwrite(temp1,cv2.imread(temp2))
        cv2.imwrite(temp2,img)
        img = QImage(temp2)
        self.common(img)
        '''
        img = cv2.imread(temp2)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #img_gaussian = cv2.GaussianBlur(gray,(3,3),0)
        img_canny = cv2.Canny(img,100,200)
        cv2.imwrite(temp1,cv2.imread(temp2))
        cv2.imwrite(temp2,img_canny)
        img = QImage(temp2)
        self.common(img)'''

    
    def sobeledge(self):
        img = cv2.imread(temp2)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gaussian = cv2.GaussianBlur(gray,(3,3),0)
        img_sobelx = cv2.Sobel(img_gaussian,cv2.CV_8U,1,0,ksize=5)
        img_sobely = cv2.Sobel(img_gaussian,cv2.CV_8U,0,1,ksize=5)
        img_sobel = img_sobelx + img_sobely
        cv2.imwrite(temp1,cv2.imread(temp2))
        cv2.imwrite(temp2,img_sobel)
        img = QImage(temp2)
        self.common(img)


    def prewittedge(self):
        img = cv2.imread(temp2)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gaussian = cv2.GaussianBlur(gray,(3,3),0)
        kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
        kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
        img_prewittx = cv2.filter2D(img_gaussian, -1, kernelx)
        img_prewitty = cv2.filter2D(img_gaussian, -1, kernely)
        img_prewitt = img_prewittx + img_prewitty
        cv2.imwrite(temp1,cv2.imread(temp2))
        cv2.imwrite(temp2, img_prewitt)
        img = QImage(temp2)
        self.common(img)





    def negative(self):
        negative = self.negative_action.isChecked()
        # # Convert numpy array to OpenCV image
        img = cv2.imread(temp2)
        img = abs(255-img)
        height,width,_ = img.shape
        # global temp1,temp2
        
        cv2.imwrite(temp1,cv2.imread(temp2))
        cv2.imwrite(temp2,img)
        img = QImage(temp2)
        self.common(img)
    
    
    
    # def smoothen(self,value):
    #     # temp1_pic = cv2.imread(temp2)
    #     # temp2_pic = cv2.imread(temp2)
    #     # temp3_pic = cv2.imread(temp2)
    #     # temp4_pic = cv2.imread(temp2)
    #     # temp5_pic = cv2.imread(temp2)
        
    #     print(value)
    #     # kernelSizes = [(1, 1), (2, 2), (3, 3),(4,4),(5,5)]
    #     # # loop over the kernel sizes
    #     # for (kX, kY) in kernelSizes:
    #     img = cv2.imread(temp2)
    #     # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     # kX = kY = value
    #     kernel2_1 = np.ones((1,1), np.float32)/25
    #     temp1_pic = cv2.filter2D(src=img, ddepth=-1, kernel=kernel2_1)
    #     kernel2_2 = np.ones((2,2), np.float32)/25
    #     temp2_pic = cv2.filter2D(src=img, ddepth=-1, kernel=kernel2_2)
    #     kernel2_3 = np.ones((3,3), np.float32)/25
    #     temp3_pic = cv2.filter2D(src=img, ddepth=-1, kernel=kernel2_3)
    #     kernel2_4 = np.ones((4,4), np.float32)/25
    #     temp4_pic = cv2.filter2D(src=img, ddepth=-1, kernel=kernel2_4)
    #     kernel2_5 = np.ones((5,5), np.float32)/25
    #     temp5_pic = cv2.filter2D(src=img, ddepth=-1, kernel=kernel2_5)


    #     # img = cv2.filter2D(src=img, ddepth=-1, kernel=kernel2)

    #     if(value == 1):
    #         temp_img = temp1_pic
    #     elif(value == 2):
    #         temp_img = temp2_pic
    #     elif(value == 3):
    #         temp_img = temp3_pic
    #     elif(value == 4):
    #         temp_img = temp4_pic
    #     elif(value == 5):
    #         temp_img = temp5_pic
    #     # cv2.imwrite(temp1,cv2.imread(temp2))
    #     # cv2.imwrite(temp2,img)
    #     temp_img = QImage(temp_img)
    #     self.common(temp_img)













    def normalize(self):
        img = cv2.imread(temp2)
        img_normalized = np.array((img - np.min(img)) / (np.max(img) - np.min(img)))
        cv2.imwrite(temp1,cv2.imread(temp2))
        cv2.imwrite(temp2,img)
        img = QImage(temp2)
        self.common(img)


    def about(self):
        QMessageBox.about(self, "About Image Viewer",
                          "<p>The <b>Image Viewer</b> example shows how to combine "
                          "QLabel and QScrollArea to display an image. QLabel is "
                          "typically used for displaying text, but it can also display "
                          "an image. QScrollArea provides a scrolling view around "
                          "another widget. If the child widget exceeds the size of the "
                          "frame, QScrollArea automatically provides scroll bars.</p>"
                          "<p>The example demonstrates how QLabel's ability to scale "
                          "its contents (QLabel.scaledContents), and QScrollArea's "
                          "ability to automatically resize its contents "
                          "(QScrollArea.widgetResizable), can be used to implement "
                          "zooming and scaling features.</p>"
                          "<p>In addition the example shows how to use QPainter to "
                          "print an image.</p>")

    def createActions(self):
        self.openAct = QAction("&Open...", self, shortcut="Ctrl+O", triggered=self.open)
        self.printAct = QAction("&Print...", self, shortcut="Ctrl+P", enabled=False, triggered=self.print_)
        self.exitAct = QAction("E&xit", self, shortcut="Ctrl+Q", triggered=self.close)
        self.zoomInAct = QAction("Zoom &In (25%)", self, shortcut="Ctrl++", enabled=False, triggered=self.zoomIn)
        self.zoomOutAct = QAction("Zoom &Out (25%)", self, shortcut="Ctrl+-", enabled=False, triggered=self.zoomOut)
        self.normalSizeAct = QAction("&Normal Size", self, shortcut="Ctrl+S", enabled=False, triggered=self.normalSize)
        self.fitToWindowAct = QAction("&Fit to Window", self, enabled=False, checkable=True, shortcut="Ctrl+F",
                                      triggered=self.fitToWindow)
        ########################################################################################################
        self.negative_action = QAction("&Negative", self, enabled=False, triggered=self.negative)
        self.cannyedgeAct = QAction("&CannyEdge", self, enabled=False, triggered=self.cannyedge)
        self.undoAct = QAction("&Undo...", self, shortcut="Ctrl+Z", triggered=self.undo)
        self.normalizeAct = QAction("&Normalize", self, enabled=False, triggered=self.normalize)
        self.sobeledgeAct = QAction("&SobelEdge", self, enabled=False, triggered=self.sobeledge)
        self.prewittedgeAct = QAction("&PrewittEdge", self, enabled=False, triggered=self.prewittedge)
        # self.smoothenAct = QAction("&Smoothen", self, enabled=False, triggered=self.smoothen)
        #########################################################################################################
        self.aboutAct = QAction("&About", self, triggered=self.about)
        self.aboutQtAct = QAction("About &Qt", self, triggered=qApp.aboutQt)

    def createMenus(self):
        self.fileMenu = QMenu("&File", self)
        self.fileMenu.addAction(self.openAct)
        self.fileMenu.addAction(self.printAct)
        self.fileMenu.addSeparator()
        self.fileMenu.addAction(self.undoAct)
        self.fileMenu.addAction(self.exitAct)

        self.viewMenu = QMenu("&View", self)
        self.viewMenu.addAction(self.zoomInAct)
        self.viewMenu.addAction(self.zoomOutAct)
        self.viewMenu.addAction(self.normalSizeAct)
        self.viewMenu.addSeparator()
        self.viewMenu.addAction(self.fitToWindowAct)


        self.editMenu = QMenu("&Edit", self)
        self.editMenu.addAction(self.negative_action)
        self.editMenu.addAction(self.cannyedgeAct)
        self.editMenu.addAction(self.sobeledgeAct)
        self.editMenu.addAction(self.prewittedgeAct)
        self.editMenu.addAction(self.normalizeAct)
        # self.editMenu.addAction(self.smoothenAct)
        self.editMenu.addSeparator()
        # self.viewMenu.addSeparator()
        # self.viewMenu.addAction(self.sharpen)
        

        self.helpMenu = QMenu("&Help", self)
        self.helpMenu.addAction(self.aboutAct)
        self.helpMenu.addAction(self.aboutQtAct)

        self.menuBar().addMenu(self.fileMenu)
        self.menuBar().addMenu(self.viewMenu)
        self.menuBar().addMenu(self.editMenu)
        self.menuBar().addMenu(self.helpMenu)

    def updateActions(self):
        self.zoomInAct.setEnabled(not self.fitToWindowAct.isChecked())
        self.zoomOutAct.setEnabled(not self.fitToWindowAct.isChecked())
        self.normalSizeAct.setEnabled(not self.fitToWindowAct.isChecked())

    def scaleImage(self, factor):
        self.scaleFactor *= factor
        self.imageLabel.resize(self.scaleFactor * self.imageLabel.pixmap().size())

        self.adjustScrollBar(self.scrollArea.horizontalScrollBar(), factor)
        self.adjustScrollBar(self.scrollArea.verticalScrollBar(), factor)

        self.zoomInAct.setEnabled(self.scaleFactor < 3.0)
        self.zoomOutAct.setEnabled(self.scaleFactor > 0.333)

    def adjustScrollBar(self, scrollBar, factor):
        scrollBar.setValue(int(factor * scrollBar.value()
                               + ((factor - 1) * scrollBar.pageStep() / 2)))
    
    



if __name__ == '__main__':
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    imageViewer = QImageViewer()
    imageViewer.show()
    sys.exit(app.exec_())








