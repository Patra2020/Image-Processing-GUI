from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap, QPalette, QPainter
from PyQt5.QtPrintSupport import QPrintDialog, QPrinter
from PyQt5.QtWidgets import QLabel, QSizePolicy, QScrollArea, QMessageBox, QMainWindow, QMenu, QAction, \
    qApp, QFileDialog
import cv2
#import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
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

        self.setWindowTitle("Image Viewer")
        self.resize(800, 600)

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
        self.prewittedgeAct.setEnabled(True)
        self.highpassfilterACT.setEnabled(True)
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


    def normalize(self):
        img = cv2.imread(temp2)
        #img = cv2.normalize(img, None, 0, 1.0,cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        img = cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
        cv2.imwrite(temp1,cv2.imread(temp2))
        cv2.imwrite(temp2,img)
        img = QImage(temp2)
        self.common(img)

    
    def highpassfilter(self):
        img = cv2.imread(temp2)/float(2**8)
        shape = img.shape[:2]
        def draw_cicle(shape,diamiter):
            assert len(shape) == 2
            TF = np.zeros(shape,dtype=bool)
            center = np.array(TF.shape)/2.0

            for iy in range(shape[0]):
                for ix in range(shape[1]):
                    TF[iy,ix] = (iy- center[0])**2 + (ix - center[1])**2 < diamiter **2
            return(TF)


        TFcircleIN   = draw_cicle(shape=img.shape[:2],diamiter=50)
        TFcircleOUT  = ~TFcircleIN

        fig = plt.figure(figsize=(30,10))
        ax  = fig.add_subplot(1,2,1)
        im  = ax.imshow(TFcircleIN,cmap="gray")
        plt.colorbar(im)
        ax  = fig.add_subplot(1,2,2)
        im  = ax.imshow(TFcircleOUT,cmap="gray")
        # plt.colorbar(im)
        # plt.show()

        fft_img = np.zeros_like(img,dtype=complex)
        for ichannel in range(fft_img.shape[2]):
            fft_img[:,:,ichannel] = np.fft.fftshift(np.fft.fft2(img[:,:,ichannel]))

        def filter_circle(TFcircleIN,fft_img_channel):
            temp = np.zeros(fft_img_channel.shape[:2],dtype=complex)
            temp[TFcircleIN] = fft_img_channel[TFcircleIN]
            return(temp)

        fft_img_filtered_IN = []
        fft_img_filtered_OUT = []
## for each channel, pass filter
        for ichannel in range(fft_img.shape[2]):
            fft_img_channel  = fft_img[:,:,ichannel]
    ## circle IN
            temp = filter_circle(TFcircleIN,fft_img_channel)
            fft_img_filtered_IN.append(temp)
    ## circle OUT
            temp = filter_circle(TFcircleOUT,fft_img_channel)
            fft_img_filtered_OUT.append(temp) 
    
        fft_img_filtered_IN = np.array(fft_img_filtered_IN)
        fft_img_filtered_IN = np.transpose(fft_img_filtered_IN,(1,2,0))
        fft_img_filtered_OUT = np.array(fft_img_filtered_OUT)
        fft_img_filtered_OUT = np.transpose(fft_img_filtered_OUT,(1,2,0))

        abs_fft_img              = np.abs(fft_img)
        abs_fft_img_filtered_IN  = np.abs(fft_img_filtered_IN)
        abs_fft_img_filtered_OUT = np.abs(fft_img_filtered_OUT)


        def imshow_fft(absfft):
            magnitude_spectrum = 20*np.log(absfft+0.00001)
            return(ax.imshow(magnitude_spectrum,cmap="gray"))

        fig, axs = plt.subplots(nrows=3,ncols=3,figsize=(15,10))
        fontsize = 15 
        for ichannel, color in enumerate(["R","G","B"]):
            ax = axs[0,ichannel]
            ax.set_title(color)
            im = imshow_fft(abs_fft_img[:,:,ichannel])
            ax.axis("off")
            if ichannel == 0:
                ax.set_ylabel("original DFT",fontsize=fontsize)
            fig.colorbar(im,ax=ax)
    
    
            ax = axs[1,ichannel]
            im = imshow_fft(abs_fft_img_filtered_IN[:,:,ichannel])
            ax.axis("off")
            if ichannel == 0:
                ax.set_ylabel("DFT + low pass filter",fontsize=fontsize)
                fig.colorbar(im,ax=ax)
    
            ax = axs[2,ichannel]
            im = imshow_fft(abs_fft_img_filtered_OUT[:,:,ichannel])
            ax.axis("off")
            if ichannel == 0:
                ax.set_ylabel("DFT + high pass filter",fontsize=fontsize)   
            fig.colorbar(im,ax=ax)
    
# plt.show()


        def inv_FFT_all_channel(fft_img):
            img_reco = []
            for ichannel in range(fft_img.shape[2]):
                img_reco.append(np.fft.ifft2(np.fft.ifftshift(fft_img[:,:,ichannel])))
            img_reco = np.array(img_reco)
            img_reco = np.transpose(img_reco,(1,2,0))
            return(img_reco)


        img_reco              = inv_FFT_all_channel(fft_img)
        img_reco_filtered_IN  = inv_FFT_all_channel(fft_img_filtered_IN)
        img_reco_filtered_OUT = inv_FFT_all_channel(fft_img_filtered_OUT)
        print(img_reco_filtered_OUT)

        ax  = fig.add_subplot(1,3,3)
        try123 = ax.imshow(np.abs(img_reco_filtered_OUT))
        ax.set_title("high pass filtered image") 

        cv2.imwrite(temp1,cv2.imread(temp2))
        cv2.imwrite(temp2, img_reco_filtered_OUT)
        img = QImage(temp2)
        self.common(try123)


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
        self.highpassfilterACT = QAction("&HighPassFilter", self, enabled=False, triggered=self.highpassfilter)
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
        self.editMenu.addSeparator()
        self.editMenu.addAction(self.highpassfilterACT)
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








