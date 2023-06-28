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
from tkinter import *
import math
import os
import matplotlib.pyplot as plt

# from tkinter import *
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
        # self.slider = QSlider(Qt.Orientation.Horizontal, self)
        # self.slider.setGeometry(5, 30, 200, 20)
        # self.slider.setRange(1, 5)
        # self.slider.setValue(1)
        # self.slider.valueChanged.connect(self.smoothen)
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
    
    def ruler(self):
        class DrawLineWidget(object):
            def __init__(self):
                self.original_image = cv2.imread(temp2)
                self.clone = self.original_image.copy()

                cv2.namedWindow('image',cv2.WINDOW_NORMAL)
                # cv2.resizeWindow('image',)
                cv2.setMouseCallback('image', self.extract_coordinates)

                # List to store start/end points
                self.image_coordinates = []

            def extract_coordinates(self, event, x, y, flags, parameters):
                # Record starting (x,y) coordinates on left mouse button click
                if event == cv2.EVENT_LBUTTONDOWN:
                    self.image_coordinates = [(x,y)]

                # Record ending (x,y) coordintes on left mouse bottom release
                elif event == cv2.EVENT_LBUTTONUP:
                        self.image_coordinates.append((x,y))
                        cv2.line(self.clone, self.image_coordinates[0], self.image_coordinates[1], (36,255,12), 2)
                        cv2.imshow("image", self.clone)           
                        
                        

                        root = Tk()
                        root.geometry("300x150")
                        root.title(" Angle Measurement ")
                        
                        

                        Output = Text(root, height = 5,
                                    width = 25,
                                    bg = "light cyan")

                        # Display = Button(root, height = 2,
                        #                 width = 20,
                        #                 text ="Show",
                        #                 command = lambda:Take_input())
                        Output.insert(END,'Angle: {}'.format(abs(math.degrees(math.atan((self.image_coordinates[1][1] - self.image_coordinates[0][1])/(self.image_coordinates[1][0] - self.image_coordinates[0][0]))) )))

    
                        Output.pack()

                        mainloop()

                        print('Starting: {}, Ending: {}'.format(self.image_coordinates[0], self.image_coordinates[1]))
                        # print()'Angle: {}'.format(abs(math.degrees(math.atan((self.image_coordinates[1][1] - self.image_coordinates[0][1])/(self.image_coordinates[1][0] - self.image_coordinates[0][0]))) ))
                        
                        
                        # print('Angle: {}'.format(abs(math.degrees(math.atan((self.image_coordinates[1][1] - self.image_coordinates[0][1])/(self.image_coordinates[1][0] - self.image_coordinates[0][0]))) )))

                            # Draw line
                         

                # Clear drawing boxes on right mouse button click
                elif event == cv2.EVENT_RBUTTONDOWN:
                    self.clone = self.original_image.copy()

            def show_image(self):
                return self.clone

        if __name__ == '__main__':

            draw_line_widget = DrawLineWidget()
            cv2.imshow('image', draw_line_widget.show_image())
            # while True:
            #     cv2.imshow('image', draw_line_widget.show_image())
            #     key = cv2.waitKey(1)

            #     # Close program with keyboard 'q'
            #     if key == ord('q'):
            #         cv2.destroyWindow("image")
            #         exit(1)


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
        self.lowpassAct.setEnabled(True)
        self.advlowpassAct.setEnabled(True)
        self.highpassAct.setEnabled(True)
        self.advhighpassAct.setEnabled(True)
        self.rulerAct.setEnabled(True)
        # self.rulerclose.setEnabled(True)
        self.smoothenAct.setEnabled(True)
        self.sharpenAct.setEnabled(True)
        self.rotateAct.setEnabled(True)
        self.rotatemergeAct.setEnabled(True)
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
        
    def zoomIn1(self):
        self.scaleImage(1.01)

    def zoomOut1(self):
        self.scaleImage(0.99)
    def zoomIn5(self):
        self.scaleImage(1.05)

    def zoomOut5(self):
        self.scaleImage(0.95)
    def zoomIn10(self):
        self.scaleImage(1.1)

    def zoomOut10(self):
        self.scaleImage(0.9)
    def zoomIn25(self):
        self.scaleImage(1.25)

    def zoomOut25(self):
        self.scaleImage(0.75)

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
        img = img_sobelx + img_sobely
        cv2.imwrite(temp1,cv2.imread(temp2))
        cv2.imwrite(temp2,img)
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
        img = img_prewittx + img_prewitty
        cv2.imwrite(temp1,cv2.imread(temp2))
        cv2.imwrite(temp2, img)
        img = QImage(temp2)
        self.common(img)


    def rotate_merge(self):

        #SELECT THE "base.bmp" file

        image1 = cv2.imread(temp2)
        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        directory = os.path.dirname(fileName)


        # base_image_path = 'base.bmp'
        # image1 = cv2.imread(base_image_path)
        # gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        base_image_extension = os.path.splitext(fileName)[1]

        finalimages = []
        i = 0

        #directory = 'RotationExperiment'
        for filename in os.listdir(directory):
            f = os.path.join(directory, filename)
            image2 = cv2.imread(f)
            gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
            wid = image2.shape[1]
            hgt = image2.shape[0]

            # displaying the dimensions
            #print(str(wid) + "x" + str(hgt))

            sift = cv2.SIFT_create(nfeatures=100)
            keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
            keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
            matches = bf.match(descriptors1, descriptors2)
            matches = sorted(matches, key=lambda x: x.distance)

            matchedPts1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
            matchedPts2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])

            tform, _ = cv2.estimateAffinePartial2D(matchedPts2, matchedPts1, method=cv2.RANSAC, confidence=0.10)
            sc = tform[0, 0]
            ss = tform[0, 1]
            scaleRecovered = np.sqrt(sc * 2 + ss * 2)
            thetaRecovered = np.degrees(np.arctan2(-ss, sc))

            print('Recovered scale:', scaleRecovered)
            print('Recovered rotation angle:', thetaRecovered)

            # Calculate the output size based on the larger image dimensions
            outputSize = (max(image1.shape[1], image2.shape[1]), max(image1.shape[0], image2.shape[0]))

            alignedImage = cv2.warpAffine(image2, tform, outputSize, flags=cv2.INTER_NEAREST)

            fig = plt.figure(figsize=(10, 7))
            rows = 1
            columns = 3

            # plt.imshow(image2)
            # plt.axis('off')
            # plt.title("Second")
            # plt.show()

            # plt.imshow(alignedImage)
            # plt.axis('off')
            # plt.title("Final")
            # plt.show()

         

            folder_path = "ROT123"  # Specify the folder path

            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
                #print("Folder created successfully.")
            # else:
                #print("Folder already exists.")

            aligned_image_path = f'ROT123/{i}{base_image_extension}'
            cv2.imwrite(aligned_image_path, alignedImage)
            i = i + 1
            wid = alignedImage.shape[1]
            hgt = alignedImage.shape[0]

            # displaying the dimensions
            #print(str(wid) + "x" + str(hgt))
            finalimages.append(alignedImage)


        directory = 'ROT123'
        merged_image = 0
        i = 0
        for filename in os.listdir(directory):
            f = os.path.join(directory, filename)
            i = i+1
        # for image in image_files:

            image = cv2.imread(f)
        # Convert images to floating-point format

        # for image in finalimages:
            image = image.astype(np.float32)/255.0
            merged_image = merged_image + image

        # Average the two images
            merged_image = merged_image/i

        # Convert the merged image back to the uint8 format
            merged_image = (merged_image * 255).astype(np.uint8)
            img = merged_image

        cv2.imwrite(temp1,cv2.imread(temp2))
        cv2.imwrite(temp2, img)
        img = QImage(temp2)
        self.common(img)



    def rotate(self):
        image2 = cv2.imread(temp2)
        gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        wid = image2.shape[1]
        hgt = image2.shape[0]
        img_folder = os.path.dirname(fileName)
        base_image_path = img_folder+'/base.bmp'
        image1 = cv2.imread(base_image_path)
        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        base_image_extension = os.path.splitext(base_image_path)[1]

        # finalimages = []
        # i = 0

        # directory = 'RotationExperiment'
        # for filename in os.listdir(directory):
        #     f = os.path.join(directory, filename)
        

            # displaying the dimensions
            #print(str(wid) + "x" + str(hgt))

        sift = cv2.SIFT_create(nfeatures=100)
        keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
        keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(descriptors1, descriptors2)
        matches = sorted(matches, key=lambda x: x.distance)

        matchedPts1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
        matchedPts2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])

        tform, _ = cv2.estimateAffinePartial2D(matchedPts2, matchedPts1, method=cv2.RANSAC, confidence=0.10)
        sc = tform[0, 0]
        ss = tform[0, 1]
        scaleRecovered = np.sqrt(sc * 2 + ss * 2)
        thetaRecovered = np.degrees(np.arctan2(-ss, sc))

        # print('Recovered scale:', scaleRecovered)
        # print('Recovered rotation angle:', thetaRecovered)

            # Calculate the output size based on the larger image dimensions
        outputSize = (max(image1.shape[1], image2.shape[1]), max(image1.shape[0], image2.shape[0]))

        img = cv2.warpAffine(image2, tform, outputSize, flags=cv2.INTER_NEAREST)

        # fig = plt.figure(figsize=(10, 7))
        # rows = 1
        # columns = 3

        #     plt.imshow(image2)
        #     plt.axis('off')
        #     plt.title("Second")
        #     plt.show()

        #     plt.imshow(alignedImage)
        #     plt.axis('off')
        #     plt.title("Final")
            # plt.show()

        #aligned_image_path = f'ROT/{i}{base_image_extension}'
        #cv2.imwrite(aligned_image_path, img)
        #i = i + 1
        wid = img.shape[1]
        hgt = img.shape[0]

            # displaying the dimensions
        #print(str(wid) + "x" + str(hgt))

        cv2.imwrite(temp1,cv2.imread(temp2))
        cv2.imwrite(temp2, img)
        img = QImage(temp2)
        self.common(img)


        #finalimages.append(alignedImage)



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
    
    def smoothen(self):
        self.smoothenAct.isChecked()
        img = cv2.imread(temp2)
        kX = kY = 5
            # apply an "average" blur to the image using the current kernel
            # size
        kernel2 = np.ones((kX,kY), np.float32)/25
            # blurred = cv2.filter2D(src=img, ddepth=-1, kernel=kernel2)
            # plt.imshow(blurred,cmap = 'gray',interpolation = 'bicubic')
        # kernel_sharp = 2*np.eye(kX) - kernel2
        img = cv2.filter2D(src=img, ddepth=-1, kernel=kernel2)
        cv2.imwrite(temp1,cv2.imread(temp2))
        cv2.imwrite(temp2,img)
        img = QImage(temp2)
        self.common(img)
    
    def sharpen(self):
        self.sharpenAct.isChecked()
        img = cv2.imread(temp2)
        kX = kY = 1
            # apply an "average" blur to the image using the current kernel
            # size
        kernel2 = np.ones((kX,kY), np.float32)/25
            # blurred = cv2.filter2D(src=img, ddepth=-1, kernel=kernel2)
            # plt.imshow(blurred,cmap = 'gray',interpolation = 'bicubic')
        kernel_sharp = 2*np.eye(kX) - kernel2
        img = cv2.filter2D(src=img, ddepth=-1, kernel=kernel_sharp)
        cv2.imwrite(temp1,cv2.imread(temp2))
        cv2.imwrite(temp2,img)
        img = QImage(temp2)
        self.common(img)

    def lowpass(self):
        self.lowpassAct.isChecked()
        img = cv2.imread(temp2)


        # do dft saving as complex output
        dft = np.fft.fft2(img, axes=(0,1))

        # apply shift of origin to center of image
        dft_shift = np.fft.fftshift(dft)

        # generate spectrum from magnitude image (for viewing only)
        mag = np.abs(dft_shift)
        spec = np.log(mag) / 20

        # create circle mask
        radius = 32
        mask = np.zeros_like(img)
        cy = mask.shape[0] // 2
        cx = mask.shape[1] // 2
        cv2.circle(mask, (cx,cy), radius, (255,255,255), -1)[0]

        # blur the mask
        # mask2 = cv2.GaussianBlur(mask, (19,19), 0)

        # apply mask to dft_shift
        dft_shift_masked = np.multiply(dft_shift,mask) / 255
        # dft_shift_masked2 = np.multiply(dft_shift,mask2) / 255


        # shift origin from center to upper left corner
        back_ishift = np.fft.ifftshift(dft_shift)
        back_ishift_masked = np.fft.ifftshift(dft_shift_masked)
        # back_ishift_masked2 = np.fft.ifftshift(dft_shift_masked2)


        # do idft saving as complex output
        img_back = np.fft.ifft2(back_ishift, axes=(0,1))
        img_filtered = np.fft.ifft2(back_ishift_masked, axes=(0,1))
        # img_filtered2 = np.fft.ifft2(back_ishift_masked2, axes=(0,1))

        # combine complex real and imaginary components to form (the magnitude for) the original image again
        img_back = np.abs(img_back).clip(0,255).astype(np.uint8)
        img = np.abs(img_filtered).clip(0,255).astype(np.uint8)
        # img_filtered2 = np.abs(img_filtered2).clip(0,255).astype(np.uint8)

        cv2.imwrite(temp1,cv2.imread(temp2))
        cv2.imwrite(temp2,img)
        img = QImage(temp2)
        self.common(img)
        
    
    def advlowpass(self):
        self.advlowpassAct.isChecked()
        img = cv2.imread(temp2)


        # do dft saving as complex output
        dft = np.fft.fft2(img, axes=(0,1))

        # apply shift of origin to center of image
        dft_shift = np.fft.fftshift(dft)

        # generate spectrum from magnitude image (for viewing only)
        mag = np.abs(dft_shift)
        spec = np.log(mag) / 20

        # create circle mask
        radius = 32
        mask = np.zeros_like(img)
        cy = mask.shape[0] // 2
        cx = mask.shape[1] // 2
        cv2.circle(mask, (cx,cy), radius, (255,255,255), -1)[0]

        # blur the mask
        mask2 = cv2.GaussianBlur(mask, (19,19), 0)

        # apply mask to dft_shift
        # dft_shift_masked = np.multiply(dft_shift,mask) / 255
        dft_shift_masked2 = np.multiply(dft_shift,mask2) / 255


        # shift origin from center to upper left corner
        back_ishift = np.fft.ifftshift(dft_shift)
        # back_ishift_masked = np.fft.ifftshift(dft_shift_masked)
        back_ishift_masked2 = np.fft.ifftshift(dft_shift_masked2)


        # do idft saving as complex output
        img_back = np.fft.ifft2(back_ishift, axes=(0,1))
        # img_filtered = np.fft.ifft2(back_ishift_masked, axes=(0,1))
        img_filtered2 = np.fft.ifft2(back_ishift_masked2, axes=(0,1))

        # combine complex real and imaginary components to form (the magnitude for) the original image again
        img_back = np.abs(img_back).clip(0,255).astype(np.uint8)
        # img = np.abs(img_filtered).clip(0,255).astype(np.uint8)
        img = np.abs(img_filtered2).clip(0,255).astype(np.uint8)

        cv2.imwrite(temp1,cv2.imread(temp2))
        cv2.imwrite(temp2,img)
        img = QImage(temp2)
        self.common(img)
    
    def highpass(self):
        img = cv2.imread(temp2)

        # do dft saving as complex output
        dft = np.fft.fft2(img, axes=(0,1))

        # apply shift of origin to center of image
        dft_shift = np.fft.fftshift(dft)

        # generate spectrum from magnitude image (for viewing only)
        mag = np.abs(dft_shift)
        spec = np.log(mag) / 20

        # create white circle mask on black background and invert so black circle on white background
        radius = 32
        mask = np.zeros_like(img)
        cy = mask.shape[0] // 2
        cx = mask.shape[1] // 2
        cv2.circle(mask, (cx,cy), radius, (255,255,255), -1)[0]
        mask = 255 - mask

        # blur the mask
        # mask2 = cv2.GaussianBlur(mask, (19,19), 0)

        # apply mask to dft_shift
        dft_shift_masked = np.multiply(dft_shift,mask) / 255
        # dft_shift_masked2 = np.multiply(dft_shift,mask2) / 255


        # shift origin from center to upper left corner
        back_ishift = np.fft.ifftshift(dft_shift)
        back_ishift_masked = np.fft.ifftshift(dft_shift_masked)
        # back_ishift_masked2 = np.fft.ifftshift(dft_shift_masked2)


        # do idft saving as complex output
        img_back = np.fft.ifft2(back_ishift, axes=(0,1))
        img_filtered = np.fft.ifft2(back_ishift_masked, axes=(0,1))
        # img_filtered2 = np.fft.ifft2(back_ishift_masked2, axes=(0,1))

        # combine complex real and imaginary components to form (the magnitude for) the original image again
        # multiply by 3 to increase brightness
        img_back = np.abs(img_back).clip(0,255).astype(np.uint8)
        img = np.abs(3*img_filtered).clip(0,255).astype(np.uint8)
        # img_filtered2 = np.abs(3*img_filtered2).clip(0,255).astype(np.uint8)


        # cv2.imshow("ORIGINAL", img)
        # cv2.imshow("SPECTRUM", spec)
        # cv2.imshow("MASK", mask)
        # cv2.imshow("MASK2", mask2)
        # cv2.imshow("ORIGINAL DFT/IFT ROUND TRIP", img_back)
        # cv2.imshow("FILTERED DFT/IFT ROUND TRIP", img_filtered)
        # cv2.imshow("FILTERED2 DFT/IFT ROUND TRIP", img_filtered2)
        cv2.imwrite(temp1,cv2.imread(temp2))
        cv2.imwrite(temp2,img)
        img = QImage(temp2)
        self.common(img)

    def advhighpass(self):
        img = cv2.imread(temp2)

        # do dft saving as complex output
        dft = np.fft.fft2(img, axes=(0,1))

        # apply shift of origin to center of image
        dft_shift = np.fft.fftshift(dft)

        # generate spectrum from magnitude image (for viewing only)
        mag = np.abs(dft_shift)
        spec = np.log(mag) / 20

        # create white circle mask on black background and invert so black circle on white background
        radius = 32
        mask = np.zeros_like(img)
        cy = mask.shape[0] // 2
        cx = mask.shape[1] // 2
        cv2.circle(mask, (cx,cy), radius, (255,255,255), -1)[0]
        mask = 255 - mask

        # blur the mask
        mask2 = cv2.GaussianBlur(mask, (19,19), 0)

        # apply mask to dft_shift
        # dft_shift_masked = np.multiply(dft_shift,mask) / 255
        dft_shift_masked2 = np.multiply(dft_shift,mask2) / 255


        # shift origin from center to upper left corner
        back_ishift = np.fft.ifftshift(dft_shift)
        # back_ishift_masked = np.fft.ifftshift(dft_shift_masked)
        back_ishift_masked2 = np.fft.ifftshift(dft_shift_masked2)


        # do idft saving as complex output
        img_back = np.fft.ifft2(back_ishift, axes=(0,1))
        # img_filtered = np.fft.ifft2(back_ishift_masked, axes=(0,1))
        img_filtered2 = np.fft.ifft2(back_ishift_masked2, axes=(0,1))

        # combine complex real and imaginary components to form (the magnitude for) the original image again
        # multiply by 3 to increase brightness
        img_back = np.abs(img_back).clip(0,255).astype(np.uint8)
        # img = np.abs(3*img_filtered).clip(0,255).astype(np.uint8)
        img = np.abs(3*img_filtered2).clip(0,255).astype(np.uint8)


        # cv2.imshow("ORIGINAL", img)
        # cv2.imshow("SPECTRUM", spec)
        # cv2.imshow("MASK", mask)
        # cv2.imshow("MASK2", mask2)
        # cv2.imshow("ORIGINAL DFT/IFT ROUND TRIP", img_back)
        # cv2.imshow("FILTERED DFT/IFT ROUND TRIP", img_filtered)
        # cv2.imshow("FILTERED2 DFT/IFT ROUND TRIP", img_filtered2)
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
    #         img = temp1_pic
    #     elif(value == 2):
    #         img = temp2_pic
    #     elif(value == 3):
    #         img = temp3_pic
    #     elif(value == 4):
    #         img = temp4_pic
    #     elif(value == 5):
    #         img = temp5_pic
    #     # cv2.imwrite(temp1,cv2.imread(temp2))
    #     # cv2.imwrite(temp2,img)
    #     img = QImage(img)
    #     self.common(img)







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
        self.zoomInAct1 = QAction("Zoom &In (1%)", self, enabled=False, triggered=self.zoomIn1)
        self.zoomOutAct1 = QAction("Zoom &Out (1%)", self, enabled=False, triggered=self.zoomOut1)
        self.zoomInAct5 = QAction("Zoom &In (5%)", self,  enabled=False, triggered=self.zoomIn5)
        self.zoomOutAct5 = QAction("Zoom &Out (5%)", self,  enabled=False, triggered=self.zoomOut5)
        self.zoomInAct10 = QAction("Zoom &In (10%)", self,  enabled=False, triggered=self.zoomIn10)
        self.zoomOutAct10 = QAction("Zoom &Out (10%)", self, enabled=False, triggered=self.zoomOut10)
        self.zoomInAct25 = QAction("Zoom &In (25%)", self, shortcut="Ctrl++", enabled=False, triggered=self.zoomIn25)
        self.zoomOutAct25 = QAction("Zoom &Out (25%)", self, shortcut="Ctrl+-", enabled=False, triggered=self.zoomOut25)
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
        self.lowpassAct = QAction("&LowPassFilter", self, enabled=False, triggered=self.lowpass)
        self.advlowpassAct = QAction("&AdvancedLowPassFilter", self, enabled=False, triggered=self.advlowpass)
        self.highpassAct = QAction("&HighPassFilter", self, enabled=False, triggered=self.highpass)
        self.advhighpassAct = QAction("&AdvancedHighPassFilter", self, enabled=False, triggered=self.advhighpass)
        self.rulerAct = QAction("&Ruler", self, enabled=False, triggered=self.ruler)
        self.smoothenAct = QAction("&Smoothen", self, enabled=False, triggered=self.smoothen)
        self.sharpenAct = QAction("&Sharpen", self, enabled=False, triggered=self.sharpen)
        self.rotateAct = QAction("&Rotate", self, enabled=False, triggered=self.rotate)
        self.rotatemergeAct = QAction("&Rotate_Merge", self, enabled=False, triggered=self.rotate_merge)
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
        self.viewMenu.addAction(self.zoomInAct1)
        self.viewMenu.addAction(self.zoomOutAct1)
        self.viewMenu.addAction(self.zoomInAct5)
        self.viewMenu.addAction(self.zoomOutAct5)
        self.viewMenu.addAction(self.zoomInAct10)
        self.viewMenu.addAction(self.zoomOutAct10)
        self.viewMenu.addAction(self.zoomInAct25)
        self.viewMenu.addAction(self.zoomOutAct25)
        self.viewMenu.addAction(self.normalSizeAct)
        self.viewMenu.addSeparator()
        self.viewMenu.addAction(self.fitToWindowAct)


        self.editMenu = QMenu("&Edit", self)
        self.editMenu.addAction(self.negative_action)
        self.editMenu.addAction(self.cannyedgeAct)
        self.editMenu.addAction(self.sobeledgeAct)
        self.editMenu.addAction(self.prewittedgeAct)
        self.editMenu.addAction(self.normalizeAct)
        self.editMenu.addAction(self.lowpassAct)
        self.editMenu.addAction(self.advlowpassAct)
        self.editMenu.addAction(self.highpassAct)
        self.editMenu.addAction(self.advhighpassAct)
        self.editMenu.addAction(self.rulerAct)
        self.editMenu.addAction(self.smoothenAct)
        self.editMenu.addAction(self.sharpenAct)
        self.editMenu.addAction(self.rotateAct)
        self.editMenu.addAction(self.rotatemergeAct)
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
        self.zoomInAct1.setEnabled(not self.fitToWindowAct.isChecked())
        self.zoomOutAct1.setEnabled(not self.fitToWindowAct.isChecked())
        self.zoomInAct5.setEnabled(not self.fitToWindowAct.isChecked())
        self.zoomOutAct5.setEnabled(not self.fitToWindowAct.isChecked())
        self.zoomInAct10.setEnabled(not self.fitToWindowAct.isChecked())
        self.zoomOutAct10.setEnabled(not self.fitToWindowAct.isChecked())
        self.zoomInAct25.setEnabled(not self.fitToWindowAct.isChecked())
        self.zoomOutAct25.setEnabled(not self.fitToWindowAct.isChecked())
        self.normalSizeAct.setEnabled(not self.fitToWindowAct.isChecked())

    def scaleImage(self, factor):
        self.scaleFactor *= factor
        self.imageLabel.resize(self.scaleFactor * self.imageLabel.pixmap().size())

        self.adjustScrollBar(self.scrollArea.horizontalScrollBar(), factor)
        self.adjustScrollBar(self.scrollArea.verticalScrollBar(), factor)

        # self.zoomInAct1.setEnabled(self.scaleFactor < 3.0)
        # self.zoomOutAct1.setEnabled(self.scaleFactor > 0.333)
        # self.zoomInAct5.setEnabled(self.scaleFactor < 3.0)
        # self.zoomOutAct5.setEnabled(self.scaleFactor > 0.333)
        # self.zoomInAct10.setEnabled(self.scaleFactor < 3.0)
        # self.zoomOutAct10.setEnabled(self.scaleFactor > 0.333)
        # self.zoomInAct25.setEnabled(self.scaleFactor < 3.0)
        # self.zoomOutAct25.setEnabled(self.scaleFactor > 0.333)

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








