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
from tkinter import Tk, simpledialog
from tkinter import messagebox
from tkinter.filedialog import askopenfilename

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
        self.trackerAct.setEnabled(True)
        # self.rulerclose.setEnabled(True)
        self.smoothenAct.setEnabled(True)
        self.sharpenAct.setEnabled(True)
        self.rotateAct.setEnabled(True)
        self.rotateaverageAct.setEnabled(True)
        self.rotatemergeAct.setEnabled(True)
        self.imagesegmentAct.setEnabled(True)
        self.averageAct.setEnabled(True)
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


    def zoomIn(self):
        root = Tk()
        root.withdraw()
        k = simpledialog.askinteger("Input", "Enter the percentage of zoom in:")
        self.scaleImage(1+(k/100))

    def zoomOut(self):
        root = Tk()
        root.withdraw()
        k = simpledialog.askinteger("Input", "Enter the percentage of zoom out:")
        self.scaleImage(1-(k/100))


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

        base_image_extension = os.path.splitext(fileName)[1]

        finalimages = []
        i = 0

            # Create the Tkinter root window
        root = Tk()

            # Hide the root window
        root.withdraw()

            # Prompt the user for input
        user_input = simpledialog.askstring("Input", "Enter the folder you want your rotated images in")
        #directory = 'RotationExperiment'
        for filename in os.listdir(directory):
            f = os.path.join(directory, filename)
            image2 = cv2.imread(f)
            gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
            wid = image2.shape[1]
            hgt = image2.shape[0]

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
            scaleRecovered = np.sqrt(sc ** 2 + ss ** 2)
            thetaRecovered = np.degrees(np.arctan2(-ss, sc))

            print('Recovered scale:', scaleRecovered)
            print('Recovered rotation angle:', thetaRecovered)

            # Calculate the output size based on the larger image dimensions
            outputSize = (max(image1.shape[1], image2.shape[1]), max(image1.shape[0], image2.shape[0]))

            alignedImage = cv2.warpAffine(image2, tform, outputSize, flags=cv2.INTER_NEAREST)

            fig = plt.figure(figsize=(10, 7))
            rows = 1
            columns = 3

         

            folder_path = user_input  # Specify the folder path

            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
                #print("Folder created successfully.")
            # else:
                #print("Folder already exists.")

            aligned_image_path = f'{user_input}/{i}{base_image_extension}'
            cv2.imwrite(aligned_image_path, alignedImage)
            i = i + 1
            wid = alignedImage.shape[1]
            hgt = alignedImage.shape[0]

            # displaying the dimensions
            #print(str(wid) + "x" + str(hgt))
            finalimages.append(alignedImage)

        directory = user_input

        def findHomography(image_1_kp, image_2_kp, matches):
            image_1_points = np.zeros((len(matches), 1, 2), dtype=np.float32)
            image_2_points = np.zeros((len(matches), 1, 2), dtype=np.float32)

            for i in range(0,len(matches)):
                image_1_points[i] = image_1_kp[matches[i].queryIdx].pt
                image_2_points[i] = image_2_kp[matches[i].trainIdx].pt


            homography, mask = cv2.findHomography(image_1_points, image_2_points, cv2.RANSAC, ransacReprojThreshold=2.0)

            return homography

        #   Align the images so they overlap properly...

        def align_images(images):

            #   SIFT generally produces better results, but it is not FOSS, so chose the feature detector
            #   that suits the needs of your project.  ORB does OK
            use_sift = True

            outimages = []

            if use_sift:
                detector = cv2.SIFT_create()
            else:
                detector = cv2.ORB_create(1000)

            #   We assume that image 0 is the "base" image and align everything to it
            print ("Detecting features of base image")
            outimages.append(images[0])
            image1gray = cv2.cvtColor(images[0],cv2.COLOR_BGR2GRAY)
            image_1_kp, image_1_desc = detector.detectAndCompute(image1gray, None)

            for i in range(1,len(images)):
                print ("Aligning image {}".format(i))
                image_i_kp, image_i_desc = detector.detectAndCompute(images[i], None)

                if use_sift:
                    bf = cv2.BFMatcher()
                    # This returns the top two matches for each feature point (list of list)
                    pairMatches = bf.knnMatch(image_i_desc,image_1_desc, k=2)
                    rawMatches = []
                    for m,n in pairMatches:
                        if m.distance < 0.7*n.distance:
                            rawMatches.append(m)
                else:
                    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                    rawMatches = bf.match(image_i_desc, image_1_desc)

                sortMatches = sorted(rawMatches, key=lambda x: x.distance)
                matches = sortMatches[0:128]



                hom = findHomography(image_i_kp, image_1_kp, matches)
                newimage = cv2.warpPerspective(images[i], hom, (images[i].shape[1], images[i].shape[0]), flags=cv2.INTER_LINEAR)

                outimages.append(newimage)
                # If you find that there's a large amount of ghosting, it may be because one or more of the input
                # images gets misaligned.  Outputting the aligned images may help diagnose that.
                # cv2.imwrite("aligned{}.png".format(i), newimage)



            return outimages

        #
        #   Compute the gradient map of the image
        def doLap(image):

            # YOU SHOULD TUNE THESE VALUES TO SUIT YOUR NEEDS
            kernel_size = 3         # Size of the laplacian window
            blur_size = 3           # How big of a kernal to use for the gaussian blur
                                    # Generally, keeping these two values the same or very close works well
                                    # Also, odd numbers, please...

            blurred = cv2.GaussianBlur(image, (blur_size,blur_size), 0)
            return cv2.Laplacian(blurred, cv2.CV_64F, ksize=kernel_size)

        
        #   This routine finds the points of best focus in all images and produces a merged result...
        def focus_stack(unimages):
            images = align_images(unimages)

            print ("Computing the laplacian of the blurred images")
            laps = []
            for i in range(len(images)):
                print ("Lap {}".format(i))
                laps.append(doLap(cv2.cvtColor(images[i],cv2.COLOR_BGR2GRAY)))

            laps = np.asarray(laps)
            print ("Shape of array of laplacians = {}".format(laps.shape))

            output = np.zeros(shape=images[0].shape, dtype=images[0].dtype)

            abs_laps = np.absolute(laps)
            maxima = abs_laps.max(axis=0)
            bool_mask = abs_laps == maxima
            mask = bool_mask.astype(np.uint8)
            for i in range(0,len(images)):
                output = cv2.bitwise_not(images[i],output, mask=mask[i])
                
            return 255-output


        def stackHDRs(image_files):
            focusimages = []
            for img in image_files:
                print ("Reading in file {}".format(img))
                imge = cv2.imread(f"{user_input}/{img}")
                crop_img = imge#[y:y+h, x:x+w]
                focusimages.append(crop_img)
                #focusimages.append(cv2.imread("Zcombi_1902_1926/{}".format(img)))

            
            merged = focus_stack(focusimages)
            cv2.imwrite("multi-focus-result.bmp", merged)
            cv2.imwrite(temp1,cv2.imread(temp2))
            cv2.imwrite(temp2, merged)
            merged = QImage(temp2)
            self.common(merged)

        # global merged
        image_files = sorted(os.listdir(user_input))
        for img in image_files:
            if img.split(".")[-1].lower() not in ["tif", "tiff", "png","jpg","jpeg","bmp"]:
                image_files.remove(img)


        stackHDRs(image_files)
        print ("FINISHED!!!")


    def rotate_avg(self):
        image1 = cv2.imread(temp2)
        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        directory = os.path.dirname(fileName)

        base_image_extension = os.path.splitext(fileName)[1]

        finalimages = []
        i = 0
        root = Tk()

            # Hide the root window
        root.withdraw()

            # Prompt the user for input
        user_input = simpledialog.askstring("Input", "Enter the folder in which you want your rotated images")
        for filename in os.listdir(directory):
            f = os.path.join(directory, filename)
            image2 = cv2.imread(f)
            gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
            wid = image2.shape[1]
            hgt = image2.shape[0]

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
            scaleRecovered = np.sqrt(sc ** 2 + ss ** 2)
            thetaRecovered = np.degrees(np.arctan2(-ss, sc))

            print('Recovered scale:', scaleRecovered)
            print('Recovered rotation angle:', thetaRecovered)

            # Calculate the output size based on the larger image dimensions
            outputSize = (max(image1.shape[1], image2.shape[1]), max(image1.shape[0], image2.shape[0]))

            alignedImage = cv2.warpAffine(image2, tform, outputSize, flags=cv2.INTER_NEAREST)

            fig = plt.figure(figsize=(10, 7))
            rows = 1
            columns = 3

         

            folder_path = user_input  # Specify the folder path

            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
                #print("Folder created successfully.")
            # else:
                #print("Folder already exists.")

            aligned_image_path = f'{user_input}/{i}{base_image_extension}'
            cv2.imwrite(aligned_image_path, alignedImage)
            i = i + 1
            wid = alignedImage.shape[1]
            hgt = alignedImage.shape[0]

            # displaying the dimensions
            #print(str(wid) + "x" + str(hgt))
            finalimages.append(alignedImage)

        directory = user_input

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
            print(image.shape)
            merged_image = merged_image + image
            print(merged_image.shape)

# Average the two images
        merged_image = merged_image/i

# Convert the merged image back to the uint8 format
        merged_image = (merged_image * 255).astype(np.uint8)

# Display the merged image
            # plt.imshow(cv2.cvtColor(merged_image, cv2.COLOR_BGR2RGB))
            # plt.axis('off')
        cv2.imwrite('merged_sperm-specimen_average.bmp',merged_image)

        cv2.imwrite(temp1,cv2.imread(temp2))
        cv2.imwrite(temp2, merged_image)
        merged_image = QImage(temp2)
        self.common(merged_image)


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


            # Calculate the output size based on the larger image dimensions
        outputSize = (max(image1.shape[1], image2.shape[1]), max(image1.shape[0], image2.shape[0]))

        img = cv2.warpAffine(image2, tform, outputSize, flags=cv2.INTER_NEAREST)
        wid = img.shape[1]
        hgt = img.shape[0]

        cv2.imwrite(temp1,cv2.imread(temp2))
        cv2.imwrite(temp2, img)
        img = QImage(temp2)
        self.common(img)


        #finalimages.append(alignedImage)

    ref_point = []
    cropping = False

 

    def average(self):
        root = Tk()

            # Hide the root window
        root.withdraw()

            # Prompt the user for input
        user_input = simpledialog.askstring("Input", "Enter the name of folder in which you have rotated images")    
        merged_image = 0
        i = 0
        directory = user_input
        for filename in os.listdir(directory):
            f = os.path.join(directory, filename)
            i = i+1
# for image in image_files:

            image = cv2.imread(f)
# Convert images to floating-point format

# for image in finalimages:
            image = image.astype(np.float32)/255.0
            #rint(image.shape)
            merged_image = merged_image + image
            #print(merged_image.shape)
        merged_image = merged_image/i

# Convert the merged image back to the uint8 format
        merged_image = (merged_image * 255).astype(np.uint8)

# Display the merged image
        cv2.imwrite('merged_sperm-specimen_average.bmp',merged_image)
        # temp1  = "temp1.jpg"
        # temp2 = "temp2.jpg"
            
        #cv2.imwrite(temp1,cv2.imread(temp2))
        cv2.imwrite(temp2, merged_image)
        merged_image = QImage(temp2)
        self.common(merged_image)



    def image_segmentation(self):
        imgsegment = self.imagesegmentAct.isChecked()
        # # Convert numpy array to OpenCV image
        image = cv2.imread(temp2)
        

        def click_and_crop(event, x, y, flags, param):
            global ref_point, cropping

            if event == cv2.EVENT_LBUTTONDOWN:
                ref_point = [(x, y)]
                cropping = True

            elif event == cv2.EVENT_LBUTTONUP:
                ref_point.append((x, y))
                cropping = False

                # Display the ROI
                # print(ref_point)
                cv2.rectangle(image, ref_point[0], ref_point[1], (0, 255, 0), 2)
                cv2.imshow("Image", image)

        # Load the image
        image_path = fileName  # Path to your image
        image = cv2.imread(image_path)
        clone = image.copy()

        cv2.namedWindow("Image")
        cv2.setMouseCallback("Image", click_and_crop)
        print("Press c to confirm template")
        print("Press r to reset")
        while True:
            cv2.imshow("Image", image)
            

            key = cv2.waitKey(1) & 0xFF

            if key == ord("r"):  # Reset the image
                image = clone.copy()

            elif key == ord("c"):  # Crop the selected ROI as template
                # print(ref_point)
                if len(ref_point) == 2:
                    template = clone[ref_point[0][1]:ref_point[1][1], ref_point[0][0]:ref_point[1][0]]
                    # cv2.imshow("Template", template)
                    cv2.imwrite("template.jpg", template)
                    print("Template selection done")
                    print("Press q")
                else:
                    print("Please select a valid ROI.")

            elif key == ord("q"):  # Quit the program
                break

        cv2.destroyAllWindows()

        def count_cells(image, template):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            h, w = template_gray.shape

            # Apply template matching
            result = cv2.matchTemplate(gray, template_gray, cv2.TM_CCOEFF_NORMED)

            # Apply a threshold to obtain the binary mask of the detected cells
            threshold = 0.7  # Adjust the threshold as per your requirement
            loc = np.where(result >= threshold)

            # Perform non-maximum suppression to remove duplicate detections
            points = np.array(loc[::-1]).T
            suppress = np.zeros((len(points),), dtype=bool)
            for i in range(len(points)):
                if not suppress[i]:
                    for j in range(i+1, len(points)):
                        if not suppress[j] and \
                        abs(points[i][0] - points[j][0]) <= w and \
                        abs(points[i][1] - points[j][1]) <= h:
                            suppress[j] = True

            # Filter out suppressed points
            filtered_points = points[np.logical_not(suppress)]

            # Count the number of cells
            cell_count = len(filtered_points)

            # Draw bounding boxes around the detected cells
            for pt in filtered_points:
                x, y = pt
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            return cell_count, image

        # Load the main image
        image_path = fileName  # Path to your image
        image = cv2.imread(image_path)

        # Load the template image
        template_path = 'template.jpg'  # Path to your template image
        template = cv2.imread(template_path)

        # Perform cell counting using template matching
        count, counted_image = count_cells(image, template)

        # Display the counted image
        # cv2.imshow('Counted Image', counted_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # Print the cell count
        print(f"Number of cells detected: {count}")
        cv2.imwrite(temp1,cv2.imread(temp2))
        cv2.imwrite(temp2,counted_image)
        counted_image = QImage(temp2)
        self.common(counted_image)



    def objectTracking(self):
        TrDict = {'csrt': cv2.legacy.TrackerCSRT_create,
                'kcf': cv2.legacy.TrackerKCF_create,
                'boosting': cv2.legacy.TrackerBoosting_create,
                'mil': cv2.legacy.TrackerMIL_create,
                'tld': cv2.legacy.TrackerTLD_create,
                'medianflow': cv2.legacy.TrackerMedianFlow_create,
                'mosse': cv2.legacy.TrackerMOSSE_create}

        trackers = cv2.legacy.MultiTracker_create()

        # Open file dialog to select video
        Tk().withdraw()  # Hide the Tkinter main window
        video_path = askopenfilename(title="Select Video")  # Show the file dialog

        video_capture = cv2.VideoCapture(video_path)

        if not video_capture.isOpened():
            print("Error opening video file:", video_path)
            exit()

        ret, frame = video_capture.read()
        
        # Prompt user to enter the value of k
        root = Tk()
        root.withdraw()
        k = simpledialog.askinteger("Input", "Enter the number of objects to be tracked:")

        messagebox.showinfo(title="Procedure", message="Select your objects with cursor and click 'enter' after each selection. Press q to exit")

        for i in range(k):
            cv2.imshow('Frame', frame)
            bbi = cv2.selectROI('Frame', frame)
            tracker_i = TrDict['csrt']()
            trackers.add(tracker_i, frame, bbi)

        while True:
            ret, frame = video_capture.read()
            if not ret:
                break

            (success, boxes) = trackers.update(frame)
            for i in range(k):
                if success:
                    (x, y, w, h) = [int(a) for a in boxes[i]]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 205, 200), 2)
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(100,205,200),2)
                    cv2.putText(frame, str(x+(w/2)) + ", ", (150,20+20*i), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),2)
                    cv2.putText(frame, str(y+(h/2)), (225,20+20*i), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),2)
                    cv2.putText(frame, "Coordinate of " + str(i+1) +": ", (5,20+20*i), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0),2)

            cv2.imshow('Frame', frame)
            key = cv2.waitKey(5) & 0xFF
            if key == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()

        
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

        cv2.imwrite(temp1,cv2.imread(temp2))
        cv2.imwrite(temp2,img)
        img = QImage(temp2)
        self.common(img)


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
        self.zoomInAct = QAction("Zoom &In (k%)", self,  enabled=False, triggered=self.zoomIn)
        self.zoomOutAct = QAction("Zoom &Out (k%)", self, enabled=False, triggered=self.zoomOut)
        self.zoomInAct25 = QAction("Zoom &In (25%)", self, shortcut="Ctrl++", enabled=False, triggered=self.zoomIn25)
        self.zoomOutAct25 = QAction("Zoom &Out (25%)", self, shortcut="Ctrl+-", enabled=False, triggered=self.zoomOut25)
        self.normalSizeAct = QAction("&Normal Size", self, shortcut="Ctrl+S", enabled=False, triggered=self.normalSize)
        self.fitToWindowAct = QAction("&Fit to Window", self, enabled=False, checkable=True, shortcut="Ctrl+F",
                                      triggered=self.fitToWindow)
        ########################################################################################################
        self.negative_action = QAction("&Negative", self, enabled=False, triggered=self.negative)
        self.cannyedgeAct = QAction("&Canny Edge", self, enabled=False, triggered=self.cannyedge)
        self.undoAct = QAction("&Undo...", self, shortcut="Ctrl+Z", triggered=self.undo)
        self.normalizeAct = QAction("&Normalize", self, enabled=False, triggered=self.normalize)
        self.sobeledgeAct = QAction("&Sobel Edge", self, enabled=False, triggered=self.sobeledge)
        self.prewittedgeAct = QAction("&Prewitt Edge", self, enabled=False, triggered=self.prewittedge)
        self.lowpassAct = QAction("&Low Pass Filter", self, enabled=False, triggered=self.lowpass)
        self.advlowpassAct = QAction("&Advanced LowPass Filter", self, enabled=False, triggered=self.advlowpass)
        self.highpassAct = QAction("&High Pass Filter", self, enabled=False, triggered=self.highpass)
        self.advhighpassAct = QAction("&Advanced HighPass Filter", self, enabled=False, triggered=self.advhighpass)
        self.rulerAct = QAction("&Ruler", self, enabled=False, triggered=self.ruler)
        self.smoothenAct = QAction("&Smoothen", self, enabled=False, triggered=self.smoothen)
        self.sharpenAct = QAction("&Sharpen", self, enabled=False, triggered=self.sharpen)
        self.rotateAct = QAction("&Rotate", self, enabled=False, triggered=self.rotate)
        self.rotateaverageAct = QAction("&Rotate Average", self, enabled=False, triggered=self.rotate_avg)
        self.rotatemergeAct = QAction("&Rotate Merge", self, enabled=False, triggered=self.rotate_merge)
        self.imagesegmentAct = QAction("&Image segmentation", self, enabled=False, triggered=self.image_segmentation)
        self.averageAct = QAction("&Image Averaging", self, triggered=self.average)
        self.trackerAct = QAction("&Multi Object Tracking", self, triggered=self.objectTracking)
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


        self.editMenu = QMenu("&Basic IP", self)
        self.editMenu.addAction(self.negative_action)
        self.editMenu.addAction(self.normalizeAct)
        self.editMenu.addAction(self.rulerAct)
        self.editMenu.addAction(self.smoothenAct)
        self.editMenu.addAction(self.sharpenAct)


        self.rotMenu = QMenu("&Advanced IP", self)
        self.rotMenu.addAction(self.rotateAct)
        self.rotMenu.addAction(self.rotateaverageAct)
        self.rotMenu.addAction(self.rotatemergeAct)
        self.rotMenu.addAction(self.imagesegmentAct)
        self.rotMenu.addAction(self.trackerAct)
        self.rotMenu.addAction(self.averageAct)
        self.rotMenu.addSeparator()
        # self.viewMenu.addSeparator()
        # self.viewMenu.addAction(self.sharpen)

        self.filterMenu = QMenu("&Filters", self)
        self.filterMenu.addAction(self.lowpassAct)
        self.filterMenu.addAction(self.advlowpassAct)
        self.filterMenu.addAction(self.highpassAct)
        self.filterMenu.addAction(self.advhighpassAct)

        self.edgeMenu = QMenu("&Edge Detection", self)
        self.edgeMenu.addAction(self.cannyedgeAct)
        self.edgeMenu.addAction(self.sobeledgeAct)
        self.edgeMenu.addAction(self.prewittedgeAct)

        self.helpMenu = QMenu("&Help", self)
        self.helpMenu.addAction(self.aboutAct)
        self.helpMenu.addAction(self.aboutQtAct)

        self.menuBar().addMenu(self.fileMenu)
        self.menuBar().addMenu(self.viewMenu)
        self.menuBar().addMenu(self.editMenu)
        self.menuBar().addMenu(self.edgeMenu)
        self.menuBar().addMenu(self.filterMenu)
        self.menuBar().addMenu(self.rotMenu)
        self.menuBar().addMenu(self.helpMenu)

    def updateActions(self):
        self.zoomInAct.setEnabled(not self.fitToWindowAct.isChecked())
        self.zoomOutAct.setEnabled(not self.fitToWindowAct.isChecked())
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


    def adjustScrollBar(self, scrollBar, factor):
        scrollBar.setValue(int(factor * scrollBar.value()
                               + ((factor - 1) * scrollBar.pageStep() / 2)))
    
    

if __name__ == '__main__':
    import sys
    from PyQt5.QtWidgets import QApplication

    temp1  = "temp1.jpg"
    temp2 = "temp2.jpg"
    app = QApplication(sys.argv)
    imageViewer = QImageViewer()
    imageViewer.show()
    sys.exit(app.exec_())








