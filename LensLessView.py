from PyQt5 import QtWidgets, QtGui, QtCore
from LedHWdriver import *
from PyQt5.QtCore import QTimer, QEventLoop
from pypylon import pylon

import time
import LenslessLite
import sys
import cv2
import os
import threading
import time
import queue
import numpy as np
import math
import ctypes


capture_thread = None
######form_class = uic.loadUiType("Lensless GUI.ui")[0]
q = queue.Queue()
running = False

def grab(queue, width, height, fps):
    global running
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    camera.Open()
    camera.ExposureAuto.Value = 'Continuous'
    camera.GainAuto.Value = 'Continuous'
    camera.Width.Value = width
    camera.Height.Value = height
    camera.Close()
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly) 
    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
    frame_set=[]
    while(running):
        frame = {}
        grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        if grabResult.GrabSucceeded():
            image = converter.Convert(grabResult)
            img = image.GetArray()
            frame["img"] = img
            if queue.qsize() < 10:
                queue.put(frame)
        grabResult.Release()
    camera.StopGrabbing()


class OwnImageWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(OwnImageWidget, self).__init__(parent)
        self.image = None
    def setImage(self, image):
        self.image = image
        sz = image.size()
        self.setMinimumSize(sz)
        self.update()
    def paintEvent(self, event):
        qp = QtGui.QPainter()
        qp.begin(self)
        if self.image:
            qp.drawImage(QtCore.QPoint(0, 0), self.image)
        qp.end()

class LenslessApp(QtWidgets.QMainWindow, LenslessLite.Ui_MainWindow):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.setupUi(self)
        self.btnLEDOn.clicked.connect(self.LEDOn)
        self.btnLEDOff.clicked.connect(self.LEDOff)
        self.btnStartCamera.clicked.connect(self.startCamera)
        self.btnStopCamera.clicked.connect(self.stopCamera)
        self.btnStartRec.clicked.connect(self.startRecord)
        self.btnCaptured.clicked.connect(self.savepic)
        self.window_width = self.widget.frameSize().width()
        self.window_height = self.widget.frameSize().height()
        self.widget = OwnImageWidget(self.widget)
        self.recTime.setText('1')
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(1)
        self.btnCaptured.setEnabled(False)
        self.btnStartRec.setEnabled(False)
        self.running = False
        self.color = "No Led"

    def LEDOn (self):
        self.LEDOff()
        self.color = "No Led"
        if self.RedLED.isChecked():
            self.RedOn()
            self.color = "Red Led"
            self.Statuslbl.setText("Red is on")
        if self.GreenLED.isChecked():
            self.GreenOn()
            self.color = "Green Led"
            self.Statuslbl.setText("Green is on")
        if self.BlueLED.isChecked():
            self.BlueOn()
            self.color = "Blue Led"
            self.Statuslbl.setText("Blue is on")
        if self.GreenLED.isChecked() & self.BlueLED.isChecked():
            self.color = "Green-Blue Led"
        if self.RedLED.isChecked() & self.BlueLED.isChecked():
            self.color = "Red-Blue Led"
        if self.RedLED.isChecked() & self.GreenLED.isChecked():
            self.color = "Red-Green Led"
        if self.RedLED.isChecked() & self.GreenLED.isChecked()& self.BlueLED.isChecked():
            self.color = "Red-Green-Blue Led"
        self.Statuslbl.setText(str(self.color) + " is On")
        
          
    def RedOn (self):
        a.digital_write(5, 1)
        
    def GreenOn (self):
        a.digital_write(6, 1)
        

    def BlueOn (self):
        a.digital_write(7, 1)
        
         
    def LEDOff (self):
        self.Statuslbl.setText("Led is off")
        a.digital_write(5, 0)
        a.digital_write(6, 0)
        a.digital_write(7, 0)
        
    def startCamera(self):
        global running
        self.btnCaptured.setEnabled(True)
        self.btnStartRec.setEnabled(True)
        running = True
        #capture_thread = threading.Thread(target=grab, args = (1, q, 648, 486, 14))
        capture_thread = threading.Thread(target=grab, args = (q, 2592, 1944, 14))
        capture_thread.start()
        self.btnStartCamera.setEnabled(False)
        self.btnStartCamera.setText('Camera On...')
        self.Statuslbl.setText("Camera is ready")
        
    def stopCamera(self):
        global running
        running = False
        self.btnStartCamera.setEnabled(True)
        self.btnStartCamera.setText('Start Live View')
        self.Statuslbl.setText("Camera is not connected")
        self.btnCaptured.setEnabled(False)
        self.btnStartRec.setEnabled(False)

    def savepic(self):
        
        frame = q.get()
        img = frame["img"]
        dirname = os.path.join(os.path.dirname(__file__),r"images")
        img_name = os.path.join(dirname,r'{}.tiff'.format(((self.color), time.strftime("%Y%m%d-%H%M%S"))))

        cv2.imwrite(img_name,img)
        
        

    def startRecord(self):
        self.btnCaptured.setEnabled(False)
        self.btnStartRec.setEnabled(False)
        self.recTime.setEnabled(False)
        #self.stopCamera()
        #time.sleep(1)
        self.Statuslbl.setText("Recording...")
        def recording():
            frame = q.get()
            img = frame["img"]
            out.write(img)
            height, width, bpc = img.shape
            bpl = bpc * width
            image = QtGui.QImage(img.data, width, height, bpl, QtGui.QImage.Format_RGB888)
            self.widget.setImage(image)
        self.btnStartCamera.setEnabled(False)    
        capturing=cv2.VideoCapture(1)
        capturing.set(3,2592)
        capturing.set(4,1944)
        frame_set=[]
        start_time=time.time()
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(('Live View {}.avi'.format((time.strftime("%Y%m%d-%H%M%S")))),fourcc, 7,(2592, 1944))
        rectime = int(self.recTime.toPlainText())*60000
        rectimer = QTimer()
        rectimer.singleShot(rectime,rectimer.stop)
        rectimer.singleShot((rectime+1000),out.release)
        rectimer.singleShot((rectime+1200),capturing.release)
        rectimer.singleShot((rectime+1800),self.startCamera)
        rectimer.singleShot(rectime+1500,self.stopCamera)
        recloop = QEventLoop()
        rectimer.timeout.connect(recording)
        rectimer.start(10)   
        recloop.exec_()
        self.Statuslbl.setText("Video saved")

    def update_frame(self):
            
        if not q.empty():
            frame = q.get()
            img = frame["img"]
            img_height, img_width, img_colors = img.shape
            scale_w = float(self.window_width) / float(img_width)
            scale_h = float(self.window_height) / float(img_height)
            scale = min([scale_w, scale_h])
            
            if scale == 0:
                scale = 1
                
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation = cv2.INTER_CUBIC)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            height, width, bpc = img.shape
            bpl = bpc * width
            image = QtGui.QImage(img.data, width, height, bpl, QtGui.QImage.Format_RGB888)
            self.widget.setImage(image)

    def closeEvent(self, event):
        global running
        running = False
    
def main():
    
    app = QtWidgets.QApplication(sys.argv)
    app_icon = QtGui.QIcon()
    app_icon.addFile('IHTLogo.gif', QtCore.QSize(32,32))
    app.setWindowIcon(app_icon)
    form = LenslessApp()
    form.setWindowTitle('Lensless Live View')
    form.show() 
    app.exec_() 
    
if __name__ == '__main__': 
    try:
        a = LedHardware()
        main()
    except:
        ctypes.windll.user32.MessageBoxA(0, "Please connect the hardware via USB port", "Hardware ERROR", 0)
    a.digital_write(5, 0)
    a.digital_write(6, 1)
    a.digital_write(7, 0)












'''copyright: Agus Budi Dharmawan, Adyasha Patra, Umang Agarwal - IHT, TU Braunschweig'''


