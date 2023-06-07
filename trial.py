# # # import numpy as np
# # # import cv2
# # # from matplotlib import pyplot as plt

# # # img1 = cv2.imread('trail.tiff')
# # # img = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
# # # # plt.imshow(img)
# # # plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
# # # # plt.show()

# # # #blurring or smoothening
# # # # kernelSizes = [(3, 3), (9, 9), (15, 15)]
# # # # # loop over the kernel sizes
# # # # for (kX, kY) in kernelSizes:
# # # # 	# apply an "average" blur to the image using the current kernel
# # # # 	# size
# # # #     kernel2 = np.ones((kX,kY), np.float32)/25
# # # #     # blurred = cv2.filter2D(src=img, ddepth=-1, kernel=kernel2)
# # # #     # plt.imshow(blurred,cmap = 'gray',interpolation = 'bicubic')
# # # #     kernel_sharp = 2*np.eye(kX) - kernel2
# # # #     sharpen = cv2.filter2D(src=img, ddepth=-1, kernel=kernel2)
    
# # # #     plt.imshow(sharpen,cmap = 'gray',interpolation = 'bicubic')

# # # #     plt.show()
# # # #     cv2.waitKey(0)

# # # kX = kY = 9
# # #         # img = cv2.imread(temp2)

# # # kernel2 = np.ones((kX,kY), np.float32)/25
# # # smooth_img = cv2.filter2D(src=img, ddepth=-1, kernel=kernel2)
# # #         # cv2.imwrite(temp1,cv2.imread(temp2))
# # #         # cv2.imwrite(temp2,img)
# # #         # img = QImage(temp2)
# # #         # self.common(img)

# # # #NORMALIZE:

# # # # img_normalized = cv2.normalize(img, None,0,1.0,cv2.NORM_MINMAX,dtype = cv2.CV_32F)
# # # # img_normalized = cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
# # # # Load image as grayscale and crop ROI

# # # # img_normalized = np.array((img - np.min(img)) / (np.max(img) - np.min(img)))

# # # # plt.imshow(img_normalized,interpolation = 'bicubic')

# # # # plt.show()

# # # # CANNY EDGE DETECTION
# # # # for i in range 
# # # # kernel2 = np.ones((8,8), np.float32)/25
# # # # sharpen = cv2.filter2D(src=img, ddepth=-1, kernel=kernel2)
# # # # edges = cv2.Canny(sharpen,80,85)
# # # plt.subplot(121),plt.imshow(img,cmap = 'gray')
# # # plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# # # plt.subplot(122),plt.imshow(smooth_img,cmap = 'gray')
# # # plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
# # # plt.show()


# # # Negative image

# # # colored_negative = abs(255-img)
# # # plt.subplot(121),plt.imshow(img,cmap = 'gray')
# # # plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# # # plt.subplot(122),plt.imshow(colored_negative,cmap = 'gray')
# # # plt.title('Negative Image'), plt.xticks([]), plt.yticks([])
# # # plt.show()


# # #High pass filter
# # # Low pass same as blurring

# # # sigma = 200
# # # img_rst = img - cv2.GaussianBlur(img,(0,0),sigma) + 127
# # # plt.subplot(121),plt.imshow(img,cmap = 'gray')
# # # plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# # # plt.subplot(122),plt.imshow(img_rst,cmap = 'gray')
# # # plt.title('High pass filter Image'), plt.xticks([]), plt.yticks([])
# # # plt.show()
# # #LOW PASS FILTER
# # # import numpy as np
# # # import cv2
# # # img = cv2.imread('nature.jpg')

# # # # do dft saving as complex output
# # # dft = np.fft.fft2(img, axes=(0,1))

# # # # apply shift of origin to center of image
# # # dft_shift = np.fft.fftshift(dft)

# # # # generate spectrum from magnitude image (for viewing only)
# # # mag = np.abs(dft_shift)
# # # spec = np.log(mag) / 20

# # # # create circle mask
# # # radius = 32
# # # mask = np.zeros_like(img)
# # # cy = mask.shape[0] // 2
# # # cx = mask.shape[1] // 2
# # # cv2.circle(mask, (cx,cy), radius, (255,255,255), -1)[0]

# # # # blur the mask
# # # mask2 = cv2.GaussianBlur(mask, (19,19), 0)

# # # # apply mask to dft_shift
# # # dft_shift_masked = np.multiply(dft_shift,mask) / 255
# # # dft_shift_masked2 = np.multiply(dft_shift,mask2) / 255


# # # # shift origin from center to upper left corner
# # # back_ishift = np.fft.ifftshift(dft_shift)
# # # back_ishift_masked = np.fft.ifftshift(dft_shift_masked)
# # # back_ishift_masked2 = np.fft.ifftshift(dft_shift_masked2)


# # # # do idft saving as complex output
# # # img_back = np.fft.ifft2(back_ishift, axes=(0,1))
# # # img_filtered = np.fft.ifft2(back_ishift_masked, axes=(0,1))
# # # img_filtered2 = np.fft.ifft2(back_ishift_masked2, axes=(0,1))

# # # # combine complex real and imaginary components to form (the magnitude for) the original image again
# # # img_back = np.abs(img_back).clip(0,255).astype(np.uint8)
# # # img_filtered = np.abs(img_filtered).clip(0,255).astype(np.uint8)
# # # img_filtered2 = np.abs(img_filtered2).clip(0,255).astype(np.uint8)


# # # cv2.imshow("ORIGINAL", img)
# # # # cv2.imshow("SPECTRUM", spec)
# # # # cv2.imshow("MASK", mask)
# # # # cv2.imshow("MASK2", mask2)
# # # # cv2.imshow("ORIGINAL DFT/IFT ROUND TRIP", img_back)
# # # cv2.imshow("FILTERED DFT/IFT ROUND TRIP", img_filtered)
# # # #  REDUCED RINGING
# # # cv2.imshow("FILTERED2 DFT/IFT ROUND TRIP", img_filtered2) 
# # # cv2.waitKey(0)
# # # cv2.destroyAllWindows()
# # import cv2

# # class DrawLineWidget(object):
# #     def __init__(self):
# #         self.original_image = cv2.imread('nature.jpg')
# #         self.clone = self.original_image.copy()

# #         cv2.namedWindow('image')
# #         cv2.setMouseCallback('image', self.extract_coordinates)

# #         # List to store start/end points
# #         self.image_coordinates = []

# #     def extract_coordinates(self, event, x, y, flags, parameters):
# #         # Record starting (x,y) coordinates on left mouse button click
# #         if event == cv2.EVENT_LBUTTONDOWN:
# #             self.image_coordinates = [(x,y)]

# #         # Record ending (x,y) coordintes on left mouse bottom release
# #         elif event == cv2.EVENT_LBUTTONUP:
# #             self.image_coordinates.append((x,y))
# #             print('Starting: {}, Ending: {}'.format(self.image_coordinates[0], self.image_coordinates[1]))
# #             import math
# #             print('Angle: {}'.format(abs(math.degrees(math.atan((self.image_coordinates[1][1] - self.image_coordinates[0][1])/(self.image_coordinates[1][0] - self.image_coordinates[0][0]))) )))

# #             # Draw line
# #             cv2.line(self.clone, self.image_coordinates[0], self.image_coordinates[1], (36,255,12), 2)
# #             cv2.imshow("image", self.clone) 

# #         # Clear drawing boxes on right mouse button click
# #         elif event == cv2.EVENT_RBUTTONDOWN:
# #             self.clone = self.original_image.copy()

# #     def show_image(self):
# #         return self.clone

# # if __name__ == '__main__':
# #     draw_line_widget = DrawLineWidget()
# #     while True:
# #         cv2.imshow('image', draw_line_widget.show_image())
# #         key = cv2.waitKey(1)

# #         # Close program with keyboard 'q'
# #         if key == ord('q'):
# #             cv2.destroyAllWindows()
# #             exit(1)


# # This will import all the widgets
# # and modules which are available in
# # tkinter and ttk module
# from tkinter import *
# from tkinter.ttk import *

# # creates a Tk() object
# master = Tk()

# # sets the geometry of main
# # root window
# master.geometry("200x200")


# # function to open a new window
# # on a button click
# def openNewWindow():
	
# 	# Toplevel object which will
# 	# be treated as a new window
# 	newWindow = Toplevel(master)

# 	# sets the title of the
# 	# Toplevel widget
# 	newWindow.title("New Window")

# 	# sets the geometry of toplevel
# 	newWindow.geometry("200x200")

# 	# A Label widget to show in toplevel
# 	Label(newWindow,
# 		text ="This is a new window").pack()


# label = Label(master,
# 			text ="This is the main window")

# label.pack(pady = 10)

# # a button widget which will open a
# # new window on button click
# btn = Button(master,
# 			text ="Click to open a new window",
# 			command = openNewWindow)
# btn.pack(pady = 10)

# # mainloop, runs infinitely
# mainloop()

import tkinter as tk


root = Tk()

# specify size of window.
root.geometry("250x170")

# Create text widget and specify size.
T = Text(root, height = 5, width = 52)

# Create label
l = Label(root, text = "Fact of the Day")
l.config(font =("Courier", 14))

Fact = """A man can be arrested in
Italy for wearing a skirt in public."""

# Create button for next text.
b1 = Button(root, text = "Next", )

# Create an Exit button.
b2 = Button(root, text = "Exit",
			command = root.destroy)

l.pack()
T.pack()
b1.pack()
b2.pack()

# Insert The Fact.
T.insert(tk.END, Fact)

tk.mainloop()

