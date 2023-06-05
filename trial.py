import numpy as np
import cv2
from matplotlib import pyplot as plt

img1 = cv2.imread('trail.tiff')
img = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
# plt.imshow(img)
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
# plt.show()

#blurring or smoothening
# kernelSizes = [(3, 3), (9, 9), (15, 15)]
# # loop over the kernel sizes
# for (kX, kY) in kernelSizes:
# 	# apply an "average" blur to the image using the current kernel
# 	# size
#     kernel2 = np.ones((kX,kY), np.float32)/25
#     # blurred = cv2.filter2D(src=img, ddepth=-1, kernel=kernel2)
#     # plt.imshow(blurred,cmap = 'gray',interpolation = 'bicubic')
#     kernel_sharp = 2*np.eye(kX) - kernel2
#     sharpen = cv2.filter2D(src=img, ddepth=-1, kernel=kernel2)
    
#     plt.imshow(sharpen,cmap = 'gray',interpolation = 'bicubic')

#     plt.show()
#     cv2.waitKey(0)

kX = kY = 9
        # img = cv2.imread(temp2)

kernel2 = np.ones((kX,kY), np.float32)/25
smooth_img = cv2.filter2D(src=img, ddepth=-1, kernel=kernel2)
        # cv2.imwrite(temp1,cv2.imread(temp2))
        # cv2.imwrite(temp2,img)
        # img = QImage(temp2)
        # self.common(img)

#NORMALIZE:

# img_normalized = cv2.normalize(img, None,0,1.0,cv2.NORM_MINMAX,dtype = cv2.CV_32F)
# img_normalized = cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
# Load image as grayscale and crop ROI

# img_normalized = np.array((img - np.min(img)) / (np.max(img) - np.min(img)))

# plt.imshow(img_normalized,interpolation = 'bicubic')

# plt.show()

# CANNY EDGE DETECTION
# for i in range 
# kernel2 = np.ones((8,8), np.float32)/25
# sharpen = cv2.filter2D(src=img, ddepth=-1, kernel=kernel2)
# edges = cv2.Canny(sharpen,80,85)
plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(smooth_img,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()


# Negative image

# colored_negative = abs(255-img)
# plt.subplot(121),plt.imshow(img,cmap = 'gray')
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(colored_negative,cmap = 'gray')
# plt.title('Negative Image'), plt.xticks([]), plt.yticks([])
# plt.show()


#High pass filter
# Low pass same as blurring

# sigma = 200
# img_rst = img - cv2.GaussianBlur(img,(0,0),sigma) + 127
# plt.subplot(121),plt.imshow(img,cmap = 'gray')
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(img_rst,cmap = 'gray')
# plt.title('High pass filter Image'), plt.xticks([]), plt.yticks([])
# plt.show()
