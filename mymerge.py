#Method1 : Image averaging
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

directory = 'qubedot_rot'
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
plt.imshow(cv2.cvtColor(merged_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
cv2.imwrite('merged_sperm-specimen_average.bmp',merged_image)
# plt.show()