import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

model = tf.keras.models.load_model('../models/set1_object_detection_Model_15_07_2021.h5')
print('Model is Loaded!')
path = '../Images/test1.jpeg'
image = load_img(path)
image = np.array(image, dtype=np.uint8)
# resize the image to fit our network
image1 = load_img(path, target_size=(224, 224))
# convert image into array and normalize it all at the same time
img_arr_norm = img_to_array(image1)/255.0

h, w, d = image.shape
print('height: ', h)
print('width: ', w)
print('Channels: ', d)
'''
plt.figure(figsize=(10, 8))
plt.imshow(image)
plt.show()
'''
#print(img_arr_norm.shape)
# we reshape because our inputs had a number of images when we trained the algorithm
# (num of images, width, height, depth)
test_arr = img_arr_norm.reshape(1, 224, 224, 3)
#print(test_arr)

# normalized values of the coordinates of the labels
normalized_coords = model.predict(test_arr)
#print(normalized_coords)

# we need to denormalize these values to be able to get real numbers
# xmin, xmax, ymin, ymax
# the fist two values need to be multiplied by the width
# the next two values need to be multiplied by the height
# this is the opposite of what we did in the preprocessing
# we create a vector and multiply it by the prediction vector (element-wise multiplication)
de_norm_array = np.array([w, w, h, h])
#print(de_norm_array)

real_coords = (normalized_coords * de_norm_array).astype(np.int32)
print(real_coords)

xmin, ymin, xmax, ymax = real_coords[0]
pt1 = (xmin, ymin)
pt2 = (xmax, ymax)

print(pt1, pt2)

cv2.rectangle(image, pt1, pt2, (0, 255, 0), 3)

plt.figure(figsize=(10, 8))
plt.imshow(image)
plt.show()

img = np.array(load_img(path))
print(img)
# crop the region of interest (roi) using the predicted coordinates
xmin, ymin, xmax, ymax = real_coords[0]
#print(xmin, ymin, xmax, ymax)
roi = img[ymin:ymax, xmin:xmax]
plt.imshow(roi)
plt.show()

roi_gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
plt.imshow(roi_gray)
plt.show()

plt.imshow(cv2.cvtColor(roi_gray, cv2.COLOR_BGR2RGB))
plt.show()


