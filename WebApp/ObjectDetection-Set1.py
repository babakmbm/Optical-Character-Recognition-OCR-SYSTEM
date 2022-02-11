import numpy as np
import os
import cv2
import pandas as pd
import xml.etree.cElementTree as xet
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import MobileNetV2, InceptionV3, InceptionResNetV2
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.callbacks import TensorBoard
import matplotlib.pyplot as plt

# Now we load the labels.csv that we created
df = pd.read_csv('../images-Set1/labels.csv')
#print(df.head(10))
#print(df.tail(10))

# The dataframe stored all of the "annotation file paths" to the annotation files
# now we need to write a function to
# extract the correct "image file path" for use in opencv
# we need to include the folder too
# we need to parse the label file to get the image name
def get_file_name(filename):
    filename = 'images-Set1/' + filename
    file_name_images = xet.parse(filename).getroot().find('filename').text
    file_path_images = os.path.join('../images-Set1', file_name_images)
    return file_path_images

# Create a list of all image file paths
image_path = list(df['filepath'].apply(get_file_name))
#print(image_path)


'''
# This next part is to test our annotations and see if we can use them with opencv
# now we need to verify and see if we can display an image using this paths
img = cv2.imread(image_path[0])
# This line is to be able to resize the window
cv2.namedWindow('sample image', cv2.WINDOW_NORMAL)
#to show the window
cv2.imshow('sample image', img)
# for window not to disappear before we press ESC
cv2.waitKey(0)
cv2.destroyAllWindows()

# we want to draw the rectangle around
# license plate using the information in labels
# Opencv accepts colours in BGR format
cv2.rectangle(img, (1093, 645), (1396, 727), (0,255,0), 3)
# This line is to be able to resize the window
cv2.namedWindow('sample image with rectangle', cv2.WINDOW_NORMAL)
#to show the window
cv2.imshow('sample image with rectangle', img)
# for window not to disappear before we press ESC
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

#############Preprocessing#################
# Now we do the preprocessing for the data



# Now we need to use our information in the dataframe to get the target values
# which are the bounding box values
# we don't need the file path values we can use the "iloc(rows,cols)" function (index from 0...)
# we choose all the rows and all columns from 1 to 4
#print(df.iloc[:,1:])
# we use .values to put the results into a single array
labels = df.iloc[:, 1:].values
#print(labels)

'''
###### normalizing input images
# we read the first image into an array
# we do this to have the images in number form
# Like i showed you in matlab if you remember
image = image_path[0]
img_arr = cv2.imread(image)
#print(img_arr)

# From the image array we can get the dimensions of each image
# and store them in variables
# height, width and depth (RGB color)
h, w, d = img_arr.shape
#print(h,w,d)

# we usually need the images in a certain size for our algorithm
# so we use load_image() function from tensor flow to load them in with a certain size
# this way the actual image is not damaged and resized
# This is one of the most used preprocessing functions in the industry
load_image = load_img(image, target_size=(224, 224))
load_image_array = img_to_array(load_image)
#print(load_image_array)

# as we can see all the values are large float values for the loaded image array
# so this needs to be normalized to help with computation.
# to do so we divide everything by 255.0
normal_load_image_array = load_image_array/255.0
#print(normal_load_image_array)

'''
####### Normalizing labels (output)
# The next thing is normalization of the bounding box coordinates
# to fit the new image sizes. for that we just divide the
# x values by the width and y values by the height and store them
# in variables nxmin, nxmax, nymin, nymax
# we call this normalizing our labels
'''
i = 0
xmin, xmax, ymin, ymax = labels[i]
nxmin, nxmax = xmin/w, xmax/w
nymin, nymax = ymin/h, ymax/h
label_normal = (nxmin, nxmax, nymin, nymax)

#print(label_normal)
'''
##### this was all for one image now we need to do the same thing
# for all of the images so we will use the a for loop that goes
# through all the image files and normalises them and stores the normalised inputs
# in an array called "data" and the normalised outputs in an array called "output"
'''
'''
data = []
output = []
for ind in range(len(image_path)):
    image = image_path[ind]
    img_arr = cv2.imread(image)
    h, w, d = img_arr.shape
    load_image = load_img(image, target_size=(224, 224))
    load_image_array = img_to_array(load_image)
    normal_load_image_array = load_image_array / 255.0
    xmin, xmax, ymin, ymax = labels[ind]
    nxmin, nxmax = xmin / w, xmax / w
    nymin, nymax = ymin / h, ymax / h
    label_normal = (nxmin, nxmax, nymin, nymax)
    # now append to the data and output arrays
    data.append(normal_load_image_array)
    output.append(label_normal)
#print(data)
#print(output)

'''Here you need to teach regression a bit'''
'''why are we using regression'''
'''what is training and validation and testing?'''
'''
# Now we are going to split the data into training and testing data
# So we can use them in our algorithm
# training data is what we use to teach the algorithm what characters in
# a license plate look like
# we will run the algorithm again using the testing data and measure the error
# to get the accuracy of our algorithm
# Now we need to split the input and output data into training and
# testing data for the training and validation steps
# do do so first we convert both data and output list to np arrays
# we will have more options with the np arrays
'''
X = np.array(data, dtype=np.float32)
y = np.array(output, dtype=np.float32)
# print(X.shape, y.shape)

# we will use the train_test_split function
# from sklearn which we have already imported
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

'''Train a deep learning model'''
'''This is called transfer learning
#Transfer learning is when you use a pre defined model on your own data
# We can use tensorflow that you already downloaded to train
# MobileNetV2, InceptionV3, InceptionResNetV2 are some of the models that we can try
# Research these models and see how they work you need to put that in your report
'''

# build our deep learning model based on InceptionResNetV2
# we will use InceptionResNetV2 algorithm which was trained on the imagenet dataset
# https://keras.io/api/applications/inceptionresnetv2/

# include_top: whether to include the fully-connected layer at the top of the network.
# because we are adding dense layers ourselves.

# weights: one of None (random initialization), 'imagenet' (pre-training on ImageNet),
# or the path to the weights file to be loaded.

# input tensor is our pictures in we created tensors for
# shape: is the shape of each input image
inception_resnet = InceptionResNetV2(weights='imagenet', include_top=False,
                                     input_tensor=Input(shape=(224, 224, 3)))

# means don't change the imageNet weights
inception_resnet.trainable = False

# we add some layers to the end of the model
endModel = inception_resnet.output

# since a CNN will return a matrix we need to flatten it first
endModel = Flatten()(endModel)

# we add two dense layers at the end of the model with relu activation
endModel = Dense(500, activation='relu')(endModel)
endModel = Dense(250, activation='relu')(endModel)

# We add another dense layer with 4
endModel = Dense(4, activation='sigmoid')(endModel)

# Define the model
model = Model(inputs=inception_resnet.input, outputs=endModel)

# compile model
model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))
# model.summary()

# this will save the data in a folder called object_detection_old
#tfb = TensorBoard('object_detection_log_15_07_2021')

# this will train the model
# history = model.fit(x=X_train, y=y_train, batch_size=10, epochs=300, validation_data=(X_test, y_test), callbacks=[tfb])
history = model.fit(x=X_train, y=y_train, batch_size=10, epochs=300, validation_data=(X_test, y_test))
model.save('models/set1_object_detection_Model_15_07_2021.h5')


