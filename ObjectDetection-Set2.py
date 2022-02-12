import numpy as np
import os
from os import path
import cv2
import pandas as pd
import xml.etree.cElementTree as xet
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import MobileNetV2, InceptionV3, InceptionResNetV2
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.models import Model


# Now we load the labels.csv that we created
df = pd.read_csv('static/datasets/images-Set2/labels.csv')
def get_file_name(filename):
    filename = 'images-Set2/' + filename
    file_name_images = xet.parse(filename).getroot().find('filename').text
    file_path_images = os.path.join('static/datasets/images-Set2/Images', file_name_images)
    return file_path_images

# Create a list of all image file paths
image_path = list(df['filepath'].apply(get_file_name))

'''
img = cv2.imread(image_path[0])
cv2.namedWindow('sample image', cv2.WINDOW_NORMAL)
cv2.imshow('sample image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.rectangle(img, (226, 125), (419, 173), (0, 255, 0), 3)
cv2.namedWindow('sample image with rectangle', cv2.WINDOW_NORMAL)
cv2.imshow('sample image with rectangle', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
#############Preprocessing#################

#print(df.iloc[:, 1:])
labels = df.iloc[:, 1:].values
#print(labels)
'''
image = image_path[0]
img_arr = cv2.imread(image)
#print(img_arr)
h, w, d = img_arr.shape
#print(h, w, d)

load_image = load_img(image, target_size=(224, 224))
load_image_array = img_to_array(load_image)
#print(load_image_array)
normal_load_image_array = load_image_array/255.0
#print(normal_load_image_array)
i = 0
xmin, xmax, ymin, ymax = labels[i]
nxmin, nxmax = xmin/w, xmax/w
nymin, nymax = ymin/h, ymax/h
label_normal = (nxmin, nxmax, nymin, nymax)

#print(label_normal)
'''
data = []
output = []
for ind in range(len(image_path)):
    image = image_path[ind]
    #print(image)
    if path.exists(image):
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
    else:
        continue


# print(data)
# print(output)
X = np.array(data, dtype=np.float32)
y = np.array(output, dtype=np.float32)
print(X.shape, y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

inception_resnet = InceptionResNetV2(weights='imagenet', include_top=False,
                                     input_tensor=Input(shape=(224, 224, 3)))
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
tfb = TensorBoard('object_detection_set2')

# this will train the model
history = model.fit(x=X_train, y=y_train, batch_size=32, epochs=300, validation_data=(X_test, y_test), callbacks=[tfb])
model.save('models/set2_object_detection_Model.h5')
