import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pytesseract as pt

def license_detection(path, model):
    image = load_img(path)
    image = np.array(image, dtype=np.uint8)
    image1 = load_img(path, target_size=(224, 224))
    img_arr_norm = img_to_array(image1) / 255.0
    h, w, d = image.shape
    test_arr = img_arr_norm.reshape(1, 224, 224, 3)
    normalized_coords = model.predict(test_arr)
    de_norm_array = np.array([w, w, h, h])
    real_coords = (normalized_coords * de_norm_array).astype(np.int32)
    xmin, ymin, xmax, ymax = real_coords[0]
    pt1 = (xmin, ymin)
    pt2 = (xmax, ymax)
    #print(pt1, pt2)
    cv2.rectangle(image, pt1, pt2, (0, 255, 0), 3)
    return image, real_coords

def main():
    model = tf.keras.models.load_model('../models/set1_object_detection_Model_15_07_2021.h5')
    print('Model is Loaded!')
    path = '../Images/test4.png'
    image, cords = license_detection(path, model)
    plt.figure()
    plt.imshow(image)
    plt.show()

    img = np.array(load_img(path))
    # crop the region of interest (roi) using the predicted coordinates
    xmin, ymin, xmax, ymax = cords[0]
    roi = img[ymin:ymax, xmin:xmax]
    plt.imshow(roi)
    plt.show()

    # extract the text from image using pytesseract
    plate_number = pt.image_to_string(roi, lang='eng', config='--psm 7 --oem 1')
    print(plate_number)


if __name__ == "__main__":
    main()
