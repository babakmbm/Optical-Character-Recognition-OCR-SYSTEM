import numpy as np
import cv2
import pytesseract as pt
import imutils
import easyocr
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def license_detection(path, filename):
    model = tf.keras.models.load_model('static/models/set1_object_detection_Model_15_07_2021.h5')
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
    # save the predicted image in predict folder
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite('static/predict/{}'.format(filename), image_bgr)
    return real_coords

def deep_ocr(path, filename):
    img = np.array(load_img(path))
    cords = license_detection(path, filename)
    xmin, ymin, xmax, ymax = cords[0]
    roi = img[ymin:ymax, xmin:xmax]
    roi_bgr = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)
    cv2.imwrite('static/roi/{}'.format(filename), roi_bgr)
    plate_number = pt.image_to_string(roi, lang='eng', config='--psm 7')
    print(plate_number)
    return plate_number


def easy_ocr(path, filename):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
    edges = cv2.Canny(bfilter, 30, 200)
    points = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(points)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    location = None
    for contour in contours:
        # it checks if the contour looks like a square or polygon
        # 10: how accurate the approximation is (we can play around with this)
        approx = cv2.approxPolyDP(contour, 10, True)
        # if the approximation has 4 points
        # then most likely it is our number plate location
        if len(approx) == 4:
            location = approx
            break

    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [location], 0, 255, -1)
    new_image = cv2.bitwise_and(img, img, mask=mask)
    (x, y) = np.where(mask == 255)
    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y))
    cropped_image = gray[x1:x2 + 1, y1:y2 + 1]
    cv2.imwrite('static/roi/{}'.format(filename), cropped_image)
    reader = easyocr.Reader(['en'])
    result = reader.readtext(cropped_image)
    text = result[0][1]
    res = cv2.putText(img, text=text, org=(approx[0][0][0], approx[1][0][1] + 60), fontFace=1, fontScale=3,
                      color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    res = cv2.rectangle(img, tuple(approx[0][0]), tuple(approx[2][0]), (0, 255, 0), 3)
    cv2.imwrite('static/easyOcrPredict/{}'.format(filename), res)
    print(text)
    return text