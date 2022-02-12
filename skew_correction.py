import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import imutils
import pytesseract as pt

def extract_contours(threshold_img):
    """This function returns the extracted contours"""
    element = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(17, 3))
    morph_img_threshold = threshold_img.copy()
    cv2.morphologyEx(src=threshold_img, op=cv2.MORPH_CLOSE, kernel=element, dst=morph_img_threshold)
    cv2.imshow("Morphed", morph_img_threshold)
    cv2.waitKey(0)

    contours, hierarchy = cv2.findContours(morph_img_threshold, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
    return contours

def ratioCheck(area, width, height):
    """This function inspects the ratio of the contour to ensure it meets the requirements
    suitable to a real license plate"""
    ratio = float(width) / float(height)
    if ratio < 1:
        ratio = 1 / ratio

    aspect = 4.7272
    min = 15 * aspect * 15  # minimum area
    max = 125 * aspect * 125  # maximum area

    rmin = 3
    rmax = 6

    if (area < min or area > max) or (ratio < rmin or ratio > rmax):
        return False
    return True

def cleanPlate(plate):
    """This function gets the countours that most likely resemeber the shape
    of a license plate"""
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    thresh = cv2.dilate(gray, kernel, iterations=1)

    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if contours:
        areas = [cv2.contourArea(c) for c in contours]
        max_index = np.argmax(areas)

        max_cnt = contours[max_index]
        max_cntArea = areas[max_index]
        x, y, w, h = cv2.boundingRect(max_cnt)

        if not ratioCheck(max_cntArea, w, h):
            return plate, None

        cleaned_final = thresh[y:y + h, x:x + w]
        cv2.imshow("Function Test", cleaned_final)
        return cleaned_final, [x, y, w, h]

    else:
        return plate, None

def skew_correction(filename):
    img = cv2.imread(filename)
    plt.imshow(img)
    plt.show()
    size = img.shape
    inverted_image = cv2.bitwise_not(img, img)

    # Edge detection
    dst = cv2.Canny(inverted_image, 50, 200, None, 3)

    # Copy edges to the images that will display the results in BGR
    cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)
    # Probabilistic hough Transform
    linesP = cv2.HoughLinesP(dst, 5, np.pi/180, 10, None, size[1]/2, 15)
    angel = 0
    #nb_lines = linesP.shape[0]
    for i in range(0, 10):
        l = linesP[i][0]
        angel += math.atan2(l[3] - l[1], l[2] - l[0])

    print(angel)
    corrected_image = imutils.rotate(img, angel)
    return corrected_image

img = cv2.imread('WebApp/static/roi/N78.jpeg')
result = cleanPlate(img)
print(result)



