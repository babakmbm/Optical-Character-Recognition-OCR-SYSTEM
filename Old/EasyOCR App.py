# insatll easyocr
# install pytorch
# install imutils
# opencv cheatsheet: https://heartbeat.fritz.ai/opencv-python-cheat-sheet-from-importing-images-to-face-detection-52919da36433
# opencv noise reduction: https://aishwaryagulve97.medium.com/smoothing-images-using-opencv-5fb6ca2cb54d

import cv2
import matplotlib.pyplot as plt
import numpy as np
import imutils
import easyocr

# Read in the image with opencv
img = cv2.imread('../Images/test3.jpeg')
# we make it into the grayscale by changing the colour space
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))
plt.show()

# apply a filter for noise reduction
# opencv documentation image filters: https://docs.opencv.org/master/d4/d86/group__imgproc__filter.html#ga9d7064d478c95d60003cf839430737ed
bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
plt.imshow(cv2.cvtColor(bfilter, cv2.COLOR_BGR2RGB))
plt.show()

'''# both images in a subplot
fig = plt.figure()
fig.add_subplot(2,1,1)
plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))
fig.add_subplot(2,1,2)
plt.imshow(cv2.cvtColor(bfilter, cv2.COLOR_BGR2RGB))
plt.show()'''

# edge detection
# Canny theory: https://docs.opencv.org/3.4/da/d22/tutorial_py_canny.html
# cv2.Canny(image, lowerThreshold, upperThreshold)
edges = cv2.Canny(bfilter, 30, 200)
plt.imshow(cv2.cvtColor(edges, cv2.COLOR_BGR2RGB))
plt.show()

# Contour detection
# cv2.CHAIN_APPROX_SIMPLE: a simple approximation of what the contour looks like
# cv2.RETR_TREE: store with a tree hierarchy
points = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print(points)
contours = imutils.grab_contours(points)
#print(contours)
# we get our top 10 contours
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

location = None;
for contour in contours:
    # it checks if the contour looks like a square or polygon
    # 10: how accurate the approximation is (we can play around with this)
    approx = cv2.approxPolyDP(contour, 10, True)
    # if the approximation has 4 points
    # then most likely it is our number plate location
    if len(approx) == 4:
        location = approx
        break

print(location)

mask = np.zeros(gray.shape, np.uint8)
new_image = cv2.drawContours(mask, [location], 0, 255, -1)
new_image = cv2.bitwise_and(img, img, mask=mask)

plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
plt.show()

(x, y) = np.where(mask == 255)
(x1, y1) = (np.min(x), np.min(y))
(x2, y2) = (np.max(x), np.max(y))
cropped_image = gray[x1:x2+1, y1:y2+1]

plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
plt.show()

reader = easyocr.Reader(['en'])
result = reader.readtext(cropped_image)
print(result)

text = result[0][1]
print('The license plate:', text)

res = cv2.putText(img, text=text, org=(approx[0][0][0], approx[1][0][1] + 60), fontFace=1, fontScale=3, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
res = cv2.rectangle(img, tuple(approx[0][0]), tuple(approx[2][0]), (0, 255, 0), 3)
plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
plt.show()



