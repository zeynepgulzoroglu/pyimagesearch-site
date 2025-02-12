import numpy as np
import cv2
import imutils

def sort_contours(cnts, method="left-to-right"):
    reverse = method in ["right-to-left", "bottom-to-top"]
    i = 1 if method in ["top-to-bottom", "bottom-to-top"] else 0
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),key=lambda b: b[1][i], reverse=reverse))
    return cnts, boundingBoxes

def draw_contour(image, c, i):
    M = cv2.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    cv2.putText(image, f"#{i + 1}", (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return image

image = cv2.imread("sort.png")
accumEdged = np.zeros(image.shape[:2], dtype="uint8")

for chan in cv2.split(image):
    chan = cv2.medianBlur(chan, 11)
    edged = cv2.Canny(chan, 50, 200)
    accumEdged = cv2.bitwise_or(accumEdged, edged)

cv2.imshow("Edge Map", accumEdged)

cnts = cv2.findContours(accumEdged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

orig = image.copy()
for i, c in enumerate(cnts):
    orig = draw_contour(orig, c, i)

cv2.imshow("Unsorted", orig)

method = "left-to-right"  # Sorting method can be changed here
cnts, boundingBoxes = sort_contours(cnts, method=method)

for i, c in enumerate(cnts):
    draw_contour(image, c, i)

cv2.imshow("Sorted", image)
cv2.waitKey(0)
