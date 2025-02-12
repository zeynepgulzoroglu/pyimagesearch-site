from imutils import perspective
import numpy as np
import cv2

notecard = cv2.imread("notecard.png")
pts = np.array([(73, 239), (356, 117), (475, 265), (187, 443)])

for (x, y) in pts:
    cv2.circle(notecard, (x, y), 5, (0, 255, 0), -1)

warped = perspective.four_point_transform(notecard, pts)

cv2.imshow("Original", notecard)
cv2.imshow("Warped", warped)
cv2.waitKey(0)