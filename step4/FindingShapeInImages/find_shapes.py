import numpy as np
import imutils
import cv2

img = cv2.imread("siyah.png")

lower = np.array([0, 0, 0])
upper = np.array([15, 15, 15])
shapeMask = cv2.inRange(img, lower, upper)

cnts = cv2.findContours(shapeMask.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
print("I found {} black shapes".format(len(cnts)))
cv2.imshow("Mask", shapeMask)

for c in cnts:
	cv2.drawContours(img, [c], -1, (0, 255, 0), 2)
	cv2.imshow("Image", img)
 
cv2.waitKey(0)
cv2.destroyAllWindows()