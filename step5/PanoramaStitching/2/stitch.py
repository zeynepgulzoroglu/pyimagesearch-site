from panorama import Stitcher
import imutils
import cv2

imageA = cv2.imread("B.png")
imageB = cv2.imread("A.png")
imageA = imutils.resize(imageA, width=400)
imageB = imutils.resize(imageB, width=400)

stitcher = Stitcher()
(result, vis) = stitcher.stitch([imageA, imageB], showMatches=True)

cv2.imshow("Image A", imageA)
cv2.imshow("Image B", imageB)
cv2.imshow("Keypoint Matches", vis)
cv2.imshow("Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()