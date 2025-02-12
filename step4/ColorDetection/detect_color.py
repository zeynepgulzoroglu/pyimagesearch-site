import cv2
import numpy as np

image = cv2.imread("renk.png")

boundaries = [
    #([17, 15, 100], [50, 56, 200]),
	([86, 31, 4], [220, 88, 50]),
	# ([25, 146, 190], [62, 174, 250]),
	# ([103, 86, 65], [145, 133, 128])
] #renk aralıkları

for (lower, upper) in boundaries:
	lower = np.array(lower, dtype = "uint8")
	upper = np.array(upper, dtype = "uint8")
	
	mask = cv2.inRange(image, lower, upper)
	output = cv2.bitwise_and(image, image, mask = mask)
	cv2.imshow("Images", np.hstack([image, output])) #np.hstack 2 resmi yanyana birleştirerek tek çıktı vermeye yarıyor

cv2.waitKey(0)
cv2.destroyAllWindows()