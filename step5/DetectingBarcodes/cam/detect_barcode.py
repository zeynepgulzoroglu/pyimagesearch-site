import simple_barcode_detection
from imutils.video import VideoStream
import time
import cv2

vs = VideoStream(src=0).start()
time.sleep(2.0)

while True:
    frame = vs.read()

    if frame is None:
        break

    box = simple_barcode_detection.detect(frame)

    if box is not None:
        cv2.drawContours(frame, [box], -1, (0, 255, 0), 2)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

vs.stop()
cv2.destroyAllWindows()