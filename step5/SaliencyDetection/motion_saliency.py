from imutils.video import VideoStream
import imutils
import time
import cv2

saliency = None
vs = VideoStream(src=0).start()
time.sleep(2.0)

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=500)

    if saliency is None:
        saliency = cv2.saliency.MotionSaliencyBinWangApr2014_create()
        saliency.setImagesize(frame.shape[1], frame.shape[0])
        saliency.init()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, saliencyMap = saliency.computeSaliency(gray)
    saliencyMap = (saliencyMap * 255).astype("uint8")

    cv2.imshow("Frame", frame)
    cv2.imshow("Saliency Map", saliencyMap)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
