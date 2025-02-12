from keyclipwriter import KeyClipWriter
from imutils.video import VideoStream
import argparse
import datetime
import imutils
import time
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,help="path to output directory")
ap.add_argument("-f", "--fps", type=int, default=20,help="FPS of output video")
ap.add_argument("-c", "--codec", type=str, default="MJPG",help="codec of output video")
ap.add_argument("-b", "--buffer-size", type=int, default=32,help="buffer size of video clip writer")
args = vars(ap.parse_args())

print("[INFO] warming up camera...")
vs = cv2.VideoCapture(0)
time.sleep(3.0)

greenLower = (29, 86, 6)
greenUpper = (64, 255, 255)
kcw = KeyClipWriter(bufSize=args["buffer_size"])
consecFrames = 0

while True:
	frame = vs.read()[1]
	frame = imutils.resize(frame, width=600)
	updateConsecFrames = True
	blurred = cv2.GaussianBlur(frame, (11, 11), 0)
	hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
	mask = cv2.inRange(hsv, greenLower, greenUpper)
	mask = cv2.erode(mask, None, iterations=2)
	mask = cv2.dilate(mask, None, iterations=2)
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)

	if len(cnts) > 0:
		c = max(cnts, key=cv2.contourArea)
		((x, y), radius) = cv2.minEnclosingCircle(c)
		updateConsecFrames = radius <= 10
		if radius > 10:
			consecFrames = 0
			cv2.circle(frame, (int(x), int(y)), int(radius),
				(0, 0, 255), 2)
			if not kcw.recording:
				timestamp = datetime.datetime.now()
				p = "{}/{}.avi".format(args["output"],timestamp.strftime("%Y%m%d-%H%M%S"))
				kcw.start(p, cv2.VideoWriter_fourcc(*args["codec"]),args["fps"])
    
	if updateConsecFrames:
		consecFrames += 1
	kcw.update(frame)
	if kcw.recording and consecFrames == args["buffer_size"]:
		kcw.finish()
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break

if kcw.recording:
	kcw.finish()

cv2.destroyAllWindows()
vs.release()