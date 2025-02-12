import argparse
import imutils
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True, type=str, help="path to input video file")
ap.add_argument("-o", "--output", required=True, type=str, help="path to output directory to store frames")
ap.add_argument("-p", "--min-percent", type=float, default=1.0, help="lower boundary of percentage of motion")
ap.add_argument("-m", "--max-percent", type=float, default=10.0, help="upper boundary of percentage of motion")
ap.add_argument("-w", "--warmup", type=int, default=200, help="# of frames to use to build a reasonable background model")
args = vars(ap.parse_args())

fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()
captured = False
total = 0
frames = 0
vs = cv2.VideoCapture(args["video"])
(W, H) = (None, None)

while True:
    (grabbed, frame) = vs.read()
    if frame is None:
        break
    orig = frame.copy()
    frame = imutils.resize(frame, width=600)
    mask = fgbg.apply(frame)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    if W is None or H is None:
        (H, W) = mask.shape[:2]
    p = (cv2.countNonZero(mask) / float(W * H)) * 100

    if p < args["min_percent"] and not captured and frames > args["warmup"]:
        cv2.imshow("Captured", frame)
        captured = True
        filename = "{}.png".format(total)
        path = os.path.sep.join([args["output"], filename])
        total += 1
        cv2.imwrite(path, orig)
    elif captured and p >= args["max_percent"]:
        captured = False

    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    frames += 1

vs.release()

# python detect_scene.py --video video.mp4 --output output