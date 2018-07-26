# import some libs
import numpy as np
from sklearn.externals import joblib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from util import *
import cv2

# setup Argument parser
parser = ArgumentParser()
parser.add_argument("-m", "--model", dest="model", metavar="FILE")
parser.add_argument("-x", "--xs", dest="xs", metavar="FILE")
args = parser.parse_args()

# load model from file
svc = joblib.load(args.model)
X_scaler = joblib.load(args.xs)

cv2.namedWindow("vehicle_detection")
cam = cv2.VideoCapture(1)

while True:
    rval, frame = cam.read()
    frame = cv2.resize(frame, (1280, 720))
    result = detect_cars(frame, svc, X_scaler)
    cv2.imshow("vehicle_detection", result)

    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
