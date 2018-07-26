# import some libs
import numpy as np
from sklearn.externals import joblib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import time
from util import *

# setup Argument parser
parser = ArgumentParser()
parser.add_argument("-f", "--filename", dest="filename", metavar="FILE")
parser.add_argument("-m", "--model", dest="model", metavar="FILE")
parser.add_argument("-x", "--xs", dest="xs", metavar="FILE")
args = parser.parse_args()

# load model from file
svc = joblib.load(args.model)
X_scaler = joblib.load(args.xs)

# load test image
image = mpimg.imread(args.filename)
t0 = time.time()
result = detect_cars(image, svc, X_scaler)
t1 = time.time()
print(t1 - t0)
plt.imshow(result)
plt.show()
