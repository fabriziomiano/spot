"""
This script loads an SVM previously trained, 
reads images from a given path, 
extracts their features and classifies them. 
It then writes a json file with the following structure:
{
    "img": {
        "class 0": probability[0],
        "class 1": probability[1]
    },
    ...
    ...
}

where : 
  - "img" is the file name of the image being labelled
  - "class i" is the i-th class
  - "probability[i]" is the i-th probability associated 
  with the prediction of the i-th class.

"""

from settings.constants import EXTENSION, N_CLUSTERS, JSON_PATH
from utils.misc import load_image, crop_to_half, fex, save_json
from sklearn.externals import joblib
from collections import defaultdict
import argparse
import sys
import os
import numpy as np

args = sys.argv
parser = argparse.ArgumentParser(
    description="""Predict labels for images within a given directory""")
parser.add_argument('test_path',
                    type=str,
                    help='Specify the path of the directory containing images to label')
parser.add_argument('svm_file',
                    type=str,
                    help='Specify the file containing the SVM')
args = parser.parse_args()
TEST_DIR = args.test_path
SVM_FILE = args.svm_file
results = defaultdict(dict)
try:
    clf = joblib.load(SVM_FILE)
    print "File " + SVM_FILE + " read succesfully"
except IOError as e:
    print "File " + SVM_FILE + " is not a valid file. Please retrain"
    raise
classes = clf.classes_
print "Classifying images..."
for img in os.listdir(TEST_DIR):
    if img.endswith(EXTENSION):
        img_path = os.path.join(TEST_DIR, img)
        image = load_image(img_path)
        # uncomment the line below if sky is always approximately in the upper half of the images
        # image = crop_to_half(image)
        test_feature = fex(image, N_CLUSTERS)
        probabilities = clf.predict_proba([test_feature])
        for i, c in enumerate(classes):
            results[img][c] = round(probabilities[0][i], 3)
save_json(JSON_PATH, results)
print "Results saved in " + JSON_PATH
