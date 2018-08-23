"""
This script loads an SVM previously trained, 
reads images from a given path, 
extracts their features to then perform the classification
and finally writes a json file with the following structure:
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

from utils.misc import load_image, load_config, crop_to_half, fex, save_json
from sklearn.externals import joblib
from collections import defaultdict
import argparse
import ConfigParser
import sys
import os
import numpy as np

args = sys.argv
parser = argparse.ArgumentParser(
    description="""Predict labels for images within a given directory""")
parser.add_argument('-c', '--config', type=str, metavar='', required=True,
                    help='Specify the path of the configuration file')
args = parser.parse_args()
CONFIG_PATH = args.config
Config = load_config(CONFIG_PATH)
if len(Config.sections()) == 0:
    print "ERROR :: Not a valid configuration file. Exiting"
    sys.exit(0)

SVM_PATH = Config.get('COMMON', 'SVM_PATH')
EXTENSION = Config.get('COMMON', 'EXTENSION')
N_CLUSTERS = int(Config.get('COMMON', 'N_CLUSTERS'))
TEST_DIR = Config.get('CLASSIFICATION', 'TEST_DIR')
results = defaultdict(dict)
n_images = 0
try:
    clf = joblib.load(SVM_PATH)
    print "INFO :: File " + SVM_PATH + " read succesfully"
    classes = clf.classes_
    print "INFO :: Classifying images in " + TEST_DIR
    for img in os.listdir(TEST_DIR):
        if img.endswith(EXTENSION):
            img_path = os.path.join(TEST_DIR, img)
            image = load_image(img_path)
            test_feature = fex(image, N_CLUSTERS)
            probabilities = clf.predict_proba([test_feature])
            for i, c in enumerate(classes):
                results[img][c] = round(probabilities[0][i], 3)
            n_images += 1
    JSON_FILENAME = Config.get('CLASSIFICATION', 'JSON_FILENAME')
    json_path = os.path.join(os.getcwd(), JSON_FILENAME)
    save_json(json_path, results)
    print "INFO :: " + str(n_images) + " images classified"
    print "INFO :: Results saved in " + json_path
except IOError as e:
    print "ERROR :: " + SVM_PATH + " not found."
