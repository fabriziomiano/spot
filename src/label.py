"""
This script loads an SVM previously trained, 
reads images from a given path, 
extracts their features to then perform the classification
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

TEST_DIR = Config.get('CLASSIFICATION', 'TEST_DIR')
SVM_FILENAME = Config.get('COMMON', 'SVM_FILENAME')
EXTENSION = Config.get('COMMON', 'EXTENSION')
N_CLUSTERS = int(Config.get('COMMON', 'N_CLUSTERS'))
results = defaultdict(dict)
try:
    clf = joblib.load(SVM_FILENAME)
    print "INFO :: File " + SVM_FILENAME + " read succesfully"
except IOError as e:
    print "ERROR :: " + SVM_FILENAME + " is not a valid file. Please retrain or check che configuration file"
    raise
classes = clf.classes_
print "INFO :: Classifying images..."
for img in os.listdir(TEST_DIR):
    if img.endswith(EXTENSION):
        img_path = os.path.join(TEST_DIR, img)
        image = load_image(img_path)
        test_feature = fex(image, N_CLUSTERS)
        probabilities = clf.predict_proba([test_feature])
        for i, c in enumerate(classes):
            results[img][c] = round(probabilities[0][i], 3)
JSON_FILENAME = Config.get('CLASSIFICATION', 'JSON_FILENAME')
json_path = os.path.join(os.getcwd(), JSON_FILENAME)
save_json(json_path, results)
print "INFO :: Results saved in " + json_path
