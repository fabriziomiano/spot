from settings.constants import EXTENSION, N_CLUSTERS, JSON_PATH
from utils.misc import load_image, crop_to_half, fex, save_json
from sklearn.externals import joblib
import argparse
import sys
import os
import numpy as np

args = sys.argv
parser = argparse.ArgumentParser(
    description="""Predict labels for all the images within a given directory""")
parser.add_argument('test_path',
                    type=str,
                    help='Specify the path of the directory containing images to label')
parser.add_argument('svm_file',
                    type=str,
                    help='Specify the file containing the SVM info')
args = parser.parse_args()

TEST_DIR = args.test_path
img_dict = dict()
results = dict()
try:
    clf = joblib.load(args.svm_file)
except Exception as e:
    print "File " + args.svm_file + " not a valid file. Please retrain"
    raise e
for img in os.listdir(TEST_DIR):
    if img.endswith(EXTENSION):
        img_path = os.path.join(TEST_DIR, img)
        image = load_image(img_path)
        # uncomment the line below if sky is always approximately in the upper half of the images
        # image = crop_to_half(image)
        test_feature = fex(image, N_CLUSTERS)
        probability = clf.predict_proba([test_feature])
        for i, classes in enumerate(clf.classes_):
            img_dict[clf.classes_[i]] = round(probability[0][i], 3)
        results[img] = img_dict
save_json(JSON_PATH, results)
