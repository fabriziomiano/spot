from utils.misc import load_image, fex, svc_param_selection
from settings.constants import EXTENSION, N_CLUSTERS, SVM_FILENAME
from sklearn.externals import joblib
from sklearn import svm
from glob import glob
import argparse
import os
import sys
import shutil
import numpy as np

"""

"""
args = sys.argv
parser = argparse.ArgumentParser(
    description="""Train an SVM using images from a given directory containing subdirectories
    whose names will be used as labels: python train_svm.py ~/dir/subdir_containing_imgs/""")
parser.add_argument('training_dataset_path', type=str,
                    help='Specify the path of the directory containing subdirectories with the training dataset')
args = parser.parse_args()
features = []
labels = []
for directory in glob(args.training_dataset_path + '*'):
    if os.path.isdir(directory):
        print "TRAINING ON " + directory
        for img in os.listdir(directory):
            if img.endswith(EXTENSION):
                img_path = os.path.join(directory, img)
                img = load_image(img_path)
                feature = fex(img, N_CLUSTERS)
                features.append(feature)
                label = directory.split('/')[-1]
                labels.append(label)
features = np.array(features)
labels = np.array(labels)
par = svc_param_selection(features, labels)
clf = svm.SVC(C=par['C'],
              gamma=par['gamma'],
              probability=True)
clf.fit(features, labels)
joblib.dump(clf, SVM_FILENAME)
print "SVM saved in " + SVM_FILENAME
