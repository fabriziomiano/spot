"""
THis script trains an SVM.   
The features are extracted using the fex() function (check utils/misc.py)
using numpy arrays as the following, e.g. for 3 clusters:
[ R1, G1, B1, %1_occurence, 
R2, G2, B2, %2_occurrence 
R3, G3, B3, %3_occurence ]
where Ri, Gi, and Bi, are the RGB coordinates of the i-th cluster centroid 
and %i_occurence is the occurrance of the i-th cluster.

A classifier clf is initialised using the svm module 
from the sci-kit learn library. In particular, a 
Support Vector Classification is chosen. The hyperparameters
(C, gamma) are chosen using the svc_param_selection(features, labels)
function (check utils/misc.py)
The fit over the extracted features and the given labels 
is then performed and the results are saved in a .pkl file

"""

from utils.misc import load_image, load_config, fex, svc_param_selection
from sklearn.externals import joblib
from sklearn import svm
from glob import glob
import ConfigParser
import argparse
import os
import sys
import shutil
import numpy as np

args = sys.argv
parser = argparse.ArgumentParser(
    description="""Train an SVM using images from a given directory containing subdirectories
    whose names will be used as labels""")
parser.add_argument('-c', '--config', type=str, metavar='', required=True,
                    help='Specify the path of the configuration file')
args = parser.parse_args()
CONFIG_PATH = args.config
Config = load_config(CONFIG_PATH)
if len(Config.sections()) == 0:
    print "ERROR :: Not a valid configuration file. Exiting"
    sys.exit(0)
features = []
labels = []
number_of_images = 0
TRAINING_PATH = Config.get('TRAINING', 'TRAINING_PATH')
EXTENSION = Config.get('COMMON', 'EXTENSION')
N_CLUSTERS = int(Config.get('COMMON', 'N_CLUSTERS'))
SVM_PATH = Config.get('COMMON', 'SVM_PATH')
pattern = os.path.join(TRAINING_PATH, '*')
for directory in glob(pattern):
    if os.path.isdir(directory):
        print "INFO :: Training on " + directory
        for img in os.listdir(directory):
            if img.endswith(EXTENSION):
                img_path = os.path.join(directory, img)
                img = load_image(img_path)
                feature = fex(img, N_CLUSTERS)
                features.append(feature)
                label = directory.split('/')[-1]
                labels.append(label)
                number_of_images += 1
features = np.array(features)
labels = np.array(labels)
print "INFO :: Images read: " + str(number_of_images)
print "INFO :: Features per image extracted: " + str(features.shape[1])
par = svc_param_selection(features, labels)
clf = svm.SVC(C=par['C'],
              gamma=par['gamma'],
              probability=True)
print "INFO :: Fitting the data"
clf.fit(features, labels)
joblib.dump(clf, SVM_PATH)
print "INFO :: SVM saved in " + SVM_PATH
