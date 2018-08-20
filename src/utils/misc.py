from sklearn import svm
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from glob import glob
from astral import Astral
from pytz import timezone
from settings.constants import DATE_FMT, EXTENSION
from sklearn.model_selection import GridSearchCV
import pytz
import shutil
import os
import cv2
import errno
import json
import numpy as np
import datetime as dt


def create_nonexistent_dir(path, exc_raise=False):
    """
    Create directory from given path
    Return True if created, False if it exists

    """
    try:
        os.makedirs(path)
        print("Created directory with path: %s\n", path)
        return path
    except OSError as e:
        if e.errno != errno.EEXIST:
            print("Could not create directory with path: %s\n", path)
            if exc_raise:
                raise
        return None


def timestamp_filter(path, timezone, city_name):
    """
    Copy images, whose filenames are a timestamp 
    of the form  %Y-%m-%d-%H-%M-%S,
    from a given path to a directory called 
    'timestamp_filtered' which is created within 
    the current working directory. 

    Arguments: 
      - path where the dataset is
      - timezone, string e.g. 'Europe/London'
      - city_name, string e.g. 'London'
    """
    cwd = os.getcwd()
    bst = pytz.timezone(timezone)
    a = Astral()
    a.solar_depression = 'civil'
    city = a[city_name]
    outdir = os.path.join(cwd, 'timestamp_filtered')
    create_nonexistent_dir(outdir)
    for img in os.listdir(path):
        if img.endswith('.jpg'):
            year = img.split('_')[2].split('-')[0]
            month = img.split('-')[1]
            day = img.split('-')[2]
            img_date = dt.date(int(year), int(month), int(day))
            sun = city.sun(date=img_date, local=True)
            ts = img.split('_')[2].strip(EXTENSION)
            timestamp_unaware = dt.datetime.strptime(ts, DATE_FMT)
            timestamp = bst.localize(timestamp_unaware)

            if sun['dawn'] < timestamp < sun['sunrise'] or\
                    sun['sunset'] < timestamp < sun['dusk']:
                shutil.copy(path + img, outdir)


def load_image(path):
    """
    Load an image from a given path
    and returns a numpy array

    """
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def crop_to_half(img, x=0, y=0):
    """
    Returns the numpy array corresponding 
    to the upper half of any given image. 
    It takes a numpy array as input

    """
    h = img.shape[0] / 2
    w = img.shape[1]
    cropped_img = img[y:y + h, x:x + w]
    return cropped_img


def fex(img, n_clusters):
    """
    Perform feature extraction on any given image (numpy array). 

    K-Means algorithm is run first with a given number of clusters, n_clusters. 
    The colour clusters are retrieved together with the cluster occurrence. 
    It returns a numpy array of the extracted features, e.g. for 3 clusters:
    [ R1, G1, B1, %1_occurence, 
    R2, G2, B2, %2_occurrence 
    R3, G3, B3, %3_occurence ]

    where Ri, Gi, and Bi, are the RGB coordinates of the i-th cluster centroid 
    and %i_occurence is the occurrance of the i-th cluster.

    """
    img = img.reshape((img.shape[0] * img.shape[1], 3))
    kmeans = KMeans(n_clusters)
    kmeans.fit(img)
    colour_clusters = kmeans.cluster_centers_
    colour_clusters = colour_clusters.astype(int)
    n_labels = np.arange(0, n_clusters + 1)
    (hist, _) = np.histogram(kmeans.labels_, bins=n_labels)
    hist = hist.astype("float")
    hist /= hist.sum()
    colour_clusters = colour_clusters[(-hist).argsort()]
    hist = hist[(-hist).argsort()]
    # creating empty chart to visualise the hist (not necessary)
    chart = np.zeros((50, 500, 3), np.uint8)
    start = 0
    cluster_frequencies = np.array([[]])
    # creating color rectangles
    for i in range(n_clusters):
        end = start + hist[i] * 500
        cluster_frequencies = np.append(cluster_frequencies, hist[i])
        # getting rgb values
        r = colour_clusters[i][0]
        g = colour_clusters[i][1]
        b = colour_clusters[i][2]
        # using cv2.rectangle to plot colour_clusters
        cv2.rectangle(chart, (int(start), 0), (int(end), 50), (r, g, b), -1)
        start = end
    outarr = np.zeros(len(colour_clusters) * 4)
    for i in range(len(colour_clusters)):
        outarr[4 * i:4 * i + 3] = colour_clusters[i]
        outarr[4 * (i + 1) - 1] = cluster_frequencies[i]
    return outarr


def svc_param_selection(features, labels):
    """
    It performs a search for the best combination
    of hyperparameters (C, gamma) for the support vector machine
    using GridSearchCV. 

    It takes two numpy arrays as input:
      - features has to be a 2D np array
      - labels has to be a 1D np array

    It returns a dictionary with 'C' and 'gamma' as keys
    and the found values as values, e.g. 

    {'C': 1, 'gamma': 0.001}

    """
    Cs = [0.001, 0.01, 0.1, 1, 10]
    gammas = [0.001, 0.01, 0.1, 1]
    param_grid = {'C': Cs, 'gamma': gammas}
    grid_search = GridSearchCV(svm.SVC(), param_grid)
    grid_search.fit(features, labels)
    par = grid_search.best_params_
    return par


def save_json(path, dictionary):
    """
    Save a json file in a given path

    """
    f = open(path, 'w')
    loader = json.dump(dictionary, f, indent=4, separators=(',', ': '))
    f.close()
