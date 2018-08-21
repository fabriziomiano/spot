from sklearn import svm
from sklearn.cluster import KMeans
from skimage.measure import compare_ssim
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
        print "Created directory with path: " + str(path)
        return path
    except OSError as e:
        if e.errno != errno.EEXIST:
            print("Could not create directory with path: %s\n", path)
            if exc_raise:
                raise
        return None


def ssim_filter(path_in, path_out, ref_image_path, threshold, tolerance):
    """
    Filter dataset of images based on the 
    SSIM score against a reference image

    Arguments: 
      path__in: origin path of the dataset
      path_out: destination path of the filtered dataset
      ref_image_path: path of the reference image
      threshold: absolute SSIM score 
      tolerance: additional tolerance on SSIM score.

    """
    ref_image = cv2.imread(ref_image_path)
    tolerance_max = threshold + tolerance
    tolerance_min = threshold - tolerance
    create_nonexistent_dir(path_out)
    for img in os.listdir(path_in):
        if img.endswith(EXTENSION):
            image_path = os.path.join(path_in, img)
            image = cv2.imread(image_path)
            score = compare_ssim(ref_image, image, multichannel=True)
            if tolerance == 0 and score <= threshold:
                shutil.copy(path_in + img, path_out)
                print img + " scored: " + str(round(score, 2))
                print "COPYING " + img + " to " + path_out
            elif tolerance_min < score < tolerance_max:
                shutil.copy(path_in + img, path_out)
                print img + " scored: " + str(round(score, 2))
                print "COPYING " + img + " to " + path_out


def timestamp_filter(path_in, path_out, timezone, city_name):
    """
    Filter dataset of images based on 
    timestamp contained in filename
    of the form, e.g. 1529293200_0_2018-06-18-04-40-00.jpg
    DATE_FMT = "%Y-%m-%d-%H-%M-%S" (check settings/constants.py)

    Arguments: 
      path_in: origin path of the dataset
      path_out: destination path of the filtered dataset
      timezone: string e.g. 'Europe/London' of the filename
      city_name: string e.g. 'London' where the picture was taken
    timezone and city_name have to match those in 
    the Astral database. Check https://astral.readthedocs.io/en/latest/

    """
    tz = pytz.timezone(timezone)
    a = Astral()
    a.solar_depression = 'civil'
    city = a[city_name]
    create_nonexistent_dir(path_out)
    for img in os.listdir(path_in):
        if img.endswith(EXTENSION):
            year = img.split('_')[2].split('-')[0]
            month = img.split('-')[1]
            day = img.split('-')[2]
            img_date = dt.date(int(year), int(month), int(day))
            sun = city.sun(date=img_date, local=True)
            ts = img.split('_')[2].strip(EXTENSION)
            timestamp_unaware = dt.datetime.strptime(ts, DATE_FMT)
            timestamp = tz.localize(timestamp_unaware)
            if sun['dawn'] < timestamp < sun['sunrise'] or\
                    sun['sunset'] < timestamp < sun['dusk']:
                shutil.copy(path_in + img, path_out)
                print "COPYING " + img + " to " + path_out


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
