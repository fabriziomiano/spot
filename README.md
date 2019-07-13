# Spotted!

## Acknowledgements
Thanks to pyimagasearch.com and many others...

## What is this? 
Spot is a tool capable of selecting interesting pictures within a given dataset.
It uses the *Structural Similarity* Index (SSIM) to quantify the differences between two images.
It is also capable of performing image classification using a Support Vector Machine (SVM).
Additionally, it provides a script to select sunset/sunrise/dawn/dusk images 
looking at the image filename, e.g. 1529032200_0_2018-06-15-04-10-00.jpg.  

## Dependencies 
It utilises:
scipy scikit-image opencv-python sklearn numpy astral pytz datetime

## How to use
Create a Python3 virtual environment following [this](https://docs.python.org/3/tutorial/venv.html).
Within the virtual environment
```pip install -r requirements.txt```

In the `COMMON` section of the conf file, specify `EXTENSION` of the image to be processed.

### SSIM SELECTION: 

In the `SSIM` section of the conf file: 
 - Specify `PATH_IN`, `PATH_OUT`, `REF_IMG_PATH`, and `THRESHOLD`.

Run 
```source select.sh <path/to/config-file> <mode>``` 
where `<mode>` has to be set to "quick". The other mode is "full", and it runs the timestamp selector first, then the ssim. 

### TIMESTAMP SELECTION:
The tool is designed to read images whose file names are date stings with a format specified in the conf file 

`DATE_FMT = %Y-%m-%d-%H-%M-%S` 

Change this at your own risk. The code hasn't been run with any other format.

In the `TIMESTAMP` section of the conf file: 
 - Specify `PATH_IN`, `PATH_OUT`, `TIMEZONE`, and `CITY_NAME`. 

Run within repo main dir 

```python src/timestamp-filter.py -c <path/to/config-file>```

### SVM
#### Training
In the `COMMON` section of the conf file: 
- specify `SVM_PATH` where the trained svm will be saved
- specify the number of clusters `N_CLUSTERS`

In the `TRAINING` section of the conf file, specify `TRAINING_PATH`. 

Run ```source trainSVM.sh <path/to/config-file>```

#### Classification
In the `CLASSIFICATION` section of the conf file: 
 - specify `TEST_DIR` (directory with images to be classified) 
 - specify `JSON_FILENAME` (file with results) 

Run ```source classify.sh <path/to/config-file>```
