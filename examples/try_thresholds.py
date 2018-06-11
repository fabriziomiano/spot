from skimage.measure import compare_ssim
from matplotlib import pyplot as plt
import argparse
import imutils
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--first", required=True,
	        help="first input image")
ap.add_argument("-s", "--second", required=True,
	        help="second")
ap.add_argument("-o", "--output", required=True,
	        help="output")
args = vars(ap.parse_args())


# load the two input images
imageA = cv2.imread(args["first"])
imageB = cv2.imread(args["second"])

# convert the images to grayscale
grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

# compute the Structural Similarity Index (SSIM) between the two
# images, ensuring that the difference image is returned
(score, diff) = compare_ssim(grayA, grayB, full=True)
diff = (diff * 255).astype("uint8")

# basic threshold
ret,thresh1 = cv2.threshold(diff,127,255,cv2.THRESH_BINARY)
ret,thresh2 = cv2.threshold(diff,127,255,cv2.THRESH_BINARY_INV)
ret,thresh3 = cv2.threshold(diff,127,255,cv2.THRESH_TRUNC)
ret,thresh4 = cv2.threshold(diff,127,255,cv2.THRESH_TOZERO)
ret,thresh5 = cv2.threshold(diff,127,255,cv2.THRESH_TOZERO_INV)
thresh6 = cv2.adaptiveThreshold(diff,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY_INV,11,2)
titles = ['Original_diff','BINARY','BINARY_INV','TRUNC','TOZERO',\
          'TOZERO_INV','ADAPTIVE_THRESH_MEAN_C']
images = [diff, thresh1, thresh2, thresh3, thresh4, thresh5, thresh6]

for i in xrange(7):
    plt.subplot(1,1,1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
    plt.savefig(args["output"] + "features_" + titles[i])

    cnts = cv2.findContours(images[i].copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]

    # loop over the contours
    for c in cnts:
        # compute the bounding box of the contour and then draw the
        # bounding box on both input images to represent where the two
        # images differ
        (x, y, w, h) = cv2.boundingRect(c)
        #cv2.rectangle(imageA, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.rectangle(imageB, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
    cv2.imwrite(args["output"] + "features_" + titles[i] + ".png", imageB)
    imageB = cv2.imread(args["second"])
