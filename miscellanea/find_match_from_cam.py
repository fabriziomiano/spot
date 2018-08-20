from itertools import izip

import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image



def take_and_save_picture(im_save):
  '''Take a picture and save it

  Args:
    im_save: filepath where the image should be stored
  '''
  camera_port = 0
  ramp_frames = 30
  cap = cv2.VideoCapture(camera_port)
  def get_image():
   retval, im = cap.read()
   return im

  for i in xrange(ramp_frames):
   temp = get_image()

  print("Taking image...")
  # Take the actual image we want to keep
  camera_capture = get_image()

  #im_save_tmp = im_save + '.jpg'
  im_save_tmp = im_save 

  # A nice feature of the imwrite method is that it will automatically choose the
  # correct format based on the file extension you provide. Convenient!
  cv2.imwrite(im_save_tmp, camera_capture)

  # You'll want to release the camera, otherwise you won't be able to create a new
  # capture object until your script exits
  # del(cap)

  img1 = cv2.imread(im_save_tmp, 0)

  edges = cv2.Canny(img1, 100, 200)
  cv2.imwrite(im_save, edges)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

#im1 = "/Users/Me/gop.jpg"
#im2 = "/Users/Me/aarthi.jpg"
im1 = raw_input('enter the path of database file: ')
im2 = raw_input('enter the path where captured image is to be saved: ')
#im1="/Users/Me/home1.png"
#im2="/Users/Me/home.png"

def compute_edges_diff(im1, im2):
  '''Compute edges diff between to image files.

  Args:
    im1: filepath to the first image
    im2: filepath to the second image

  Returns:
    float: percentage of difference between images
  '''
#for no_file1 in range(0,10):
  #template = cv2.imread('numbers1/{no_file}.png'.format(no_file=no_file1),0)
  i1 = Image.open(im1)
  i2 = Image.open(im2)
  assert i1.mode == i2.mode, "Different kinds of images."
  assert i1.size == i2.size, "Different sizes."

  pairs = izip(i1.getdata(), i2.getdata())
  if len(i1.getbands()) == 1:
      # for gray-scale jpegs
      dif = sum(abs(p1-p2) for p1,p2 in pairs)
  else:
      dif = sum(abs(c1-c2) for p1,p2 in pairs for c1,c2 in zip(p1,p2))

  ncomponents = i1.size[0] * i1.size[1] * 3
  diff = (dif / 255.0 * 100) / ncomponents
  return diff

def main():
  #capture_img = "/Users/Me/home1.png"
  capture_img = raw_input('enter path of the file from database')
  #img_to_compare = "/Users/Me/Documents/python programs/compare/img2.jpg"
  take_and_save_picture(capture_img)
  diff = compute_edges_diff(im1, im2)
  print "Difference (percentage):", diff
  if diff > 0.5:
   print im1
  else :
   print im2

if __name__ == '__main__':
  main()

#del(cap)
