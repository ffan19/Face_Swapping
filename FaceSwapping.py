from skimage.feature import corner_harris, corner_peaks
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.interpolation import geometric_transform
from sklearn.preprocessing import normalize
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import random
import math
from numpy.linalg import inv
import pdb
import cv2
import imageio
import dlib

# Pretrained model
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
SCALE_FACTOR = 1
FEATHER_AMOUNT = 11

# Amount of blur to use during colour correction, as a fraction of the
# pupillary distance.
COLOUR_CORRECT_BLUR_FRAC = 0.7


FACE_POINTS = list(range(17, 68))
MOUTH_POINTS = list(range(48, 61))
LEFT_EYE_POINTS = list(range(42, 48))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_BROW_POINTS = list(range(17, 22))
NOSE_POINTS = list(range(27, 35))
JAW_POINTS = list(range(0, 17))


# Points used to line up the images.
ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS +
                RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS)

# Points from the second image to overlay on the first. The convex hull of each
OVERLAY_POINTS = [
    LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS,
    NOSE_POINTS + MOUTH_POINTS
]


# returns a list of rectangles, each of which corresponding with a face in the image
detector = dlib.get_frontal_face_detector()
# feature extractor requires a rough bounding box as input to the algorithm, use pre-trained model
predictor = dlib.shape_predictor(PREDICTOR_PATH)


class TooManyFaces(Exception):
  pass


class NoFaces(Exception):
  pass


'''
  Takes an image in the form of a numpy array, and returns a 68x2 element matrix, each
  row of which corresponding with the x, y coordinates of a particular feature point in
  the input image
'''


def get_landmarks(im):

  rects = detector(im, 1)

  if len(rects) > 1:
    raise TooManyFaces
  if len(rects) == 0:
    raise NoFaces

  return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])

# Ordinary Procrustes Analysis to align faces


def transformation_from_points(points1, points2):
  # Standarize points by subtracting mean and dividing by standard deviation
  points1 = points1.astype(np.float64)
  points2 = points2.astype(np.float64)

  c1 = np.mean(points1, axis=0)
  c2 = np.mean(points2, axis=0)
  points1 -= c1
  points2 -= c2

  s1 = np.std(points1)
  s2 = np.std(points2)
  points1 /= s1
  points2 /= s2

  U, S, Vt = np.linalg.svd(points1.T * points2)
  R = (U * Vt).T

  return np.vstack([np.hstack(((s2 / s1) * R,
                               c2.T - (s2 / s1) * R * c1.T)),
                    np.matrix([0., 0., 1.])])

# Affine transformation


def warp_im(im, M, dshape):
  output_im = np.zeros(dshape, dtype=im.dtype)
  cv2.warpAffine(im,
                 M[:2],
                 (dshape[1], dshape[0]),
                 dst=output_im,
                 borderMode=cv2.BORDER_TRANSPARENT,
                 flags=cv2.WARP_INVERSE_MAP)
  return output_im


# Color correction using Gaussian blur


def correct_colours(im1, im2, landmarks1):
  blur_amount = COLOUR_CORRECT_BLUR_FRAC * np.linalg.norm(
      np.mean(landmarks1[LEFT_EYE_POINTS], axis=0) -
      np.mean(landmarks1[RIGHT_EYE_POINTS], axis=0))
  blur_amount = int(blur_amount)
  if blur_amount % 2 == 0:
    blur_amount += 1
  im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
  im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)

  # Avoid divide-by-zero errors.
  im2_blur = np.add(im2_blur, 128 * (im2_blur <= 1.0), out=im2_blur, casting="unsafe")  # 128 * (im2_blur <= 1.0)

  return (im2.astype(np.float64) * im1_blur.astype(np.float64) /
          im2_blur.astype(np.float64))


# Blend features from second image to first
def draw_convex_hull(im, points, color):
  points = cv2.convexHull(points)
  cv2.fillConvexPoly(im, points, color=color)


def get_face_mask(im, landmarks):
  im = np.zeros(im.shape[:2], dtype=np.float64)

  for group in OVERLAY_POINTS:
    draw_convex_hull(im,
                     landmarks[group],
                     color=1)

  im = np.array([im, im, im]).transpose((1, 2, 0))

  im = (cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0) > 0) * 1.0
  im = cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)

  return im


def read_im_and_landmarks(fname):
  im = cv2.resize(fname, (fname.shape[1] * SCALE_FACTOR,
                          fname.shape[0] * SCALE_FACTOR))
  s = get_landmarks(im)

  return im, s


def annotate_landmarks(im, landmarks):
  im = im.copy()
  for idx, point in enumerate(landmarks):
    pos = (point[0, 0], point[0, 1])
    cv2.putText(im, str(idx), pos,
                fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                fontScale=0.4,
                color=(0, 0, 255))
    cv2.circle(im, pos, 3, color=(0, 255, 255))
  return im


pic_list1 = []
pic_list2 = []

vidcap1 = cv2.VideoCapture('CIS581Project4PartCDatasets/Medium/LucianoRosso1.mp4')
vidcap2 = cv2.VideoCapture('CIS581Project4PartCDatasets/Easy/FrankUnderwood.mp4')
count = 0
success = True
while success:
  success, image1 = vidcap1.read()
  print('Read a new frame: ', success)

  pic_list1.append(image1)

  count += 1

count = 0
success = True
while success:
  success, image2 = vidcap2.read()
  print('Read a new frame2: ', success)

  pic_list2.append(image2)

  count += 1


output_images = []
for i in range(0, len(pic_list2)):
  im1, landmarks1 = read_im_and_landmarks(pic_list1[i])
  im2, landmarks2 = read_im_and_landmarks(pic_list2[i])
  print(i)
  M = transformation_from_points(landmarks1[ALIGN_POINTS], landmarks2[ALIGN_POINTS])

  mask = get_face_mask(im2, landmarks2)
  warped_mask = warp_im(mask, M, im1.shape)
  combined_mask = np.max([get_face_mask(im1, landmarks1), warped_mask], axis=0)

  warped_im2 = warp_im(im2, M, im1.shape)
  warped_corrected_im2 = correct_colours(im1, warped_im2, landmarks1)

  output_im = im1 * (1.0 - combined_mask) + warped_corrected_im2 * combined_mask
  output_images.append(output_im)

  imageio.mimsave('./face_swap_7.gif', output_images)

  cv2.imwrite('output7.jpg', output_im)
  print("Finished")
