import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from skimage.feature import hog
from PIL import Image
from sklearn import svm
import time
import random

def train_imgs_blue(path,clf):
    labels = []
    features_list = []
    hog_list = []

    for root, dirs, files in os.walk("./Training"):
        for name in files:
            file_path = root+os.sep+name
            if name.endswith((".ppm",".jpg")) and root in path:
                imag = np.asarray(Image.open(file_path))
                img_resized = cv2.resize(imag,(64,64))
                img_gray = cv2.cvtColor(img_resized,cv2.COLOR_BGR2GRAY)

                feat_vec = hog(img_gray, orientations=4, pixels_per_cell=(4, 4),
                            cells_per_block=(2, 2), transform_sqrt = False, block_norm="L1", visualize=False, multichannel=False, feature_vector=True)

                features_list.append(feat_vec)
                labels.append(int(root[-2:]))

    clf.fit(features_list, labels)
    return clf



def predict_img_blue(imag_,clf):

    features_list_test = []
    ground_truth = []
    hog_list_ = []

    img_gray_ = cv2.cvtColor(imag_,cv2.COLOR_BGR2GRAY)

    feat_vec_ = hog(img_gray_, orientations=4, pixels_per_cell=(4, 4),
                cells_per_block=(2, 2), transform_sqrt = False, block_norm="L1", visualize=False, multichannel=False, feature_vector=True)
    features_list_test.append(feat_vec_)
    pred = clf.predict(features_list_test)

    class_probabilities = clf.predict_proba(features_list_test)

    return pred, class_probabilities


def train_imgs_red(path,clf):
    labels = []
    features_list = []
    hog_list = []

    for root, dirs, files in os.walk("./Training"):
        for name in files:
            file_path = root+os.sep+name
            if name.endswith((".ppm",".jpg")) and root in path:
                imag = np.asarray(Image.open(file_path))
                img_resized = cv2.resize(imag,(64,64))
                img_gray = cv2.cvtColor(img_resized,cv2.COLOR_BGR2GRAY)

                feat_vec = hog(img_gray, orientations=4, pixels_per_cell=(8, 8),
                            cells_per_block=(1, 1), transform_sqrt = True, block_norm="L1", visualize=False, multichannel=False, feature_vector=True)

                features_list.append(feat_vec)
                # hog_list.append(hog_img)
                labels.append(int(root[-2:]))

    clf.fit(features_list, labels)
    return clf


def predict_img_red(imag_,clf):

    features_list_test = []
    ground_truth = []
    hog_list_ = []

    img_gray_ = cv2.cvtColor(imag_,cv2.COLOR_BGR2GRAY)

    feat_vec_ = hog(img_gray_, orientations=4, pixels_per_cell=(8, 8),
                cells_per_block=(1, 1), transform_sqrt = True, block_norm="L1", visualize=False, multichannel=False, feature_vector=True)

    features_list_test.append(feat_vec_)
    pred = clf.predict(features_list_test)

    return pred
