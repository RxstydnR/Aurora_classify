# -*- coding: utf-8 -*-
import glob
import json
import os
import random
import time
import warnings
import cv2
import yaml

import numpy as np
import seaborn as sns
from PIL import Image
import matplotlib
from matplotlib import pyplot as plt
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from skimage.color import rgb2hsv

warnings.simplefilter(action = "ignore", category = FutureWarning)
warnings.filterwarnings('ignore')
matplotlib.use('Agg')
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

 
'''''''''''''''''''''''''''''''''''''''''''''
    HSV Color Space
        H:0~360
        S:0~100
        V:0~100    
    Aurora
        H:60~180
        S:25~97 
        V:20~90
    CV2 HSV color range
        H:0~180
        S:0~255 
        V:0~255
'''''''''''''''''''''''''''''''''''''''''''''
def hsv_classification(X):
    """ Aurora and Cloud Classification by HSV value.

    Args:
        X (arr): RGB image

    Returns:
        y_pred: Judge of Aurora (0) or Cloud (1)
    """
    
    HSV_MIN = np.array([60/2, 25*255/100, 20*255/100])  # 論文の閾値をcv2のスケールに合わせたもの
    HSV_MAX = np.array([180/2, 97*255/100, 90*255/100]) # 論文の閾値をcv2のスケールに合わせたもの

    y_pred = []
    for x in X:
        x = cv2.cvtColor(x, cv2.COLOR_RGB2HSV) # RGB -> HSV

        # Mask HSV Region (output 0 or 255)
        mask_hsv = cv2.inRange(x, HSV_MIN, HSV_MAX)

        # HSVの値によりAurora or Cloudの判別
        area = np.count_nonzero(mask_hsv == 255)

        threshold = 100 # Threshold
        if area >= threshold: 
            y_pred.append(0) # Aurora判定
        else:
            y_pred.append(1) # Cloud判定
    
    return np.array(y_pred)


def hsv_histogram(X,Y,save_path='histgram_result'):

    os.makedirs(save_path, exist_ok=True)

    c_dic = {
        0:"aurora",
        1:"cloud"
    }

    for i,(x,y) in enumerate(zip(X,Y)):
        
        # RGB ヒストグラム
        img_rgb = x.copy()
        r, g, b = img_rgb[:,:,0], img_rgb[:,:,1], img_rgb[:,:,2]

        hist_b = cv2.calcHist([b],[0],None,[256],[0,256])
        hist_g = cv2.calcHist([g],[0],None,[256],[0,256])
        hist_r = cv2.calcHist([r],[0],None,[256],[0,256])
        plt.figure()
        plt.plot(hist_r, color='r', label="r")
        plt.plot(hist_g, color='g', label="g")
        plt.plot(hist_b, color='b', label="b")
        plt.legend()
        plt.savefig(histogram_save_path+'/RGB_{}_{}.jpg'.format(i, c_dic[y]))
        plt.clf()
        plt.close()
    
        # HSV ヒストグラム
        img_hsv = cv2.cvtColor(x, cv2.COLOR_RGB2HSV)
        h, s, v = img_hsv[:,:,0], img_hsv[:,:,1], img_hsv[:,:,2]
        hist_h = cv2.calcHist([h],[0],None,[256],[0,256])
        hist_s = cv2.calcHist([s],[0],None,[256],[0,256])
        hist_v = cv2.calcHist([v],[0],None,[256],[0,256])
        plt.figure()
        plt.plot(hist_h, color='r', label="h")
        plt.plot(hist_s, color='g', label="s")
        plt.plot(hist_v, color='b', label="v")
        plt.legend()
        plt.savefig(histogram_save_path+'/HSV_{}_{}.jpg'.format(i, c_dic[y]))
        plt.clf()
        plt.close()
