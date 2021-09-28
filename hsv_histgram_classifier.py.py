import os
import warnings
import cv2
import argparse

import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
# from skimage.color import rgb2hsv

from utils import get_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
warnings.filterwarnings('ignore')


'''''''''''''''''''''''''''''''''''''''''''''
    HSV Color Space
        H:0~360
        S:0~100
        V:0~100    
    Aurora
        H:60~180
          and 
        S:25~97 
          and 
        V:20~90
    CV2 HSV color range
        H:0~180
          and 
        S:0~255 
          and 
        V:0~255
'''''''''''''''''''''''''''''''''''''''''''''
def hsv_classification(x_test):
    ''' Classification by hsv value '''
    
    y_pred = []
    
    # HSV_MIN = np.array([60/2, 25*255/100, 20*255/100]) 
    # HSV_MAX = np.array([180/2, 97*255/100, 90*255/100])
    
    HSV_MIN = np.array([20,50,40]) # 上の計算をして丸めた閾値
    HSV_MAX = np.array([100, 250, 230])

    for x in tqdm(x_test):

        x = cv2.cvtColor(x, cv2.COLOR_RGB2HSV) 

        # max_h=max(x[:,:,0].max(),max_h)
        # max_s=max(x[:,:,1].max(),max_s)
        # max_v=max(x[:,:,2].max(),max_v)
        
        # Mask HSV Region (output 0 or 255)
        mask_hsv = cv2.inRange(x, HSV_MIN, HSV_MAX)
        area = np.count_nonzero(mask_hsv == 255)

        if area>=100: # Threshold
            y_pred.append(0) # predict as Aurora class
        else:
            y_pred.append(1) # predict as Cloud class
    
    return y_pred


def hsv_histogram(X,Y,save_path):

    aurora_cnt=0
    cloud_cnt=0
    for x,y in tqdm(zip(X,Y)):
        
        # RGB histogram
        img = x.copy()
        r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]

        hist_b = cv2.calcHist([b],[0],None,[256],[0,256])
        hist_g = cv2.calcHist([g],[0],None,[256],[0,256])
        hist_r = cv2.calcHist([r],[0],None,[256],[0,256])
        plt.figure()
        plt.plot(hist_r, color='r', label="r")
        plt.plot(hist_g, color='g', label="g")
        plt.plot(hist_b, color='b', label="b")
        plt.legend()
        if y==0:    
            plt.savefig(save_path+f'/rgb_aurora_{aurora_cnt}.jpg')
        else:
            plt.savefig(save_path+f'/rgb_cloud_{cloud_cnt}.jpg')

        
        # HSV histogram
        img2 = cv2.cvtColor(x, cv2.COLOR_RGB2HSV)
        h, s, v = img2[:,:,0], img2[:,:,1], img2[:,:,2]
        hist_h = cv2.calcHist([h],[0],None,[256],[0,256])
        hist_s = cv2.calcHist([s],[0],None,[256],[0,256])
        hist_v = cv2.calcHist([v],[0],None,[256],[0,256])
        plt.figure()
        plt.plot(hist_h, color='r', label="h")
        plt.plot(hist_s, color='g', label="s")
        plt.plot(hist_v, color='b', label="v")
        plt.legend()
        if y==0:    
            plt.savefig(save_path+'/hsv_aurora_{}.jpg'.format(aurora_cnt))
            aurora_cnt+=1
        else:
            plt.savefig(save_path+'/hsv_cloud_{}.jpg'.format(cloud_cnt))
            cloud_cnt+=1

    return print("visualization of hsv histogram is done...")


def main():

    label_to_class = {
        'Aurora': 0,
        'Cloud': 1
    }

    # get image data
    X,Y = get_data(opt.data, ["aurora","cloud"])
    X = (X*255.).astype("uint8")
    Y = np.argmax(Y,1)
    print(X.shape,Y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=opt.test_size, random_state=2021, stratify=Y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=2021)
    print("train:{}, val:{}, test:{}, all:{}".format(len(X_train),len(X_test),len(X_val), len(X_train)+len(X_test)+len(X_val)))

    # Visualization of HSV histogram
    save_path = os.path.join(opt.save_dir, 'Histgram_result')
    os.makedirs(save_path, exist_ok=True)
    hsv_histogram(X_test, y_test, save_path)

    # Classification
    y_pred = hsv_classification(X_test)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))



if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Aurora & Cloud Classification by HSV Info')
    parser.add_argument('--data', required=True, type=str, help='Path to data directory.')
    parser.add_argument('--test_size', type=float, default=0.2, help='Ratio of Test data')
    parser.add_argument('--save_dir', type=str, default="Results", help='Path to a save folder (the folder will be made if it dose not exist).')
    opt = parser.parse_args()

    main()