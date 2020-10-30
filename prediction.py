# -*- coding: utf-8 -*-

import glob
import os
import random
import warnings
import argparse

import keras
import numpy as np
import seaborn as sns
import tensorflow as tf
import matplotlib
from matplotlib import pyplot as plt
from keras import layers
from keras.utils import to_categorical
from keras.models import Model, load_model
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split

from data import data_extract_shuffle, load_npy_data, img_resize, get_data

matplotlib.use('Agg')
warnings.simplefilter(action = "ignore", category = FutureWarning)
warnings.filterwarnings('ignore')
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
 
parser = argparse.ArgumentParser(
            prog='prediction.py',
            usage='Aurora classification with a trained model.',
            description='Aurora classification experiment parameters.'
            )

parser.add_argument('--data_path', type=str, required=True, help='The path of test data.')
parser.add_argument('--model_path', type=str, required=True, help='The path of trained model.')
parser.add_argument('--save_path', type=str, required=True, help='The path of folder to save results.')
parser.add_argument('--image_size', type=int, default=256, help='The size of image.')
parser.add_argument('--n_data', type=int, default=0, help='The number of each class data.')

opt = parser.parse_args()


if __name__=="__main__":

    os.makedirs(opt.save_path, exist_ok=False)
    
    # Class to label
    label_to_class = {
        'Aurora':          0,
        'AuroraCloud':     1,
        'AuroraMoon':      2,
        'AuroraMoonCloud': 3,
        'Cloud':           4,
        'Empty':           5,
        'Fine':            6
    }
    
    if os.path.splitext(opt.data_path) == ".npy":
        X,Y = load_npy_data(opt.data_path)
        X = np.array(X)
        Y = np.array(Y)
    else:
        X,Y = get_data(opt.data_path, opt.image_size)

    X = X.astype(np.float32)/255

    # Resize
    if X[0].shape[0] != opt.image_size:
        X = img_resize(X,opt.image_size)
    
    # Make the number of data equal. (データの数を揃える.)
    X, Y = data_extract_shuffle(X,Y,n=opt.n_data)


    # Get trained model
    model = load_model(opt.model_path, compile=False)

    # Prediction
    y_pred = model.predict(X)
    y_pred = np.argmax(y_pred,1)
    y_test = np.argmax(Y,1)
    
    # Evaluation
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"Accuracy {acc} \n")
    print(f"Report {report} \n")
        
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8,8))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')
    ax.set_xticklabels(label_to_class.keys(), rotation=45)
    ax.set_yticklabels(label_to_class.keys(), rotation=0)
    ax.set_ylabel("True Class")
    ax.set_xlabel("Prediction Class")
    ax.set_ylim(len(cm), 0) 
    plt.tight_layout()
    plt.savefig(opt.save_path+"/ConfusionMatrix.jpg")
    plt.clf()
    plt.close()