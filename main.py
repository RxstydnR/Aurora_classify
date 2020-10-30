# -*- coding: utf-8 -*-
import csv
import datetime
import glob
import json
import os
import random
import time
import yaml
import warnings
import argparse

import keras
import numpy as np
import seaborn as sns
import tensorflow as tf
from keras import backend as K
from keras import layers
from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint
from keras.models import Model
from keras.preprocessing.image import array_to_img, img_to_array, load_img, save_img
from keras.utils.np_utils import to_categorical
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split

import matplotlib
from matplotlib import pyplot as plt

from model import get_model
from data import data_extract_shuffle, load_npy_data, img_resize, get_data

matplotlib.use('Agg')
warnings.simplefilter(action = "ignore", category = FutureWarning)
warnings.filterwarnings('ignore')
os.environ["CUDA_VISIBLE_DEVICES"] = "4"


parser = argparse.ArgumentParser(
            prog='main.py',
            usage='Aurora classification with deep learning model.',
            description='Aurora classification experiment parameters.'
            )

parser.add_argument('--data_path', type=str, required=True, help='The path of data.')
parser.add_argument('--save_path', type=str, required=True, help='The path of folder to save results.')
parser.add_argument('--image_size', type=int, default=256, help='The size of image.')
parser.add_argument('--color', type=int, default=3, help='The color of image.')
parser.add_argument('--n_data', type=int, default=0, help='The number of each class data.')
parser.add_argument('--model', type=str, default="ResNet18", choices=['MobileNet','MobileNetV2','ResNet50','ResNet18'], help='deep learning model.')
parser.add_argument('--test_size', type=float, default=0.2, help='size of test data')
parser.add_argument('--vali_size', type=float, default=0.1, help='size of validation data')
parser.add_argument('--batchsize', type=int, default=32, help='training batch size')
parser.add_argument('--epochs', type=int, default=30, help='training epochs')
parser.add_argument('--random_state', type=int, default=0, help='random state')

opt = parser.parse_args()

if __name__=="__main__":

    os.makedirs(opt.save_path, exist_ok=False)

    # ラベルクラス対応表
    label_to_class = {
        'Aurora':          0,
        'AuroraCloud':     1,
        'AuroraMoon':      2,
        'AuroraMoonCloud': 3,
        'Cloud':           4
        #'Empty':           5,
        #'Fine':            6
    }

    if os.path.splitext(opt.data_path) == ".npy":
        X,Y = load_npy_data(opt.data_path)
        X = np.array(X)
        Y = np.array(Y)
    else:
        X,Y = get_data(opt.data_path, opt.image_size)

    assert X[0].shape[0] == X[0].shape[1], "Image's width and height have to be equal in this program."

    # Resize
    if X[0].shape[0] != opt.image_size:
        X = img_resize(X,opt.image_size)
    
    # Make the number of data equal. (データの数を揃える.)
    X, Y = data_extract_shuffle(X,Y,n=opt.n_data)
    
    X = X.astype(np.float32)/255
    classes = len(set(Y))
    Y = to_categorical(Y, classes)
    
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=opt.test_size, random_state=opt.random_state)
    print(f"train:{len(x_train)}, test:{len(x_test)}, All:{X}")
    
    # Model
    image_shape = (opt.image_size, opt.image_size, opt.color) 
    model = get_model(_input_shape=image_shape, num_classes=classes, model_name=opt.model)
    
    early_stopper = EarlyStopping(monitor='val_acc', min_delta=0.05, patience=5, restore_best_weights=True)
    csv_logger = CSVLogger(opt.save_path+'/history.csv', separator=',', append=False)
    
    history = model.fit(
                x_train, y_train,
                batch_size=opt.batch_size,
                epochs=opt.epochs,
                verbose=2,
                callbacks=[early_stopper, csv_logger],
                validation_split=opt.val_size,
    )
    
    # Save model
    model.save(opt.save_path+"/model.h5", include_optimizer=False)

    y_pred = model.predict(x_test)
    y_pred = np.argmax(y_pred,1)
    y_test = np.argmax(y_test,1)

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

    # Accuracy
    plt.figure()
    plt.plot(history.history['accuracy'], label="Acc")
    plt.plot(history.history['val_accuracy'], label="Val_Acc")
    plt.title(f'{opt.model} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper left')
    plt.savefig(opt.save_path+'/Accuracy.jpg')
    plt.clf()
    plt.close()

    # Loss
    plt.figure()
    plt.plot(history.history['loss'], label="Loss")
    plt.plot(history.history['val_loss'], label="Val_Loss")
    plt.title(f'{opt.model} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper left')
    plt.savefig(opt.save_path+'/Loss.jpg')
    plt.clf()
    plt.close()