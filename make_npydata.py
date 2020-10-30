import argparse
import glob
import os
import warnings
import cv2
import numpy as np
from tqdm import tqdm 

from PIL import Image
from tqdm import tqdm

warnings.simplefilter('ignore')
warnings.filterwarnings('ignore')


def make_npy_data(data_path,save_path,img_size):
    """ Make npy Image and label dataset.
        This makes loading image dataset faster, but not necessary.
    """

    X = []
    Y = []
    for i,class_ in enumerate(glob.glob(data_path+"/*")):
        for img in tqdm(glob.glob(class_+"/*jpg")):
            x = Image.open(img).resize(size=(img_size,img_size))
            X.append(np.array(x))
            Y.append(i)

    assert len(X) == len(Y), "Length of Data and Label are not equel."    

    Data = np.array([X,Y])

    print(f"Save \"data.npy\" on {save_path}.")
    print(f"Data shape is {Data.shape}.")
    np.save(save_path+"/data", Data)

    return


parser = argparse.ArgumentParser(prog='make_npy_data.py')
parser.add_argument('--data_path', type=str, required=True, help='The path of image data.')
parser.add_argument('--save_path', type=str, required=True, help='The path of save folder.')
parser.add_argument('--img_size', type=int, default=256, help='Images will be saved at this size.')
opt = parser.parse_args()

if __name__=="__main__":
    """ Convert your image data to npy file.
    """
    os.makedirs(opt.save_path, exist_ok=False)
    make_npy_data(opt.data_path,opt.save_path,opt.img_size)
