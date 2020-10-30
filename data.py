from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def load_npy_data(data_path):
    """ Load image and label data from .npy data(データ読み込み)
    """
    Data = np.load(data_path, allow_pickle=True)
    X,Y = Data[0], Data[1]
    
    return [X,Y]


def get_data(data_path,img_size):

    X = []
    Y = []
    for i,class_ in enumerate(glob.glob(data_path+"/*")):
        for img in tqdm(glob.glob(class_+"/*jpg")):
            x = Image.open(img).resize(size=(img_size,img_size))
            X.append(np.array(x))
            Y.append(i)
    assert len(X) == len(Y), "Length of Data and Label are not equel."    

    return np.array(X), np.array(Y)




def img_resize(X, IMG_SIZE):
    """ Resizing a NumPy Array Image. (NumPy配列画像のリサイズ)
    """
    resized_X = []
    for img in X:
        img = Image.fromarray(np.uint8(img))
        resized_img = img.resize(size=(IMG_SIZE,IMG_SIZE))
        resized_X.append(np.array(resized_img))
    return resized_X



def data_extract_shuffle(X, Y, n):
    """ Randomly extracted data. (データをランダムに抽出)
    """
    # np.random.seed(0)
    p = np.random.permutation(len(X))
    X = X[p]
    X = X[:n]
    Y = Y[p]
    Y = Y[:n]
    return [X, Y]
