import numpy as np
import cv2
from os import listdir
from os.path import isfile, join
from tqdm import tqdm

paths = {"train": "data/train", "val": "data/val"}

def process(path):
    img = cv2.imread(path)
    if img is not None:
        # Create new tuple with new width and height
        newDim = (250, 200)
        # Resize img to avoid using too much memory
        resizedImg = cv2.resize(img, newDim)
        # Grayscale image
        gray = cv2.cvtColor(resizedImg, cv2.COLOR_RGB2GRAY)
        return gray
    return None

def prepData(path):
    x_data = []
    y_data = []
    trainMode = False
    x_path = "x_val.npy"
    y_path = "y_val.npy"
    if path == "data/train":
        trainMode = True
        x_path = "x_train.npy"
        y_path = "y_train.npy"
    addonPaths = ["/NORMAL/", "/PNEUMONIA/"]
    for idx, addon in enumerate(addonPaths):
        for f in listdir(path+addon):
            if isfile(join(path+addon, f)):
                img = process(path+addon+f)
                if img is not None:
                    x_data.append(img)
                    y_data.append(idx)

    # Convert data lists to numpy arrays (will be needed when saving)
    x_data = np.array(x_data, dtype=np.float32)
    y_data = np.array(y_data, dtype=np.long)
    # Save the data arrays so that you wont have to prepare it again
    np.save(x_path, x_data)
    np.save(y_path, y_data)


if __name__ == '__main__':
    print("preparing data")
    # use "train" for training data and "val" for validation data.
    prepData(paths["train"])
    print("Data prepared and saved")
