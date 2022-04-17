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
        newDim = (150, 150)
        # Resize img to avoid using too much memory
        resizedImg = cv2.resize(img, newDim)
        return resizedImg
    return None

def prepData(path):
    x_data = []
    y_data = []
    addonPaths = ["/NORMAL/", "/PNEUMONIA/"]
    for idx, addon in tqdm(enumerate(addonPaths)):
        for f in listdir(path+addon):
            if isfile(join(path+addon, f)):
                img = process(path+addon+f)
                if img is not None:
                    if path == paths['train']:
                        # If training data is being generated/processed we want to use data augmentation by rotating images clockwise and counterclockwise to get more data.
                        horizontalImg1 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                        horizontalImg2 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                        x_data.append(img)
                        y_data.append(idx)
                        x_data.append(horizontalImg1)
                        y_data.append(idx)
                        x_data.append(horizontalImg1)
                        y_data.append(idx)
                    else:
                        x_data.append(img)
                        y_data.append(idx)

    # Convert data lists to numpy arrays (will be needed when saving)
    x_data = np.array(x_data, dtype=np.float32)
    y_data = np.array(y_data, dtype=np.long)

    return x_data, y_data
