#Import Libraries
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize
import os
import cv2
from training import *

IMAGE_SIZE=128
IMAGE_CHANNELS = 38
WEIGHT_PATH="ResUnet_weights.h5"

LABELS = {
            'Specular reflection' : 0,
            'Artery, ICG':          1,
            'Vein':                 2,
            'Stroma, ICG':          3,
            'Blue dye':             4,
            'Red dye':              5,
            'ICG':                  6,
            'Stroma':               7,
            'Umbilical cord':       8,
            'Suture':               9,
            'Artery':               10,
         }
class_colors = np.array([
            [255,0,0],
            [128,128,128],
            [0,128,0],
            [131,255,51],
            [128,0,128],
            [0,0,255],
            [255,255,0],
            [0,100,100],
            [165,42,42],
            [255,0,255],
            [255,165,0],[0,0,0]])
def getResNet50_Unet_model():
    model = build_resnet50_unet(11)
    model.load_weights(WEIGHT_PATH)
    return model

def segmentation_image(file_path):
    spim, wavelength, rgb_img, metadata = read_stiff(file_path)
    IMAGE_SIZE = 128
    resized_image = resize(spim,(IMAGE_SIZE,IMAGE_SIZE))
    test_image = resized_image
    test_image = np.expand_dims(test_image, axis=0)
    model = getResNet50_Unet_model()
    pred_mask = model.predict(test_image)
    pred_mask = np.expand_dims(np.squeeze(np.argmax(pred_mask, axis=3)),axis=-1)
    combined_predmask = np.zeros((test_image.shape[1:3]))
    combined_predmask = np.stack((combined_predmask,)*3, axis=-1)


    xy_list = np.where(pred_mask!=0)
    x_list = xy_list[0]
    y_list = xy_list[1]

    for x, y in zip(x_list, y_list):
        idxpred = pred_mask[x,y,0]
        combined_predmask[x,y,:] = class_colors[idxpred]
    #plt.imshow(combined_predmask)
    cv2.imwrite('pred_mask.jpg',combined_predmask)