# -*- coding: utf-8 -*-
# Copyright © 2021 Thong Duy Nguyen
# Email: thongngu@student.uef.vn

import numpy as np
import os
import glob
import tensorflow as tf
import config
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from spectral_tiffs import read_mtiff, read_stiff
from skimage.transform import resize
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt


class dataset:
    def __init__(self, data_dir, labels):
        self.image_dir = os.path.join(data_dir, "images") 
        self.mask_dir = os.path.join(data_dir, "masks")
        self.labels = labels
        self.image_files = glob.glob(os.path.join(self.image_dir,"*.tif"))
        self.image_list = []
        self.image_name_list = []
        self.mask_list = []
        self.input_size = config.INPUT_SIZE
        self.band_reduction = config.BAND_REDUCTION
        self.number_of_channel = config.NUMBER_OF_CHANNEL

    def data_preprocessing(self): 
        for image_file in self.image_files:
            # Get spectral image, wavelength,rgb image and metadata
            spectral_image, w_lenght, rgb_image, metadata = read_stiff(image_file)
            image_name = os.path.basename(image_file)
            image_name_w_out_ext, ext = os.path.splitext(image_name)
            # Create mask image path
            mask_name = image_name_w_out_ext + "_masks" + ext
            mask_file = os.path.join(self.mask_dir, mask_name)
            # Get mask image dictionary
            mask_image_dict = read_mtiff(mask_file)
            # Create an empty mask with the same spectral image size
            combined_mask = np.zeros((spectral_image.shape[:2]))
            for key in mask_image_dict:
                # Get label code by label name
                label_code = self.get_label_code(key)
                # Convert pixel value to integer
                spectral_mask_image = mask_image_dict[key].astype(int)
                # Replace pixel having value 1 by label code
                spectral_mask_image [spectral_mask_image == 1] = label_code + 1
                # Get max value to stack the mask images
                combined_mask = np.maximum(combined_mask, spectral_mask_image)

            self.image_list.append([image_name_w_out_ext, resize(spectral_image,(self.input_size,self.input_size))])
            self.mask_list.append([image_name_w_out_ext + "_masks", resize(combined_mask,(self.input_size,self.input_size))])
        return self.image_list, self.mask_list
        
    def get_label_code(self, label_name):
        return self.labels[label_name]
    
    def generate_dataset(self, validation_set = False):
        X = self.image_list
        y = self.mask_list
        for _y in y:
            _y[1] = _y[1].astype(int) 

        if validation_set == False:
            x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle=True)
            x_train = np.array(x_train, dtype=object)
            x_test = np.array(x_test, dtype=object)
            y_train = np.array(y_train, dtype=object)
            y_test = np.array(y_test, dtype=object)

            if self.band_reduction == True:
                for x in x_train:
                    x[1] = self.pca_custom(x[1])
                for x in x_test:
                    x[1] = self.pca_custom(x[1])

            return x_train, x_test, y_train, y_test
        else: 
            x_trainval, x_test, y_trainval, y_test = train_test_split(X, y, test_size = 0.2, shuffle=True)
            x_train, x_val, y_train, y_val = train_test_split(x_trainval, y_trainval, test_size = 0.2, shuffle=True)
            x_train = np.array(x_train, dtype=object)
            x_val = np.array(x_val, dtype=object)
            x_test = np.array(x_test, dtype=object)
            y_train = np.array(y_train, dtype=object)
            y_val = np.array(y_val, dtype=object)
            y_test = np.array(y_test, dtype=object)

            if self.band_reduction == True:
                for x in x_train:
                    x[1] = self.pca_custom(x[1])
                for x in x_val:
                    x[1] = self.pca_custom(x[1])
                for x in x_test:
                    x[1] = self.pca_custom(x[1])

            return x_train, x_val, x_test, y_train, y_val, y_test

    def pca(self, x):
        imgs = []
        for x_i in x:
            x_i = x_i.reshape(x_i.shape[0]*x_i.shape[1], -1)
            pca = PCA(n_components=3)
            principalComponents = pca.fit_transform(x_i)
            img = principalComponents.reshape(256,256,3)
            imgs.append(img)
        return np.array(imgs)

    def pca_custom(self, x):
        x = x.reshape(x.shape[0]*x.shape[1], -1)
        pca = PCA(n_components=3)
        principalComponents = pca.fit_transform(x)
        img = principalComponents.reshape(self.input_size,self.input_size,self.number_of_channel)
        return img