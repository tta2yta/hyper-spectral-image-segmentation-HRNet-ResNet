# -*- coding: utf-8 -*-
# Copyright © 2021 Thong Duy Nguyen
# Email: thongngu@student.uef.vn

from re import S
import __init__
lib_path = __init__.get_libpath()
import config
import cv2
import os
import numpy as np
from PyQt5.QtCore import QBuffer
from PIL import Image
from spectral_tiffs import read_mtiff, read_stiff
from matplotlib import cm


import numpy as np
import cv2
import os
import glob
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow import keras 
from spectral_tiffs import read_mtiff, read_stiff
from skimage.transform import resize
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import config

class HRNet:
    def __init__(self):
        self.image_list = []
        self.mask_list = []
        self.image_paths = []
        self.dataset = []
        self.predmask_list = []
        self.gtmask_list = []
        self.batch_size = config.BATCH_SIZE
        self.model_path = config.MODEL_PATH
        self.colors = config.COLORS
        self.input_size = config.INPUT_SIZE
        self.image_w = config.IMAGE_W
        self.image_h = config.IMAGE_H
        self.labels = config.LABELS
        self.band_reduction = config.BAND_REDUCTION
        self.number_of_channel = config.NUMBER_OF_CHANNEL
        self.visualize_dir = config.VISUALIZE_DIR

    def visualize(self, paths):
        self.image_paths = paths
        self.get_mask()
        self.generate_dataset()
        self.predict()
        return self.predmask_list, self.gtmask_list

    def predict(self):
        # Load model
        model = keras.models.load_model(self.model_path)

        # Create directory for saving visualized image
        if not os.path.exists(self.visualize_dir):
            os.makedirs(self.visualize_dir)

        for name, image, mask in self.dataset:
            name = name.cpu().numpy()
            name = name.decode("utf-8")

            image = image[np.newaxis,:,:,:]
            
            gt_mask = mask[:,:,:]
            gt_mask = gt_mask.cpu().numpy()
            gt_mask = np.expand_dims(np.argmax(gt_mask, axis=-1), axis=-1)
            combined_gtmask = np.zeros((self.image_w, self.image_h))
            combined_gtmask = np.stack((combined_gtmask,)*3, axis=-1)

            pred_mask = model.predict(image)
            pred_mask = np.expand_dims(np.squeeze(np.argmax(pred_mask, axis=3)),axis=-1)
            combined_predmask = np.zeros((image.shape[1:3]))
            combined_predmask = np.stack((combined_predmask,)*3, axis=-1)

            np.stack((np.zeros((image.shape[1:3])),)*3, axis=-1)

            xy_list = np.where(pred_mask!=0)
            x_list = xy_list[0]
            y_list = xy_list[1]

            for x, y in zip(x_list, y_list):
                idxpred = pred_mask[x,y,0]
                combined_predmask[x,y,:] = self.colors[idxpred]
            combined_predmask = resize(combined_predmask,(self.image_w, self.image_h))
            self.predmask_list.append(combined_predmask)
            # cv2.imwrite(self.visualize_dir+"/{}_combined_predmask.png".format(name), combined_predmask)

            # combined_predmask = Image.fromarray(combined_predmask.astype('uint8'), 'RGB')
            # combined_predmask.save(self.visualize_dir+"/{}_combined_predmask.png".format(name))

            
            
            xy_list = np.where(gt_mask!=0)
            x_list = xy_list[0]
            y_list = xy_list[1]

            for x, y in zip(x_list, y_list):
                idx = gt_mask[x,y,0].astype(int)
                combined_gtmask[x,y,:] = self.colors[idx]
            self.gtmask_list.append(combined_gtmask)
            # cv2.imwrite(self.visualize_dir+"/{}_combined_gtmask.png".format(name), combined_gtmask)

            # combined_gtmask = Image.fromarray(combined_gtmask.astype('uint8'), 'RGB')
            # combined_gtmask.save(self.visualize_dir+"/{}_combined_predmask.png".format(name))

    def get_mask(self): 
        for image_file in self.image_paths:
            # Get spectral image, wavelength,rgb image and metadata
            spectral_image, w_lenght, rgb_image, metadata = read_stiff(image_file)
            image_name = os.path.basename(image_file)
            image_name_w_out_ext, ext = os.path.splitext(image_name)
            # Get mask image
            mask_name = image_name_w_out_ext + "_masks" + ext
            mask_file = lib_path + "/dataset/masks/" + mask_name
            mask_image_dict = read_mtiff(mask_file)
            # Create an empty mask with the same spectral image size
            combined_mask = np.zeros((spectral_image.shape[:2]))
            for key in mask_image_dict:
                label_code = self.get_label_code(key)
                spectral_mask_image = mask_image_dict[key].astype(int)
                spectral_mask_image [spectral_mask_image == 1] = label_code + 1
                # Get max value for stacked labels
                combined_mask = np.maximum(combined_mask, spectral_mask_image)
            
            self.image_list.append([image_name_w_out_ext, resize(spectral_image,(self.input_size,self.input_size))])
            self.mask_list.append([image_name_w_out_ext + "_masks", combined_mask])

            # self.image_list.append([image_name_w_out_ext, spectral_image])
            # self.mask_list.append([image_name_w_out_ext + "_masks", combined_mask])
    
    def generate_dataset(self):
        X = self.image_list
        y = self.mask_list
        for _y in y:
            _y[1] = _y[1].astype(int) 
        
        X = np.array(X, dtype=object)
        y = np.array(y, dtype=object)
        
        if self.band_reduction == True:
            for x in X:
                x[1] = self.pca_custom(x[1])
            
        self.data_pipeline(X,y)
    
    def data_pipeline(self, X, Y):
        x = []
        for x_i in X:
            x.append(x_i[1])
        x = np.array(x)

        y = []
        for y_i in Y:
            y.append(y_i[1])
        y = np.array(y)
        y = to_categorical(y)

        dataset_x = tf.data.Dataset.from_tensor_slices(x)
        dataset_y = tf.data.Dataset.from_tensor_slices(y)
        names = tf.data.Dataset.from_tensor_slices(X[:,0])
        self.dataset = tf.data.Dataset.zip((names, dataset_x, dataset_y))
        self.dataset = self.dataset.prefetch(2)

    def get_label_code(self, label_name):
        return self.labels[label_name]

    def pca_custom(self, x):
        x = x.reshape(x.shape[0]*x.shape[1], -1)
        pca = PCA(n_components=3)
        principalComponents = pca.fit_transform(x)
        img = principalComponents.reshape(self.input_size,self.input_size,3)
        return img

class UNet:
    def __init__(self):
        self.image_list = []
        self.mask_list = []
        self.image_paths = []
        self.dataset = []
        self.predmask_list = []
        self.gtmask_list = []
        self.name_list = []
        self.batch_size = config.BATCH_SIZE
        self.model_path = config.MODEL_PATH
        self.colors = config.COLORS
        self.input_size = config.INPUT_SIZE
        self.image_w = config.IMAGE_W
        self.image_h = config.IMAGE_H
        self.labels = config.LABELS

    def visualize(self, paths):
        self.image_paths = paths
        self.get_mask()
        self.generate_dataset()
        self.predict()
        return self.predmask_list, self.gtmask_list

    def predict(self):
        # Load model
        model = keras.models.load_model(self.model_path)

        for image, mask in zip(self.image_list, self.mask_list):
            name = self.name_list[self.image_list.tolist().index(image.tolist())] 

            image = np.expand_dims(image, axis=0)

            gt_mask = mask 
            gt_mask = np.expand_dims(mask, axis=-1)
            combined_gtmask = np.zeros((self.image_w, self.image_h))
            combined_gtmask = np.stack((combined_gtmask,)*3, axis=-1)

            pred_mask = model.predict(image)        
            pred_mask = np.expand_dims(np.squeeze(np.argmax(pred_mask, axis=3)),axis=-1)
            combined_predmask = np.zeros((image.shape[1:3]))
            combined_predmask = np.stack((combined_predmask,)*3, axis=-1)

            xy_list = np.where(pred_mask!=0)
            x_list = xy_list[0]
            y_list = xy_list[1]
            for x, y in zip(x_list, y_list):
                idxpred = pred_mask[x,y,0]
                combined_predmask[x,y,:] = self.colors[idxpred]
            combined_predmask = resize(combined_predmask,(self.image_w, self.image_h))
            self.predmask_list.append(combined_predmask)
            # cv2.imwrite(config.VISUALIZE_DIR+"/{}_combined_predmask.png".format(name), combined_predmask)
            
            xy_list = np.where(gt_mask!=0)
            x_list = xy_list[0]
            y_list = xy_list[1]

            for x, y in zip(x_list, y_list):
                idx = gt_mask[x,y,0].astype(int)
                combined_gtmask[x,y,:] = self.colors[idx]
            combined_gtmask = resize(combined_gtmask,(self.image_w, self.image_h))
            self.gtmask_list.append(combined_gtmask)
            # cv2.imwrite(config.VISUALIZE_DIR+"/{}_combined_gtmask.png".format(name), combined_gtmask)

    def get_mask(self): 
        for image_file in self.image_paths:
            # Get spectral image, wavelength,rgb image and metadata
            spectral_image, w_lenght, rgb_image, metadata = read_stiff(image_file)
            image_name = os.path.basename(image_file)
            image_name_w_out_ext, ext = os.path.splitext(image_name)
            # Get mask image
            mask_name = image_name_w_out_ext + "_masks" + ext
            mask_file = lib_path + "/dataset/masks/" + mask_name
            mask_image_dict = read_mtiff(mask_file)
            # Create an empty mask with the same spectral image size
            combined_mask = np.zeros((spectral_image.shape[:2]))
            for key in mask_image_dict:
                label_code = self.get_label_code(key)
                spectral_mask_image = mask_image_dict[key].astype(int)
                spectral_mask_image [spectral_mask_image == 1] = label_code + 1
                # Get max value for stacked labels
                combined_mask = np.maximum(combined_mask, spectral_mask_image)
            
            self.image_list.append(resize(spectral_image,(self.input_size,self.input_size)))
            self.mask_list.append(combined_mask)
            self.name_list.append(image_name_w_out_ext)
    
    def generate_dataset(self):
        X = self.image_list
        y = self.mask_list
        for _y in y:
            _y = _y.astype(int) 
        
        self.image_list = np.array(X)
        y = np.array(y)

    def get_label_code(self, label_name):
        return self.labels[label_name]

# imgs = glob.glob("/Users/liam/Documents/COSI Resources/Semester 3 - UEF/Industrial Project/HSI_Analyzer/sources/libs/dataset/images/*.*")

# model = HRNet()
# model.visualize(imgs)

