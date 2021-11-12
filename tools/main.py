# -*- coding: utf-8 -*-
# Copyright © 2021 Thong Duy Nguyen
# Email: thongngu@student.uef.vn

import argparse
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from numpy.core.defchararray import mod
from matplotlib import pyplot as plt

import __init__

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow import keras 
from sklearn.model_selection import train_test_split
# from keras.models import load_model

import time
import datetime
import os
import cv2
from PIL import Image
from sklearn.decomposition import PCA

import dataset 
from dataset import dataset
import config
import model
import metric


class hsi_rentina:
    def __init__(self):
        # Get configuration
        self.batch_size = config.BATCH_SIZE
        self.number_of_epoch = config.NUMBER_OF_EPOCH
        self.checkpoint_dir = config.CHECKPOINT_DIR
        self.log_dir = config.LOG_DIR
        self.colors = config.COLORS
        self.model_path = config.MODEL_PATH
        self.visualize_dir = config.VISUALIZE_DIR
        self.number_of_channel = config.NUMBER_OF_CHANNEL 
        self.input_size = config.INPUT_SIZE

    def train(self, X_train, X_test, Y_train, Y_test):
        # Get Dataset
        x_train = X_train
        y_train = Y_train
        x_test = X_test
        y_test = Y_test

        # Pre-process data and pass data to tensors
        train_dataset = self.data_pipeline(x_train, y_train)
        test_dataset = self.data_pipeline(x_test, y_test)
   
        # Intitialize threshold for Mean of IoU at 0.4
        mean_iou = metric.MeanIoU(11, 0.4)

        # Initilize model HRNet
        HR_Net = model.hrnet_keras()
        HR_Net.summary()
        # Optimizer = Adam, Loss function = Categorical Cross Entropy
        HR_Net.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[mean_iou])
        # Fit data into the network
        history = HR_Net.fit(train_dataset, 
                             initial_epoch = 0, 
                             epochs = self.number_of_epoch, 
                             validation_data = test_dataset,
                             callbacks = [self.save_checkpoints(), self.draw_tensorboard()])
    
    # def evaluate(self, x_eval, y_eval):
    #     # Load trained model
    #     model = keras.models.load_model(self.model_path)
    #     # Prepare input data
    #     import pdb
    #     pdb.set_trace()
    #     eval_dataset = self.data_pipeline(x_eval, y_eval, evaluate=True)
        
    #     for names, image, mask in eval_dataset.take(1):    
    #         for i in range(self.batch_size):   
    #             name = names[i].cpu().numpy()
    #             name = name.decode("utf-8")
    #             result  = model.evaluate(x=image[i][np.newaxis,:,:,:], y=mask[i][np.newaxis,:,:,:], batch_size=self.batch_size, verbose=1)
    #             print(result)

    def predict(self, x_test):
        # Load model
        model = keras.models.load_model(self.model_path)

        # Create directory for saving visualized image
        if not os.path.exists(self.visualize_dir):
            os.makedirs(self.visualize_dir)

        # Prepare input data
        test_dataset = self.data_pipeline1(x_test)
 
        for name, image in test_dataset:
            # Get image name
            name = name.cpu().numpy()
            name = name.decode("utf-8")

            # Get image
            image = image[np.newaxis,:,:,:]

            # Get predicted image
            pred_mask = model.predict(image)
            pred_mask = np.expand_dims(np.squeeze(np.argmax(pred_mask, axis=3)),axis=-1)
            combined_predmask = np.zeros((self.input_size, self.input_size))
            combined_predmask = np.stack((combined_predmask,)*3, axis=-1)

            # Find where the position of pixel having value != 0
            xy_list = np.where(pred_mask!=0)
            x_list = xy_list[0]
            y_list = xy_list[1]

            # Fill found positions with color codes.
            for x, y in zip(x_list, y_list):
                idxpred = pred_mask[x,y,0]
                combined_predmask[x,y,:] = self.colors[idxpred]
            # cv2.imwrite(self.visualize_dir+"/{}_combined_predmask.png".format(name), combined_predmask)
            combined_predmask = Image.fromarray(combined_predmask.astype('uint8'), 'RGB')
            combined_predmask.save(self.visualize_dir+"/{}_combined_predmask.png".format(name))

    def data_pipeline(self, X, Y, evaluate=False):
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
        if evaluate == True:
            names = tf.data.Dataset.from_tensor_slices(X[:,0])
            dataset = tf.data.Dataset.zip((names, dataset_x, dataset_y))
        else:
            dataset = tf.data.Dataset.zip((dataset_x, dataset_y))
            dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(2)
        return dataset
    
    def data_pipeline1(self, X):
        x = []
        for x_i in X:
            x.append(x_i[1])
        x = np.array(x)

        dataset_x = tf.data.Dataset.from_tensor_slices(x)
        names = tf.data.Dataset.from_tensor_slices(X[:,0])
        dataset = tf.data.Dataset.zip((names, dataset_x))
        dataset = dataset.prefetch(2)
        return dataset

    def get_label_code(self, label_name):
        return self.labels[label_name]

    def save_checkpoints(self):
        # Create directory to save checkpoints
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        
        checkpoint_path = self.checkpoint_dir + 'weights.{epoch:04d}-{val_loss:.2f}.hdf5'
        checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                       mode='auto', 
                                       monitor='val_loss')
        return checkpoint

    def draw_tensorboard(self):
        logdir = os.path.join(self.log_dir, str(self.input_size) + "_{}_".format(self.number_of_channel) + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir,
                                                            histogram_freq=1, 
                                                            write_graph=True)
        return tensorboard_callback

def parse_args():
    parser = argparse.ArgumentParser(description='Train HRNet on custom dataset')
 
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'predict' or 'evaluate'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/dataset/",
                        help='Directory of the Balloon dataset')
    parser.add_argument('--weights', required=False,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file")
    # parser.add_argument('--logs', required=False,
    #                     default=DEFAULT_LOGS_DIR,
    #                     metavar="/path/to/logs/",
    #                     help='Logs and checkpoints directory (default=logs/)')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    # Load data
    load_data = dataset(config.DATA_DIR, config.LABELS)
    load_data.data_preprocessing()
    x_train, x_test, y_train, y_test = load_data.generate_dataset()

    # Initialize model
    model = hsi_rentina()

    if args.command == "train":
        model.train(x_train, x_test, y_train, y_test)
    # elif args.command == "evaluate":
    #     model.evaluate(x_train, y_train)
    elif args.command == "predict":
        model.predict(x_train)

if __name__ == '__main__':
    main()
