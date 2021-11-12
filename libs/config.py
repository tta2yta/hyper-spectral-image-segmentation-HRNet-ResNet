# -*- coding: utf-8 -*-
# Copyright © 2021 Thong Duy Nguyen
# Email: thongngu@student.uef.vn

import os
import __init_lib__
lib_path = __init_lib__.get_libpath()

#DATASET
DATA_DIR = lib_path + "/dataset/"
COLORS = [  [0,0,0],
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
            [255,165,0]]

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

BATCH_SIZE = 32

NUMBER_OF_EPOCH = 100

LOG_DIR =  lib_path + "/../logs"

MODEL_NAME = "hrnet_256x256x38.hdf5"

MODEL_PATH = lib_path + "/model/" + MODEL_NAME

VISUALIZE_DIR =  lib_path + "/../hret_256x256x38" 

INPUT_SIZE = 256

IMAGE_W = 1024

IMAGE_H = 1024

NUMBER_OF_CHANNEL = 38# 3

BAND_REDUCTION = False # For doing PCA

CHECKPOINT_DIR = lib_path + "/../checkpoints_{}x{}x{}/".format(INPUT_SIZE,INPUT_SIZE,NUMBER_OF_CHANNEL)

BN_MOMENTUM = 0.1

BN_EPSILON = 1e-5

MODEL_OPTION = "hrnet" #"unet"