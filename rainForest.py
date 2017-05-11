# KSU Data Science Team
# developed: Steffen Lim (slanimero@gmail.com) & John Ware (johnware55@gmail.com)
# Used as the kaggle competition
# Designed for the Amazon Satellite image classification

import os
import json
import numpy as np
from dataPreprocessing import load_X_train_data, load_X_test_data, load_Y_data, list_to_vec
# from model import class_model

from keras.applications.xception import Xception
from keras.models import Model
from keras.layers import Input

PATH = os.path.dirname(os.path.abspath(__file__))
number_of_images = 1000

train_path = os.path.join(PATH, )
train = load_X_train_data(train_path, number_of_images)

label_path = os.path.join(PATH, "train_v2.csv")
labels = load_Y_data(label_path, number_of_images)

_input = Input((256, 256, 3))
model = Xception(include_top=False, weights='imagenet', input_tensor=_input, input_shape=(256, 256, 3)))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])
model.fit(Xtr, Ytr, validation_split=0.2, batch_size=128, nb_epoch=30, verbose=1)
