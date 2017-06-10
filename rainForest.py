# KSU Data Science Team
# developed: Steffen Lim (slanimero@gmail.com) & John Ware (johnware55@gmail.com)
# Used as the kaggle competition
# Designed for the Amazon Satellite image classification

import os
import json
import numpy as np
from dataPreprocessing import load_X_train_data, load_Y_data
from model import class_model

PATH = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(PATH, "train-tif-v2")
label_path = os.path.join(PATH, "train_v2.csv")

model_type = 1

model = class_model(input_shape=(256, 256, 3))
model.create_model(model_type=model_type)

begin = 0
number_of_images = 5000
split_ratio = 0.1
split = int(split_ratio*number_of_images)
while begin < 40000:
    train, train_names = load_X_train_data(train_path, begin, number_of_images)
    train = np.array(train)
    train_names = np.array(train_names)
    print("train shape", train.shape)
    print("train names shape", train_names.shape)

    labels = load_Y_data(label_path, begin, number_of_images)
    print("labels shape", labels.shape)

    # 3 channel split
    channel_split = 1
    _train = train.take((0, 1, 3), axis=3)
    # for channel_split in range(4):
    # if channel_split == 0:
    #     _train = train.take((0, 1, 2), axis=3)
    # elif channel_split == 1:
    #     _train = train.take((0, 1, 3), axis=3)
    # elif channel_split == 2:
    #     _train = train.take((0, 2, 3), axis=3)
    # else:
    #     _train = train.take((1, 2, 3), axis=3)

    x_train, y_train, x_val, y_val = _train[split:], labels[split:], _train[:split], labels[:split] 

    history = model.train_model(x_train, y_train, validation=(x_val, y_val), save_path=os.path.join(PATH, "saved_models", "s"+str(begin)+"_m_"+str(model_type)+"_c_"+str(channel_split)))

    with open(os.path.join(PATH, "saved_models", "s"+str(begin)+"_m_"+str(model_type)+"_c_"+str(channel_split)+".json"), 'w') as f:
        data = []
        accuracy = model.kaggle_metric(input_val=x_val, labels_val=y_val)
        data.append(accuracy)
        data.append(history)
        json.dump(data, f, indent=4)
        print(accuracy)
    
    # number of images loop
    begin += number_of_images