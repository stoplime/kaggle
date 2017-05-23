# KSU Data Science Team
# developed: Steffen Lim (slanimero@gmail.com) & John Ware (johnware55@gmail.com)
# Used as the kaggle competition
# Designed for the Amazon Satellite image classification

import os
import json
import numpy as np
from dataPreprocessing import load_X_train_data, load_X_test_data, load_Y_data
from model import class_model

PATH = os.path.dirname(os.path.abspath(__file__))
begin = 5000
number_of_images = 5000

train_path = os.path.join(PATH, "train-tif-v2")
train, train_names = load_X_train_data(train_path, begin, number_of_images)
train = np.array(train)
train_names = np.array(train_names)
print("train shape", train.shape)
print("train names shape", train_names.shape)

label_path = os.path.join(PATH, "train_v2.csv")
labels = load_Y_data(label_path, begin, number_of_images)

model_type = 0

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


split = int(0.2*number_of_images)
x_train, y_train, x_val, y_val = _train[split:], labels[split:], _train[:split], labels[:split] 

# _input = Input((256, 256, 3))
model = class_model(input_shape=(256, 256, 3))
model.create_model()

history = model.train_model(x_train, y_train, validation=(x_val, y_val), save_path=os.path.join(PATH, "saved_models", "model_"+str(model_type)+"_channel_"+str(channel_split)))

with open(os.path.join(PATH, "saved_models", "model_"+str(model_type)+"_channel_"+str(channel_split)+".json"), 'w') as f:
    data = []
    accuracy = model.kaggle_metric(input_val=x_val, labels_val=y_val)
    data.append(accuracy)
    data.append(history)
    json.dump(data, f, indent=4)
    print(accuracy)
