# KSU Data Science Team
# developed: Steffen Lim (slanimero@gmail.com) & John Ware (johnware55@gmail.com)
# Used as the kaggle competition
# Designed for the Amazon Satellite image classification

import os
import json
import numpy as np
from dataPreprocessing import load_X_train_data, load_Y_data, load_data_dynamic
from model import class_model
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

PATH = os.path.dirname(os.path.abspath(__file__))
test_path = os.path.join(PATH, "test-jpg")
train_path = os.path.join(PATH, "train-jpg")
lable_path = os.path.join(PATH, "train_v2.csv")

model_type = 0

# model = class_model(input_shape=(256, 256, 3))
# model.create_model(model_type=model_type)

number_of_images = 5000
split_ratio = 0.1
split = int(split_ratio*number_of_images)

def sequential_training():
    begin = 0
    while begin < 40000:
        train, train_names = load_X_train_data(train_path, begin, number_of_images)
        train = np.array(train)
        train_names = np.array(train_names)
        print("train shape", train.shape)
        print("train names shape", train_names.shape)

        labels = load_Y_data(lable_path, begin, number_of_images)
        print("labels shape", labels.shape)

        # 3 channel split
        channel_split = 1
        _train = train.take((0, 1, 3), axis=-1)
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

        with open(os.path.join(PATH, "saved_models", "_m_"+str(model_type)+"_c_"+str(channel_split)+".json"), 'w') as f:
            data = []
            accuracy = model.kaggle_metric(input_val=x_val, labels_val=y_val)
            data.append(accuracy)
            data.append(history)
            json.dump(data, f, indent=4)
            print(accuracy)
        
        # number of images loop
        begin += number_of_images
# creates a generator for data acquisition
def dynamic_training():
    channel_split = 1
    total_num_images = 40478
    batch_size = 32
    val_num = 5000
    val_begin = total_num_images-val_num
    model_name = str(model_type)+"_bi_multiDropout2_3"
    save_path = os.path.join(PATH, "saved_models", model_name)

    logging = TensorBoard()
    checkpoint = ModelCheckpoint(str(save_path)+".h5", monitor='val_FScore2', save_weights_only=False, save_best_only=True)
    # early_stopping = EarlyStopping(monitor='val_FScore2', min_delta=0.01, patience=5, verbose=1, mode='max')

    data_gen = load_data_dynamic(train_path, lable_path=lable_path, batch_size=batch_size, val_split=val_begin)
    print("Loading validation data")
    x_val, train_names = load_X_train_data(train_path, val_begin, val_num)
    x_val = np.array(x_val)
    x_val = x_val.take((0, 1, 2), axis=-1)
    #x_val = np.swapaxes(x_val, -1, 1)
    y_val = load_Y_data(lable_path, val_begin, val_num)
    # print("val", x_val.shape)
    history = model.get_model().fit_generator(data_gen, steps_per_epoch=total_num_images//batch_size, epochs=15, validation_data=(x_val, y_val), max_q_size=128, verbose=1, callbacks=[logging, checkpoint])
    # total_num_images//batch_size

    with open(os.path.join(PATH, "saved_models", model_name+".json"), 'w') as f:
        data = []
        accuracy = model.kaggle_metric(input_val=x_val, labels_val=y_val)
        print(accuracy)
        data.append(accuracy)
        data.append(history.history)
        json.dump(data, f, indent=4)
    
    test_gen = load_data_dynamic
    with open(os.path.join(PATH, "saved_models", "valPred_"+model_name+".json"), 'w') as f:
        pred = model.get_model().predict(x_val, batch_size=32, verbose=1)
        json.dump(pred.tolist(), f, indent=4)

if __name__ == "__main__":
    model_type = 'inceptionv3MOD'
    model = class_model(input_shape=(256, 256, 3))
    model.create_model(model_type=model_type)
    # model.create_model(model_type=model_type, load_weights=os.path.join(PATH, "saved_models", "resnet50_shallow.h5"))
    # model.load_weights(os.path.join(PATH, "saved_models", "resnet50_shallow.h5"))
    dynamic_training()