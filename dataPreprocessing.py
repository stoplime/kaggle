# Data arrangement and processing
import os
import numpy as np
import tifffile as tiff
import csv
import time
import progressbar as pb
import pandas as pd


def load_data_dynamic(path, lable_path=None, batch_size=8, val_split=None):
    data_path = path
    if lable_path == None:
        lable_path = path
    
    value_max = 65535
    val_split = 3900
    _list = os.listdir(data_path)
    _list_n = [(int(''.join(list(filter(str.isdigit, x)))), _list[i]) for i, x in enumerate(_list)]
    _list_n = sorted(_list_n, key=getkey)

    labels = ['primary', 'clear', 'agriculture', 'road', 'water', 'partly_cloudy', 'cultivation', 'habitation', 'haze', 'cloudy', 'bare_ground', 'selective_logging', 'artisinal_mine', 'blooming', 'slash_burn', 'blow_down', 'conventional_mine']
    while 1:
        with open(lable_path) as f:
            reader = csv.reader(f)
            next(reader)
            # for b in range(batch_size):
            #     print("batch: ", b)
            x_train = []
            y_train = []
            for i, _filename in enumerate(_list_n):
                if val_split != None:
                    if i >= val_split:
                        break
                filename = _filename[1]
                img = np.array(tiff.imread(os.path.join(data_path, filename))) / value_max
                img = img.take((0, 1, 3), axis=-1)
                row = next(reader)
                tags = row[-1].split(' ')
                vec = np.zeros((len(labels),), dtype=int)
                for j in range(len(labels)):
                    if labels[j] in tags:
                        vec[j] = 1
                    else:
                        vec[j] = 0
                x_train.append(img)
                y_train.append(vec)
                if i % batch_size == 0:
                    # print("batch: ", i)
                    x_train = np.asarray(x_train)
                    y_train = np.asarray(y_train)
                    yield (x_train, y_train)
                    x_train = []
                    y_train = []

    # while 1:
    # f = open(path)
    # for line in f:
    #     # create numpy arrays of input data
    #     # and labels, from each line in the file
    #     x1, x2, y = process_line(line)
    #     yield ({'input_1': x1, 'input_2': x2}, {'output': y})
    # f.close()

def load_X_train_data(path, begin_images, n_images):
    # Chose path to the folder containing the training data in .jpg format:
    train_data_path = path
    # Chose number of images to load.
    # Type 'all' to load all the images in the folder, or 'half' to load half of them

    print('Loading Train Data: ')
    X_train, X_name_of_each_train = load_jpg_images(train_data_path, begin_images, n_images)
    X_train = np.array(X_train)

    print('Shape or train images array: ', X_train.shape)
    return X_train, X_name_of_each_train

def load_Y_data(path, begin_images, n_images):
    # Chose path to the .csv file containing the labels: 
    csv_path = path

    image_and_tags = csv_reader(csv_path)[begin_images:begin_images+n_images]
    labels = label_lister(image_and_tags)
    Y_train = list_to_vec(image_and_tags['tags'], labels)
    return Y_train

def getkey(item):
    return item[0]

def load_jpg_images(folder, start, N):
    _list = os.listdir(folder)
    
    # print("**** DATA: ", _list[0])
    _list_n = [(int(''.join(list(filter(str.isdigit, x)))), _list[i]) for i, x in enumerate(_list)]
    # print(_list_n[0])
    _list_n = sorted(_list_n, key=getkey)
    pbar = pb.ProgressBar(widgets=[pb.Percentage(), pb.Bar(), ], max_value=N).start() # max_value=len(list)).start()
    images = []
    filenames = []
    for i, _filename in enumerate(_list_n, start=start):
        if i >= N + start:
            break
        filename = _filename[1]
        img = np.array(tiff.imread(os.path.join(folder, filename)))/255
        if img is not None:
            images.append(img)
            filenames.append(filename)
        pbar.update(i-start)
    pbar.finish()
    return images, filenames

def csv_reader(file_labels):
    with open(file_labels) as f:
        CSVread = pd.read_csv(f)
    print('Labels succesfully loaded')
    return CSVread

def label_lister(labels_df):
    label_list = []
    for tag_str in labels_df.tags.values:
        labels = tag_str.split(' ')
        for label in labels:
            if label not in label_list:
                label_list.append(label)
    return label_list

def list_to_vec(list_img_labels, all_labels):
    number_of_labels = len(all_labels)
    number_of_pics = len(list_img_labels)
    vec = np.zeros([number_of_pics, number_of_labels], dtype=int)
    # ['haze', 'primary', 'agriculture', 'clear', 'water', 'habitation', 'road', 'cultivation', 'slash_burn', 'cloudy', 'partly_cloudy',
    # 'conventional_mine', 'bare_ground', 'artisinal_mine', 'blooming', 'selective_logging', 'blow_down']
    print('Translating lables into vectors:')
    pbar = pb.ProgressBar(widgets=[pb.Percentage(), pb.Bar(), ], max_value=(number_of_pics-1)).start()
    list_img_labels = [labels.split(' ') for labels in list_img_labels]
    for i in range(number_of_pics):
        pbar.update(i)
        for j in range(number_of_labels):
            if all_labels[j] in list_img_labels[i]:
                vec[i][j] = 1
            else:
                vec[i][j] = 0
    pbar.finish()
    return vec


