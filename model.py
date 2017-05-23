# Keras model Design
# Using the Keras implementation of the Xception Model

import json
import numpy as np

from sklearn.metrics import fbeta_score

from keras.models import Model
from keras.layers import Input
from keras import metrics, losses

from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

from keras.applications.xception import Xception
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg19 import VGG19 
from keras.applications.vgg16 import VGG16

class class_model(object):
    def __init__(self, input_shape=(256, 256, 4), output_classes=17):
        self.input_tensor = Input(input_shape)
        self.output_size = output_classes

    def create_model(self, model_type='xception'):
        if(model_type == 'inceptionv3'):
            self.model = InceptionV3(include_top=True, weights=None, input_tensor=self.input_tensor, classes=self.output_size)
        elif(model_type == 'resnet50'):
            self.model = ResNet50(include_top=True, weights=None, input_tensor=self.input_tensor, classes=self.output_size)
        elif(model_type == 'vgg19'):
            self.model = VGG19(include_top=True, weights=None, input_tensor=self.input_tensor, classes=self.output_size)
        elif(model_type == 'vgg16'):
            self.model = VGG16(include_top=True, weights=None, input_tensor=self.input_tensor, classes=self.output_size)
        else:
            self.model = Xception(include_top=True, weights=None, input_tensor=self.input_tensor, classes=self.output_size)

        self.model.compile(loss=losses.binary_crossentropy, optimizer='adam', metrics=[metrics.binary_accuracy])

    def train_model(self, input_train, labels, validation=None, save_path=None):
        num_epochs = 2
        batch_size = 8

        logging = TensorBoard()
        if save_path != None:
            checkpoint = ModelCheckpoint(str(save_path)+".h5", monitor='val_loss', save_weights_only=True, save_best_only=True)
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')

        if validation==None:
            history = self.model.fit(input_train, labels, validation_split=0.2, batch_size=batch_size, epochs=num_epochs, verbose=1, callbacks=[logging, checkpoint, early_stopping])
        else:
            history = self.model.fit(input_train, labels, validation_data=validation, batch_size=batch_size, epochs=num_epochs, verbose=1, callbacks=[logging, checkpoint, early_stopping])
        return history.history

    def get_model(self):
        return self.model
        
    def kaggle_metric(self, input_val, labels_val):
        p_val = self.model.predict(input_val, batch_size=128)
        return fbeta_score(labels_val, np.array(p_val) > 0.2, beta=2, average='samples')

