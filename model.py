# Keras model Design
# Using the Keras implementation of the Xception Model

from keras.applications.xception import Xception
from keras.models import Model
from keras.layers import Input

class class_model(object):
    def __init__(self):
        _input = Input((256, 256, 3))
        self.model = Xception(include_top=False, weights='imagenet', input_tensor=_input, input_shape=(256, 256, 3)))

