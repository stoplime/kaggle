import os
import numpy as np
import tifffile as tiff
import csv
import time
import progressbar as pb
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

path = os.getcwd()
filename = os.path.join(path, "train-jpg", "train_10.jpg")
a = np.array(Image.open(filename))
a = a.take((0, 1, 2), axis=-1)
image = mpimg.imread(filename)
print(a)
print(a.shape)
plt.axis("off")
plt.imshow(image)
plt.show()
