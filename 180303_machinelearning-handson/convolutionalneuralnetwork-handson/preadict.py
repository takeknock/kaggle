# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 07:51:16 2018

@author: train2
"""

import numpy as np
import sys
from keras.preprocessing import image
from keras.models import load_model

filename = sys.argv[1]

model = load_model('./resnet50_best.h5')

img = image.load_img(filename, target_size=(224,224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = x / 255.0

preds = np.around(model.predict(x), decimals=3)
print(preds)