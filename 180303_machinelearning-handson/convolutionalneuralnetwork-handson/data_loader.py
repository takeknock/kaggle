# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 05:17:19 2018

@author: train2
"""


from keras.preprocessing.image import ImageDataGenerator

#image_w = 64
#image_h = 64
#batch_size = 20


image_w = 224
image_h = 224
batch_size = 20

def get_and_refine_image_data(filepath):
    generator = ImageDataGenerator(rescale=1/255, zoom_range=0.2, shear_range=0.2, horizontal_flip=True)
    data = generator.flow_from_directory(filepath, 
                                         target_size=(image_w, image_h),
                                         batch_size=batch_size)
    return data

def get_train_datagenerator():
    filepath = '../Caltech101_ValTrain/train'
    data = get_and_refine_image_data(filepath)
    return data

def get_validation_datagenerator():
    filepath = '../Caltech101_ValTrain/validation'
    data = get_and_refine_image_data(filepath)
    return data
    