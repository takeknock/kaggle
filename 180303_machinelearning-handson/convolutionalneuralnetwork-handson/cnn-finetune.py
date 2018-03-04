# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 07:10:20 2018

@author: train2
"""

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.applications.resnet50 import ResNet50
from keras.models import Model

from data_loader import *

image_w = 224
image_h = 224
batch_size = 20


train_generator = get_train_datagenerator()
validation_generator = get_validation_datagenerator()

class_num = len(train_generator.class_indices)
training_num = len(train_generator.classes)
validation_num = len(validation_generator.classes)
steps_per_epoch_ = (int)(training_num/batch_size)
validation_steps_ = (int)(validation_num/batch_size)
    
base_model = ResNet50(weights='imagenet',
                      include_top=False,
                      input_shape=(image_w, image_h, 3))

classifier_model = Sequential()
classifier_model.add(Flatten(input_shape=base_model.output_shape[1:]))
classifier_model.add(Dense(class_num, activation='softmax'))

model = Model(input=base_model.input, output=classifier_model(base_model.output))

for layer in model.layers[:174]:
    layer.trainable = False
    
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
checkpointer = ModelCheckpoint('resnet50_best.h5', verbose=1, save_best_only=True)
tb_cb = TensorBoard(log_dir='tb_log')

model.fit_generator(generator=train_generator,
                    steps_per_epoch=steps_per_epoch_,
                    epochs=10,
                    callbacks=[checkpointer, tb_cb],
                    validation_data=validation_generator,
                    validation_steps=validation_steps_)
