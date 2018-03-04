# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 05:14:50 2018

@author: train2
"""


from data_loader import *
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.callbacks import TensorBoard

def train(model):
    return model

def evaluate(model):
    return model    


def refine(data):
    return data 

def create_model(class_num):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(image_w, image_h, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(class_num, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model
    

def main():
    image_w = 64
    image_h = 64
    batch_size = 20
    
    train_generator = get_train_datagenerator() 
    validation_generator = get_validation_datagenerator()   
    
    class_num = len(train_generator.class_indices)
    training_num = len(train_generator.classes)
    validation_num = len(validation_generator.classes)
    steps_per_epoch_ = (int)(training_num/batch_size)
    validation_steps_ = (int)(validation_num/batch_size)
    
    model = create_model(class_num)
    
    tb_cb = TensorBoard(log_dir='log2/')
    model.fit_generator(generator=train_generator, 
              steps_per_epoch=steps_per_epoch_,
              epochs=50,
              validation_data=validation_generator,
              validation_steps=validation_steps_,
              callbacks=[tb_cb])


if __name__ == '__main__':
    main()