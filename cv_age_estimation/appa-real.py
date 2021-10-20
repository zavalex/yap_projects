from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import ResNet50

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_test(path):
    df = pd.read_csv(path+'/labels.csv')
    validation_datagen = ImageDataGenerator(
        validation_split=0.25,
        rescale=1./255)

    val_datagen_flow = validation_datagen.flow_from_dataframe(
        dataframe=df,
        directory=path+'/final_files',
        x_col='file_name',
        y_col='real_age',
        target_size=(224, 224),
        batch_size=16,
        class_mode='raw',
        subset='validation',
        seed=127) 
    return val_datagen_flow


def load_train(path):
    df = pd.read_csv(path+'/labels.csv')

    train_datagen = ImageDataGenerator(
        validation_split=0.25,
        rescale=1./255
        #horizontal_flip=True,
        #vertical_flip=True,
        )

    train_datagen_flow = train_datagen.flow_from_dataframe(
        dataframe=df,
        directory=path+'/final_files',
        x_col='file_name',
        y_col='real_age',
        target_size=(224, 224),
        batch_size=16,
        class_mode='raw',
        subset='training',
        seed=127)
    return train_datagen_flow


def create_model(input_shape):
    backbone = ResNet50(input_shape=input_shape,
                 include_top=False,
                 weights='imagenet') 
    
    model = Sequential()
    model.add(backbone)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(units=1, activation='relu')) 

    optimizer = Adam(lr=0.0001) 
    model.compile(optimizer=optimizer, loss='mse', 
              metrics=['mae'])
 

    return model


def train_model(model, train_data, test_data, batch_size=None, epochs=5, 
                steps_per_epoch=None, validation_steps=None):
    if steps_per_epoch is None:
        steps_per_epoch = len(train_data)
    if validation_steps is None:
        validation_steps = len(test_data)                
    model.fit(train_data,
              validation_data=test_data,
              batch_size=batch_size, epochs=epochs,
              steps_per_epoch=steps_per_epoch,
              validation_steps=validation_steps,
              verbose=2, shuffle=True)
    return model