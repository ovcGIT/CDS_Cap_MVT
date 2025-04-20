# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 10:03:30 2025

@author: Omkar
"""
import sys
sys.path.append('C:/Users/Omkar/Documents/teach_learn/IISc/capstone/26')
from master_utils import *

import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_curve, auc, confusion_matrix, roc_auc_score, roc_curve, classification_report
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from keras.applications import ResNet50
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
import pandas as pd
import pickle
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import AveragePooling2D, Flatten, Dropout, BatchNormalization, Dense, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint

# objects = ['grid','leather','tile','wood']
objects = ['bottle','cable','capsule','hazelnut','metal_nut','pill','screw', 'toothbrush', 'transistor', 'zipper']

for obj in objects:
    mvtobj_train_images, mvtobj_train_labels = load_mvt(base_folder, [obj], 'train')
    mvtobj_test_images, mvtobj_test_labels = load_mvt(base_folder, [obj], 'test')
    mvtobj_GT_images, mvtobj_GT_labels = load_mvt(base_folder, [obj], 'ground_truth')
    
    train_img = mvtobj_train_images
    train_lab = mvtobj_train_labels
    test_img = mvtobj_test_images
    test_lab = mvtobj_test_labels
    train_GT_img = mvtobj_GT_images
    train_GT_lab = mvtobj_GT_labels
    
    base_model = tf.keras.applications.MobileNetV2(
        weights="imagenet", 
        include_top=False, 
        input_shape=(224, 224, 3)
    )
    
    for layer in base_model.layers:
        layer.trainable = False
    
    par = 3  
    combined_images, combined_labels = createsamples(train_img, train_lab, test_img, test_lab, par)
    
    #Transfer learning
    headmodel = base_model.output
    headmodel = AveragePooling2D(pool_size = (4,4))(headmodel)
    headmodel = Flatten(name= 'flatten')(headmodel)
    headmodel = Dense(256, activation = "relu")(headmodel)
    headmodel = Dropout(0.3)(headmodel)
    headmodel = Dense(1, activation = 'sigmoid')(headmodel)
    
    model2 = Model(inputs = base_model.input, outputs = headmodel)
    model2.compile(loss = 'binary_crossentropy', optimizer='Nadam', metrics= ["accuracy"])
    
    earlystopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
    checkpointer = ModelCheckpoint(filepath="mobilenetv2_weights.hdf5", verbose=1, save_best_only=True)
    
    train_images, val_images, train_labels, val_labels = train_test_split(
        combined_images, combined_labels, test_size=0.2, random_state=42  # 20% for validation
    )
    
    history = model2.fit(
        train_images, train_labels,
        batch_size=32,  # Define your batch size
        epochs=200,      # Number of epochs
        validation_data=(val_images, val_labels),  # Use validation split
        shuffle=True,   # Shuffle the data
        callbacks=[checkpointer, earlystopping]
    )

    model2.save(f"{obj}_model_200.h5")
    model_history[obj] = history.history  # Save training history
    
    print(f"Completed processing for {obj}.\n")

print("Finished processing all objects.")

modelloc = r"C:\Users\Omkar\Documents\teach_learn\IISc\capstone\26\out\26_master_objects\\"
objects = ["bottle"]
for obj in objects:
    mvtobj_test_images, mvtobj_test_labels = load_mvt(base_folder, [obj], 'test')

    model = load_model(modelloc + obj + "_model_200.h5")
    print(f"Loaded {obj}")
    
    pred = model.predict(mvtobj_test_images)   
    stsh = 0.15
    outstat(mvtobj_test_labels, pred,stsh,obj)    

for h5_model_file in os.listdir(modelloc):
    if h5_model_file.endswith("_200.h5"):
        h5_model_path = os.path.join(modelloc, h5_model_file)

        model = tf.keras.models.load_model(h5_model_path, compile=False)

        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()

        tflite_model_file = h5_model_file.replace(".h5", ".tflite")
        
        tflite_model_path = os.path.join(modelloc, tflite_model_file)
        
        with open(tflite_model_path, "wb") as f:
            f.write(tflite_model)

        print(f"Converted {h5_model_file} to {tflite_model_file}")















