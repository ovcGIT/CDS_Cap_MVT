# -*- coding: utf-8 -*-
"""
Created on Sun Mar 30 17:34:52 2025
Utility file for capstone

@author: Omkar
"""
import matplotlib.pyplot as plt
import numpy as np

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
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import AveragePooling2D, Flatten, Dropout, BatchNormalization, Dense, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint


base_folder = r'C:\Users\Omkar\Documents\teach_learn\IISc\capstone\26\data\MVT'
outpath = "C:\\Users\\Omkar\\Documents\\teach_learn\\IISc\\capstone\\26\\out\\REStuned"
dproc = 'C:/Users/Omkar/Documents/teach_learn/IISc/capstone/26/data' + '/proc'

# Parameters
resol = 224

def load_mvt(base_folder,objects,data_type):
    images = []
    labels = []
    
    for item in objects: 
        item_folder = os.path.join(base_folder, item, data_type)
    
        if os.path.isdir(item_folder):
            for sub_folder in os.listdir(item_folder):
                sub_folder_path = os.path.join(item_folder, sub_folder)
                for filename in os.listdir(sub_folder_path):
                    img_path = os.path.join(sub_folder_path, filename)
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.resize(img, (resol, resol)) / 255.0
                        images.append(img)
                        if sub_folder == 'good':
                            labels.append(0)  # Non-defective label
                        else:
                            labels.append(1)  # Defective label
    return np.array(images), np.array(labels) 

def rgb(df):
    return np.repeat(df,3,axis=-1)

def createsamples(df_sampled, df_sampled_labels,df_2, df_2_labels,par):
    #par = 0 : use test set
    
    if par !=0:
        sample_size = len(df_sampled) // par 
    
        indices = np.random.choice(len(df_sampled), size=sample_size, replace=False)
        sampled_images = df_sampled[indices]
        sampled_labels = df_sampled_labels[indices]
        df_out = np.concatenate((df_2, sampled_images), axis=0)
        df_out_labels = np.concatenate((df_2_labels, sampled_labels), axis=0)

    else:
        df_out = df_2
        df_out_labels = df_2_labels
    return df_out, df_out_labels

def outstat(tst_lab, pred,tsh,obj_name):
        
    fpr, tpr, thresholds_roc = roc_curve(tst_lab, pred)
    roc_auc = auc(fpr, tpr)   
    
    # Plot the ROC Curve
    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title(f'{obj_name}: ROC')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.show()

    pred_label = (pred > tsh).astype(int)  
    conf_matrix = confusion_matrix(tst_lab, pred_label)
    cr = classification_report(tst_lab, pred_label)
    print(cr)
    print(conf_matrix)
    
def plt_vl(hist):
    train_loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    
    plt.plot(val_loss, label='Validation Loss', color='orange')
    plt.plot(train_loss, label='Training Loss', color='blue', linestyle='dashed')
    plt.title('Validation Loss vs. Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def plt_vl_dict(hist, nameofobj):
    train_loss = hist['loss']
    val_loss = hist['val_loss']
    
    plt.plot(val_loss, label='Validation Loss', color='orange')
    plt.plot(train_loss, label='Training Loss', color='blue', linestyle='dashed')
    plt.title(f'{nameofobj} - Validation Loss vs. Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
