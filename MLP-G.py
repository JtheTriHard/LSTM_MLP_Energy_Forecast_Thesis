# -*- coding: utf-8 -*-
# Author: Joey G.
# Runs in Spyder 4.1. using Python 3.7

import time
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler,RobustScaler

np.random.seed(123)
tf.random.set_seed(123)

# Confirm working directory
print("Working Directory: ", os.getcwd())
# Verify GPUs usable by tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# =============================================================================
# PREPARE DATA

# Load data
df = pd.read_csv('YOUR_DIRECTORY')
# Rename columns with characters not accepted by tf
df = df.rename(columns={'building_type_Single-Family Home 001 (Master)':'building_type_Single-Family Home'})
# Reorganize features in order expected by model, drop unused features
new_order = ['dataid','Hour','Weekday','building_type_Apartment',
             'building_type_Single-Family Home', 'building_type_Town Home',
             'state_California','state_Texas', 'grid_fixed','Dew Point']
df = df[new_order]
# Remap IDs to consecutive values, to be embedded later
df['dataid'], remap = pd.factorize(df['dataid'])
remap = remap.to_list()
# Reserve ID = 0 for new households, shift IDs accordingly
df['dataid'] = df['dataid']+1
remap.insert(0,'Unknown ID') #index = new ID, entry = original ID

# Create 90%-10% train-validation set
nseries = df['dataid'].nunique() # unique IDs in dataset
ID_length = 8760
df_train, df_test = pd.DataFrame([]), pd.DataFrame([])
for i in range(nseries):
    start = i*ID_length
    end = (i+1)*ID_length
    split = start + int(ID_length*0.9)
    df_curTrain = df.iloc[start:split]
    df_curTest = df.iloc[split:end]
    df_train = pd.concat([df_train,df_curTrain],axis=0)
    df_test = pd.concat([df_test,df_curTest],axis=0)

# Scale data 
scale_cols = ['grid_fixed','Dew Point']
scaler = MinMaxScaler()
df_train[scale_cols] = scaler.fit_transform(df_train[scale_cols])
df_test[scale_cols] = scaler.transform(df_test[scale_cols])

# Transform into LSTM shape. Will have (lags + steps_ahead) LESS samples than input
# MLP uses this by using a slice for each feature for its respective input layers
# Given LSTM [sample 0:i, timesteps, features 0:j] -> MLP uses [i, 24, j] which is a vector
# Modified from TensorFlow.org
lags = 24 # past obs to use
steps_ahead = 24 # period to forecast
def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, single_step=False):
    dataset = dataset.values
    target = target.values
    data = []
    labels = []
    start_index = start_index + history_size # starts (lags) after first obs
    if end_index is None:
        end_index = len(dataset) - target_size # ends (steps_ahead) before last obs
    for i in range(start_index, end_index):
        indices = range(i-history_size, i)
        data.append(dataset[indices])
        if single_step:
            labels.append(target[i+target_size])
        else:
            labels.append(target[i:i+target_size])
    return np.array(data), np.array(labels)

# Transform each ID in the training set individually and append
# This prevents the ends of one ID from being used as obs for an adjacent ID
x_train, y_train = [],[]
train_size = int(len(df_train)/nseries)
for i in range(nseries):
    df_temp = df_train.iloc[i*train_size:(i+1)*train_size].reset_index(drop=True)
    x_temp, y_temp = multivariate_data(df_temp,df_temp['grid_fixed'],0,None,lags,steps_ahead,single_step=False)
    x_train.append(x_temp)
    y_train.append(y_temp)
x_train,y_train = np.concatenate(x_train).astype('float32'),np.concatenate(y_train).astype('float32')
# For test set, kept in separate loop due to previous approaches
test_size = int(len(df_test)/nseries)
x_test, y_test = [],[]
for i in range(nseries):
    df_temp = df_test.iloc[i*test_size:(i+1)*test_size].reset_index(drop=True)
    x_temp, y_temp = multivariate_data(df_temp,df_temp['grid_fixed'],0,None,lags,steps_ahead,single_step=False)
    x_test.append(x_temp)
    y_test.append(y_temp)
x_test, y_test = np.concatenate(x_test).astype('float32'),np.concatenate(y_test).astype('float32')

# =============================================================================
# MODEL

# Features to be embedded
cat_cols = ['dataid','Hour','Weekday']
# Features to send as regular numeric input
num_cols = ['building_type_Apartment','building_type_Single-Family Home',
            'building_type_Town Home','state_California','state_Texas',
            'grid_fixed','Dew Point']

# Parameters
n_epochs = 200
embed_dim = 3
neurons = 64
batchsize= 16384

# Multi-Headed MLP using Keras Functional API
# Each feature feeds into a head, which each feeds into a separate MLP architecture
tf.keras.backend.clear_session()
# Heads for embedded features
inputs, embeddings = [], []
for i in cat_cols:
    cat_input = tf.keras.layers.Input(shape=(lags,), name="".join([i.replace(" ", ""),"_inp"]))
    cat_dim  = df[i].nunique()
    if i == 'dataid':
        cat_dim = cat_dim + 1 # +2 when holding out entire ID
    inputs.append(cat_input)
    cat_input = tf.keras.layers.Embedding(cat_dim, embed_dim, input_length = lags,
            name="".join([i.replace(" ", ""),"_embed"]))(cat_input)
    cat_input = tf.reshape(cat_input,[-1,lags*embed_dim])
    cat_input = tf.keras.layers.Dense(neurons,activation='tanh')(cat_input)
    cat_input = tf.keras.layers.Dropout(0.1)(cat_input)
    cat_input = tf.keras.layers.Dense(neurons,activation='tanh')(cat_input)
    cat_input = tf.keras.layers.Dropout(0.1)(cat_input)
    cat_input = tf.keras.layers.Dense(neurons,activation='tanh')(cat_input)
    cat_input = tf.keras.layers.Dropout(0.1)(cat_input)
    #cat_input = tf.keras.layers.Dense(neurons,activation='tanh')(cat_input)
    #cat_input = tf.keras.layers.Dropout(0.1)(cat_input)
    embeddings.append(cat_input)
    
# Heads for all other features
for j in num_cols:
    num_input = tf.keras.layers.Input(shape=(lags), 
          name="".join([j.replace(" ", ""),"_input"]))
    inputs.append(num_input)
    num_input = tf.keras.layers.Dense(neurons,activation='tanh')(num_input)
    num_input = tf.keras.layers.Dropout(0.1)(num_input)
    num_input = tf.keras.layers.Dense(neurons,activation='tanh')(num_input)
    num_input = tf.keras.layers.Dropout(0.1)(num_input)
    num_input = tf.keras.layers.Dense(neurons,activation='tanh')(num_input)
    num_input = tf.keras.layers.Dropout(0.1)(num_input)
    #num_input = tf.keras.layers.Dense(neurons,activation='tanh')(num_input)
    #num_input = tf.keras.layers.Dropout(0.1)(num_input)
    embeddings.append(num_input)
    
# Combine all heads and compile
input_layer = tf.keras.layers.Concatenate(name="concat")(embeddings)
output_layer = tf.keras.layers.Dense(steps_ahead,name='output_layer')(input_layer)
model = tf.keras.Model(inputs, output_layer, name = "MLP-C-G")
model.compile(optimizer='adam',loss='mse')
model.summary()
tf.keras.utils.plot_model(model, 'YOUR_DIRECTORY', show_shapes=True)
# Fit
train_input = [x_train[:,:,0],x_train[:,:,1],x_train[:,:,2],x_train[:,:,3],
               x_train[:,:,4],x_train[:,:,5],x_train[:,:,6],x_train[:,:,7],
               x_train[:,:,8],x_train[:,:,9]]
test_input = [x_test[:,:,0],x_test[:,:,1],x_test[:,:,2],x_test[:,:,3],
              x_test[:,:,4],x_test[:,:,5],x_test[:,:,6],x_test[:,:,7],
              x_test[:,:,8],x_test[:,:,9]]
history = model.fit(train_input,y_train,batch_size=batchsize, epochs=n_epochs,validation_data=(test_input,y_test))

# =============================================================================
# EVALUATE

# Predict hourly consumption for test set
yhat_test = model.predict(test_input)

# Unscale data
def unscale(result,steps):
    if steps == 1: #single step
        if result.shape != (len(result),1): # if inputs
            temp = result.flatten() # (obs,)
            temp = np.vstack([temp,temp]) # (scaled features, obs). match to num of scaled features
            temp = scaler.inverse_transform(temp.T) # unscale
            temp = temp[:,0] # keep only the target feature column
            temp = temp.reshape(len(result),-1) # return to original shape
        else: # if outputs
            temp = np.hstack([result,result]) # match to num of scaled features
            temp = scaler.inverse_transform(temp)
            temp = temp[:,0]
    else: #multi step
        temp = result.flatten()
        temp = np.vstack([temp,temp]) # match to num of scaled features
        temp = scaler.inverse_transform(temp.T)
        temp = temp[:,0]
        temp = temp.reshape(-1,steps)
    return temp

yhat_unscaled = unscale(yhat_test, steps_ahead)
ytest_unscaled = unscale(y_test, steps_ahead)
xtest_unscaled = x_test.copy()
xtest_unscaled[:,:,8] = unscale(x_test[:,:,8], steps_ahead) # note that only target var is unscaled

# Find unscaled MSE per step ahead and average
diff = ytest_unscaled - yhat_unscaled
squared = diff ** 2
summed = np.sum(squared,axis=0)
mse_multi = summed / len(diff)
mse_avg = mse_multi.mean() # same as above when single step forecast

# Find unscaled MSE per ID
ytest_split = ytest_unscaled.reshape(nseries, -1)
yhat_split = yhat_unscaled.reshape(nseries, -1)
diff_split = ytest_split - yhat_split
squared_split = diff_split ** 2
summed_split = np.sum(squared_split,axis=1)
mse_split = summed_split / ytest_split.shape[1]

# MSE sanity checks
mse_check = mse_split.mean()
mse_all = squared.mean()

# Define plotting functions
def plot_train_history(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epoch_range = range(len(loss))
    plt.figure()
    plt.plot(epoch_range, loss, label='Training loss')
    plt.plot(epoch_range, val_loss, label='Validation loss')
    #plt.title(title)
    plt.ylabel('MSE (Scaled)')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()

def multiplot(past_obs, true_obs, predicted_obs):
    plt.figure()
    plt.plot(range(-lags,0), past_obs, label='Prior Steps')
    plt.plot(range(0,steps_ahead), true_obs,
             label='True Future')
    if predicted_obs.any():
        plt.plot(range(0,steps_ahead), predicted_obs,
                 label='Predicted Future')
    plt.ylabel('Grid Consumption (kWh)')
    plt.xlabel('Time (hr)')
    plt.legend(loc='upper left')
    plt.show()
    
def singlestep_plot(true_obs, predicted_obs):
    plt.figure(figsize=(10,5))
    plt.plot(range(0,len(true_obs)), true_obs, label='True Future')
    plt.plot(range(0,len(predicted_obs)), predicted_obs, label='Predicted Future')
    plt.ylabel('Grid Consumption (kWh)')
    plt.xlabel('Time (hr)')
    plt.legend(loc='upper left')
    plt.show()
    
def plot_embeddings(layer,categories,offset):
    u=model.layers[layer]
    weights = np.array(u.get_weights())
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(weights[0,:,0], weights[0,:,1], weights[0,:,2])
    labels = list(range(0+offset,categories+offset))
    for x, y, z, label in zip(weights[0,:,0], weights[0,:,1], weights[0,:,2], labels):
        ax.text(x, y, z, label)
    #for angle in range(0, 360):
        #ax.view_init(30, angle)
        #plt.draw()
        #plt.pause(.001)

# Plot train-validation loss
plot_train_history(history)

# Plot plots most appropriate for forecast horizon
plot_days = 7
if steps_ahead == 1:
    # First week of forecasts, by day
    for i in range(plot_days):
        singlestep_plot(ytest_unscaled[i*24:(i+1)*24].tolist(),yhat_unscaled[i*24:(i+1)*24].tolist()) 
    # Last week of forecasts
    for i in range(-plot_days,0):
        singlestep_plot(ytest_unscaled[i*24:(i+1)*24].tolist(),yhat_unscaled[i*24:(i+1)*24].tolist())
else:    
    # First week of forecasts
    for i in range(plot_days):
        multiplot(xtest_unscaled[i*steps_ahead,:,8],ytest_unscaled[i*steps_ahead,:],yhat_unscaled[i*steps_ahead,:])
    # Last week of forecasts
    for i in range(-plot_days,0):
        multiplot(xtest_unscaled[i*steps_ahead,:,8],ytest_unscaled[i*steps_ahead,:],yhat_unscaled[i*steps_ahead,:])

# Plot MSE of each household
plt.figure()
plt.bar(x=range(1,len(mse_split)+1),height=mse_split)
plt.xlabel('Household ID')
plt.ylabel('MSE $(kWh^2)$')
plt.show()

# Plot embeddings
# 3 = dataid, 4 = hour, 5 = DoW
plot_embeddings(3,nseries,1)
plot_embeddings(4,24,0)
plot_embeddings(5,7,0)

# Save model
#if steps_ahead==1:
#    model.save('YOUR_DIRECTORY')
#else:
#    model.save('YOUR_DIRECTORY') 
