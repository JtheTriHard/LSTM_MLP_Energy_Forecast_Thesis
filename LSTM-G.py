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

#np.random.seed(123)
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
# Only Temperature was found to be a weather predictor of grid consumption
new_order = ['dataid','Hour','Weekday','building_type_Apartment',
             'building_type_Single-Family Home', 'building_type_Town Home',
             'state_California','state_Texas', 'grid_fixed','Temperature']
df = df[new_order]

# Remap IDs to consecutive values, to be embedded later
df['dataid'], remap = pd.factorize(df['dataid'])
remap = remap.to_list()
# Reserve ID = 0 for new households, shift IDs accordingly
df['dataid'] = df['dataid']+1
remap.insert(0,'Unknown ID') #index = new ID, entry = original ID

# Adjust Month 1-12 to 0-11
#df['Month']=df['Month']-1

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
scale_cols = ['grid_fixed','Temperature']
scaler = MinMaxScaler()
df_train[scale_cols] = scaler.fit_transform(df_train[scale_cols])
df_test[scale_cols] = scaler.transform(df_test[scale_cols])

# Transform into LSTM shape. Will have (lags + steps_ahead) LESS samples than input
# Modified from TensorFlow.org
lags = 24 # past obs to use
steps_ahead = 1 # period to forecast
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
            'grid_fixed','Temperature']

# Parameters
n_epochs = 500
batchsize = 16384
embed_dim = 3
n_num_feats = len(num_cols) # num features that are not embedded

tf.keras.backend.clear_session()
# Define embedded features input
inputs, embeddings = [], []
for i in cat_cols:
    cat_input = tf.keras.layers.Input(shape=(lags,), name="".join([i.replace(" ", ""),"_inp"]))
    cat_dim  = df[i].nunique()
    if i == 'dataid':
        cat_dim = cat_dim + 1 # +2 when holding out entire ID
    inputs.append(cat_input)
    embeddings.append(tf.keras.layers.Embedding(cat_dim, embed_dim, input_length = lags,
            name="".join([i.replace(" ", ""),"_embed"]))(cat_input))
    
# Define numeric features input    
num_input = tf.keras.layers.Input(shape=(lags,n_num_feats), name="num_input")
inputs.append(num_input)
embeddings.append(num_input)
    
# LSTM using Keras Functional API
# training = True due to approach for model uncertainty estimation
input_layer = tf.keras.layers.Concatenate(name="concat")(embeddings)
new_layer= tf.keras.layers.LSTM(64,activation='tanh',return_sequences=True,name='LSTM_1')(input_layer)
new_layer = tf.keras.layers.Dropout(0.1,name='Dropout_1')(new_layer,training=True)
new_layer= tf.keras.layers.LSTM(64,activation='tanh',name='LSTM_2')(new_layer)
new_layer = tf.keras.layers.Dropout(0.1,name='Dropout_2')(new_layer,training=True)
#new_layer= tf.keras.layers.LSTM(64,activation='tanh',name='LSTM_3')(new_layer)
#new_layer = tf.keras.layers.Dropout(0.1,name='Dropout_3')(new_layer)
#new_layer= tf.keras.layers.LSTM(32,activation='tanh',name='LSTM_4')(new_layer)
output_layer = tf.keras.layers.Dense(steps_ahead,name='output_layer')(new_layer)
model = tf.keras.Model(inputs, output_layer, name = "LSTM-C-G")
model.compile(optimizer='adam',loss='mse')
model.summary()
tf.keras.utils.plot_model(model, 'YOUR_DIRECTORY', show_shapes=True)
history = model.fit([x_train[:,:,0],x_train[:,:,1],x_train[:,:,2],
                     x_train[:,:,3:]],y_train,batch_size=batchsize, epochs=n_epochs,
    validation_data=([x_test[:,:,0],x_test[:,:,1],x_test[:,:,2], x_test[:,:,3:]],y_test))

# =============================================================================
# EVALUATE

# Predict hourly consumption for test set
yhat_test = model.predict([x_test[:,:,0], x_test[:,:,1], x_test[:,:,2], x_test[:,:,3:]])

# Unscale data
def unscale(result,steps):
    if steps == 1: #single step
        if result.shape != (len(result),1): # if inputs
            temp = result.flatten() # (obs,)
            temp = np.vstack([temp,temp]) # (2, obs). Change if scaler was fit on more than 2 features
            temp = scaler.inverse_transform(temp.T) # unscale
            temp = temp[:,0] # keep only the target feature column
            temp = temp.reshape(len(result),-1) # return to original shape
        else: # if outputs
            temp = np.hstack([result,result]) # change based on scaler fit
            temp = scaler.inverse_transform(temp)
            temp = temp[:,0]
    else: #multi step
        temp = result.flatten()
        temp = np.vstack([temp,temp]) # change based on scaler fit
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

# Sanity checks
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
    for angle in range(0, 360):
        ax.view_init(30, angle)
        plt.draw()
        plt.pause(.001)

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
    # Plots all prediction. Remember that these are the final months of all IDs at once
    singlestep_plot(ytest_unscaled,yhat_unscaled)
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
#model.save('YOUR_DIRECTORY')

# =============================================================================
# UNCERTAINTY ESTIMATION

t = time.time()
# Load fitted model using dropout at training
model = tf.keras.models.load_model('YOUR_DIRECTORY')

# Split held-out dataset into validation for noise term, test for model uncertainty term
# Validation set: first 80% of held out set for each ID
# Test set: last 20% of held out set for each ID
df_held = df_test.copy().reset_index(drop=True)
df_v, df_t = pd.DataFrame([]), pd.DataFrame([])
for i in range(nseries):
    start = i*test_size
    end = (i+1)*test_size
    split = start + int(test_size*0.8)
    df_curV = df_held.iloc[start:split]
    df_curT = df_held.iloc[split:end]
    df_v = pd.concat([df_v,df_curV],axis=0)
    df_t = pd.concat([df_t,df_curT],axis=0)

# Reshape dataframes into inputs
x_v, y_v = [],[]
length_v = int(len(df_v)/nseries)
for i in range(nseries):
    df_temp = df_v.iloc[i*length_v:(i+1)*length_v].reset_index(drop=True)
    x_temp, y_temp = multivariate_data(df_temp,df_temp['grid_fixed'],0,None,lags,steps_ahead,single_step=False)
    x_v.append(x_temp)
    y_v.append(y_temp)
x_v, y_v = np.concatenate(x_v).astype('float32'),np.concatenate(y_v).astype('float32')

# Predict for validation set
in_v = [x_v[:,:,0],x_v[:,:,1],x_v[:,:,2],x_v[:,:,3:]]
yhat_v = model.predict(in_v)
# Unscale
yhat_v = unscale(yhat_v, steps_ahead)
y_v = unscale(y_v, steps_ahead)

# Calculate noise term per ID using validation test
noise = y_v - yhat_v
noise = noise**2
length_noise = int(len(noise)/nseries)
for i in range(nseries):
    cur_noise = noise[i*length_noise:(i+1)*length_noise]
    noise[i*length_noise:(i+1)*length_noise] = np.sum(cur_noise)/length_noise
noise, ind = np.unique(noise,return_index=True)
noise = noise[np.argsort(ind)]

# Run Monte Carlo simulations with dropout and input uncertainty
# Done for all samples in test set at once
samples = len(df_t)
iterations = 1000
mc_forecasts = []
length_t = int(len(df_t)/nseries)
for j in range(iterations):
    # Create a dataframe with forecast errors
    df_dist = pd.DataFrame(list(zip(df_t['grid_fixed'],
        np.random.normal(0,2,samples) # temperature
        #np.random.normal(0,1,samples), # wind speed
        #np.random.normal(0,2,samples), # dew point
        #np.random.normal(0,1,samples)
        ))) # visibility
    df_dist = scaler.transform(df_dist) # converted to array
    # Add simulated forecast errors to original data
    df_mc=df_t.copy().reset_index(drop=True)
    df_mc['Temperature'] = df_mc['Temperature'] + df_dist[:,1]
    #df_mc['Wind Speed'] = df_mc['Wind Speed'] + df_dist[:,2]
    #df_mc['Dew Point'] = df_mc['Dew Point'] + df_dist[:,3]
    #df_mc['Visibility'] = df_mc['Visibility'] + df_dist[:,4]
    # Re-construct test dataset in each iteration
    x_mc, y_mc = [],[]
    for i in range(nseries):
        df_temp = df_mc.iloc[i*length_t:(i+1)*length_t].reset_index(drop=True)
        x_temp, y_temp = multivariate_data(df_temp,df_temp['grid_fixed'],0,None,lags,steps_ahead,single_step=False)
        x_mc.append(x_temp)
        y_mc.append(y_temp)
    x_mc, y_mc = np.concatenate(x_mc).astype('float32'),np.concatenate(y_mc).astype('float32')
    in_mc = [x_mc[:,:,0],x_mc[:,:,1],x_mc[:,:,2],x_mc[:,:,3:]]
    # Forecast
    yhat_mc = model.predict(in_mc)
    # Unscale 
    yhat_mc = unscale(yhat_mc, steps_ahead)
    y_mc = unscale(y_mc, steps_ahead)
    # Store results to use as a distribution later
    mc_forecasts.append(yhat_mc)

# Calculate model uncertainty term per step forecast
mc_forecasts = np.array(mc_forecasts) # (iterations, forecasts)
model_uncertainty = np.sum((mc_forecasts - mc_forecasts.mean(axis=0))**2,axis=0)/iterations

# Forecast using original model without dropout, using only test samples
x_original, y_original = [],[]
for i in range(nseries):
    df_temp = df_t.iloc[i*length_t:(i+1)*length_t].reset_index(drop=True)
    x_temp, y_temp = multivariate_data(df_temp,df_temp['grid_fixed'],0,None,lags,steps_ahead,single_step=False)
    x_original.append(x_temp)
    y_original.append(y_temp)
x_original, y_original = np.concatenate(x_original).astype('float32'),np.concatenate(y_original).astype('float32')
in_original = [x_original[:,:,0],x_original[:,:,1],x_original[:,:,2],x_original[:,:,3:]]
old_model = tf.keras.models.load_model('YOUR_DIRECTORY')
predictions = old_model.predict(in_original)
# Unscale
predictions = unscale(predictions, steps_ahead)
y_original = unscale(y_original, steps_ahead)

# Combine uncertainty terms
PI_length = int(len(predictions)/nseries)
standard_error = []
for i in range(nseries):
    cur_obs = model_uncertainty[i*PI_length:(i+1)*PI_length]
    standard_error.append(cur_obs + noise[i])
standard_error = np.sqrt(np.array(standard_error)).flatten()
# Predictor prediction intervals
zscore = 1.645
term = zscore * standard_error
PI_upper = predictions + term
PI_lower = predictions - term

# Calculate coverage across all predictions
PI_check = (y_original <= PI_upper) & (y_original >= PI_lower)
coverage = np.count_nonzero(PI_check)
coverage = coverage/len(PI_check) 
# Calculate coverage within each ID
coverage_IDs = []
for i in range(nseries):
    cur_PI = PI_check[i*PI_length:(i+1)*PI_length]
    cur_coverage = np.count_nonzero(cur_PI)
    cur_coverage = cur_coverage/PI_length
    coverage_IDs.append(cur_coverage)
coverage_check = sum(coverage_IDs)/nseries # sanity check
plt.figure()
plt.bar(x=range(1,len(coverage_IDs)+1),height=coverage_IDs)
plt.xlabel('Household ID')
plt.ylabel('Proportion Covered By Prediction Interval')
plt.axhline(y=0.9,linewidth=1,color='r')
plt.show()

# Plot forecasts and prediction intervals
def interval_plot(true_obs, predicted_obs, interval):
    plt.figure(figsize=(10,5))
    plt.plot(range(0,len(true_obs)), true_obs, label='True Future')
    plt.plot(range(0,len(predicted_obs)), predicted_obs, label='Predicted Future')
    plt.fill_between(range(0,len(predicted_obs)), predicted_obs - interval, predicted_obs + interval, alpha=0.2)
    plt.ylabel('Grid Consumption (kWh)')
    plt.xlabel('Time (hr)')
    plt.legend(loc='upper left')
    plt.show()

t_mc = time.time() - t

# =============================================================================
# OLD CODE

# Difference time series
#diff_cols = ['grid_fixed','Temperature','Wind Speed'] # features to be differenced
#groups = df.groupby('dataid')
#df[diff_cols] = groups[diff_cols].diff()
#df[diff_cols] = df[diff_cols].fillna(0) # fill newly created NaNs

# Check stationarity: stationary after first-order differencing
#from statsmodels.tsa.stattools import adfuller
#X = df_test['grid_fixed'].values
#result = adfuller(X,autolag=None, maxlag=336) #had to limit to two week lag, otherwise MemoryError
#print('ADF Statistic: %f' % result[0])
#print('p-value: %f' % result[1])
#print('Critical Values:')
#for key, value in result[4].items():
#	print('\t%s: %.3f' % (key, value))
