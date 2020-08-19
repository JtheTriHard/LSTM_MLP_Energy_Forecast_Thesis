# -*- coding: utf-8 -*-
# Author: Joey G.
# Runs in Spyder 4.1. using Python 3.7

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Confirm working directory
print("Working Directory: ", os.getcwd())

# =============================================================================
# PREPARE DATA

df = pd.read_csv('YOUR_DIRECTORY')

# Create persistance forecast within each household ID
df_in = df[['dataid','grid_fixed']]
groups = df_in.groupby('dataid')
df_in['t-1'] = pd.DataFrame(groups['grid_fixed'].shift(1))
df_in.rename(columns={'grid_fixed':'t'}, inplace=True)
df_in = df_in[['t-1','t']]
df_in = df_in.dropna()

# Separate into training/test sets
nseries = df['dataid'].nunique() # unique IDs in dataset
ID_length = int(len(df_in)/nseries)
df_train, df_test = pd.DataFrame([]), pd.DataFrame([])
for i in range(nseries):
    start = i*ID_length
    end = (i+1)*ID_length
    split = start + int(ID_length*0.9)
    df_curTrain = df_in.iloc[start:split]
    df_curTest = df_in.iloc[split:end]
    df_train = pd.concat([df_train,df_curTrain],axis=0)
    df_test = pd.concat([df_test,df_curTest],axis=0)

# =============================================================================
# RUN FOR 1 STEP AHEAD

# Get predictions
ylabels = df_test['t'].values
yhat = df_test['t-1'].values

# Plot example forecast
def singlestep_plot(true_obs, predicted_obs):
    plt.plot(range(0,len(true_obs)), true_obs, label='True Future')
    plt.plot(range(0,len(predicted_obs)), predicted_obs, label='Predicted Future')
    plt.legend(loc='upper left')
    plt.ylabel('Grid Consumption (kWh)')
    plt.xlabel('Time (hr)')
    plt.show()
singlestep_plot(ylabels[0:24],yhat[0:24]) # plot first day as check

# Overall MSE
diff = ylabels - yhat
squared = diff ** 2
summed = np.sum(squared)
mse_single = summed / len(diff)

# MSE separated by ID:
squared_IDs = squared.reshape(nseries,-1)
summed_IDs = np.sum(squared_IDs,axis=1)
mse_IDs = summed_IDs / (squared_IDs.shape[1])

# =============================================================================
# RUN FOR 24 STEPS AHEAD

# Transform into LSTM shape. Will have (lags + steps_ahead) LESS samples than input
lags = 24 # past obs to use
steps_ahead = 24 # period to forecast

# rows: 0>23 for past, 24>47 for future
# columns: j*829 > (j+1)*829
seq = df_test['t'].values
for j in range(nseries):
    cur = seq[j*876:(j+1)*876] # select obs of current ID
    store = cur.copy() # end result for current ID
    for i in range(lags+steps_ahead-1):
        temp = np.roll(cur,shift=i+1)
        store = np.vstack((temp, store))
    store = store.T
    store = store[47:,:]
    if j==0:
        df_multi = store
    else:
        df_multi = np.vstack((df_multi,store))
yhat_multi = df_multi[:,0:24]
ylabels_multi = df_multi[:,24:]

# Plot example forecast
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
multiplot(df_test.values[0:24,1],ylabels_multi[0,:],yhat_multi[0,:])

# Overall MSE
diff_multi = ylabels_multi - yhat_multi
squared_multi = diff_multi ** 2
summed_multi = np.sum(squared_multi,axis=0)
mse_multi = summed_multi / len(diff_multi)

# MSE separated by ID:
squared_split = squared_multi.reshape(nseries,-1)
summed_split = np.sum(squared_split,axis=1)
mse_split = summed_split / squared_split.shape[1]

# Sanity checks
mse_avg = squared_multi.mean()
mse_check = mse_split.mean()
