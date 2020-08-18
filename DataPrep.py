# -*- coding: utf-8 -*-
# Author: Joey G.
# Runs in Spyder 4.1. using Python 3.7.

import time
import os
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Confirm working directory
print("Working Directory: ", os.getcwd())

start_time = time.time()

# Load household energy use (Source: Pecan Street, Limited University Access)
# 15 minute resolution is aggregated into hourly resolution to match resolution
# of weather datasets. Second & minute resolution is also available.
df_TX = pd.read_csv('YOUR_DIRECTORY')
df_CA = pd.read_csv('YOUR_DIRECTORY')
# drop 9836 for missing a full week of data

# Load hourly-resolution weather (Source: https://climate.usu.edu/)
df_weather_TX = pd.read_csv('YOUR_DIRECTORY')
df_weather_CA = pd.read_csv('YOUR_DIRECTORY')

# =============================================================================
# CLEAN METADATA
    
# Load metadata
df_meta = pd.read_csv('YOUR_DIRECTORY')

# Retrieve IDs of available houses
ID_CA = df_CA.drop_duplicates(subset='dataid')
ID_TX = df_TX.drop_duplicates(subset='dataid')
df_ID = ID_CA.append(ID_TX)
df_ID = df_ID['dataid']

# Keep only necessary metadata
df_meta = df_meta[df_meta.dataid.isin(df_ID)]
keep_meta = ['dataid','building_type','state','amount_of_south_facing_pv',
             'amount_of_west_facing_pv','amount_of_east_facing_pv']
df_meta = df_meta[keep_meta]
avail_meta = df_meta.count()

# =============================================================================
# CLEAN WEATHER

def clean_weather(df_weather, drop_cols):
    df_out = df_weather.drop(drop_cols,axis = 1)
    # Convert str date to datetime
    df_out['Day'] = df_out['Day'].str[:-4] # remove timezone
    df_out['Day'] = pd.to_datetime(df_out['Day'], format = '%Y-%m-%d %I:%M %p')
    df_out['Day'] = df_out['Day'].dt.floor('h') # round down minutes
    df_out = df_out.replace('S', np.nan) # source uses S = missing
    num_col = ['Wind Speed','Visibility',
               'Temperature','Dew Point']
    df_out[num_col] = df_out[num_col].apply(pd.to_numeric)
    df_out.rename(columns={'Day':'Time'}, inplace=True)
    # Resample to hourly-resolution by taking mean of observations
    df_out = df_out.resample('H',on='Time').mean().reset_index()
    # Fill NaN with value from previous day, by row
    NaN_bool = df_out.isnull().any(axis=1)
    NaN_idx = df_out[NaN_bool].index.tolist()
    for i in range(len(NaN_idx)):
        df_out.iloc[NaN_idx[i]] = df_out.iloc[NaN_idx[i]].fillna(value=df_out.iloc[NaN_idx[i]-24])
    return df_out
drop_cols = ['Wind Gust', 'Wind Direction','Precipitation Type', 'Altimeter','Precipitation']
df_weather_TX = clean_weather(df_weather_TX,drop_cols)
df_weather_CA = clean_weather(df_weather_CA,drop_cols)

# =============================================================================
# CONSTRUCT DATASETS

# FOR GRID MODEL:
# Features: dataid, building_type, time, day, month, weather, state
# Out: grid_fixed.
# For Checking: city, datetime.

# FOR PV MODEL:
# Features: dataid, amount_of_south_facing_pv, amount_of_west_facing_pv, 
#           amount_of_east_facing_pv, time, month, weather, solar_fixed, state
# Out: solar_fixed
# For Checking: city, datetime.
# Only TX homes available.

# Cleans consumption data and adds time features
def add_time(df, keep_cols):
    # Reformat data and resample to hourly resolution
    df_out = df[keep_cols]
    df_out = df_out.fillna(0)
    df_out['local_15min'] = df_out['local_15min'].str[:-3] # remove timezone adjustment
    df_out['local_15min'] = pd.to_datetime(df_out['local_15min'],
              format = '%Y-%m-%d %H:%M:%S')
    df_out = df_out.set_index('local_15min').groupby('dataid').resample('H').sum().drop(['dataid'], 1).reset_index()    
    df_out.rename(columns={'local_15min':'Time'}, inplace=True) # above resets name for some reason?        
    return df_out

keep_cols = ['dataid', 'local_15min', 'grid', 'solar','solar2','car1','drye1','dryg1']
df_TX_time = add_time(df_TX, keep_cols)
df_CA_time = add_time(df_CA, keep_cols)

# Append weather vars based on date
df_TX_time_weather = pd.merge(df_TX_time,df_weather_TX,on='Time')
df_TX_time_weather = df_TX_time_weather.sort_values(['dataid','Time'])
df_TX_time_weather.reset_index(drop=True, inplace=True)
df_CA_time_weather = pd.merge(df_CA_time,df_weather_CA,on='Time')
df_CA_time_weather = df_CA_time_weather.sort_values(['dataid','Time'])
df_CA_time_weather.reset_index(drop=True, inplace=True)

# Fill missing timesteps
# Warning: The following is just a prototype, does not work for all cases
def fix_steps(df,parallel,missing,leap):
    df_out = df.copy()
    if parallel == True: # Parallel: All time series are in same timespan
        if missing == True: # Missing: There are missing timesteps
            t1 = df_out['Time'].min()
            t2 = df_out['Time'].max()
            expected_range = pd.date_range(start=t1,end=t2,freq='H')
            df_out = df_out.groupby('dataid').apply(lambda x : x.set_index('Time').reindex(expected_range)).drop('dataid', 1).reset_index()
            df_out.rename(columns={'level_1':'Time'},inplace=True)
    if leap == True: # Leap: There is a leap day that needs to be taken out
        remove_range = pd.date_range(start='2016-02-29',periods=24,freq='H') # only leap yr in data
        indices = df_out[df_out['Time'].isin(remove_range)].index.tolist() 
        df_out.drop(indices,inplace=True)
    # Fill NaN using value of previous day
    df_out = df_out.fillna(value=df_out.shift(periods=24))
    return df_out

df_TX_time_weather = fix_steps(df_TX_time_weather, parallel = True, missing = True, leap = False)
df_CA_time_weather = fix_steps(df_CA_time_weather, parallel = False, missing = False, leap = True)

# =============================================================================
# PREPARE DATASETS FOR NEURAL NETWORK

def transform_data(df, drop_cols):    
    df_out = df.copy()
    # Convert to kWh
    for i in ['grid','solar','solar2','car1','drye1','dryg1']:
        df_out[i] = df_out[i]/4
    # Grid is net consumption. Negate effects of PV. Remove EV charging
    df_out['grid_fixed'] = df_out['grid'] + df_out['solar'] + df_out['solar2'] - df_out['car1']
    df_out['solar_fixed'] = df_out['solar'] + df_out['solar2']
    # Create hour, day, month columns from timestamp
    df_out['Weekday'] = df_out['Time'].dt.dayofweek # 0 = Monday
    df_out['Month'] = df_out['Time'].dt.month
    df_out['Hour'] = df_out['Time'].dt.hour         
    # sin/cos transform time
    df_out['Hour_sin'] = np.sin(2 * np.pi * df_out['Hour']/24.0)
    df_out['Hour_cos'] = np.cos(2 * np.pi * df_out['Hour']/24.0)
    df_out['DoW_sin'] = np.sin(2 * np.pi * df_out['Weekday']/7.0)
    df_out['DoW_cos'] = np.cos(2 * np.pi * df_out['Weekday']/7.0)   
    # Drop unnecessary columns
    df_out = df_out.drop(drop_cols,axis=1) 
    return df_out

drop_cols = ['grid','solar','solar2','car1']
df_TX_time_weather = transform_data(df_TX_time_weather, drop_cols)
df_CA_time_weather = transform_data(df_CA_time_weather, drop_cols)

# Add meta data features corresponding to dataid
df_TX_complete = pd.merge(df_TX_time_weather,df_meta,on='dataid')
df_CA_complete = pd.merge(df_CA_time_weather,df_meta,on='dataid')
# Fill PV NaNs
pv_labels = ['amount_of_south_facing_pv','amount_of_east_facing_pv','amount_of_west_facing_pv']
df_TX_complete[pv_labels] = df_TX_complete[pv_labels].fillna(value=0)  
df_CA_complete[pv_labels] = df_CA_complete[pv_labels].fillna(value=0)  
             
# Combined dataset with state and btype one-hot encoded
df_all = pd.concat([df_TX_complete,df_CA_complete])
dums = pd.get_dummies(df_all[['building_type','state']])
df_all = pd.concat([df_all,dums],axis=1).drop(['building_type','state'],axis=1)

# Generate plots for continuous vars for all households to check inconsistencies
#df_elec_all[['dataid','grid_fixed','solar_fixed','Temperature']].groupby('dataid').plot(subplots=True)
# Remove unusable IDs
remove_ID = [203,1450,1524,2606,3864,3938,4934,7114,6547,9278,9922,4767,2361,6139]
df_elec_all = df_all[~df_all['dataid'].isin(remove_ID)]
keep_ID = df_meta[df_meta['amount_of_south_facing_pv'].isna() == False].dataid
df_solar_all = df_all[df_all['dataid'].isin(keep_ID)]
df_solar_all = df_solar_all[df_solar_all['dataid']!=2335] # week long gap

t_total = time.time()-start_time

# =============================================================================
# EXPORTING

# Export datasets
df_elec_all.to_csv('YOUR_DIRECTORY', index=False,header=True) # 34 IDs
df_solar_all.to_csv('YOUR_DIRECTORY', index=False,header=True) # 14 IDs, TX Single Home only

# Scatter matrix
df_plot = df_all[['grid_fixed','solar_fixed','Temperature','Hour','Month']]
plt.figure()
sns.pairplot(df_plot)
plt.show()

# Correlation matrix
df_plot = df_all[['grid_fixed','solar_fixed','Temperature','Wind Speed','Dew Point','Visibility']]
plt.figure()
sns.heatmap(df_plot.corr(), annot = True, cmap='coolwarm')
plt.show()

# All continuous features in stacked plot, for an example ID
df_plot = df_all[df_all['dataid']==661]
df_plot = df_plot[['grid_fixed','solar_fixed','Temperature','Wind Speed', 'Dew Point', 'Visibility']]
plt.figure()
df_plot.plot(figsize=(10,10) ,subplots=True)
plt.xlabel('Time (hr)')
plt.show()

# Autocorrelations
plt.figure()
plt.acorr(df_elec_all['grid_fixed'], maxlags=24)
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.title('Grid Consumption Autocorrelation')
plt.show()
plt.figure()
plt.acorr(df_solar_all['solar_fixed'], maxlags=24)
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.title('Solar PV Autocorrelation')
plt.show()
# add for dryer

def plot_violin(df,group,dependent,x_label, y_label, divide, category):
    plt.figure(figsize=(10,5))
    sns.set(style='white')
    if divide == True:
        sns.violinplot(x = group, y = dependent, data = df, hue = category, split=True,palette='muted')
    else:
        sns.violinplot(x=group, y=dependent, data=df, palette='muted')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()  
    
# Monthly grid violin plot grouped by state, separate plots by building type
# Apartments
df_plot = df_elec_all[df_elec_all['building_type_Apartment']==1]
plot_violin(df_plot,'Month','grid_fixed','Month','Grid Consumption (kWh)', False, None)
# Single Home
df_plot = df_elec_all[df_elec_all['building_type_Single-Family Home 001 (Master)']==1]
plot_violin(df_plot,'Month','grid_fixed','Month','Grid Consumption (kWh)', False, None)
# Town Home
df_plot = df_elec_all[df_elec_all['building_type_Town Home']==1]
plot_violin(df_plot,'Month','grid_fixed','Month','Grid Consumption (kWh)', False, None)

# Monthly solar violin plot (14 TX)
plot_violin(df_solar_all,'Month','solar_fixed','Month','Solar PV Generation (kWh)', False, None)

# Plot example of sin/cos transform of hour
hrs = pd.DataFrame(range(0,24))
hrs_sin = np.sin(2 * np.pi * hrs/24).values
hrs_cos = np.cos(2 * np.pi * hrs/24).values
fig, ax = plt.subplots()
ax.scatter(hrs_sin, hrs_cos)
for i in range(0,24):
    ax.annotate(i, (hrs_sin[i], hrs_cos[i]))
