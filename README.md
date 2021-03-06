# LSTM_MLP_Energy_Forecast_Thesis

## Overview
Contains the majority of the code written when completing a thesis for the MScBA: Master's in Management degree at the Rotterdam School of Management, Erasmus University Rotterdam (RSM EUR). The topic was inspired by the Municipality of Rotterdam (Gemeente Rotterdam) to contribute to their research in the implications of recent EU energy transition initiatives. Datasets were obtained from Pecan Street Inc. (limited university access) and the Utah State University Climate Center (open-source).

The study developed "collective" multivariate-input LSTM and MLP neural networks to forecast the electricity consumption and solar PV generation of multiple households at once. The 1-hour and 24-hour ahead forecast horizons were studied using lag observations containing the past 24 hours. Features included household characteristics and weather variables. Entity embedding was done for the ordinal and high-cardinality features (aka the household ID) to reduce sparsity and investigate whether the models would be able to learn relationships. Prediction intervals were built incorporating uncertainty due to the model, the inputs, and inherent noise. The inherent noise was estimated using the residuals of a held-out set, and the model/input uncertainty was estimating by conducting Monte Carlo simulations in which noise distributions were added to the weather features to simulate forecast errors and dropout was activated at the prediction stage to simulate model misspecification.

## Results

I find that LSTMs outperform MLPs for all forecast horizons and target variables, and that both neural networks outperform a naive persistence benchmark. Furthermore, both models are able to recognize the cyclic natural of the time variables (hour, day of week), with the LSTM producing more defined patterns. The mapping of the household ID embeddings were not interpretable. The models perform better for predicting solar PV generation than electricity consumption, but the opposite is true when forecasting 24 hours ahead. The constructed prediction intervals successfully achieve the desired 90% average coverage, but fall short for individual households with volatile time series. However, the widths of the intervals tend to be too large to be informative if deployed. This was found to be due to the standard error due to noise being significantly larger than due to the model/input uncertainty, suggesting that the models can be further improved to lessen the residuals used to calculate the noise and also that the predictions may not be sensitivity to small errors in weather inputs. I am confident that the performance could have been further improved given longer time-series. Only a year of data was provided for each household, which resulted in the final month of the year being used for testing. Thus, the month could not be incorporated as a feature, and the neural networks had to predict on a period of the year it had not yet encountered. Nevertheless, the results are positive. The full thesis document can be provided upon request.

Some following improvements and extensions are recommended: differencing, ARIMA benchmark, automated hyperparameter tuning, automated feature extraction, more training data.

## Contents

DataPrep.py - The code used to clean and combine the datasets provided by Pecan Street and USU Climate Center, along with some plots for exploratory data analysis.

LSTM-G.py - TensorFlow LSTM for predicting grid consumption.

LSTM-S.py - TensorFlow LSTM for predicting solar PV generation.

MLP-G.py - TensorFlow MLP for predicting grid consumption.

MLP-S.py - TensorFlow MLP for predicting solar PV generation.

Because the LSTMs were found to perform best, the prediction intervals built using Monte Carlo simulations were only done for the LSTM models, and can be found in the final section of the files.
