# Survey-of-M4-competition

This repository is to study the datset of [M4 competition](https://github.com/Mcompetitions/M4-methods/tree/master/Dataset). There are 100000 time-series where they belong to the different periods ( Hourly, Daily, Weekly, Monthly, Quarterly, Yearly). We adopt the different deep-learning models to perform the predictions and compare their performance.

## Methods:
- CNN1D
- CNN2D
- LSTM
- BiLSTM
- CNNLSTM
- CNNBiLSTM
- Transformer

## Metrics: 
- SMAPE
- MASE
- OWA
- MAE
- RMSE

## Result summary: 
In this study, the models give the best performance for the time-series with the hourly period. However, the training time for each deep learning models is comparatively long.

## Usage:
Run the command: “python train.py” with the following arguments:
--model (CNN1D, CNN2D, LSTM, BiLSTM, CNNLSTM, CNNBiLSTM, Transformer)
--period (Hourly, Daily, Weekly, Monthly, Quarterly, Yearly)
--device (cpu or cuda)

In addition, the model parameters can be modified by modifying the config, ModelParameters.ini.



