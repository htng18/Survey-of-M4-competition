import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import entropy
from statsmodels.tsa.seasonal import seasonal_decompose


def smape(test, pred):
    '''
      Compute SMAPE
    '''
    return 100/len(test) * np.sum(2 * np.abs(test - pred) / (np.abs(test) + np.abs(pred)))

def mase(test, pred, insample, period):
    '''
      Compute MASE
    '''
    periodicity = {"Yearly":1, "Quarterly":4, "Monthly":12, "Weekly":1, "Daily":1, "Hourly":24}
    m = periodicity[period]
    d = np.sum(np.abs(insample[:-m]-np.roll(insample, -m)[:-m]))/(len(insample[:-m]))
    return np.sum(np.abs(test - pred))/d/len(test)

def compute_metrics(net, scale, inputdata, test, insample, target, params, period, device):
    '''
      Compute all metrics (SMAPE, MASE, OWA, MAE and RMSE)
    '''
    with torch.no_grad():
        net.eval()
        if params["model"]=="Transformer":
            x_batch = torch.tensor(inputdata,dtype=torch.float32).permute(1,0,2).to(device)
        else:
            x_batch = torch.tensor(inputdata,dtype=torch.float32).to(device)
        if params["model"] in ["CNN1D", "CNN2D"]:
                output = net(x_batch)
        elif params["model"] in ["LSTM", "BiLSTM", "CNNLSTM", "CNNBiLSTM"]:
            output = net(x_batch)[0]
        elif params["model"]=="Transformer":
            x_batch = torch.tensor(inputdata,dtype=torch.float32).permute(1,0,2).to(device)
            X_mask = generation_mask(test.shape[0]).to(device)
            output = net(x_batch, X_mask)

    if params["model"]=="CNN2D":
        pred = scale[target].inverse_transform(output.cpu().detach().numpy()[:,-1,-1,:].reshape(-1,1)).flatten()
    else:
        pred = scale[target].inverse_transform(output.cpu().detach().numpy()[:,-1,:].reshape(-1,1)).flatten()

    SMAPE = smape(test, pred)
    MASE = mase(test, pred, insample, period)
    # Naive forcast with the previous 2 steps
    navie2 = np.concatenate([insample[-2:], test[:-2]])
    # Compute OWA
    if smape(test, navie2)==0 or mase(test, navie2, insample, period)==0:
        OWA = (SMAPE + MASE)/2
    else:
        OWA = (SMAPE/smape(test, navie2) + MASE/mase(test, navie2, insample, period))/2
    MAE = mean_absolute_error(test, pred)
    RMSE = mean_squared_error(test, pred, squared=False)
    return SMAPE, MASE, OWA, MAE, RMSE

class FeatureEngineering(object):
    def __init__(self, target, degree=2, lag=2, num_lag=10, window=3, num_window=5):
        self.target = target
        self.degree = degree
        self.lag = lag
        self.num_lag = num_lag
        self.window = window
        self.num_window = num_window
        
        
    def polytime(self, data):
        '''
        Add the polynomials of features such as x, x**2 and x**3, etc.

        Args:
          data (pandas DataFrame): the input DataFrame with the different features
          target (str): The column name of the time-series forecasting.
          degree (int): The degree of polynomial to be added for the features.

        Returns:
          pandas DataFrame: The dataframe with the added polynomials of features.

        '''
        for i in range(2,self.degree+1):
            x = [j**i for j in data[self.target]]
            data['x'+str(i)] = x
        # Arrange the "target" column to the last column
        column = list(data.columns)
        column.remove(self.target)
        column = column + [self.target]
        data = data[column]
        return data
    
    def lag_generation(self, data):
        '''
        Add the time-lagged features of the time series with column name, "target"

        Args:
          data (pandas DataFrame): the input DataFrame with the different features
          target (str): The column name of the time-series forecasting.
          lag (int): The time unit of lags for the time-lagged features.
          num_lag (int): The number of time-lag features to be added.

        Returns:
          pandas DataFrame: The dataframe with the added time-lagged features.

        '''
        lag = [i for i in range(self.lag, (self.num_lag+1)*self.lag, self.lag)]
        for i in lag:
            data['lag'+str(i)] = data[self.target].shift(i).fillna(0.0)
        column = list(data.columns)
        column.remove(self.target)
        column = column + [self.target]
        data = data[column]
        return data
    
    def seasonal_decomposition(self, data, period):
        data['seasonal'+str(period)] = seasonal_decompose(data[self.target], period=period).seasonal.fillna(0.0)
        data['resid'+str(period)] = seasonal_decompose(data[self.target], period=period).resid.fillna(0.0)
        return data
    
    def rollingwindow_generation(self, data, metrics='mean'):
        '''
        Add the moving-average features of the time series with column name, 
        "target" with the different window size.

        Args:
          data (pandas Dataframe): the input DataFrame with the different features.
          target (str): The column name of the time-series forecasting.
          window (int): The window size of moving average.
          num_window (int): the number of moving-average features to be added, where 
                            the window size is multiple of window.

        Returns:
          pandas DataFrame: The dataframe with the added moving-average features.

        '''
        rollingwindow = [i for i in range(self.window, (self.num_window+1)*self.window, self.window)]
        if metrics=='mean':
            for i in rollingwindow:
                data['rollingwindow_'+metrics+'_'+str(i)] = data[self.target].rolling(i).mean().fillna(0.0)
        elif metrics=='std':
            for i in rollingwindow:
                data['rollingwindow_'+metrics+'_'+str(i)] = data[self.target].rolling(i).std().fillna(0.0)
        elif metrics=='max':
            for i in rollingwindow:
                data['rollingwindow_'+metrics+'_'+str(i)] = data[self.target].rolling(i).max().fillna(0.0)
        elif metrics=='min':
            for i in rollingwindow:
                data['rollingwindow_'+metrics+'_'+str(i)] = data[self.target].rolling(i).min().fillna(0.0)
        elif metrics=='entropy':
            for i in rollingwindow:
                data['rollingwindow_'+metrics+'_'+str(i)] = data[self.target].rolling(i).apply(entropy).fillna(0.0)
        column = list(data.columns)
        column.remove(self.target)
        column = column + [self.target]
        data = data[column]
        return data
    
    def featuregeneration(self, data):
        size = data.shape[0]
        data = self.polytime(data)
        data = self.lag_generation(data)
        for i in ['mean','std','max','min','entropy']:
            data = self.rollingwindow_generation(data, metrics=i)
        if size < 100:
            data = self.seasonal_decomposition(data, period=2)
        else:
            for i in [2,5,10]:
                data = self.seasonal_decomposition(data, period=i)
        return data 


def scalefeature(data):
    '''
    Scaling the features of the datasets
    
    Args:
      data (pandas Dataframe): the input dataframe, train
      
    Returns:
      numpy array: dataset with the scaling features
      dictionary: scale with MinMaxScaler for the different features
    
    '''
    
    column = data.columns

    scale = {}
    temp = pd.DataFrame()
    for col in column:
        scale[col] = MinMaxScaler(feature_range=(0,1))
        temp[col] = scale[col].fit_transform(data[col].values.reshape(-1,1)).reshape(-1)
    data = temp.values

    data = pd.DataFrame(data)
    data.columns = column 

    return data, scale


def feature_selection(data, target, corr_degree=0.4, method='spearman'):
    corr = data.corr(method=method).abs().sort_values(by=[target], ascending=False)[target]
    column = list(corr.where(corr>corr_degree).dropna().index)
    column.remove(target)
    data = data[column+[target]].to_numpy()
    return data

def feature_selection2(data, target, num_feature=30, method='spearman'):
    corr = data.corr(method=method).abs().sort_values(by=[target], ascending=False)[target]
    column = list(corr.dropna().index)
    column.remove(target)
    column = column[:(num_feature-1)]
    data = data[column+[target]].to_numpy()
    return data

def featurelabel_generation(data, input_steps, output_steps):
    '''
    Prepare the features X and label y for the input shape of the deep-learning model
    
    Args:
      data (numpy array): the input features 
      input_steps (int): the number of steps for model training
      output_steps (int): the number of steps for predictions
      
    Returns:
      numpy array: feature X and label y are returned.
      
    '''
    X, y = list(), list()
    for i in range(len(data)):
        endindex = i + input_steps
        out_endindex = endindex + output_steps-1
        if out_endindex > len(data):
            break
        data_x, data_y = data[i:endindex, :-1], data[endindex-1:out_endindex, -1]
        X.append(data_x)
        y.append(data_y)
    return np.array(X), np.array(y)


def featurelabel_generation2(data, input_steps, output_steps):
    '''
    Prepare the features X and label y for the input shape of the deep-learning model
    
    Args:
      data (numpy array): the input features 
      input_steps (int): the number of steps for model training
      output_steps (int): the number of steps for predictions
      
    Returns:
      numpy array: feature X and label y are returned.
      
    '''
    X, y = list(), list()
    for i in range(len(data)):
        endindex = i + input_steps
        out_endindex = i + input_steps + output_steps
        if out_endindex > len(data):
            break
        #data_x, data_y = data[i:endindex], data[i+output_steps:out_endindex]
        data_x, data_y = data[i:endindex], data[endindex:out_endindex]
        X.append(data_x)
        y.append(data_y)
    X = np.array(X)
    y = np.array(y)
    return X, y


def generation_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask