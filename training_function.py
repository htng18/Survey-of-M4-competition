import torch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utilities import FeatureEngineering, scalefeature, feature_selection, featurelabel_generation
from utilities import feature_selection2, featurelabel_generation2, generation_mask
from models import CNN, LSTM, BiLSTM, CNNLSTM, CNNBiLSTM, CNN2D, Transformer

model_dict = {"CNN1D":CNN, "LSTM":LSTM, "BiLSTM":BiLSTM, "CNNLSTM":CNNLSTM, 
            "CNNBiLSTM":CNNBiLSTM, "CNN2D":CNN2D, "Transformer":Transformer}

def trainer(train, test, target, params, model, model_params, device):
    
    params['wait'] = 0
    params['best_loss'] = np.Inf
    params['epochs_no_improve'] = 0
    params['early_stop'] = False
    
    train = train.T.reset_index(drop=True)[[target]].dropna()
    test = test.T.reset_index(drop=True)[[target]].dropna()

    size = train.shape[0]
    output_steps = test.shape[0]
    
    
    if size/output_steps <= 3:
        input_steps = 2
    elif (size/output_steps > 3) and (size/output_steps <= 6):
        input_steps = 3
    elif (size/output_steps > 6) and (size/output_steps < 10):
        input_steps = 5
    else:
        input_steps = test.shape[0]

        
    lag = 2
    num_lag = int(size/3/lag)
    window = 5
    num_window = int(size/2/window)
    feature_eng = FeatureEngineering(target, degree=4, lag=lag, num_lag=num_lag, window=window, num_window=num_window)

    train = feature_eng.featuregeneration(train)
    train, scale = scalefeature(train)
    train = feature_selection(train, target, corr_degree=0.3)
    test = test.values.flatten()
    
    inputdata = train[-input_steps:,:]
    inputdata = inputdata.reshape(1, inputdata.shape[0], inputdata.shape[1])
    inputdata = inputdata[:,:,:-1]



    if size/output_steps > 6:
        validation = train[-(input_steps+output_steps):,:]
        validation_X, validation_y = featurelabel_generation(validation, input_steps, output_steps)

        train = train[:-(input_steps+output_steps),:]
        train_X, train_y = featurelabel_generation(train, input_steps, output_steps)
    else:
        validation_X, validation_y = featurelabel_generation(train, input_steps, output_steps)
        train_X, train_y = featurelabel_generation(train, input_steps, output_steps)
    
    n_features = train_X.shape[2]

    if params["model"] == "CNN2D":
        train_X = train_X.reshape(train_X.shape[0], train_X.shape[1], 1, train_X.shape[2])
        validation_X = validation_X.reshape(validation_X.shape[0], validation_X.shape[1], 1, validation_X.shape[2])
        inputdata= inputdata.reshape(inputdata.shape[0], inputdata.shape[1], 1, inputdata.shape[2])
    
    net = model(n_features,input_steps,output_steps,model_params).to(device)
    criterion = torch.nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=5e-4)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience = 2)
    
    net, loss_dict = train_func(net, optimizer, criterion, scheduler, train_X, train_y, validation_X, validation_y, params, device)

    return net, loss_dict, scale, inputdata, test


def train_func(net, optimizer, criterion, scheduler, train_X, train_y, validation_X, validation_y, params, device):
    loss_dict = {}
    loss_dict['train loss'] = []
    loss_dict['val loss'] = []
    
    for epoch in range(params['epochs']):
        net.train()
        for b in range(0,len(train_X),params['batch_size']):
            optimizer.zero_grad() 
            X = train_X[b:b+params['batch_size']]
            y = train_y[b:b+params['batch_size']]

            x_batch = torch.tensor(X,dtype=torch.float32).to(device)
            y_batch = torch.tensor(y,dtype=torch.float32).to(device)

            if params["model"] in ["CNN1D", "CNN2D"]:
                output = net(x_batch)
            elif params["model"] in ["LSTM", "BiLSTM", "CNNLSTM", "CNNBiLSTM"]:
                output = net(x_batch)[0]
        
            
            #train_loss = criterion(output, y_batch)
            if params["model"]=="CNN2D":
                train_loss = criterion(output[:,-1,-1,:], y_batch)
            else:
                train_loss = criterion(output[:,-1,:], y_batch)
                print(output[:,-1,:].size(), y_batch.size())
            train_loss.backward()
            optimizer.step()
            

        with torch.no_grad():
            net.eval()
            for b in range(0,len(validation_X),params['batch_size']):
                X = validation_X[b:b+params['batch_size']]
                y = validation_y[b:b+params['batch_size']]    

                x_batch = torch.tensor(X,dtype=torch.float32).to(device)   
                y_batch = torch.tensor(y,dtype=torch.float32).to(device)

                if params["model"] in ["CNN1D", "CNN2D"]:
                    output = net(x_batch)
                elif params["model"] in ["LSTM", "BiLSTM", "CNNLSTM", "CNNBiLSTM"]:
                    output = net(x_batch)[0]
                
                if params["model"]=="CNN2D":
                    val_loss = criterion(output[:,-1,-1,:], y_batch)
                else:
                    val_loss = criterion(output[:,-1,:], y_batch)
        curr_lr = optimizer.param_groups[0]['lr']
        print('epoch:' , epoch , 'train loss:' , round(train_loss.item(),3), 
              'val loss:', round(val_loss.item(),3), 'lr:',curr_lr)
        loss_dict['train loss'].append(train_loss.item())
        loss_dict['val loss'].append(val_loss.item())
        scheduler.step(val_loss)

        if val_loss < params['best_loss']:
            params['epochs_no_improve'] = 0
            params['best_loss'] = val_loss
        else:
            params['epochs_no_improve'] += 1
        #params['wait'] += 1
        if epoch > 5 and params['epochs_no_improve'] == params['patience_loss']:
            params['early_stop'] = True
            break
        else:
            continue
    return net, loss_dict


def trainer2(train, test, target, params, model, model_params, device):
    
    params['best_loss'] = np.Inf
    params['epochs_no_improve'] = 0
    params['early_stop'] = False
    
    train = train[[target]].dropna()
    test = test[[target]].dropna()

    size = train.shape[0]
    output_steps = test.shape[0]

    if size/output_steps <= 3:
        input_steps = 1
    elif (size/output_steps > 3) and (size/output_steps <= 6):
        input_steps = 2
    elif (size/output_steps > 6) and (size/output_steps < 10):
        input_steps = 3
    else:
        input_steps = test.shape[0]

    

    input_steps = test.shape[0]

    #input_steps, output_steps = test.shape[0], test.shape[0]
    model_params["max_seq_len"] = input_steps
    #print("input_steps", input_steps, "output_steps", output_steps)
    
    #size = train.shape[0]
    lag = 2
    num_lag = int(size/3/lag)
    window = 5
    num_window = int(size/2/window)
    feature_eng = FeatureEngineering(target, degree=4, lag=lag, num_lag=num_lag, window=window, num_window=num_window)

    train = feature_eng.featuregeneration(train)
    #print(train.shape)
    if train.shape[1] < model_params["num_features"]:
        model_params["num_features"] = int(np.floor(train.shape[1]/10)*10)

    train, scale = scalefeature(train)
    train = feature_selection2(train, target, model_params["num_features"])
    test = test.values.flatten()
    
    inputdata = train[-input_steps:,:]
    inputdata = inputdata.reshape(1, inputdata.shape[0], inputdata.shape[1])
    inputdata = inputdata[:,:]

    if size/output_steps > 6:
        validation = train[-(input_steps+output_steps):,:]
        validation_X, validation_y = featurelabel_generation2(validation, input_steps, output_steps)

        train = train[:-(input_steps+output_steps),:]
        train_X, train_y = featurelabel_generation2(train, input_steps, output_steps)
    else:
        validation_X, validation_y = featurelabel_generation2(train, input_steps, output_steps)
        train_X, train_y = featurelabel_generation2(train, input_steps, output_steps)

    net = model(model_params).to(device)
    criterion = torch.nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=5e-4)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience = 2)
    
    net, loss_dict = train_func2(net, optimizer, criterion, scheduler, train_X, train_y, validation_X, validation_y, output_steps, params, device)

    return net, loss_dict, scale, inputdata, test

def train_func2(net, optimizer, criterion, scheduler, train_X, train_y, validation_X, validation_y, output_steps, params, device):
    loss_dict = {}
    loss_dict['train loss'] = []
    loss_dict['val loss'] = []
    
    X_mask = generation_mask(train_X.shape[1])
    X_mask = X_mask.to(device)
    
    for epoch in range(params['epochs']):
        net.train()
        for b in range(0,len(train_X),params['batch_size']):
            optimizer.zero_grad() 
            X = train_X[b:b+params['batch_size']]
            y = train_y[b:b+params['batch_size']]

            x_batch = torch.tensor(X,dtype=torch.float32).permute(1,0,2).to(device)    
            y_batch = torch.tensor(y,dtype=torch.float32).permute(1,0,2).to(device)

            output = net(x_batch, X_mask) 
            
            train_loss = criterion(output[-output_steps:,0,0], y_batch[-output_steps:,0,-1]) 
            train_loss.backward()
            optimizer.step()
        
        with torch.no_grad():
            net.eval()
            for b in range(0,len(validation_X),params['batch_size']):
                X = validation_X[b:b+params['batch_size']]
                y = validation_y[b:b+params['batch_size']]    

                x_batch = torch.tensor(X,dtype=torch.float32).permute(1,0,2).to(device)           
                y_batch = torch.tensor(y,dtype=torch.float32).permute(1,0,2).to(device)

                output = net(x_batch, X_mask)
                
                val_loss = criterion(output[-output_steps:,0,0], y_batch[-output_steps:,0,-1])
        curr_lr = optimizer.param_groups[0]['lr']
        print('epoch:' , epoch , 'train loss:' , round(train_loss.item(),3), 'val loss:', round(val_loss.item(),3), 'lr:',curr_lr)
        loss_dict['train loss'].append(train_loss.item())
        loss_dict['val loss'].append(val_loss.item())
        scheduler.step(val_loss)

        if val_loss < params['best_loss']:
            params['epochs_no_improve'] = 0
            params['best_loss'] = val_loss
        else:
            params['epochs_no_improve'] += 1
            
        if params['epochs_no_improve'] == params['patience_loss']:
            params['early_stop'] = True
            break
        else:
            continue
        
    return net, loss_dict