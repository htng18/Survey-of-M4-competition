import torch
import torch.nn as nn
import numpy as np

class CNN(nn.Module):
    def __init__(self, n_features, seq_length, output_steps, params):
        super(CNN, self).__init__()
        self.n_features = n_features
        self.seq_len = seq_length
        self.output_steps = output_steps
        self.num_hidden1 = params['num_hidden1'] 
        self.num_hidden2 = params['num_hidden2']
        
        self.conv1d = nn.Conv1d(in_channels=seq_length, 
                             out_channels=self.num_hidden1, 
                             kernel_size=1,
                             stride=1)
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(self.n_features, self.num_hidden2)
        #self.linear1 = nn.Linear(self.n_features*self.num_hidden1, self.num_hidden2)
        self.linear2 = nn.Linear(self.num_hidden2, self.output_steps)
    
    def forward(self, x):
        batch_size, _, _ = x.size()
        x = self.conv1d(x)
        #x = self.conv1d(x).contiguous().view(batch_size,-1)
        x = self.relu(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

class CNN2D(nn.Module):
    def __init__(self,n_features,seq_length,output_steps, params):
        super(CNN2D, self).__init__()
        self.n_features = n_features
        self.seq_len = seq_length
        self.num_hidden1 = params["num_hidden1"]
        self.num_hidden2 = params["num_hidden2"]
        self.output_steps = output_steps
        
        self.conv2d = nn.Conv2d(in_channels=seq_length, 
                             out_channels=self.num_hidden1, 
                             kernel_size=1,
                             stride=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.linear1 = nn.Linear(self.n_features, self.num_hidden2)
        self.linear2 = nn.Linear(self.num_hidden2, self.output_steps)
    
    def forward(self, x):
        x = self.conv2d(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class LSTM(nn.Module):
    #def __init__(self,n_features,seq_length,output_steps, num_hidden, num_layers):
    def __init__(self,n_features,seq_length,output_steps, params):
        super(LSTM, self).__init__()
        self.n_features = n_features
        self.seq_len = seq_length
        self.num_hidden = params['num_hidden']
        self.num_layers = params['num_layers']
        self.output_steps = output_steps
    
        self.lstm = nn.LSTM(input_size = n_features, 
                                 hidden_size = self.num_hidden,
                                 num_layers = self.num_layers, 
                                 batch_first = True)
        self.linear = nn.Linear(self.num_hidden, self.output_steps)
    
    def forward(self, x, hidden=None):        
        batch_size, seq_len, _ = x.size()
        if hidden==None:
            hidden_state = torch.zeros(self.num_layers,batch_size,self.num_hidden)
            cell_state = torch.zeros(self.num_layers,batch_size,self.num_hidden)
            self.hidden = (hidden_state, cell_state)
        else:
            self.hidden = hidden
        
        lstm_out, self.hidden = self.lstm(x,self.hidden)
        
        x = lstm_out
        #x = lstm_out.contiguous().view(batch_size,-1)
        return self.linear(x), self.hidden

class BiLSTM(nn.Module):
    #def __init__(self,n_features,seq_length,output_steps, num_hidden, num_layers):
    def __init__(self,n_features,seq_length,output_steps, params):
        super(BiLSTM, self).__init__()
        self.n_features = n_features
        self.seq_len = seq_length
        self.num_hidden = params['num_hidden']
        self.num_layers = params['num_layers']
        self.output_steps = output_steps
    
        self.lstm = nn.LSTM(input_size = n_features, 
                                 hidden_size = self.num_hidden,
                                 num_layers = self.num_layers, 
                                 batch_first = True,
                                 bidirectional = True)
        self.linear = nn.Linear(2*self.num_hidden, self.output_steps)
    
    def forward(self, x, hidden=None):        
        batch_size, seq_len, _ = x.size()
        if hidden==None:
            hidden_state = torch.zeros(2*self.num_layers,batch_size,self.num_hidden)
            cell_state = torch.zeros(2*self.num_layers,batch_size,self.num_hidden)
            self.hidden = (hidden_state, cell_state)
        else:
            self.hidden = hidden
        lstm_out, self.hidden = self.lstm(x,self.hidden)
        
        x = lstm_out
        #x = lstm_out.contiguous().view(batch_size,-1)
        return self.linear(x), self.hidden


class CNNLSTM(nn.Module):
    def __init__(self,n_features,seq_length,output_steps, params):
        super(CNNLSTM, self).__init__()
        self.n_features = n_features
        self.seq_len = seq_length
        self.num_hidden = params["num_hidden"] 
        self.num_layers = params["num_layers"]
        self.output_steps = output_steps
        
        self.conv1d = nn.Conv1d(in_channels=seq_length, 
                             out_channels=self.num_hidden, 
                             kernel_size=1,
                             stride=1)
        self.lstm = nn.LSTM(input_size = n_features, 
                                 hidden_size = self.num_hidden,
                                 num_layers = self.num_layers, 
                                 batch_first = True)
        self.linear = nn.Linear(self.num_hidden, self.output_steps)
       
   
    def forward(self, x, hidden=None):
        batch_size, seq_len, _ = x.size()
        if hidden==None:
            hidden_state = torch.zeros(self.num_layers,batch_size,self.num_hidden)
            cell_state = torch.zeros(self.num_layers,batch_size,self.num_hidden)
            self.hidden = (hidden_state, cell_state)
        else:
            self.hidden = hidden
        x = self.conv1d(x)
        lstm_out, self.hidden = self.lstm(x,self.hidden)
        #x = lstm_out.contiguous().view(batch_size,-1)
        x = lstm_out

        return self.linear(x), self.hidden


class CNNBiLSTM(nn.Module):
    def __init__(self,n_features,seq_length,output_steps, params):
        super(CNNBiLSTM, self).__init__()
        self.n_features = n_features
        self.seq_len = seq_length
        self.num_hidden = params["num_hidden"] 
        self.num_layers = params["num_layers"]
        self.output_steps = output_steps
        
        self.conv1d = nn.Conv1d(in_channels=seq_length, 
                             out_channels=self.num_hidden, 
                             kernel_size=1,
                             stride=1)
        self.lstm = nn.LSTM(input_size = n_features, 
                                 hidden_size = self.num_hidden,
                                 num_layers = self.num_layers, 
                                 batch_first = True,
                                 bidirectional = True)
        self.linear = nn.Linear(2*self.num_hidden, self.output_steps)
       
   
    def forward(self, x, hidden=None):
        batch_size, seq_len, _ = x.size()
        if hidden==None:
            hidden_state = torch.zeros(2*self.num_layers,batch_size,self.num_hidden)
            cell_state = torch.zeros(2*self.num_layers,batch_size,self.num_hidden)
            self.hidden = (hidden_state, cell_state)
        else:
            self.hidden = hidden
        x = self.conv1d(x)
        lstm_out, self.hidden = self.lstm(x,self.hidden)
        x = lstm_out

        return self.linear(x), self.hidden




class PositionalEncoder(nn.Module):

    def __init__(self, max_seq_len= 5000, d_model= 512):
        super().__init__()
        self.d_model = d_model
        position = torch.arange(max_seq_len).unsqueeze(1)
        exp_input = torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model)
        div_term = torch.exp(exp_input) 
        pe = torch.zeros(max_seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term) 
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe) 
        
    def forward(self, x):
        x = x + self.pe[:x.size(1), :].squeeze(1)
        return x


class Transformer(nn.Module):
    '''
      Reference: https://github.com/nklingen/Transformer-Time-Series-Forecasting
                 https://medium.com/mlearning-ai/transformer-implementation-for-time-series-forecasting-a9db2db5c820
                
    '''
    

    def __init__(self, params): 
        super(Transformer, self).__init__() 

        self.positional_encoding_layer = PositionalEncoder(
            d_model=params["num_features"],
            max_seq_len=params["max_seq_len"])
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=params["num_features"], 
            nhead=params["num_heads"],
            dropout=0.1)
        self.encoder = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer,
            num_layers=params["num_layers"])
        self.decoder = nn.Linear(
            params["num_features"], 1)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-0.1, 0.1)

    def forward(self, src, src_mask):

        src = self.positional_encoding_layer(src)
        output = self.encoder(src, src_mask)

        output = self.decoder(output)

        return output