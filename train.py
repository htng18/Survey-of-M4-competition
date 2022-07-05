import argparse
import configparser
import pandas as pd
import torch
from training_function import model_dict, trainer, train_func, trainer2, train_func2
from utilities import compute_metrics
import gc
from os.path import exists
import time
import warnings
warnings.filterwarnings('ignore')
torch.manual_seed(21)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="CNN1D")
    parser.add_argument("--period", type=str, default="Hourly")
    parser.add_argument("--device", type=str, default="cpu")
    args, _ = parser.parse_known_args()
    model = model_dict[args.model]
    
    config = configparser.ConfigParser()
    config.read('ModelParameters.INI')
    params = {key:int(value) for key, value in config.items('Parameters')}
    params["model"] = args.model
    model_params = {key:int(value) for key, value in config.items(args.model)}

    if args.model=="Transformer":
        Trainer = trainer2
    else:
        Trainer = trainer

    file_path = 'Dataset/{}/{}-train.csv'.format("Train", args.period)
    train = pd.read_csv(file_path)
    file_path = 'Dataset/{}/{}-test.csv'.format("Test", args.period)
    test = pd.read_csv(file_path)
    train.index = train['V1']
    target_list = list(train.index)
    train = train.drop('V1', axis = 1).T.reset_index(drop=True)
    test.index = test['V1']
    test = test.drop('V1', axis = 1).T.reset_index(drop=True)
    
    file_path = "result/performance_{}_{}.csv".format(args.period, args.model)
    
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available() and args.device=="cuda":
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    for target in target_list:
        print('Target', target)
        start = time.time()
        net, loss_dict, scale, inputdata, testdata = Trainer(train, test, target, params, model, model_params, device)
        insample = train[[target]].dropna().values.flatten()
        SMAPE, MASE, OWA, MAE, RMSE = compute_metrics(net, scale, inputdata, testdata, insample, target, params, args.period, device)
        performance = pd.DataFrame(index=["SMAPE", "MASE", "OWA", "MAE", "RMSE"])
        performance[target] = [SMAPE, MASE, OWA, MAE, RMSE]
        performance = performance.T
        print("\n")
        print("SMAPE:", round(SMAPE,2), "MASE:", round(MASE,2), "OWA:", round(OWA,2), "MAE:", round(MAE,2), "RMSE:", round(RMSE,2))
        print("\n")
        end = time.time()
        print("Elapsed time: ", end - start)
        
        if exists(file_path):
            performance.to_csv(file_path, mode="a", header=False)
        else:
            performance.to_csv(file_path, mode="w", header=True)

        del net
        torch.cuda.empty_cache()
        gc.collect()


