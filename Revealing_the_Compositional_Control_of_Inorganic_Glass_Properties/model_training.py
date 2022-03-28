import sys
agrs = sys.argv
from functools import partial
import pandas as pd
import numpy as np
import MLPipeline as MLP
import json
import os

# 1. Data processing

name = 'Thermal_shock_resistance.pkl'

P1 = MLP.Pipe(name="%s"%name[:-4], output="1b_processed_data/%s"%name[:-4])
P1.add(MLP.data_cleaning.checksum())
P1.add(MLP.data_cleaning.drop_duplicates())
P1.add(MLP.data_cleaning.feature_selection())
P1.add(MLP.data_cleaning.data_spliting())

data = pd.read_pickle("1a_raw_data/%s"%name)
X = data[data.columns[:-1]]
y = data[data.columns[-1:]]
X_,y_ = P1((X.round(3), y))


# 2. Neural Network Model Training

optuna_NN = MLP.tunning.optuna_NN
def NN_params(trial, batch_size=[1], minN=1, maxN=10):
    params = {
    "epochs": 300,
    "batch_size": trial.suggest_categorical("batch_size", batch_size),
    "n_layers": trial.suggest_int("n_layers", 1, 2),
    "drop": trial.suggest_categorical("drop", [True, False]),
    "drate": trial.suggest_categorical("drate", [i for i in np.arange(0.1,0.4,0.1)]),
    "norm": trial.suggest_categorical("norm", [False, True]),
    "activation":  trial.suggest_categorical("activation", ["LeakyReLU", "ReLU"]),
    "opt": trial.suggest_categorical("opt", ["Adam", "SGD"]),
    "opt_params": {},
    }
    params["layers"] = [trial.suggest_int("L{}".format(i), minN, maxN) for i in range(params["n_layers"])]
    if params["opt"]=="SGD":
        params["opt_params"] = {
            "lr": trial.suggest_float("lr", 1e-5, 0.1, log=True),
            "momentum": trial.suggest_float("momentum", 9e-5, 0.9, log=True),
            }
    if params["opt"]=="Adam":
        params["opt_params"] = {
            "lr": trial.suggest_float("lr", 1e-3, 0.1, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 0.001, log=True),
            }
    return params


name = 'Thermal_shock_resistance'

X_ = pd.read_csv("1b_processed_data/Thermal_shock_resistance/03_Pipeline_%s_Node_train_test_split_train_split_X.csv"%name)
y_ = pd.read_csv("1b_processed_data/Thermal_shock_resistance/03_Pipeline_%s_Node_train_test_split_train_split_y.csv"%name)

N = X_.shape[0]
D_in = X_.shape[1]
if N<100:
    batch_size = [8]
else:
    batch_size = [int(i) for i in list(2**np.arange(np.floor(np.log2(0.02*N)),np.floor(np.log2(0.04*N))+1))]

minN = int(np.floor(D_in/2))
maxN = D_in*2

P2 = MLP.Pipe(name="Thermal_shock_resistance", output="2a_nn_results/Thermal_shock_resistance")
P2.add(MLP.data_cleaning.normalize_data())
P2.add(optuna_NN(partial(NN_params, batch_size=batch_size, minN=minN, maxN=maxN), CV=True, n_trials=100))
P2((X_, y_))

# 3. XGBoost Model Training

optuna_XGBoost = MLP.tunning.optuna_XGBoost

def xgboost_params(trial):
    param = {
        "random_state": trial.suggest_int("random_state", 1, 1000),
        "objective": "reg:squarederror",
        "booster": trial.suggest_categorical("booster", ["gbtree", "dart"]),
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        "n_estimators" : 300,
        "subsample": trial.suggest_float("subsample",0.7,1),
        "colsample_bytree": trial.suggest_float("colsample_bytree",0.7,1),
        "reg_alpha": trial.suggest_float("reg_alpha",1e-4,1, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda",1e-4,1, log=True)
    }
    if param["booster"] == "gbtree" or param["booster"] == "dart":
        param["max_depth"] = trial.suggest_int("max_depth", 1, 9)
        param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
        param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
        param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
    if param["booster"] == "dart":
        param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
        param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
        param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
        param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)
    return param

name = 'Thermal_shock_resistance'

X_ = pd.read_csv("1b_processed_data/Thermal_shock_resistance/03_Pipeline_%s_Node_train_test_split_train_split_X.csv"%name)
y_ = pd.read_csv("1b_processed_data/Thermal_shock_resistance/03_Pipeline_%s_Node_train_test_split_train_split_y.csv"%name)

P1 = MLP.Pipe(name="Thermal_shock_resistance", output="2b_xgb_results/Thermal_shock_resistance")
P1.add(MLP.data_cleaning.normalize_data())
P1.add(optuna_XGBoost(xgboost_params, CV=True, n_trials=1000))
P1((X_, y_))
