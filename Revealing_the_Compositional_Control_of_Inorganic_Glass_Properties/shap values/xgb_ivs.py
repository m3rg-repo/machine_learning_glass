import pickle
import pandas as pd
import numpy as np
import xgboost
import json

def loadfile(filename):
    with open(filename, "rb") as f:
        out = pickle.load(f)
    return out

def save2file(filename, *args, **kwargs):
    with open(filename, "wb") as f:
        out = pickle.dump(args, f, **kwargs)
    return out

def batching(x, p, axis=0):
    N = len(x)
    chunksize = int(N/p) + 1
    m = int(N/chunksize)
    if m != N/chunksize:
        m += 1
    return [np.take(x, list(range(i*chunksize,min(N, chunksize*(i+1)))), axis=axis) for i in range(m)]

def loadmodel(file):
    model = loadfile(file)
    # monkey patch
    booster = model.get_booster()
    model_bytearray = booster.save_raw()[4:]
    booster.save_raw = lambda : model_bytearray
    return model

class Model():
    def __init__(self, modelfile, datafile, mean_std_file):
        self.X = pd.read_csv(datafile).values
        self.MS = json.load(open(mean_std_file))
        self.mean = self.MS["means"][:-1]
        self.std = self.MS["stds"][:-1]
        self.X = (self.X-self.mean)/self.std
        self.model = loadmodel(modelfile)
    def cal_ivs(self, X):
        return self.model.get_booster().predict(xgboost.DMatrix(X), pred_interactions=True)
