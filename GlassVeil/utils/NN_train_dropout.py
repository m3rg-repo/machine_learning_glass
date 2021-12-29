from __future__ import print_function
try:
    import __builtin__
except ImportError:
    # Python 3
    import builtins as __builtin__

try:
    __builtin__.print = _print
except:
    pass

_print = __builtin__.print

def print(*args, **kwargs):
    """My custom print() function."""
    global logfile
    _print(*args, **kwargs)
    with open(logfile,'a+') as f:
        f.write('\n\n'+str(now())+'\n')
        for v in args:
            f.write(v.__str__()+' ')

__builtin__.print = print

from dropout_norm import DropoutNormNet, split, DoML
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import datetime
from sklearn.metrics import r2_score
import pickle
from sklearn.model_selection import KFold


logfile='logfile.log'

now = datetime.datetime.now

def myloss(y,ypred):
    return ((y-ypred)**2).sum(axis=0)/y.shape[0]


def dothis(file):
    global Print, print, logfile, layer
    logfile = file+'_log.txt'
    print(file)

    data = np.loadtxt('./'+file.split('_')[0]+'.csv',skiprows=1,delimiter=',')
    train_data, test_data, _, _ = split(data,data)

    np.savetxt('./train_data_'+file+'.csv',train_data)
    np.savetxt('./test_data_'+file+'.csv',test_data)

    data.shape
    train_data.shape
    test_data.shape

    means = train_data.mean(dim=0, keepdim=True)
    stds = train_data.std(dim=0, keepdim=True)
    mask = stds<=0.0001
    stds[mask] = 1
    normalized_data = (train_data - means) / stds
    normalized_test = (test_data - means) / stds

    Xtest = normalized_test[:,:-1]
    ytest = normalized_test[:,-1]

    N = Xtest.shape[0]

    kf = KFold(n_splits=4)
    models = []
    k = 0
    for train_index, test_index in kf.split(normalized_data):
        k+=1
        print("Kfold: ",k)
        Xtrain, Xval = normalized_data[train_index,:-1], normalized_data[test_index,:-1]
        ytrain, yval = normalized_data[train_index,-1], normalized_data[test_index,-1]

        mlc = DoML()
        models.append(mlc)
        mlc.means = means
        mlc.stds = stds
        mlc.set_data([Xtrain, Xval, ytrain, yval])
        drop = False
        norm = False
        if 'drop' in file:
            drop = True
        if 'norm' in file:
            norm = True
        mlc.model = DropoutNormNet(Xtrain.shape[1], 1, layer, [0.2], dropout=drop, batch_norm=norm)
        mlc.loss = myloss
        mlc.optimizer = torch.optim.Adam(mlc.model.parameters(),lr=0.001,weight_decay=0.001)
        mlc.dataloader_params.update({"batch_size":int(N/40),"drop_last":True})


        mlc.train(N=200)
        m, s = means[0,-1], stds[0,-1]
        mlc.ytest = m+s*ytest
        mlc.ytest_pred = m+s*mlc.model(Xtest).detach().numpy()

        mlc.cal_R2()

        mlc.plot_loss()
        plt.savefig(file+'_loss.png',dpi=100)
        plt.clf()

        # plt.hist(((mlc.yval-mlc.model(mlc.Xval))).detach().numpy()*s.item(),bins=50)
        # plt.savefig(file+'_error_hist.png',dpi=100)
        # plt.clf()
        #
        # plt.plot(mlc.yval*s+m,m+s*mlc.model(mlc.Xval).detach().numpy(),'o',alpha=0.1)
        # lim = [(m+s*mlc.yval).min(),(m+s*mlc.yval).max()]
        # plt.plot(lim,lim,)
        # plt.savefig(file+'_yvsy_val.png',dpi=100)
        # plt.clf()
        #
        # plt.plot(mlc.ytest,mlc.ytest_pred,'o',alpha=0.1)
        # lim = [mlc.ytest.min(),mlc.ytest.max()]
        # plt.plot(lim,lim)
        # plt.savefig(file+'_yvsy_test.png',dpi=100)
        # plt.clf()
        # print('R2(test): ',r2_score(ytest,mlc.model(Xtest).detach().numpy()))

    avg_val_score = np.mean([m.R2_val for m in models])
    avg_train_score = np.mean([m.R2_train for m in models])
    print("Avg val R2: ", avg_val_score)
    print("Avg train R2: ", avg_train_score)
    with open(file+'_models.pickle','bw+') as f:
        pickle.dump(models,f)

layers = {'den': [19, 17, 12] ,
'YM': [20, 20, 20] ,
'H': [29, 29] ,
'SM': [19, 9, 20] ,
'TEC': [65, 65, 65, 65] ,
'TG': [20, 15, 1] ,
'LT': [18, 19, 4] ,
'RI': [6, 9, 16] ,}

for file in ['H']: #['den','YM','H','SM','TEC','TG','LT','RI']:
    layer = layers[file]
    dothis(file+'_drop')
    dothis(file+'_drop_norm')


#
