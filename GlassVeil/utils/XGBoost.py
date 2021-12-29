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
logfile='logfile.log'
import datetime
now = datetime.datetime.now
import pickle
import os

from numpy import loadtxt
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from matplotlib import pyplot as plt
from scipy.stats import uniform, randint

def report_best_scores(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
    return results['params'][candidates[0]]


def train_XGB(file):
    # load data
    full_dataset = loadtxt(file+'.csv', delimiter=",",skiprows=1)
    dataset, test_dataset = train_test_split(full_dataset,test_size=0.2)

    # split data into X and y
    X = dataset[:,:-1]
    y = dataset[:,-1:]


    model = XGBRegressor(objective="reg:squarederror")
    params = {
        "colsample_bytree": uniform(0.7, 0.3),
        "gamma": uniform(0, 0.5),
        "learning_rate": uniform(0.03, 0.3), # default 0.1
        #"max_depth": randint(2, 6), # default 3
        "n_estimators": randint(100, 150), # default 100
        "subsample": uniform(0.6, 0.4)
    }

    search = RandomizedSearchCV(model, param_distributions=params, n_iter=200, cv=3, verbose=1, n_jobs=1, return_train_score=True)
    search.fit(X, y)

    params = report_best_scores(search.cv_results_, 1)
    # params = {}
    params.update({'lambda':1,'max_depth':3})

    kfold = KFold(n_splits=4, shuffle=True,)
    scores = []
    results_ = []
    yvsy = []

    ind = 0
    for train_index, test_index in kfold.split(X):
        ind += 1
        X_train, X_val = X[train_index], X[test_index]
        y_train, y_val = y[train_index], y[test_index]

        eval_set = [(X_train, y_train), (X_val, y_val)]
        model = XGBRegressor(objective="reg:squarederror",**params)

        model.fit(X_train, y_train, eval_metric=["rmse", "mae"], eval_set=eval_set, verbose=True, early_stopping_rounds=10)

        y_pred_val = model.predict(X_val)
        y_pred_train = model.predict(X_train)

        y_pred_test = model.predict(test_dataset[:,:-1])

        yvsy.append([test_dataset[:,-1:],y_pred_test])

        scores.append((r2_score(*yvsy[-1]),r2_score(y_val, y_pred_val), r2_score(y_train, y_pred_train)))
        print(scores[-1])

        # retrieve performance metrics
        results = model.evals_result()
        results_.append(results)
        epochs = len(results['validation_0']['rmse'])
        x_axis = range(0, epochs)

        # plot rmse
        fig, ax = plt.subplots()
        ax.plot(x_axis, results['validation_0']['rmse'], label='Train')
        ax.plot(x_axis, results['validation_1']['rmse'], label='Test')
        ax.legend()
        plt.ylabel('RMSE')
        plt.xlabel('Epoch')
        plt.title(ind)
        plt.show()

        # plot mae
        fig, ax = plt.subplots()
        ax.plot(x_axis, results['validation_0']['mae'], label='Train')
        ax.plot(x_axis, results['validation_1']['mae'], label='Test')
        ax.legend()
        plt.ylabel('MAE')
        plt.xlabel('Epoch')
        plt.title(ind)
        plt.show()

    pickle.dump([yvsy, results_, scores, params],open(file+'.pickle','wb+'))

    test_score, val_score, train_score, = np.array(scores).mean(axis=0)

    print('Train : ',train_score)
    print('Val : ',val_score)
    print('Test : ',test_score)


for file in ['den','YM','H','SM','TEC','TG','LT','RI']:
    logfile = file+'_log.txt'
    try:
        os.rename(logfile,logfile+'.bak')
    except:
        pass
    print(file)
    train_XGB(file)
    try:
        os.remove(logfile+'.bak')
    except:
        pass





#
#
