import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
import pickle
from sklearn.metrics import r2_score, mean_squared_error, make_scorer
from sklearn.model_selection import GridSearchCV

df = pd.read_csv('Density_gp.csv')
X = df.values[:,:-1]
y = df.values[:,-1]

optim_model = None
mse_previous = 1000000000
r2_score_test_opt_rnitr = 0
r2_score_train_opt_rnitr = 0
mse_all_opt_rnitr = 0
mse_train_opt_rnitr = 0
mse_test_opt_rnitr = 0
optim_seed = 0
def do_nn(X,y, hl, lr, rn_iter):
    global optim_model, mse_previous, r2_score_test_opt_rnitr,r2_score_train_opt_rnitr,mse_all_opt_rnitr,mse_train_opt_rnitr,mse_test_opt_rnitr,optim_seed
    for i in range(rn_iter):
        np.random.seed(1+i*1231)
        model = MLPRegressor(max_iter=1000, solver='lbfgs' ,learning_rate_init=lr, hidden_layer_sizes=hl)
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3,)
        model.fit(x_train,y_train)

        r2_score_test = model.score(x_test,y_test)
        r2_score_train = model.score(x_train,y_train)
        mse_all = mean_squared_error(model.predict(X),y)
        mse_train = mean_squared_error(model.predict(x_train),y_train)
        mse_test = mean_squared_error(model.predict(x_test),y_test)
        if (mse_previous>mse_all) and (mse_train<mse_test):
            mse_previous = mse_all
            optim_model = model
            r2_score_test_opt_rnitr = r2_score_test
            r2_score_train_opt_rnitr = r2_score_train
            mse_all_opt_rnitr = mse_all
            mse_train_opt_rnitr = mse_train
            mse_test_opt_rnitr = mse_test
            optim_seed = 1+i*1231

    return optim_model,r2_score_test_opt_rnitr,r2_score_train_opt_rnitr,mse_all_opt_rnitr,optim_seed

mse_all_opt_rnitr_prev = 10000000
for j in range(1,25,1):
    for i in ([j],[j,j],[j,j,j],[j,j,j,j],[j,j,j,j,j]):
        optim_model,r2_score_test_opt_rnitr,r2_score_train_opt_rnitr,mse_all_opt_rnitr,optim_seed= do_nn(X,y, hl=i, lr=0.005, rn_iter=30)
        if (mse_all_opt_rnitr_prev>mse_all_opt_rnitr) and (r2_score_train_opt_rnitr>r2_score_test_opt_rnitr):
            mse_all_opt_rnitr_prev = mse_all_opt_rnitr
            optim_optim_moddel = optim_model
            optim_optim_seed = optim_seed
            with open('Density_model', 'wb') as f:
                pickle.dump(optim_optim_moddel,f)
            f.close
            with open('Density_seed', 'wb') as f:
                pickle.dump(optim_optim_seed,f)
            f.close
