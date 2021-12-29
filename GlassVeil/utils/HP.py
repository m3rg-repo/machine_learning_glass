# -*- coding: utf-8 -*-

import numpy as np
import pickle
import os

from sklearn.neural_network import MLPRegressor as NN
from sklearn.model_selection import RandomizedSearchCV
from sklearn.datasets import load_digits
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

# build a MLPRegressor
class new_NN(NN):
    def fit(self, x, y):
        self.random_val = None
        self.x_copy = None
        self.y_copy = None
        # perform training with multiple random starts
        # get the initial state of the RNG
        mae = 100000000
        for i in range(10):
            st0 = np.random.get_state()
            temp_reg = NN(**self.get_params()).fit(x,y)
            mae_ = np.abs(y-temp_reg.predict(x)).mean()
            if mae > mae_:
                mae = mae_
                self.random_val = st0
                self.x_copy = x
                self.y_copy = y
                self.optim_model = temp_reg
        self.set_params(**self.optim_model.get_params())
        self.MSE = mean_squared_error(self.y_copy,self.predict(self.x_copy))
        self.MAE = np.abs(self.y_copy-self.predict(self.x_copy)).mean()
        self.R2 = self.optim_model.score(self.x_copy,self.y_copy)
        print('MSE: ', self.MSE)
        print('R2: ', self.R2)
        print('MAE: ', mae)
        return self

    def score(self, X, y, sample_weight=None):
        return -np.abs(y-self.predict(X)).mean()

    def predict(self, X):
        return self.optim_model.predict(X)


# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

def tunning(X, y, params, n_iter_rand_search=100, multiple_times=1, prefix=""):
    print('\nRandomizedSearchCV\n')
    print('Multiple times: ', multiple_times)
    Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,test_size=0.2)
    for i in range(multiple_times):
        reg = new_NN(max_iter=1000,)

        # run randomized search
        random_search = RandomizedSearchCV(reg, param_distributions=params, n_iter=n_iter_rand_search, cv=4, iid=False, verbose=5, return_train_score=True,)

        random_search.fit(Xtrain, ytrain)

        print('\nReport\n')
        report(random_search.cv_results_)
        print("R2 test: ",random_search.best_estimator_.optim_model.score(Xtest,ytest))

        save_RS_models(random_search,prefix)


def save_RS_models(random_search,prefix):
    for i in range(100):
        if not(os.path.exists('./'+prefix+'random_search{}.pickle'.format(i))):
            with open('./'+prefix+'random_search{}.pickle'.format(i),'wb') as f:
                pickle.dump(random_search, f)
            model = random_search.best_estimator_
            print('Best score: ', random_search.best_score_,'\n')
            break
