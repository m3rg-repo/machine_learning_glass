import sys
sys.path.append('/Users/sureshbishnoi/Owncloud/python_packages')
import pandas as pd
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
import pickle

from ml import MLmodule
from shadow.plot import *
from shadow import plot
import importlib
importlib.reload(plot)
importlib.reload(MLmodule1)
S = MLmodule1.sess

from sklearn.linear_model import ARDRegression, LinearRegression, Lasso, Ridge, Lars
from sklearn.neural_network import MLPRegressor as nn
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel,RBF, ConstantKernel as C,Matern
from sklearn.metrics import r2_score as r2

################################################################################
##data_visulation starts
np.random.seed(5111)
opf=pd.read_csv("input_data.csv")

X = opf[opf.columns[0:-1]].values
y = opf[opf.columns[-1]].values
X.shape

################################################################################
###Regressian analysis starts

T2 = S()
T2.features = X
T2.properties = y

T2.train(model = LinearRegression,rn_iter = 30)
T2.plot1(label = "density (g/cm$^3$)",inter = 2,range = [0,10],set_range = True,model_name = 'LinearRegression')
T2.plot2(label = "density *(g/cm$^3$)",inter = 2,range = [0,10],set_range = True,den_scale_bool=True,den_scale = 2,model_name = 'LinearRegression')

T2.train(xg_model = True,rn_iter = 1)
T2.plot1(label = "density (g/cm$^3$)",inter = 2,range = [0,10],set_range = True,model_name = 'xg_boost')
T2.plot2(label = "density *(g/cm$^3$)",inter = 2,range = [0,10],set_range = True,den_scale_bool=True,den_scale = 2,model_name = 'xg_boost')
