import math
import torch
torch.potrs = torch.cholesky_solve  ## setting the varible name to the varible name in old launch of torch
import gpytorch

import csv
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import r2_score as r2
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import KFold

ran = np.int(np.random.random()*1000000)

data = (pd.read_csv('YM_gp.csv').values)
X = data[:,:-1]
# normalize input data in 0 to 1 range(X consist the values of the glass compositions in percentage(values ranges foe a composition is [0% to 100%]))
X = X - 0
X = 2 * (X / 100) - 1
y = data[:,-1] #property value in GPa

my_score = []
my_score.append(['seed','fold','epoch','R2train','R2val','R2test'])
my_loss = []
my_loss.append(['seed','fold','epoch','tr_loss','val_loss'])


for i in range(ran,ran+10):
    global opt_mse_all,val_score_prev
    seed_val = i
    torch.manual_seed(seed_val)

    with gpytorch.settings.use_toeplitz(False), gpytorch.settings.fast_pred_var(), gpytorch.settings.max_root_decomposition_size(100):
        # Use the first 70% of the data for training, and the last 30% for testing.
        train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.15, random_state=seed_val)

        train_x = torch.Tensor(train_x)
        train_y = torch.Tensor(train_y)
        test_x = torch.Tensor(test_x)
        test_y = torch.Tensor(test_y)
        kf = KFold(n_splits=5,shuffle=True, random_state=seed_val)
        j=0
        for train_index, val_index in kf.split(train_x):
            j+=1
            tr_x, val_x = train_x[train_index], train_x[val_index]
            tr_y, val_y = train_y[train_index], train_y[val_index]

            data_dim = tr_x.size(-1)

            class LargeFeatureExtractor(torch.nn.Sequential):
              def __init__(self):
                  super(LargeFeatureExtractor, self).__init__()
                  self.add_module('linear1', torch.nn.Linear(data_dim, 1000))
                  self.add_module('relu1', torch.nn.ReLU())
                  self.add_module('linear2', torch.nn.Linear(1000, 1000))
                  self.add_module('relu2', torch.nn.ReLU())
                  self.add_module('linear3', torch.nn.Linear(1000, 500))
                  self.add_module('relu3', torch.nn.ReLU())
                  self.add_module('linear4', torch.nn.Linear(500, 50))
                  self.add_module('relu4', torch.nn.ReLU())
                  self.add_module('linear5', torch.nn.Linear(50, 2))

            feature_extractor = LargeFeatureExtractor()

            class GPRegressionModel(gpytorch.models.ExactGP):
              def __init__(self, train_x, train_y, likelihood):
                  super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
                  self.mean_module = gpytorch.means.ConstantMean()
                  self.covar_module = gpytorch.kernels.GridInterpolationKernel(
                      gpytorch.kernels.RBFKernel(),
                      grid_size=100, num_dims=2
                  )

                  # Also add the deep net
                  self.feature_extractor = feature_extractor

              def forward(self, x):
                  # We're first putting our data through a deep net (feature extractor)
                  # We're also scaling the features so that they're nice values
                  projected_x = self.feature_extractor(x)
                  projected_x = projected_x - projected_x.min(0)[0]
                  projected_x = 2 * (projected_x / projected_x.max(0)[0]) - 1

                  # The rest of this looks like what we've seen
                  mean_x = self.mean_module(projected_x)
                  covar_x = self.covar_module(projected_x)
                  return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

            likelihood = gpytorch.likelihoods.GaussianLikelihood()

            model = GPRegressionModel(tr_x, tr_y, likelihood)
            # Find optimal model hyperparameters
            model.train()
            likelihood.train()

            # Use the adam optimizer
            # Add weight decay to the feature exactractor ONLY
            optimizer = torch.optim.Adam([
              {'params': model.mean_module.parameters()},
              {'params': model.covar_module.parameters()},
              {'params': model.likelihood.parameters()},
              {'params': model.feature_extractor.parameters(), 'weight_decay': 1e-3}
            ], lr=0.005)

            # "Loss" for GPs - the marginal log likelihood
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

            opt_mse_all=1000000
            val_score_prev = 0
            def train(training_iterations=600):
                global opt_mse_all,val_score_prev
                for z in range(training_iterations):
                    optimizer.zero_grad()
                    output = model(tr_x)
                    loss = -mll(output, tr_y)
                    loss.backward()
                    print(z,'mll_tr_loss',loss.item()) #Maximum Likelihood loss
                    with torch.no_grad():
                        model.eval()
                        tr_score = r2(tr_y.cpu().numpy(),model(tr_x).mean.cpu().numpy())
                        val_score = r2(val_y.cpu().numpy(),model(val_x).mean.cpu().numpy())
                        test_score = r2(test_y.cpu().numpy(),model(test_x).mean.cpu().numpy())
                        mse_tr = mse(tr_y.cpu().numpy(),model(tr_x).mean.cpu().numpy())
                        mse_val = mse(val_y.cpu().numpy(),model(val_x).mean.cpu().numpy())
                        mse_all = (mse_tr*len(tr_y)+mse_val*len(val_y))/(len(tr_y)+len(val_y))
                        print(z,'mse_tr_loss',mse_tr,'mse_val_loss',mse_val)  #MSE loss for training and validation
                        my_score.append([seed_val,j,z,tr_score,val_score,test_score])
                        my_loss.append([seed_val,j,z,mse_tr,mse_val])
                        if (mse_all<opt_mse_all) and (val_score>val_score_prev) and (tr_score>val_score):
                            opt_mse_all = mse_all
                            val_score_prev = val_score
                            torch.save(model.state_dict(), './YM_model') #storing stat dictionary of best model
                            import pickle
                            f = open('YM_seed', 'wb')
                            pickle.dump([seed_val,j],f) #storing seed value and K-fold value(j) for best model

                    with open('./my_score.csv', 'w', newline='') as myfile: #dumping score values in csv file
                        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
                        wr.writerows(my_score)

                    with open('./my_loss.csv', 'w', newline='') as mfile: #dumping loss values in csv file
                        wr = csv.writer(mfile, quoting=csv.QUOTE_ALL)
                        wr.writerows(my_loss)

                    optimizer.step()
                    model.train()
                    likelihood.train()

            # Sometimes we get better performance on the GPU when we don't use Toeplitz math
            # for SKI. This flag controls that
            with gpytorch.settings.use_toeplitz(False):
              train()
