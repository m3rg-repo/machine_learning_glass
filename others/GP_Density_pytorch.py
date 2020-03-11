import math
import torch
import gpytorch

import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import r2_score as r2
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_log_error as MSLE

ran = np.int(np.random.random()*1000000)
data = (pd.read_csv('Density_gp.csv').values)
X = data[:,:-1]
X = X - X.min(0)[0]
X = 2 * (X / X.max(0)[0]) - 1
y = data[:,-1]


global opt_loss, opt_model,opt_mll,seed_val

seed_val = ran
torch.manual_seed(seed_val)


with gpytorch.settings.use_toeplitz(False), gpytorch.settings.fast_pred_var(), gpytorch.settings.max_root_decomposition_size(100):


    # Use the first 70% of the data for training, and the last 30% for testing.
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.30, random_state=seed_val)

    train_x = torch.Tensor(train_x)
    train_y = torch.Tensor(train_y)
    test_x = torch.Tensor(test_x)
    test_y = torch.Tensor(test_y)

    data_dim = train_x.size(-1)

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
    model = GPRegressionModel(train_x, train_y, likelihood)

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
    ], lr=0.01)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    opt_loss = 100
    opt_mse_all=1000000
    opt_MAE_all=1000000
    opt_MSLE_all=1000000
    def train(training_iterations=300):
      global opt_loss,opt_model,eval_model,opt_mse_all,opt_eval_model,opt_MAE_all,opt_MSLE_all
      for i in range(training_iterations):
          optimizer.zero_grad()
          output = model(train_x)
          loss = -mll(output, train_y)
          loss.backward()
          if loss<opt_loss:
            opt_loss = loss
            opt_model = model
            eval_model = model

            with torch.no_grad():
              eval_model.eval()
              train_score = r2(train_y.cpu().numpy(),eval_model(train_x).mean.cpu().numpy())
              test_score = r2(test_y.cpu().numpy(),eval_model(test_x).mean.cpu().numpy())
              mse1 = mse(train_y.cpu().numpy(),eval_model(train_x).mean.cpu().numpy())
              mse2 = mse(test_y.cpu().numpy(),eval_model(test_x).mean.cpu().numpy())
              mse_all = (mse1*len(train_y)+mse2*len(test_y))/(len(train_y)+len(test_y))
              torch.save(model.state_dict(), './eval_model')
              import pickle
              f = open('seed', 'wb')
              pickle.dump([seed_val,i],f)
              if (mse_all<opt_mse_all) and (train_score>test_score):
                  opt_mse_all = mse_all
                  opt_eval_model = model
                  torch.save(model.state_dict(), './eval_model_mse')
                  import pickle
                  f = open('seed_mse', 'wb')
                  pickle.dump([seed_val,i],f)

              MAE1 = MAE(train_y.cpu().numpy(),eval_model(train_x).mean.cpu().numpy())
              MAE2 = MAE(test_y.cpu().numpy(),eval_model(test_x).mean.cpu().numpy())
              MAE_all = (MAE1*len(train_y)+MAE2*len(test_y))/(len(train_y)+len(test_y))
              if (MAE_all<opt_MAE_all) and (train_score>test_score):
                 opt_MAE_all = MAE_all
                 opt_eval_model = model
                 torch.save(model.state_dict(), './eval_model_MAE')
                 import pickle
                 f = open('seed_MAE', 'wb')
                 pickle.dump([seed_val,i],f)

              MSLE1 = MSLE(train_y.cpu().numpy(),eval_model(train_x).mean.cpu().numpy())
              MSLE2 = MSLE(test_y.cpu().numpy(),eval_model(test_x).mean.cpu().numpy())
              MSLE_all = (MSLE1*len(train_y)+MSLE2*len(test_y))/(len(train_y)+len(test_y))
              if (MSLE_all<opt_MSLE_all) and (train_score>test_score):
                opt_MSLE_all = MSLE_all
                opt_eval_model = model
                torch.save(model.state_dict(), './eval_model_MSLE')
                import pickle
                f = open('seed_MSLE', 'wb')
                pickle.dump([seed_val,i],f)


          optimizer.step()
          model.train()
          likelihood.train()

    # Sometimes we get better performance on the GPU when we don't use Toeplitz math
    # for SKI. This flag controls that
    with gpytorch.settings.use_toeplitz(False):
      train()
