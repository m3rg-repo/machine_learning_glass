import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from torch.utils.data import TensorDataset, DataLoader
from shadow.plot import panel,xlabel,ylabel,legend_on,set_things

class DropoutNormNet(nn.Module):
    def __init__(self,D_in,D_out,layers=[1],dropout_rate=[0.2],batch_norm=True,dropout=True):
        super(DropoutNormNet, self).__init__()
        if len(layers)!=len(dropout_rate):
            dropout_rate = dropout_rate*len(layers)
        self.seq = nn.Sequential()
        for a,b,p,n in zip([D_in]+layers[:-1],layers,dropout_rate,range(1+len(layers))):
            self.seq.add_module("L{}".format(n),nn.Linear(a,b))
            if dropout:
                self.seq.add_module("D{}".format(n),nn.Dropout(p=p))
            self.seq.add_module("A{}".format(n),nn.ReLU())
            if batch_norm:
                self.seq.add_module("BN{}".format(n),nn.BatchNorm1d(b))
        self.seq.add_module("LN",nn.Linear(layers[-1],D_out))

    def forward(self,x):
        return self.seq(x)

def split(X,y,test_size=0.2):
    return [torch.tensor(i).float() for i in train_test_split(X,y,test_size=0.2)]

class DoML():
    def __init__(self,):
        self.model = None
        self.data = None
        self.loss = None
        self.optimizer = None
        self.dataloader_params = {"batch_size":10}

    def set_data(self,data):
        self.Xtrain, self.Xval, self.ytrain, self.yval = data
        self.ytrain = self.ytrain.reshape(-1,1)
        self.yval = self.yval.reshape(-1,1)

    def training(self):
        loss_ = []
        self.model.train()
        dataset = TensorDataset(self.Xtrain,self.ytrain)
        loader = DataLoader(dataset,**self.dataloader_params)
        for batch_idx, data in enumerate(loader):
            x,y = data
            self.optimizer.zero_grad()
            l = self.loss(y,self.model(x).view(y.size())) #.sum()/self.dataloader_params["batch_size"]
            loss_ += [l.item()]
            l.backward()
            self.optimizer.step()
        self.model.eval()
        return np.mean(loss_)

    def cal_loss_val(self):
        self.model.eval()
        l = self.loss(self.yval,self.model(self.Xval).view(self.yval.size()))
        return l.item()

    def cal_loss_train(self):
        self.model.eval()
        l = self.loss(self.ytrain,self.model(self.Xtrain).view(self.ytrain.size()))
        return l.item()

    def train(self,N=100,n_print=10):
        print(self.model)
        self.train_loss, self.val_loss = [], []
        for epoch in range(N):
            self.training()
            self.train_loss += [self.cal_loss_train()]
            self.val_loss += [self.cal_loss_val()]
            if epoch%n_print==0:
                print("Epoch: ", epoch, "Training loss: ", self.train_loss[-1], "Validation loss: ", self.val_loss[-1])
                self.cal_R2()

        self.model.eval()

    def plot_loss(self,clip0=0,clip1=-1):
        fig, [ax] = panel(1,1)
        ax.plot(range(len(self.train_loss[clip0:clip1])),self.train_loss[clip0:clip1],label="Training")
        ax.plot(range(len(self.val_loss[clip0:clip1])),self.val_loss[clip0:clip1],label="Validation")
        xlabel("Epoch")
        ylabel("MSE Loss")
        legend_on(ax)

    def cal_R2(self,):
        self.model.eval()
        self.R2_val = r2_score(self.yval,self.model(self.Xval).detach().numpy().reshape(self.yval.shape))
        self.R2_train = r2_score(self.ytrain,self.model(self.Xtrain).detach().numpy().reshape(self.ytrain.shape))
        print("Score(val): ",self.R2_val)
        print("Score(train): ",self.R2_train)



#
