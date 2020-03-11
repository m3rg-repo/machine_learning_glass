import pandas as pd
import numpy as np
import pickle

import sys
sys.path.append('/Users/sureshbishnoi/Owncloud/python_packages')

from shadow.plot import *
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import KFold
import xgboost as xgb



class sess():
    def __init__(self,):
        self.features = None
        self.properties = None
        self.data = None
        self.r2_score_test = 0
        self.r2_score_train = 0
        self.optim_model = None
        self.seed = 0

    def data_loader(self,features=None, properties=['Density at RT'],dropna=True):

        X = pickle.load(open('/Users/sureshbishnoi/Owncloud/Shared/Machine_learning/database/compositions.pkl2','rb'))
        X.columns = np.array(X.columns).astype(np.int)
        features1 = True
        if features==None:
            features = list(X.columns[0:231])
            features1 = None

        X.index = X.index.values.astype(np.int)
        X.insert(loc=0, column='g_ids', value=X.index.values)

        first = True
        count = 0
        for p in properties:
            y = pickle.load(open('/Users/sureshbishnoi/Owncloud/Shared/Machine_learning/database/Properties/{}.pickle'.format(p),'rb'))
            y = pd.DataFrame(np.vstack([y['g_ids'],y['prop_val']]).T,columns=['g_ids',p]).dropna().astype(np.float)
            if first:
                Y = y.copy()
                count+=1
                first = False
            else:
                Y = Y.merge(y,on='g_ids',how='outer')
                count+=1
        print('properties: ', Y.shape)

        data = X.merge(Y,on='g_ids',how='inner')

        if features1==None:
            data = data.iloc[:,np.r_[0:232,-1]]
            mask = (data[[1,2,3,5,6,8,9,40,58,61,81]]).values.max(axis=1)>10
            data = data[mask]

        mask = data[features].values.sum(axis=1)>98
        data = data[mask]
        mask = data[features].values.sum(axis=1)<101
        data = data[mask]


        data = data.groupby(features,as_index=False).agg(self.mean_dup)

        if dropna:
            data = data.dropna()

        self.data = data
        self.features_name = features
        self.features = data[features].values.copy()
        self.properties = data[properties].values.copy()

        print('Total examples: ',len(data))
        print('Features: ',self.features.shape)
        print('Properties: ',self.properties.shape)
        print('Do NaN exist?: ', ~dropna)




    def train(self,model=None,rn_iter=10,xg_model = False,*args,**kwargs):
        self.r2_score_test = 0
        self.r2_score_train = 0
        self.mse_previous = 1000000000000
        self.optim_model = None
        if 1:
            for i in range(rn_iter):
                np.random.seed(1+i*1231)
                if xg_model:
                    self.model = xgb.XGBRegressor(objective="reg:linear")
                else:
                    self.model = model(*args,**kwargs)

                self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.features, self.properties, test_size=0.2,)
                kf = KFold(n_splits=5,shuffle=True, random_state = np.random.seed(1+i*1231))
                for train_index, val_index in kf.split(self.x_train):
                    self.x_tr, self.x_val = self.x_train[train_index], self.x_train[val_index]
                    self.y_tr, self.y_val = self.y_train[train_index], self.y_train[val_index]
                    self.model.fit(self.x_tr,self.y_tr)
                    self.r2_score_tr = self.model.score(self.x_tr,self.y_tr)
                    self.r2_score_val = self.model.score(self.x_val,self.y_val)
                    self.mse_train = mse(self.model.predict(self.x_train),self.y_train)
                    self.mse_tr = mse(self.model.predict(self.x_tr),self.y_tr)
                    self.mse_val = mse(self.model.predict(self.x_val),self.y_val)
                    if (self.mse_previous > self.mse_train) and (self.mse_tr < self.mse_val):
                        self.mse_previous = self.mse_train
                        self.optim_model = self.model
                        # self.x_tr_optim = self.x_tr
                        # self.x_val_optim = self.x_val
                        self.x_train_optim = self.x_train
                        self.x_test_optim = self.x_test
                        # self.y_tr_optim = self.y_tr
                        # self.y_val_optim = self.y_val
                        self.y_train_optim = self.y_train
                        self.y_test_optim = self.y_test
                        self.seed = 1+i*1231

            self.r2_score_test = self.optim_model.score(self.x_test_optim,self.y_test_optim)
            self.r2_score_train = self.optim_model.score(self.x_train_optim,self.y_train_optim)
            self.y_train_pred = self.optim_model.predict(self.x_train_optim)
            self.y_test_pred = self.optim_model.predict(self.x_test_optim)
            if not xg_model:
                print('{} Successfully trained.'.format(model.__name__))
            print(' Test: {}\n Train: {}'.format(self.r2_score_test, self.r2_score_train))

        else:
            pass


    def plot(self,):
        fig, [ax] = panel(1,1,dpi=100,r_p=0.12,l_p=0.12,b_p=0.1,t_p=0.05)
        y_train,y_train_pred,y_test,y_test_pred = self.y_train_optim,self.y_train_pred,self.y_test_optim,self.y_test_pred

        ax.plot(y_train,y_train_pred,'ob', label='Training (R$^2={:0.2f}$)'.format(self.r2_score_train),mec='g')
        ax.plot(y_test,y_test_pred,'sr',label='Test (R$^2={:0.2f}$)'.format(self.r2_score_test),mec='g')
        min_ = min((y_train.min()),(y_test.min()))-order(y_train.max())
        max_ = max((y_train.max()),(y_test.max()))+order(y_train.max())
        ax.plot([min_,max_],[min_,max_],':k')
        xlim([min_,max_])
        ylim([min_,max_])
        xlabel('Measured values')
        ylabel('Predicted values')
        legend_on(ax=ax,loc=4)
        plt.savefig('Y_pr vs Y_ac.png')

    def plot1(self, label = "Young's modulus (GPa)",inter = 20,range = [0,200],set_range = False,model_name = 'LinearRegression'):
        import matplotlib.ticker as plticker
        fig, [ax] = panel(1,1,dpi=500,r_p=0.12,l_p=0.16,b_p=0.1,t_p=0.05)
        y_train,y_train_pred,y_test,y_test_pred = self.y_train_optim,self.y_train_pred,self.y_test_optim,self.y_test_pred

        # ax.plot(y_train,y_train_pred,'ob', label='Training (R$^2={:0.2f}$)'.format(self.r2_score_train),mec='g')
        ax.plot(y_test,y_test_pred,'sr',mec = 'k')
        ax.plot([], [], 'k',ls='none', mew=0,label='Training (R$^2$={:.2f})'.format(self.r2_score_train),alpha=0.1)
        ax.plot([], [], 'k',ls='none', mew=0,label='Test (R$^2$={:.2f})'.format(self.r2_score_test),alpha=0.1)

        min_ = min((y_train.min()),(y_test.min()))-order(y_train.max())
        max_ = max((y_train.max()),(y_test.max()))+order(y_train.max())
        rng = [min_,max_]
        if set_range:
            rng = range
        ax.plot(rng,rng,':k')
        loc = plticker.MultipleLocator(base=inter)
        ax.xaxis.set_major_locator(loc)
        ax.yaxis.set_major_locator(loc)
        xlim(rng)
        ylim(rng)
        xlabel('Measured '+label)
        ylabel('Predicted '+label)
        legend_on(ax=ax,loc=4)
        plt.savefig(model_name+'_Y_pr vs Y_ac.png')

    # def predict(self,x):
    #     return self.model.predict(x)

    def plot2(self, s=70,den_scale_bool=False,den_scale = 1,inter = 20,label="Young's modulus *(GPa)",range = [0,200],set_range=False,model_name = 'LinearRegression'):
        import matplotlib.ticker as plticker
        fig, [ax] = panel(1,1,dpi=500,r_p=0.12,l_p=0.16,b_p=0.1,t_p=0.05)
        y_train_pred,y_train,y_test_pred,y_test = self.y_train_optim,self.y_train_pred,self.y_test_optim,self.y_test_pred
        scale = 1
        min_ = min((y_train.min()),(y_test.min()))-order(y_train.max())
        max_ = max((y_train.max()),(y_test.max()))+order(y_train.max())
        rng = [min_,max_]
        if set_range:
            rng = range
        ax.plot([], [], 'k',ls='none', mew=0,label='Training (R$^2$={:.2f})'.format(self.r2_score_train),alpha=0.1)
        ax.plot([], [], 'k',ls='none', mew=0,label='Test (R$^2$={:.2f})'.format(self.r2_score_test),alpha=0.1)

        # im_data, xedges, yedges= np.histogram2d(y_train.ravel()/scale, y_train_pred.ravel()/scale, bins=(s,s), range=np.array([rng,rng]), density=1)
        im_data, xedges, yedges= np.histogram2d(y_test.ravel()/scale, y_test_pred.ravel()/scale, bins=(s,s), range=np.array([rng,rng]), density=1)

        # im_data += im_data2

        if den_scale_bool:
            mask = im_data > (im_data.mean()+den_scale*im_data.std())
            im_data[mask] = im_data.mean()+den_scale*im_data.std()

        im_data /= im_data.max()

        im_data = im_data[::-1,:]

        from matplotlib import cm
        from matplotlib.colors import ListedColormap
        ocean = cm.get_cmap('gist_heat', 256)
        c_data = ocean(np.linspace(0, 1, 256))
        mycm = ListedColormap(c_data[::-1,:])

        cb = ax.imshow(im_data, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap = mycm)

        xlim(rng)
        ylim(rng)
        ax.plot(rng,rng,':',c='gray')
        xlabel('Measured '+label.replace('*',''))
        ylabel('Predicted '+label.replace('*',''))
        loc = plticker.MultipleLocator(base=inter)
        ax.xaxis.set_major_locator(loc)
        ax.yaxis.set_major_locator(loc)
        legend_on(loc=4)

        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(cb, cax=cax,)

        e = (y_test.ravel()-y_test_pred.ravel())/scale

        std = e.std()

        ins_ax = inset([0.03,0.67,0.3,0.3], ax=ax)

        plt.sca(ins_ax)

        _ = ins_ax.hist(e,bins=100,fc='k', density=True)


        rectangle(-1.65*std,1.65*std,0,_[0].max()*1.2,ax=ins_ax,color='r')

        ins_ax.yaxis.set_label_position("right")
        ins_ax.yaxis.tick_right()

        xlim([-5*std,5*std])

        ylabel('PDF')
        xlabel(r'$\varepsilon$ '+label.split('*')[-1])
        plt.savefig(model_name+'_Y_ac vs Y_pr_with_pdf.png')


    def reject_outliers(self,data, m = 2.):
        d = np.abs(data - np.median(data))
        mdev = np.median(d)
        s = d/mdev if mdev else 0.
        return (s<m)

    def mean_dup(self,x_):
        global reject_outliers
        if 1==len(np.unique(x_.values)):
            return x_.values[0]
        else:
            x = x_.values[self.reject_outliers(x_.values.copy())]
            x_mean = x.mean()
            mask = (x_mean*0.975 <= x) & (x <= x_mean*1.025)
            return x[mask].mean()

    def sudo_clean(self,):
        self.data = None
        self.r2_score_test = 0
        self.r2_score_train = 0
        self.optim_model = None
        self.seed = 0

    def clean(self,):
        self.sudo_clean()
        self.features = None
        self.properties = None



def order(num):
    num=num+50
    for i,j in zip([1,10,100,1000,10000,100000],[0.1,1,10,100,1000,10000]):
        if int(num/i) < 10 and int(num/i) >= 1:
            return j
