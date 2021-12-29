import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

from ml import MLmodule
from plot import *
import importlib

S = MLmodule.sess

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
opf=pd.read_csv("Density_gp.csv")

X = opf[opf.columns[0:-1]].values
y = opf[opf.columns[-1]].values
X.shape

### histogram for Density value vs frequence
plt.hist(y,bins=20)
fig,axs = panel(1,1,dpi=500,r_p=0.05,l_p=0.15,b_p=0.1,t_p=0.05)
l = axs[0].hist(y,fc='r',ec='k',bins = list(np.arange(1.5,9,.5)))
xlabel("Density (g/cm$^3$)")
ylabel('Frequency')
xlim([0,9])
ylim([0,7000])
# yticks(np.arange(0, 1400, 200))
xticks(np.arange(0,9.1,1))
plot.add_value_labels(axs[0],precision=0)
plt.savefig('Density_GP_cleaned_hist.png')

### histogram for nonzero composition vs frequence
plt.hist(sum((X!=0).T))
fig,axs = panel(1,1,dpi=500,r_p=0.05,l_p=0.15,b_p=0.1,t_p=0.05)
l = axs[0].hist(sum((X !=0).T),fc='r',ec='k',bins = list(np.arange(.5,16,1)),label='Number of component')
xlabel("Number of component")
ylabel('Frequency')
xlim([0,16])
ylim([0,9000])
yticks(np.arange(0, 9001, 1000))
axs[0].minorticks_off()
xticks(np.arange(0,16.1,1))
plot.add_value_labels(axs[0],precision=0)
plt.savefig('Density_GP_cleaned_hist_NoOfComponent.png')

##data_visulation ends
################################################################################
###Regressian analysis starts

T2 = S()
T2.features = X
T2.properties = y

T2.train(model = LinearRegression,rn_iter = 30)
T2.plot1(label = "density (g/cm$^3$)",inter = 2,range = [0,10],set_range = True,model_name = 'LinearRegression')

T2.plot2(label = "density *(g/cm$^3$)",inter = 2,range = [0,10],set_range = True,den_scale_bool=True,den_scale = 2,model_name = 'LinearRegression')

###1. Neural network Training
def do_nn_for_Hl_vs_R2(HL_neurons = range(3,4,1)):
    ## finding the optimum neurons and hl
    score_ttl = []
    train_score = []
    test_score = []
    for j in HL_neurons:
        k = 0
        opt_score = -90
        for i in ([j],[j,j],[j,j,j],[j,j,j,j]):
            T2.train(model=nn,rn_iter=30 ,max_iter=1000,
                    learning_rate_init=0.0005, solver='lbfgs',
                    hidden_layer_sizes=i)
            score = T2.optim_model.score(T2.features,T2.properties)
            if score>opt_score and k!=4:
                opt_score = score
                opt_train_score = T2.r2_score_train
                opt_test_score = T2.r2_score_test
                m = i

            k = k+1
            if k==4:
                print('i am in')
                score_ttl.append(opt_score)
                train_score.append(opt_train_score)
                test_score.append(opt_test_score)
                print(m)

    ## optimum neurons vs  score plot
    m = HL_neurons
    fig,[axs] = panel(1,1,dpi=100,r_p=0.12,l_p=0.15,b_p=0.1,t_p=0.05)
    ax2,ax = twinx(axs)
    l1 = ax.plot(m,train_score,'sb',label=r'\textbf{Training}')
    ax.plot(m,train_score,'-b')
    xlabel('Number of neurons',ax=axs)
    ylabel('R$^{2}$ Training',ax=ax,color = 'b')
    l2 = ax2.plot(m,test_score,'^r',label=r'\textbf{Test}')
    ax2.plot(m,test_score,'-r')
    ylabel('R$^{2}$ Test',ax=ax2, color = 'r')
    ax.set_ylim([min(min(ax.get_ylim()),min(ax2.get_ylim())),max(max(ax.get_ylim()),max(ax2.get_ylim()))])
    ax2.set_ylim([min(min(ax.get_ylim()),min(ax2.get_ylim())),max(max(ax.get_ylim()),max(ax2.get_ylim()))])
    ls = l1+l2
    l_names = [l.get_label() for l in ls]
    plt.legend(ls,l_names,loc=4,frameon=False)

do_nn_for_Hl_vs_R2(range(1,8,2))

T2.train(model=nn,rn_iter=30 ,max_iter=1000,
        learning_rate_init=.001, solver='lbfgs',
        hidden_layer_sizes=[5,5])
T2.plot()

###2. Polynomial regressian
def polynomial_regression(degrees = range(2,6,1)):

    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import make_pipeline

    np.random.seed(12354678)
    x_train, x_test, y_train, y_test = train_test_split(T2.features, T2.properties,test_size=0.3)
    # Polynomial Regression-nth order

    r2_train = []
    r2_test = []
    mse_train = []
    mse_test =[]
    for degree in degrees:
        model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        model.fit(x_train,y_train)
        y_plot = model.predict(x_test)

        r2_train.append(model.score(x_train, y_train))
        r2_test.append(model.score(x_test, y_test))
        mse_train.append(mse(y_train,model.predict(x_train)))
        mse_test.append(mse(y_test,model.predict(x_test)))

        fig, [ax] = panel(1,1)

        ax.plot(y_train,model.predict(x_train),'ob', label="degree %d" % degree +' ; Training (R$^2={:0.2f}$)'.format(model.score(x_train, y_train),mec='g'))
        ax.plot(y_test,model.predict(x_test),'sr',label="degree %d" % degree + ' ; Test (R$^2={:0.2f}$)'.format(model.score(x_test, y_test),mec='g'))
        min_ = min(min(y_train),min(y_test))-1
        max_ = max(max(y_train),max(y_test))+1
        ax.plot([min_,max_],[min_,max_],':k')
        # xlim([min_,max_])
        # ylim([min_,max_])
        xlabel('Actual values')
        ylabel('Predicted values')
        legend_on(ax=ax,loc=4)



    fig,[axs] = panel(1,1,dpi=500,r_p=0.12,l_p=0.12,b_p=0.1,t_p=0.05)
    ax2,ax = twinx(axs)
    min_ = min(min(mse_train),min(mse_test))-10
    max_ = max(max(mse_train),max(mse_test))+10


    l1 = ax.plot([i for i in degrees],mse_train,'^b',label=r'\textbf{MSE (Train)}')
    ax.plot([i for i in degrees],mse_train,'-b')
    xlabel('Polynomial order',ax=axs)
    ylabel('MSE Training(g/cm$^3$)',ax=axs,color='b')
    ax.set_ylim(min_,max_)


    l2 = ax2.plot([i for i in degrees],mse_test,'sr',label=r'\textbf{MSE (Test)}')
    ax2.plot([i for i in degrees],mse_test,'-r')
    ylabel('MSE Test(g/cm$^3$)',ax=ax2,color='r')
    ax2.set_ylim(min_,max_)
    return r2_train,r2_test,mse_train,mse_test

polynomial_regression(degrees = range(1,3,1))

###3. Gaussian process regressian
def GPR():

    def save_this(seed_val,gpr,X_train,X_test,y_train,y_test):

        optim_kernel_ = gpr.kernel_
        optim_alpha_ = gpr.alpha_
        y_train_mean = y_train.mean()

        X_train_ = X_train
        X_test_ = X_test
        y_train_ = y_train
        y_test_ = y_test


        model = [optim_kernel_,optim_alpha_,y_train_mean,seed_val]
        data = [X_train_,X_test_,y_train_,y_test_]


        f = open('my_model_{}'.format(nproc),'wb')
        g = open('my_data_{}'.format(nproc),'wb')
        import pickle

        pickle.dump(model,f)
        pickle.dump(data,g)

        f.close()
        g.close()




    def do_this():

        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=None)

        kernel = 1* RBF([10]*X_train.shape[1], (1e-5,1e5)) + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1))

        gpr= GaussianProcessRegressor(kernel=kernel,alpha=1.0e-5, n_restarts_optimizer=RESTARTS,normalize_y=True,optimizer='fmin_l_bfgs_b')

        gpr.fit(X_train, y_train)

        y_pred = gpr.predict(X)

        mse_all = mse(y,y_pred)
        R_test = gpr.score(X_test,y_test)
        R_tr = gpr.score(X_train,y_train)
        return mse_all, R_test, R_tr, gpr, X_train, X_test, y_train, y_test
    mse_all = 48564548

    _R_test = -100
    loops = LOOPS
    nproc = 982
    for i in range(loops*nproc,loops+loops*nproc):
        seed_val = 99*i*2+1
        np.random.seed(seed_val)
        mse_all, R_test, R_tr, gpr, X_train, X_test, y_train, y_test = do_this()
        if mse_all<mse_pr and (R_tr > R_test):
            mse_pr = mse_all
            print(mse_all, R_test, R_tr)
            gpr_opt = gpr
