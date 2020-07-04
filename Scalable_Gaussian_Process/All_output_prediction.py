## Creating Class and loading of the model
import json
all_name = json.load(open("./read_me.json",'r'))
for z_name in ["BM","Density","H","LT","SM","TG","YM","TEC","RI"]:

    prop_name = all_name[z_name][0]['short_name']
    prop_label = all_name[z_name][1]['label_name']
    prop_scale = all_name[z_name][2]['prop_scale']
    Fig1_xlim = all_name[z_name][3]['Fig1'][0]['x_lim']
    Fig1_ylim = all_name[z_name][3]['Fig1'][1]['y_lim']
    Fig1_xticks = all_name[z_name][3]['Fig1'][2]['ticks_x']
    Fig1_bins = all_name[z_name][3]['Fig1'][3]['bins']
    Fig2_xlim = all_name[z_name][4]['Fig2'][0]['x_lim']
    Fig2_ylim = all_name[z_name][4]['Fig2'][1]['y_lim']
    Fig2_xticks = all_name[z_name][4]['Fig2'][2]['ticks_x']
    Fig3_margin = all_name[z_name][5]['Fig3'][0]['margin']
    Fig3_inter = all_name[z_name][5]['Fig3'][1]['inter']
    den_scale_bool = all_name[z_name][5]['Fig3'][2]['den_scale_bool']
    den_scale = all_name[z_name][5]['Fig3'][3]['den_scale']



    import sys
    sys.path.append('/Users/sureshbishnoi/Owncloud/python_packages')
    %matplotlib inline
    from matplotlib import pyplot as plt
    from ml import MLmodule
    from shadow.plot import *
    from shadow import plot
    import importlib
    importlib.reload(plot)
    importlib.reload(MLmodule)
    set_things()
    linestyles()

    import warnings
    warnings.filterwarnings("ignore")


    import math
    import torch
    import gpytorch

    import pandas as pd
    import numpy as np
    from sklearn.model_selection import KFold
    from sklearn.metrics import r2_score as r2
    from sklearn.model_selection import train_test_split


    data = pd.read_csv(prop_name+'_gp.csv').values
    X = data[:,:-1]
    X = X - 0
    X = 2 * (X / 100) - 1
    y = data[:,-1]

    ### histogram for BM value vs frequence
    # plt.hist(y,bins=20)
    fig,axs = panel(1,1,dpi=50,r_p=0.05,l_p=0.15,b_p=0.1,t_p=0.05)
    l = axs[0].hist(y/prop_scale,fc='r',ec='k',bins = Fig1_bins)#list(np.arange(10,111,10)))
    xlabel(prop_label.replace('*',''))
    ylabel('Frequency')
    xlim(Fig1_xlim)
    ylim(Fig1_ylim)
    # yticks(np.arange(0, 1400, 200))

    xticks(Fig1_xticks)
    # plot.add_value_labels(axs[0],precision=0)
    plt.savefig('./Result_plots/Fig1/'+prop_name+'_prop_vs_freq_hist.png',dpi=50)
    plt.show()


    ### histogram for nonzero composition vs frequence
    # plt.hist(sum(((data[:,:-1])!=0).T))
    fig,axs = panel(1,1,dpi=50,r_p=0.05,l_p=0.15,b_p=0.1,t_p=0.05)
    l = axs[0].hist(sum(((data[:,:-1]) !=0).T),fc='r',ec='k',bins = list(np.arange(.5,11,1)),label='Number of component')
    xlabel("Number of component")
    ylabel('Frequency')
    xlim(Fig2_xlim)
    ylim(Fig2_ylim)
    # yticks(np.arange(0, 1400, 200))
    axs[0].minorticks_off()
    xticks(Fig2_xticks)
    # plot.add_value_labels(axs[0],precision=0)
    plt.savefig('./Result_plots/Fig2/'+prop_name+'_#comp_vs_freq_hist.png',dpi=50)
    plt.show()

    import pickle
    with open(prop_name+'_mse','rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        opt_seed = u.load()

    seed_val = opt_seed[0]
    fold_val = opt_seed[1]
    torch.manual_seed(seed_val)


    with torch.no_grad(), gpytorch.settings.use_toeplitz(False), gpytorch.settings.fast_pred_var(), gpytorch.settings.max_root_decomposition_size(100):

        train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.15, random_state=seed_val)
        train_x = torch.Tensor(train_x)
        train_y = torch.Tensor(train_y)
        test_x = torch.Tensor(test_x)
        test_y = torch.Tensor(test_y)
        kf = KFold(n_splits=5,shuffle=True, random_state=seed_val)
        j=0
        for train_index, val_index in kf.split(train_x):
            j+=1
            if j==fold_val:
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

        ## Reading the saved model by defining TheModelClass and then load the model on taht class
        device = torch.device('cpu')    ## model loaded on cpu but it was saved on gpu
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        my_model = GPRegressionModel(tr_x, tr_y, likelihood)   ## TheModelClass
        my_model.load_state_dict(torch.load(prop_name+'_model',map_location='cpu'))   ## loded the saved model
        my_model.eval()   ## evaluation the loaded model


    # =====================================================================================================
    ## Predicting the Bulk modulus


    def den_plot(ax,y_pr_tr,y_pr_val,y_pr_test,y_tr,y_val,y_test,r2_tr,r2_val,r2_test,rng,s=70,den_scale_bool=False):
        import matplotlib.ticker as plticker
        scale = 1
        ax.plot([], [], 'k',ls='none', mew=0,label='Training (R$^2$={:.2f})'.format(r2_tr),alpha=0.1)
        ax.plot([], [], 'k',ls='none', mew=0,label='Validation (R$^2$={:.2f})'.format(r2_val),alpha=0.1)
        ax.plot([], [], 'k',ls='none', mew=0,label='Test (R$^2$={:.2f})'.format(r2_test),alpha=0.1)

        im_data, xedges, yedges= np.histogram2d(y_tr.ravel()/scale, y_pr_tr.ravel()/scale, bins=(s,s), range=np.array([rng,rng]), density=1)
        im_data1, xedges1, yedges1= np.histogram2d(y_val.ravel()/scale, y_pr_val.ravel()/scale, bins=(s,s), range=np.array([rng,rng]), density=1)
        im_data2, xedges2, yedges2= np.histogram2d(y_test.ravel()/scale, y_pr_test.ravel()/scale, bins=(s,s), range=np.array([rng,rng]), density=1)

        im_data = im_data+im_data1+im_data2

        if den_scale_bool:
            mask = im_data > (im_data.mean()+den_scale*im_data.std())
            im_data[mask] = im_data.mean()+den_scale*im_data.std()

        im_data /= im_data.max()

        # im_data = im_data[::-1,:]
        from matplotlib import cm
        from matplotlib.colors import ListedColormap
        ocean = cm.get_cmap('gist_heat', 256)
        c_data = ocean(np.linspace(0, 1, 256))
        mycm = ListedColormap(c_data[::-1,:])
        cb = ax.imshow(im_data, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap = mycm)

        xlim(rng)
        ylim(rng)
        ax.plot(rng,rng,':',c='gray')
        # xticks(Fig1_xticks)
        # yticks(Fig1_xticks)

        xlabel('Measured '+prop_label.replace('*',''))
        ylabel('Predicted '+prop_label.replace('*',''))
        loc = plticker.MultipleLocator(base=inter)
        ax.xaxis.set_major_locator(loc)
        ax.yaxis.set_major_locator(loc)
        legend_on(loc=4)

        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(cb, cax=cax,)

        e = (y_tr.ravel()-y_pr_tr.ravel())/scale
        std = e.std()
        ins_ax = inset([0.03,0.67,0.3,0.3], ax=ax)
        plt.sca(ins_ax)
        _ = ins_ax.hist(e,bins=100,fc='k', density=True)

        rectangle(-1.65*std,1.65*std,0,_[0].max()*1.2,ax=ins_ax,color='r')

        ins_ax.yaxis.set_label_position("right")
        ins_ax.yaxis.tick_right()

        xlim([-5*std,5*std])

        ylabel('PDF')
        xlabel(r'$\varepsilon$ '+prop_label.split(' ')[-1])


    with torch.no_grad(), gpytorch.settings.use_toeplitz(False), gpytorch.settings.fast_pred_var(), gpytorch.settings.max_root_decomposition_size(100):

        # ## predict and ploat the graphs
        y_tr_pr = my_model(tr_x).mean.cpu().numpy()/prop_scale
        y_val_pr = my_model(val_x).mean.cpu().numpy()/prop_scale
        y_test_pr = my_model(test_x).mean.cpu().numpy()/prop_scale
        tr_y=tr_y/prop_scale
        val_y=val_y/prop_scale
        test_y=test_y/prop_scale
        fig,[ax] = panel(1,1,dpi=50,r_p=0.12,l_p=0.12,b_p=0.1,t_p=0.05)
        ax.plot(tr_y.cpu().numpy(),y_tr_pr,'ob', label= 'R$^2$(training={:0.2f})'.format(r2(tr_y.cpu().numpy(),y_tr_pr)),mec='k')
        ax.plot(val_y.cpu().numpy(),y_val_pr,'og', label= 'R$^2$(Validation={:0.2f})'.format(r2(val_y.cpu().numpy(),y_val_pr)),mec='k')
        ax.plot(test_y.cpu().numpy(),y_test_pr,'sr',label='R$^2$(test={:0.2f})'.format(r2(test_y.cpu().numpy(),y_test_pr)),mec='k')
        min_ = min(y/prop_scale)-Fig3_margin
        max_ = max(y/prop_scale)+Fig3_margin
        ax.plot([min_,max_],[min_,max_],':k')

        xlim([min_,max_])
        ylim([min_,max_])

        # xticks(Fig1_xticks)
        # yticks(Fig1_xticks)

        xlabel('Measured '+prop_label.replace('*',''))
        ylabel('Predicted '+prop_label.replace('*',''))
        legend_on(ax=ax,loc=2)
        plt.savefig('./Result_plots/Fig0/'+prop_name+'_Ypr_vs_Yac_mse_RBF.png',dpi=50)
        plt.show()

        ## Y_ac vs Y_pr plot with heatmap and erroe inset
        rng = [min_,max_]
        r2_tr = r2(tr_y.cpu().numpy(),y_tr_pr)
        r2_val = r2(val_y.cpu().numpy(),y_val_pr)
        r2_test = r2(test_y.cpu().numpy(),y_test_pr)
        fig,[ax] = panel(1,1,dpi=50,r_p=0.12,l_p=0.12,b_p=0.1,t_p=0.05)
        inter=Fig3_inter
        den_plot(ax,tr_y.cpu().numpy(),val_y.cpu().numpy(),test_y.cpu().numpy(),y_tr_pr,y_val_pr,y_test_pr,r2_tr,r2_val,r2_test,rng,s=70,den_scale_bool=den_scale_bool)
        plt.savefig('./Result_plots/Fig3/'+prop_name+'_Ypr_vs_Yac_mse_scale_mse_RBF.png',dpi=50)
        plt.show()


    exp_data_f = pd.read_csv(prop_name+'_gp.csv').values[:,:-1]
    mask = exp_data_f[:,[0,1,7]].sum(axis=1)==100
    mask.sum()
    exp_data_f = exp_data_f[mask,:]
    exp_data_f_f = exp_data_f
    exp_data_f_f = exp_data_f_f - 0
    exp_data_f_f = 2 * (exp_data_f_f / 100) - 1
    exp_data_f_f = torch.Tensor(exp_data_f_f)

    exp_y = pd.read_csv(prop_name+'_gp.csv').values[:,-1]/prop_scale
    exp_y = exp_y[mask]

    aug_list = []
    z_data = np.arange(1,100.1,0.5)

    for i in z_data:
        for j in z_data:
            if (100-i-j)>0:
                aug_list.append([i,j,100-i-j])

    aug_data = np.array(aug_list)
    aug_data.shape
    aug_data_f = np.zeros([len(aug_data),train_x.size(-1)])
    aug_data_f[:,0] = aug_data[:,0]#SiO2
    aug_data_f[:,1] = aug_data[:,1]#B2O3
    aug_data_f[:,7] = aug_data[:,2]#Na2O


    aug_data_f_f = aug_data_f
    aug_data_f_f = aug_data_f_f - 0
    aug_data_f_f = 2 * (aug_data_f_f / 100) - 1
    aug_data_f_f = torch.Tensor(aug_data_f_f)


    with torch.no_grad(), gpytorch.settings.use_toeplitz(False), gpytorch.settings.fast_pred_var(), gpytorch.settings.max_root_decomposition_size(100):
        exp_y_pred = my_model(exp_data_f_f).mean.cpu().numpy()/prop_scale
        aug_y = my_model(aug_data_f_f).mean.cpu().numpy()/prop_scale
        std_exp = my_model(exp_data_f_f).stddev.cpu().detach().numpy()/prop_scale
        std_aug = my_model(aug_data_f_f).stddev.cpu().detach().numpy()/prop_scale

    std_exp[np.isnan(std_exp)]=0
    std_aug[np.isnan(std_aug)]=0

    exp_y.shape
    plt.plot(exp_y,exp_y_pred,'.')
    plt.plot(exp_y,exp_y)
    plt.show()

    def NBS_ternary(fill_coordinate,fill_val,ac_coordinate,ac_val,y_label):
        import importlib
        import ternary
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import matplotlib
        import random
        import math
        import pickle
        importlib.reload(ternary)

        cm = plt.cm.get_cmap('jet')
        scale=100
        fig, tax = ternary.figure(scale=scale)
        fig.set_size_inches(10, 9)

        #Set the minv and maxv values below according to the range of values of your property
        minv=min(fill_val.min(),ac_val.min())
        maxv=max(fill_val.max(),ac_val.max())

        tax.scatter(fill_coordinate,s=10,marker='s',vmin=minv,vmax=maxv,colormap=plt.cm.jet,colorbar=True,c=fill_val,cmap=plt.cm.jet,)
        tax.scatter(ac_coordinate,s=40,marker='s',vmin=minv,vmax=maxv,colormap=plt.cm.jet,c=ac_val,cmap=plt.cm.jet,edgecolor='k')
        # Decoration.
        tax.boundary(linewidth=2)
        tax.right_axis_label('B$_2$O$_3$',fontsize=20,offset=0.1)
        tax.left_axis_label('Na$_2$O',fontsize=20,offset=0.1)
        tax.bottom_axis_label('SiO$_2$',fontsize=20,offset=-0.1)
        tax.gridlines(multiple=10, color="gray")
        tax.ticks(axis='lbr', linewidth=1, multiple=20,fontsize=16,offset=0.02)
        tax.get_axes().axis('off')
        tax.get_axes().axis('equal')
        axs =  plt.gcf().get_children()[-1]
        axs.set_position([0.82,0.24,1, 0.525])

        #Change the ylabel according to your property.
        axs.set_ylabel(y_label.replace('*',''), fontsize=20,labelpad=20)

    NBS_ternary(fill_coordinate=aug_data_f[:,[0,1,7]], fill_val=aug_y, ac_coordinate=exp_data_f[:,[0,1,7]], ac_val=exp_y,y_label=prop_label.replace('*',''))
    plt.savefig('./Result_plots/Fig4/'+prop_name+'_mean_ternary_NBS_RBF.png',dpi=50)
    plt.show()
    try:
        NBS_ternary(fill_coordinate=aug_data_f[:,[0,1,7]], fill_val=std_aug, ac_coordinate=exp_data_f[:,[0,1,7]], ac_val=std_exp,y_label="Standard deviation ("+prop_label.split('(')[1])
        plt.savefig('./Result_plots/Fig5/'+prop_name+'_STD_ternary_NBS_RBF.png',dpi=50)
        plt.show()
    except:
        NBS_ternary(fill_coordinate=aug_data_f[:,[0,1,7]], fill_val=std_aug, ac_coordinate=exp_data_f[:,[0,1,7]], ac_val=std_exp,y_label="Standard deviation")
        plt.savefig('./Result_plots/Fig5/'+prop_name+'_STD_ternary_NBS_RBF.png',dpi=50)
        plt.show()




    my_loss = pd.read_csv(prop_name+'_loss.csv').values
    fig,[ax] = panel(1,1,dpi=50,r_p=0.12,l_p=0.12,b_p=0.1,t_p=0.05)
    ax.plot(my_loss[:,1],my_loss[:,2],c='r',label = "Training",)
    ax.plot(my_loss[:,1],my_loss[:,3],c='b',label = "Validation",)
    xlabel("epoch")
    ylabel("Loss(GPa)")
    legend_on()
    # ax.set_yticks(np.arange(50,451,50))
    # ax.set_ylim([10,440])
    plt.savefig('./Result_plots/Fig6/'+prop_name+'_loss_vs_epoch.png',dpi=50)
    plt.show()


    def comp_vs_freq(prop_Short_name = 'TEC'):

        name_ids = {}
        name_ele = {}

        for i in pd.read_csv('./all_component.csv').values:
            try:
                name_ids[i[1]] = int(i[0])
                name_ele[i[0]] = i[1]
            except:
                name_ids[i[1]] = None
                name_ele[i[0]] = None

        def formula(list1):
            return [name_ids[i] for i in list1]

        def reverse_formula(list2):
            return [name_ele[j] for j in [str(i) for i in list2]]


        freq_of_nonzero_composition = []
        for i in range(data[:,:-1].shape[1]):
            freq_of_nonzero_composition.append((data[:,i]!=0).sum())


        y_hight = np.array(freq_of_nonzero_composition)

        formula_name = reverse_formula(pd.read_csv(prop_Short_name+'_gp.csv').columns[:-1])

        latex_formula_name = []
        for i in formula_name:
            latex_formula_name.append(i.replace('2', '$_2$').replace('3', '$_3$').replace('5', '$_5$'))

        sorted_Latex_formula_name = [x for _,x in sorted(zip(y_hight,latex_formula_name))[::-1]]

        fig,[ax] = panel(1,1,dpi=50,r_p=0.12,l_p=0.12,b_p=0.1,t_p=0.05,figsize=(17,4))
        plt.bar(range(1,(data[:,:-1].shape[1])+1), sorted(y_hight)[::-1], width = 0.5, color='r')
        plt.xlim(min(range(0,(data[:,:-1].shape[1]))), max(range(0,(data[:,:-1].shape[1]))))
        plt.ylabel('Frequency')

        my_xticks = sorted_Latex_formula_name
        plt.xticks(range(1,(data[:,:-1].shape[1])+1), my_xticks)
        plt.xticks(rotation=90)
        from matplotlib.lines import Line2D
        custom_lines = [Line2D([0], [0], color='w', lw=1)]
        ax.legend(custom_lines,[prop_label.split(' (')[0].capitalize()],loc=1)
        # ax.minorticks_off()
        ax.xaxis.set_tick_params(which='minor', bottom=False)
        # plt.savefig('Result_plots/try/'+prop_Short_name+'_nonzero_composition_vs_freq')
        plt.show()

    comp_vs_freq(prop_Short_name = prop_name)
