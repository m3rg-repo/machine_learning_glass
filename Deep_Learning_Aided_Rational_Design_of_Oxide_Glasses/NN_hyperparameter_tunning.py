import sys
args = sys.argv
jobid = args[1]

Prop =["Vickers_Hardness_(Typical)"]
LOG_FILE = args[1]+'.log'
data_path = '../../Data/'

from sklearn.neural_network import MLPRegressor as NN
import MLmodule as ML
import matplotlib.pyplot as plt
import os.path
import pickle
import HP
import numpy as np
import sys

old_stdout = sys.stdout
log_file = open(LOG_FILE,"w")


max_layer_size = 20
n = max_layer_size + 1
# specify parameters and distributions to sample from
params = {'solver':['lbfgs', 'sgd', 'adam'],
          'learning_rate':['adaptive','constant'],
          'hidden_layer_sizes':[(x, y, z) for x in range(1,n) for y in range(1,n) for z in range(1,n)],
          'alpha':np.random.uniform(0.0001, 1, 100),
          'activation':['tanh', 'relu'],}
n_iter_rand_search = 100
multiple_times = 3



try:
    sys.stdout = log_file
    print("========================\n\n"+"".join(Prop)+"\n\n========================")

    _columns = [1, 2, 3, 4, 5, 6, 7, 8, 9, 23, 24, 25,31, 35, 36, 37, 39, 40, 53, 55, 58, 60, 61, 71,75, 79, 81, 82, 83, 84, 92, 93, 101, 111, 208,218, 268]

    if not sys.warnoptions:
        import warnings
        warnings.simplefilter("ignore")



    if os.path.exists(data_path+Prop[0].replace(' ','_')+'.pickle'):
        print('Using existing saved data.')
        with open(data_path+Prop[0].replace(' ','_')+'.pickle','rb') as f:
            S1 = pickle.load(f)
            f.close()
    else:
        S1 = ML.sess()
        S1.data_loader(features = _columns,properties = Prop )
        with open(data_path+Prop[0].replace(' ','_')+'.pickle','wb+') as f:
            pickle.dump(S1,f)
            f.close()


    print('Before cleaning:')
    print(S1.features.shape)
    print(S1.properties.shape)

    ## Data cleaning
    mean = S1.properties.mean()
    scale = 10**int(np.log10(mean))
    print('Scale: ', scale)
    S1.properties /= scale
    mean = S1.properties.mean()
    std = S1.properties.std()
    Zs = (S1.properties-mean)/std
    mask = (Zs<3) & (Zs>-3)

    plt.hist(S1.properties)
    plt.savefig('./hist.png')
    plt.show()

    plt.hist(S1.properties[mask])
    plt.savefig('./hist_after_3s.png')
    plt.show()

    S1.features = S1.features[mask.ravel(),:]
    S1.properties = S1.properties[mask].ravel()

    print('After cleaning:')
    print(S1.features.shape)
    print(S1.properties.shape)

    X = S1.features
    y = S1.properties

    HP.tunning(X,y,params,n_iter_rand_search=n_iter_rand_search, multiple_times=multiple_times, prefix=jobid+'_')

    print('It ran properly.')
except:
    print('Did not run properly.')

sys.stdout = old_stdout
log_file.close()
