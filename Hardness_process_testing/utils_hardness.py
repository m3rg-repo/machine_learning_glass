import shap
import os
import pickle
from new_plot import *

data_path = os.path.join(os.getcwd(),'shap_data')

def load_explainer(path=os.path.join(data_path,'hardness_explainer.pkl')):
    #loading explainer 
    with open(path,'rb') as f:
        explainer = pickle.load(f)
    return explainer

def load_shap_values(path=os.path.join(data_path,'hardness_shap_values.pkl')):
    #loading shap values
    with open(path,'rb') as f:
        shap_values = pickle.load(f)
    return shap_values

def load_values(path=os.path.join(data_path,'values.pkl')):
    #loading values
    with open(path,'rb') as f:
        values = pickle.load(f)
    return values

def load_featuers(path=os.path.join(data_path,'features.pkl')):
    #loading rendered feature names
    with open(path,'rb') as f:
        features = pickle.load(f)
    return features

def make_hill(chemicals):
    
    '''
    Function for converting chemical formula to hill notation
    takes input as a list. For example, 
    >> chemicals =['Al2O3']
    >> make_hill(chemicals)
    Output: ['Al$_2$O$_3$']
    '''
    
    hill_all = []
    for col in chemicals:
        hill = ''
        for ele in col:
            if ele in ['2','3','4','5','6','7','8','9','10','11','12','13']:
                hill = hill + '$_' + ele + '$'
            else:
                hill = hill + ele
#         print(hill)
        hill_all.append(hill)
    return hill_all