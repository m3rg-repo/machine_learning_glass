import pandas as pd
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import os
import pandas as pd
import numpy as np
import json
from new_plot import *
set_things()


#function to convert chemical formulas into Hill notation for final graphs
def make_hill(x):
    lable = ''
    for i in x:
        if i  in ['2','3','4','5','6','7','8','9']:
            subs = "$_" + i  + "$"
            lable = lable + subs
        else:
            lable = lable + i
    return lable

#setting random state for reproducibility
np.random.seed(20201219)

count = 0
#reading all the compositions from csv file
seed_df = pd.read_csv('all_compositions.csv')
seed_df = seed_df.dropna() #sanity check
# seed_df = seed_df.sample(50000) #sampling to check on randomly selected compositions


#setting parameters for KMeans
k = 30
kmeans = KMeans(n_clusters=k, max_iter=1000).fit(seed_df)
seed_df["clusters"] = kmeans.labels_
seed_df.clusters = seed_df.clusters + 1

#preparing labels for coloring tsne plots

#getting cluster label and number of points caputred
info = seed_df['clusters'].value_counts().sort_values()
info = pd.DataFrame(info)

info['PERCENTAGE'] = 100*info.clusters/info.clusters.sum()
info = info.sort_values('PERCENTAGE',ascending=False)

to_color = info[info['PERCENTAGE'] > 3].index.to_list()

print(len(to_color),to_color)

    # info

def set_label(x):
    if x in to_color:
        return x
    else:
        return 0

seed_df['color'] = seed_df['clusters'].apply(set_label) #coloring done


#getting first three components of each clusters
unique_clusters = seed_df.color.unique()
labelling = {} #storing cluster id and corresponding glass family
for num in to_color:
    if num != 0:
        tempDF = seed_df[seed_df['clusters'] == num]
        tempDF = tempDF.drop(['clusters','color'],axis=1)
        name = ','.join(tempDF.astype(bool).sum(axis=0).sort_values(ascending=False)[:4].index.to_list())
        labelling[num] =  name
        print(num, name)

with open(str(seed_df.shape[0]) + 'all_tsne_label.json','w') as f:
    json.dump(labelling,f)

#RUNNING TSNE
X_embedded = TSNE(n_components=2,n_jobs=-1,perplexity=40,learning_rate=100).fit_transform(seed_df[seed_df.columns[:-2]])

x_coordinate = {}
y_coordinate = {}
for key in labelling.keys():
    x_coordinate[key]  = X_embedded[:,0][seed_df['clusters']==key].mean()
    y_coordinate[key]  = X_embedded[:,1][seed_df['clusters']==key].mean()

tsne_df = pd.DataFrame(data=X_embedded, columns=['x1','x2'])
tsne_df['color'] =  seed_df['color'].values
tsne_df.to_csv(str(seed_df.shape[0])+'all_tsne_data.csv',index=False)

NUM_COLORS = len(labelling.keys())
cm = plt.get_cmap('jet') #'gist_ncar')
# cm = plt.get_cmap('nipy_spectral')
clrs = [cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)]

s = 10
to_color = list(labelling.keys())
print(to_color)
mask = tsne_df.color==0
plt.scatter(tsne_df.x1[mask], tsne_df.x2[mask], c='k',alpha=0.3,s=s)

for i in range(len(to_color)):
    c = int(to_color[i])
    mask = tsne_df.color==c
    if c == 0:
        pass
    else:
        plt.scatter(tsne_df.x1[mask], tsne_df.x2[mask],color=clrs[i],s=s, label=make_hill(labelling[c]))

mask = tsne_df.color==0
plt.scatter(tsne_df.x1[mask].iloc[0], tsne_df.x2[mask].iloc[0], c='k',alpha=0.3,s=s,label='misc')
plt.legend(markerscale=5,loc=[1.01,0])
# plt.title('all_compositions')
plt.box(on=None)
plt.axis('off')
plt.savefig(str(seed_df.shape[0])+'all_tsne.png',bbox_inches='tight')
plt.show()
plt.close()
