import numpy as np
import pandas as pd
from sklearn.metrics import r2_score as r2
from sklearn.model_selection import train_test_split
import pickle
import os


plot_ = 'test/'
try:
    os.mkdir(plot_)
except:
    print('Directory exists.')


# Sodium Aluminoborate Na Al B
# Calcium Aluminoborate Ca Al B
# Magnesium Aluminosilicate Mg Al Si
# Phosphosilicate  P Si
# Lead silicate Pb Si
# Vanado borate V B
# Sodium borosilicate Na B Si
# Lead tellurite Pb Te
# Magnesium tellurite Mg Te
# Sodium Tellurite Na Te
# Vanado tellurite V te
# Potasium borate K B

glasses = ['Na2O+Al2O3_+B2O3_',
           'CaO+Al2O3_+B2O3_',
          'MgO+Al2O3_+SiO2_',
          'P2O5_+SiO2_',
          'PbO+SiO2_',
          'V2O3+B2O3_',
          'Na2O+B2O3_+SiO2_',
          'PbO+TeO2_',
          'MgO+TeO2_',
          'Na2O+TeO2_',
          'V2O3+TeO2_',
          'K2O+B2O3_']

P2 = 'TG/'
P1 = 'YM/'

name_dict = {P1:P1[:-1],P2:P2[:-1]}


def make_X(formula=['SiO2_+Al2O3_+Na2O+K2O']):
    xs = []
    col_ids = pd.read_csv(database+'csv_files/Components_id.csv')
    col_ids.index = col_ids['Desc']
    for f in formula:
        comps = f.strip().split('+')
        NF = []
        NM = []
        for comp in comps:
            if comp[-1]=='_':
                NF.append(comp[:-1])
            else:
                NM.append(comp)

        ids = col_ids.loc[NF+NM]['ID'].values
        X_ = np.array(compositions(NF,NM))
        X = pd.DataFrame(np.zeros((len(X_),len(names1)-1)),columns=[int(i) for i in names1[:-1]])
        print('NF: ',NF,'NM: ',NM)
        print('X shape:', X_.shape)
        print('x shape:', X.shape)
        X[ids] = X_
        xs.append(X)
    print('Total columns=',len(names1)-1)
    return xs


    def comps(w,Sum,low=30,s=1):
        if w==0:
            return [[]]
        if w==1:
            return [[Sum]]
        elif w==2:
            return [[i/s,Sum-i/s] for i in range(low*s,s*Sum-s*low+1)]
        elif w==3:
            return [[i/s,j/s,Sum-i/s-j/s] for i in range(s*low,s*Sum) for j in range(s*low,max(s*low,s*Sum-s*i-s*low+1))]
        elif w==4:
            return [[i/s,j/s,k/s,Sum-i/s-j/s-k/s] for i in range(s*low,s*Sum) for j in range(s*low,max(s*low,s*Sum-s*i)) for k in range(s*low,max(s*low,s*Sum-s*i-s*j-s*low+1)) ]
        else:
            pass

    def compositions(NF,NM,NF_sum_low=60,NF_low=30,NM_low=5):
        all = []
        for a in [[i, 100-i] for i in range(NF_sum_low,100-NM_low*len(NM)+1)]:
            for i in comps(len(NF),a[0],low=NF_low,s=100):
                for j in comps(len(NM),a[1],low=NM_low,s=100):
                    all.append(i+j)
        return all


Xs = make_X(glasses)


names_ = iter(["Sodium-aluminoborate", "Calcium-aluminoborate", "Magnesium-aluminosilicate", "Phosphosilicate", "Lead-silicate", "Vanado-borate", "Sodium-borosilicate", "Lead-tellurite", "Magnesium-tellurite", "Sodium-tellurite", "Vanado-tellurite", "Potasium-borate"])
name_pos = iter([(800,190),(1250,195),(1250,85),(1000,40),(1150,50),(440,20),(650,125),(550,110),(920,30),(480,75),(520,95),(700,10),])
patch_pos = iter([((800,190),(820,150)), ((970,195),(900,190)),((1120,85),(1120,100)),((650,65),(600,125)), ((570,60),(540,110)), ((550,50),(500,90)), ((500,60),(450,75)),((350,25),(400,35)), ((600,25),(600,10)), ((1000,50),(900,65)), ((720,50),(810,40)), ((750,40),(750,30))])


fig, [ax] = panel(1,1,figsize=(12,9))

scale0 = 1
scale1 = 1

for x in Xs:
    pos = list(next(name_pos))
    text(*pos,r"\textbf{{{}}}".format(next(names_)),va='center',ha='right',fontsize=16)
    arrow(pos=next(patch_pos),arrowstyle='-')

    ax.plot(model1.predict(x)/scale1,model0.predict(x)/scale0,'o',ms=15,alpha=1)


limx = [250,1250]
limy = [0,210]

for r in np.arange(0.05,0.3,0.05):
    X = np.linspace(*limx,100).reshape(-1,1)
    ax.plot(X,X*r,'--k',alpha=0.5,lw=1)
    ypos = min(X[-1]*r,205)
    xpos = ypos/r
    if r==0.25:
        str1 = r"\textbf{E/T$_g$ (GPa/K) = }"+r"\textbf{{{:.2f}}}".format(r)
    else:
        str1 = r"\textbf{{{:.2f}}}".format(r)
    t = text(xpos,ypos,str1,va='top',ha='right',fontsize=16)
    mask = (X<limx[1])&(X>limx[0])
    mask1 = (X*r>limy[0])&(X*r>limy[0])
    mask = mask&mask1
    X = X[mask].reshape(-1,1)
    data = np.hstack([X,X*r])
    np.savetxt(plot_+'/{}_{}_{:.3f}'.format(P2[:-1],P1[:-1],r)+'.csv',data,header='{}, {}'.format(P2[:-1],P1[:-1]),comments='',delimiter=',',fmt='%.3f')

yticks([0,50,100,150,200])
xlim(limx)
ylim(limy)
ylabel("Young's modulus, E (GPa)")
xlabel("Glass transition temperature, T$_g$ (k)")
ax.set_axisbelow(False)
plt.grid('on',ls='--',c='k',alpha=0.5)
fig.savefig(plot_+'fig.png')
