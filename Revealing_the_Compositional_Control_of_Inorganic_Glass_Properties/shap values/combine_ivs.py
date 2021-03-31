import sys
import os
from xgb_ivs import *

folder = sys.argv[1]

os.chdir(folder)

kv_pair = {}


for file in os.listdir():
    if ("-" in file):
        key = file.split("-")[0].split("_")[-1]
        kv_pair[key] = file

keys = list([int(i) for i in kv_pair.keys()])
keys.sort()
keys = [str(i) for i in keys]

print("Last: ", keys[-1])

values = []
for i in keys:
    print(i)
    val, = loadfile(kv_pair[i])
    values += [val]

import numpy as np

data = np.concatenate(values, axis=0) 

print("Total points: ", len(data))

save2file("../../IVS/{}_ivs.pkl".format(folder.strip("/")), data, protocol=4)






