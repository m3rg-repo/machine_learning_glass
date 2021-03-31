import os
import sys
import numpy as np
from mpi4py import MPI
from xgb_ivs import Model, batching, save2file

comm = MPI.COMM_WORLD

modelfolder, datafolder = sys.argv[1:]
datafile = datafolder+"/train_split_X.csv"
r = 1
for file in os.listdir(modelfolder):
    if "scores.txt" in file:
        scores = np.genfromtxt(modelfolder+"/"+file, skip_header=2)
        r = np.argmax(scores[:,-1]) + 1
for file in os.listdir(modelfolder):
    if "{}_model.pkl".format(r) in file:
        modelfile = modelfolder+"/"+file 
for file in os.listdir(modelfolder):
    if "means_and_stds.json" in file:
        ms_file = modelfolder+"/"+file

nprocs = comm.Get_size()
rank = comm.Get_rank()

m = Model(modelfile, datafile, ms_file)

if rank==0:
    print("\n===========================\nUsing nprocs: ", nprocs)
    print("\nUsing datafile: ", datafile)
    print("\nUsing modelfile: ", modelfile)
    print("\nTotal data points: ", m.X.shape[0])

from functools import partial
def f(x, m=None):
    print("cpu: ", MPI.COMM_WORLD.Get_rank(), x.shape)
    return m.cal_ivs(x)

batches = batching(m.X, nprocs)

ivs = partial(f,m=m)(batches[rank])
ivs = comm.gather(ivs, root=0)

if rank==0:
    outfile = modelfile[:-4]+"_ivs.pkl"
    save2file(outfile, np.concatenate(ivs))
    print("\nIVS saved to: ", outfile, "\n===========================")
