import os
import shutil

def isprop(file):
    if os.path.isdir(file):
        stds = [j for j in os.listdir(file) if "stds" in j]
        best = [j for j in os.listdir(file) if "ivs" in j]
        if 1==len(stds):
            best_ = "_".join(best[0].split("_model_ivs")[0].split("_")[-2:]) + "_model.pkl"
            best = [j for j in os.listdir(file) if best_ in j][0]
            print(file)
            print(best)
            print(stds[0])
            shutil.copyfile(file+"/"+best, "../best/"+file+"_best_model.pkl")
            shutil.copyfile(file+"/"+stds[0], "../best/"+file+"_means_and_stds.json")
            
        

for i in os.listdir("./"):
    isprop(i)
    



