import time
import numpy as np
from sklearn.metrics import accuracy_score

from woody import WoodClassifier
from woody.data import *
import sys
nr_tree =int(sys.argv[1])
f_type = int(sys.argv[2])

seed = 0

if (f_type == 0):
        Xtrain, ytrain, Xtest, ytest = susy(train_size=500000, seed=seed)
elif(f_type == 1):
        Xtrain, ytrain, Xtest, ytest = covtype(train_size=400000, seed=seed)

if Xtrain.dtype != np.float32:
        Xtrain = Xtrain.astype(np.float32) 
        ytrain = ytrain.astype(np.float32) 
        Xtest = Xtest.astype(np.float32) 
        ytest = ytest.astype(np.float32) 

print("")
print("Number of training patterns:\t%i" % Xtrain.shape[0])
print("Number of test patterns:\t%i" % Xtest.shape[0])
print("Dimensionality of the data:\t%i" % Xtrain.shape[1])
model = WoodClassifier(
            n_estimators=nr_tree,
            criterion="gini",
            max_features=None,
            min_samples_split=2,
            n_jobs=4,
            seed=seed,
            bootstrap=True,
            tree_traversal_mode="dfs",
            tree_type="standard",
            min_samples_leaf=1,
            float_type="double",
            max_depth=None,
            verbose=1)

fit_start_time = time.time()
model.fit(Xtrain, ytrain)
fit_end_time = time.time()

if (f_type == 0):
        file_name = "./models/model_susy" + str(nr_tree) + "tree_4jobs.data"
elif(f_type == 1):
        file_name = "./models/model_" + str(nr_tree) + "tree_4jobs.data"

model.save(file_name)