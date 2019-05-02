import time
import numpy as np
#from sklearn.metrics import accuracy_score

from woody import WoodClassifier
from woody.data import *

seed = 0

#Xtrain, ytrain, Xtest, ytest = covtype(train_size=400000, seed=seed)
Xtrain, ytrain, Xtest, ytest = susy(train_size=4000000, seed=seed)
if Xtrain.dtype != np.float32:
        Xtrain = Xtrain.astype(np.float32) 
        ytrain = ytrain.astype(np.float32) 
        Xtest = Xtest.astype(np.float32) 
        ytest = ytest.astype(np.float32) 

model = WoodClassifier.load('./model_susy8tree.data')
nr_classes = len(np.unique(ytrain)) +1 #not sure if accurate
model.compile_store_v2(Xtrain,nr_classes)

cpu_train = model.predict(Xtrain)
cpu_test = model.predict(Xtest)
#model.cuda_v2(Xtrain)
assert np.allclose(cpu_train, model.cuda_v2(Xtrain)) == True, "Failed for train set"

print("All tests passed")      
