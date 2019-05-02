import time
import numpy as np
from sklearn.metrics import accuracy_score

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
model.compile_and_Store(Xtrain,nr_classes)

cpu_train = model.predict(Xtrain)
cpu_test = model.predict(Xtest)
#print(cpu_train)

assert np.allclose(cpu_train, model.cuda_predict(Xtrain)) == True, "cuda_predict failed for train set"
assert np.allclose(cpu_train, model.cuda_pred_tree_mult(Xtrain,10)) == True, "cuda_pred_tree_mult failed for train set"
assert np.allclose(cpu_train, model.cuda_pred_forest(Xtrain)) == True, "cuda_pred_forest failed for train set"
assert np.allclose(cpu_train, model.cuda_pred_forest_mult(Xtrain,10)) == True, "cuda_pred_forest_mult failed for train set"
assert np.allclose(cpu_train, model.cuda_pred_all(Xtrain)) == True, "cuda_pred_all failed for train set"
assert np.allclose(cpu_train, model.cuda_pred_all_mult(Xtrain,2)) == True, "cuda_pred_all failed for train set"
assert np.allclose(cpu_train, model.cuda_pred_1block1X(Xtrain)) == True, "cuda_pred_1block1X failed for train set"
#something is not right here when n_estimators = 128...
assert np.allclose(cpu_train, model.cuda_pred_1block_multX(Xtrain)) == True, "cuda_pred_1block1X failed for train set"

assert np.allclose(cpu_test, model.cuda_predict(Xtest)) == True, "cuda_predict failed for test set"
assert np.allclose(cpu_test, model.cuda_pred_tree_mult(Xtest,10)) == True, "cuda_pred_tree_mult failed for train set"
assert np.allclose(cpu_test, model.cuda_pred_forest(Xtest)) == True, "cuda_pred_forest failed for test set"
assert np.allclose(cpu_test, model.cuda_pred_forest_mult(Xtest,10)) == True, "cuda_pred_forest_mult failed for test set"
assert np.allclose(cpu_test, model.cuda_pred_all(Xtest)) == True, "cuda_pred_all failed for test set"
assert np.allclose(cpu_test, model.cuda_pred_all_mult(Xtest,2)) == True, "cuda_pred_all failed for test set"
assert np.allclose(cpu_test, model.cuda_pred_1block1X(Xtest)) == True, "cuda_pred_1block1X failed for test set"
assert np.allclose(cpu_test, model.cuda_pred_1block_multX(Xtest)) == True, "cuda_pred_1block1X failed for test set"

print("All tests passed")      
