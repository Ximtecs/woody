import time
import numpy as np
from sklearn.metrics import accuracy_score

from woody import WoodClassifier
from woody.data import *


seed = 0

Xtrain, ytrain, Xtest, ytest = covtype(train_size=400000, seed=seed)
#Xtrain, ytrain, Xtest, ytest = susy(train_size=4000000, seed=seed)
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
            n_estimators=2,
            criterion="gini",
            max_features=None,
            min_samples_split=2,
            n_jobs=1,
            seed=seed,
            bootstrap=True,
            tree_traversal_mode="dfs",
            tree_type="standard",
            min_samples_leaf=1,
            float_type="double",
            max_depth=None,
            verbose=1)

fit_start_time = time.time()
#model.fit(Xtrain, ytrain)
fit_end_time = time.time()
#model.save("./model_susy2tree.data")
model = WoodClassifier.load('./model_8tree.data')
print("Number of estimators: \t\t%i\n" % model.n_estimators)

nr_classes = len(np.unique(ytrain)) +1 #not sure if accurate
model.compile_and_Store(Xtrain,nr_classes)
#model.store_numpy_forest()
cpu_train_start = time.time()
ypreds_train = model.predict(Xtrain) 
cpu_train_end = time.time()

gpu_train_start = time.time()

#gpu_ypreds_train = model.cuda_predict(Xtrain) 
#gpu_ypreds_train = model.cuda_pred_forest(Xtrain) 
#gpu_ypreds_train = model.cuda_pred_forest_mult(Xtrain,10) 
#gpu_ypreds_train = model.cuda_pred_all(Xtrain) 
gpu_ypreds_train = model.cuda_pred_1block1X(Xtrain) 
#gpu_ypreds_train = model.cuda_pred_1block_multX(Xtrain) 
gpu_train_end = time.time()


test_start_time = time.time()
ypred_test = model.predict(Xtest)
test_end_time = time.time()
#print(ypred_test[84703])
gpu_test_start_time = time.time()
#gpu_ypred_test = model.cuda_predict(Xtest)
#gpu_ypred_test = model.cuda_pred_forest(Xtest)
#gpu_ypred_test = model.cuda_pred_forest_mult(Xtest,10)
#gpu_ypred_test = model.cuda_pred_all(Xtest)
gpu_ypred_test = model.cuda_pred_1block1X(Xtest)
#gpu_ypred_test = model.cuda_pred_1block_multX(Xtest)
gpu_test_end_time = time.time()


cpu_time = test_end_time - test_start_time
gpu_time = gpu_test_end_time - gpu_test_start_time
speedup = cpu_time / gpu_time

cpu_train_time = cpu_train_end - cpu_train_start
gpu_train_time = gpu_train_end - gpu_train_start
train_speedup = cpu_train_time / gpu_train_time

print("Training time:         %f" % (fit_end_time - fit_start_time))
print("CPU Testing time:      %f" % (test_end_time - test_start_time)) 
print("GPU Testing time:      %f" % (gpu_test_end_time - gpu_test_start_time)) 
print("Speedup:               %f" % (speedup) )
print("training Speedup:      %f" % (train_speedup) )
print("CPU Training accuracy: %f" % (accuracy_score(ypreds_train, ytrain)))
print("CPU Testing accuracy:  %f" % (accuracy_score(ypred_test, ytest)))
print("GPU Training accuracy: %f" % (accuracy_score(gpu_ypreds_train, ytrain)))
print("GPU Testing accuracy:  %f" % (accuracy_score(gpu_ypred_test, ytest)))

#tests.test_output(gpu_ypred_test,ypred_test)
# model.draw_tree(0, fname="./wood_tree.pdf")        
