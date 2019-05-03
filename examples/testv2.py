import time
import numpy as np
#from sklearn.metrics import accuracy_score

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

model = WoodClassifier.load('./model_32tree.data')
nr_classes = len(np.unique(ytrain)) +1 #not sure if accurate
print("Number of training patterns:\t%i" % Xtrain.shape[0])
print("Number of test patterns:\t%i" % Xtest.shape[0])
print("Dimensionality of the data:\t%i" % Xtrain.shape[1])
print("Number of estimators: \t\t%i" % model.n_estimators)
print("Number of classes: \t\t%i\n" % nr_classes)	


model.compile_store_v2(Xtrain,nr_classes)

cpu_train_start = time.time()
print("Calcuation predictions for training set on CPU\n")
cpu_train = model.predict(Xtrain)
cpu_train_end = time.time()
print("")
cpu_test_start = time.time()
print("Calcuation predictions for testing set on CPU\n")
cpu_test = model.predict(Xtest)
cpu_test_end = time.time()
print("")
gpu_train_start = time.time()
print("Calcuation predictions for training set on GPU\n")
gpu_train = model.cuda_v2(Xtrain)
gpu_train_end = time.time()
print("")
gpu_test_start = time.time()
print("Calcuation predictions for testing set on GPU\n")
gpu_test = model.cuda_v2(Xtest)
gpu_test_end  = time.time()
print("")
assert np.allclose(cpu_train, gpu_train) == True, "Failed for train set"
assert np.allclose(cpu_test, gpu_test) == True, "Failed for test set"
print("All tests passed - GPU and CPU yeild identical result") 
print("")
cpu_train_time = cpu_train_end - cpu_train_start
cpu_test_time = cpu_test_end - cpu_test_start
gpu_train_time = gpu_train_end - gpu_train_start
gpu_test_time = gpu_test_end - gpu_test_start

train_speedup = cpu_train_time / gpu_train_time
test_speedup = cpu_test_time / gpu_test_time

print("CPU Training time:     %f" % (cpu_train_time))
print("CPU Testing time:      %f" % (cpu_test_time) )
print("GPU Training time:     %f" % (gpu_train_time) )
print("GPU Testing time:      %f" % (gpu_test_time) )
print("Training Speedup:      %f" % (train_speedup))
print("Testing Speedup:       %f" % (test_speedup) )
