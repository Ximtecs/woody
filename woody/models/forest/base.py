#
# Copyright (C) 2015-2017 Fabian Gieseke <fabian.gieseke@di.ku.dk>
# License: GPL v2
#

import os
import abc
import math
import numpy as np
import cPickle as pickle

from .util import PickableWoodyRFWrapper, ensure_data_types
from woody.util.array import transpose_array
from woody.util import draw_single_tree

#Cuda stuff - for testing cuda still works
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
from scipy.stats import mode
import time
import string

class Wood(object):
    """
    Random forest implementation.
    """
    __metaclass__ = abc.ABCMeta

    ALLOWED_FLOAT_TYPES = ['float',
                           'double',
                           ]
    TREE_TRAVERSAL_MODE_MAP = {"dfs": 0,
                               "node_size": 1,
                               "prob": 2,
                               }
    CRITERION_MAP = {"mse": 0,
                     "gini": 1,
                     "entropy": 2,
                     "even_mse": 3,
                     "even_gini": 3,
                     "even_entropy": 3,
                     }
    LEARNING_TYPE_MAP = {"regression": 0,
                         "classification": 1,
                         }
    TREE_TYPE_MAP = {"standard":0,
                     "randomized":1,
                     }
    LEAF_STOP_MODE_MAP = {"all":0,
                          "ignore_impurity":1,
                          }
    TRANSPOSED_MAP = {False: 0,
                      True: 1,
                      }
    
    def __init__(self,
                 seed=0,
                 n_estimators=10,
                 min_samples_split=2,
                 max_features=None,
                 bootstrap=True,
                 max_depth=None,
                 min_samples_leaf=1,
                 learning_type=None,
                 criterion=None,
                 tree_traversal_mode="dfs",
                 leaf_stopping_mode="all",
                 tree_type="randomized",
                 float_type="double",
                 patts_trans=True,
                 do_patts_trans=True,
                 lam_criterion = 0.0, 
                 n_jobs=1,                                  
                 verbose=1,
                 left_ids=None,
                 right_ids=None,
                 features=None,
                 thres_or_leafs=None,
                 tree_nodes_sum=None,
                 ):

        self.seed = seed
        self.n_estimators = n_estimators
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.learning_type = learning_type
        self.criterion = criterion
        self.tree_traversal_mode = tree_traversal_mode
        self.leaf_stopping_mode = leaf_stopping_mode
        self.tree_type = tree_type
        self.float_type = float_type
        self.patts_trans = self.TRANSPOSED_MAP[patts_trans]
        self.do_patts_trans = do_patts_trans
        self.lam_criterion = lam_criterion
        self.n_jobs = n_jobs
        self.verbose = verbose
        

        # set numpy float and int dtypes
        if self.float_type == "float":
            self.numpy_dtype_float = np.float32
        else:
            self.numpy_dtype_float = np.float64
        self.numpy_dtype_int = np.int32
                        
        assert self.float_type in self.ALLOWED_FLOAT_TYPES

        
        
    def __del__(self):
        """ Destructor taking care of freeing
        internal and external (Swig) resources.
        """

        if hasattr(self, 'wrapper_params'):
            
            self.wrapper.module.free_resources_extern(self.wrapper.params,
                                                      self.wrapper.forest)

    def get_params(self, deep=True):
        
        return {"seed": self.seed, 
                "n_estimators": self.n_estimators, 
                "min_samples_split": self.min_samples_split, 
                "max_features": self.max_features, 
                "bootstrap": self.bootstrap, 
                "max_depth": self.max_depth, 
                "min_samples_leaf": self.min_samples_leaf, 
                "learning_type": self.learning_type, 
                "criterion": self.criterion,
                "leaf_stopping_mode": self.leaf_stopping_mode, 
                "tree_traversal_mode": self.tree_traversal_mode, 
                "tree_type": self.tree_type, 
                "float_type": self.float_type, 
                "patts_trans": self.patts_trans, 
                "do_patts_trans": self.do_patts_trans,
                "lam_criterion": self.lam_criterion, 
                "n_jobs": self.n_jobs, 
                "verbose": self.verbose, 
                }

    def set_params(self, **parameters):
        """ Updates local parameters
        """
        
        for parameter, value in parameters.items():
            self.setattr(parameter, value)
        
    def fit(self, X, y, indices=None):
        """ If indices is not None, then 
        consider X[indices] instead of X 
        (in-place).
        """

        if X.shape[0] != y.shape[0]:
            raise ValueError("Dimensions not equal:" 
                             "X.shape[0]=%i != y.shape[0]=%i" %
                             (X.shape[0], y.shape[0]))
            
        #if self.tree_type == "standard" and self.bootstrap == False:
        #    raise Exception("No randomness given: bootstrap=%s and tree_type=%s" % (str(self.bootstrap), str(self.tree_type)))

        # convert input data to correct types and generate local
        # copies to prevent destruction of objects
        X, y = ensure_data_types(X, y, self.numpy_dtype_float)
        
        # transform some parameters
        if self.max_features == None:
            max_features = X.shape[1]
        elif isinstance(self.max_features, int):
            if self.max_features < 1 or self.max_features > X.shape[1]:
                raise Exception("max.features=%i must "
                                "be >= 1 and <= X.shape[1]=%i" % 
                                (self.max_features, X.shape[1]))
            max_features = self.max_features
        elif self.max_features == "sqrt":
            max_features = int(math.sqrt(X.shape[1]))
        elif self.max_features == "log2":
            max_features = int(math.log(X.shape[1]), 2)
        else:
            max_features = 1

        # set max_depth
        max_depth = ((2 ** 31) - 1 if self.max_depth is None else self.max_depth)

        if self.min_samples_leaf <= 0:
            raise ValueError("min_samples_leaf must be greater than zero!")
    
        self.wrapper = PickableWoodyRFWrapper(self.float_type)
        
        if self.do_patts_trans == True:
            XT = np.empty(X.shape, dtype=X.dtype)
            transpose_array(X, XT)
            X = XT

        self.wrapper.module.init_extern(self.seed, 
                                        self.n_estimators, 
                                        self.min_samples_split, 
                                        max_features, 
                                        self.bootstrap, 
                                        max_depth, 
                                        self.min_samples_leaf, 
                                        self.LEARNING_TYPE_MAP[self.learning_type], 
                                        self.CRITERION_MAP[self.criterion], 
                                        self.TREE_TRAVERSAL_MODE_MAP[self.tree_traversal_mode], 
                                        self.LEAF_STOP_MODE_MAP[self.leaf_stopping_mode],
                                        self.TREE_TYPE_MAP[self.tree_type], 
                                        self.n_jobs, 
                                        self.verbose, 
                                        self.patts_trans, 
                                        self.wrapper.params, 
                                        self.wrapper.forest, 
                                        )
        
        self.wrapper.params.lam_crit = self.lam_criterion
                    
        if indices is not None:
            use_indices = 1
            indices = np.array(indices).astype(dtype=np.int32)
            indices_weights = np.ones(indices.shape, dtype=np.int32)
            if indices.ndim == 1:
                indices = indices.reshape((1, len(indices)))
                indices_weights = indices_weights.reshape((1, len(indices_weights)))
                
            if (indices.shape[0] != self.n_estimators) or \
                (indices_weights.shape[0] != self.n_estimators):
                raise Exception("""
                    Both 'indices' and 'indices_weights' must be of shape 
                    (n_estimators, x), but are of shape %s and %s, respectively!
                """ % (str(indices.shape), str(indices_weights.shape)))
                                
        else:
            # dummy parameters
            use_indices = 0
            indices = np.empty((0, 0), dtype=np.int32)
            indices_weights = np.empty((0, 0), dtype=np.int32)
        
        self.wrapper.module.fit_extern(X, y, indices, indices_weights, use_indices, self.wrapper.params, self.wrapper.forest)

        return self

    def predict(self, X, indices=None):
        """
        """
        
        if X.dtype != self.numpy_dtype_float:
            X = X.astype(self.numpy_dtype_float)      
        
        if indices is None: 
            indices = np.empty((0, 0), dtype=np.int32)
        else:
            indices = np.array(indices).astype(dtype=np.int32)
            if indices.ndim == 1:
                indices = indices.reshape((1, len(indices)))   
                
        preds = np.ones(X.shape[0], dtype=self.numpy_dtype_float)
        
        #cpu_start = time.time()
        self.wrapper.module.predict_extern(X, preds, indices, self.wrapper.params, self.wrapper.forest)
        #cpu_end = time.time()
        #print("cpu time:\t\t%f" % (cpu_end - cpu_start))

        return preds

    def predict_all(self, X, indices=None):
        """
        """
        if X.dtype != self.numpy_dtype_float:
            X = X.astype(self.numpy_dtype_float)      
        
        if indices is None: 
            indices = np.empty((0, 0), dtype=np.int32)
        else:
            indices = np.array(indices).astype(dtype=np.int32)
            if indices.ndim == 1:
                indices = indices.reshape((1, len(indices)))   
                
        preds = np.ones((X.shape[0], self.n_estimators), dtype=self.numpy_dtype_float)
        
        self.wrapper.module.predict_all_extern(X, preds, indices, self.wrapper.params, self.wrapper.forest)

        return preds

    #This function uses a kernel which predicts for a single tree
    #The kernel is then sequentially called on each tree to get the final prediction
    def cuda_predict(self, X):
        mod = SourceModule("""
        __global__ void cuda_query_tree(float *predictions, 
            int* left_ids, int* right_ids, int* features, float* thres_or_leaf, int *Xtest, int* params)
        {
            register unsigned int i, node_id;
            i = threadIdx.x + blockDim.x * blockIdx.x;             
            node_id = 0;
            if (i < params[0]){
                while (left_ids[node_id] != 0) {
                    if (Xtest[params[1]*i+features[node_id]] <= thres_or_leaf[node_id]) {
                        node_id = left_ids[node_id];
                    } else {
                        node_id = right_ids[node_id];
                    }
                }
                predictions[i] = thres_or_leaf[node_id];
            }
        }

        """)
        cuda_predict = mod.get_function("cuda_query_tree")        

        all_preds = []#np.ones((X.shape[0],self.n_estimators), dtype=np.float32)
        for i in xrange (self.n_estimators):
            preds = np.ones(X.shape[0], dtype=np.float32)
            params = np.array([X.shape[0],X.shape[1],self.n_estimators], dtype=np.int32)
            left_ids, right_ids, features, thres_or_leaf, leaf_criterion = self.tree_as_arrays(i)
            nr_grids = int(float(X.shape[0])/1024.0+1)
            X_1D = np.array(X.ravel(),dtype=np.int32)
            max_threads = 1024
            cuda_start = time.time()
            cuda_predict(drv.Out(preds), drv.In(left_ids), drv.In(right_ids), drv.In(features), drv.In(thres_or_leaf),
            drv.In(X_1D), drv.In(params),block=(max_threads,1,1), grid=(nr_grids,1)) 
            cuda_end = time.time()
            print("cuda time:      %f" % (cuda_end - cuda_start))
            all_preds.append(np.array(preds,np.int32))  
        all_preds = np.array(all_preds,np.int32)
        combined_preds = mode(all_preds)[0][0]
        return combined_preds

    #this function has a cuda kernel were each thread predicts for a single instance on all trees
    def cuda_pred_forest(self,X):
        mod = """
        #include <math.h>
        #define X_dim0 $X_dim0
        #define X_dim1 $X_dim1
        #define N_estimators $N_estimators

        __device__ int max_class(int* preds, int thread_nr){
            int counts[N_estimators]; //assumes more tree than classes?? how many classes are there?
            for(int i = 0; i < N_estimators; i++){
                counts[i] = 0;
            }
            for(int i = 0; i < N_estimators; i++){
                int index = preds[i];
                counts[index] += 1;
            }
            int max_class = 0;
            int max_count = 0;
            for(int i = 0; i < N_estimators; i++){
                if(counts[i] > max_count){
                    max_count = counts[i];
                    max_class = i;
                }
            }
            return max_class;
        }

        __global__ void cuda_query_forest(int *predictions,
            int* left_ids, int* right_ids, int* features, float* thres_or_leaf, int *Xtest, int* params, int* displacements)
        {
            register unsigned int i, node_id;
            i = threadIdx.x + blockDim.x * blockIdx.x;

            int pred_local[N_estimators];

            int X_local[X_dim1];
            for(int k = 0; k < X_dim1; k++){
                X_local[k] = Xtest[X_dim1*i+k];
            }

            for(int j = 0; j < N_estimators; j++){                
                node_id = 0 + displacements[j];
                if (i < X_dim0){
                    while (left_ids[node_id] != 0) {
                        if (X_local[features[node_id]]<= thres_or_leaf[node_id]) {
                            node_id = left_ids[node_id] + displacements[j];
                        } else {
                            node_id = right_ids[node_id] + displacements[j];
                        }
                    }
                pred_local[j] = round(thres_or_leaf[node_id]);
                }

            }
            predictions[i] = max_class(pred_local,i);
        }
        """

        start_time = time.time()
        mod = string.Template(mod)
        code = mod.substitute(X_dim0 = X.shape[0], X_dim1 = X.shape[1], N_estimators = self.n_estimators)
        module = SourceModule(code)
        cuda_pred_forest = module.get_function("cuda_query_forest")
        end_time = time.time()
        print("test time:\t\t%f\n" % (end_time - start_time))


        if X.dtype != np.int32:
            X = X.astype(np.int32) 


        preds = np.ones(X.shape[0], dtype=np.int32)
        params = np.array([X.shape[0],X.shape[1],self.n_estimators], dtype=np.int32)
        max_threads = 1024
        nr_grids = int(float(X.shape[0])/float(max_threads)+1)
        #cuda_start = time.time()
        cuda_pred_forest(drv.Out(preds), drv.In(self.left_ids), drv.In(self.right_ids), drv.In(self.features), drv.In(self.thres_or_leafs),
            drv.In(X), drv.In(params), drv.In(self.tree_nodes_sum),block=(max_threads,1,1), grid=(nr_grids,1))
        #cuda_end = time.time()
        #print("cuda time:\t\t%f\n" % (cuda_end - cuda_start))
        return preds

    def cuda_pred_forest_mult(self,X):
        mod = """
        #define X_dim0 $X_dim0
        #define X_dim1 $X_dim1
        #define N_estimators $N_estimators
        #define X_per_threads $X_per_threads
        __global__ void cuda_query_forest(float *predictions, 
            int* left_ids, int* right_ids, int* features, float* thres_or_leaf, int *Xtest, int* params, int* displacements)
        {
            register unsigned int i, node_id;
            i = threadIdx.x + blockDim.x * blockIdx.x;
            int X_local[X_dim1*X_per_threads];
            for(int k = 0; k < X_dim1*X_per_threads; k++){
                if(X_dim1*i*X_per_threads+k < X_dim0 * X_dim1){
                    X_local[k] = Xtest[X_dim1*i*X_per_threads+k];
                }
            }

            for(int k = 0; k < X_per_threads; k++){
                if (i * X_per_threads + k < X_dim0){
                    for(int j = 0; j < N_estimators; j++){                
                        node_id = 0 + displacements[j];
                        while (left_ids[node_id] != 0) {
                            if (X_local[k*X_dim1 + features[node_id]]<= thres_or_leaf[node_id]) {
                                node_id = left_ids[node_id] + displacements[j];
                            } else {
                                node_id = right_ids[node_id] + displacements[j];
                            }
                        }
                        predictions[j*X_dim0+i*X_per_threads+k] = thres_or_leaf[node_id];

                    }
                }
            }
        }
        """
        X_per_threads = 1
        mod = string.Template(mod)
        code = mod.substitute(X_dim0 = X.shape[0], X_dim1 = X.shape[1], N_estimators = self.n_estimators, X_per_threads = X_per_threads)
        module = SourceModule(code)
        cuda_pred_forest = module.get_function("cuda_query_forest")


        if X.dtype != np.int32:
            X = X.astype(np.int32) 

        preds = np.ones(X.shape[0]*self.n_estimators, dtype=np.float32)
        params = np.array([X.shape[0],X.shape[1],self.n_estimators], dtype=np.int32)
        max_threads = 1024
        nr_grids = int(float(X.shape[0])/float(max_threads*X_per_threads)+1)
        cuda_start = time.time()
        cuda_pred_forest(drv.Out(preds), drv.In(self.left_ids), drv.In(self.right_ids), drv.In(self.features), drv.In(self.thres_or_leafs),
            drv.In(X), drv.In(params), drv.In(self.tree_nodes_sum),block=(max_threads,1,1), grid=(nr_grids,1))
        cuda_end = time.time()
        print("cuda time:\t\t%f\n" % (cuda_end - cuda_start))
        preds = np.reshape(np.array(preds,np.int32),(-1,X.shape[0]))
        #print(preds)
        start_time = time.time()
        combined_preds = mode(preds)[0][0]
        end_time = time.time()
        print("test time:\t\t%f\n" % (end_time - start_time))
        return combined_preds


    #sanity check that all attributes work correctly
    def python_predict(self,left_ids, right_ids, features, thres_or_leaf, leaf_criterion, Xtest, dims):
        preds = np.zeros(dims[0],np.int32)
        for i in xrange (dims[0]):
            node_id = 0
            while (left_ids[node_id] != 0):
                if i == 1:
                    print("Xval is: " + str(Xtest[dims[1]*i+features[node_id]]))
                    print("thres_or_leaf is: " + str(thres_or_leaf[node_id]))
                    print("node_id is: " + str(node_id))
                    print("left_ids is: " + str(left_ids[node_id]))
                    print("right_ids is: " + str(right_ids[node_id]))
                    print("feature is: " + str(features[node_id]))
                    print("")
                if Xtest[dims[1]*i+features[node_id]] <= thres_or_leaf[node_id]:
                    node_id = left_ids[node_id]
                else:
                    node_id = right_ids[node_id]
            preds[i] = thres_or_leaf[node_id]
        return preds


    def store_numpy_forest(self):
        tree_nodes_counter = np.zeros(self.n_estimators,np.int32)
        tree_nodes_sum = np.zeros(self.n_estimators,np.int32)
        for i in xrange (self.n_estimators):
            tree = self.get_wrapped_tree(i)
            tree_nodes_counter[i] = tree.node_counter
            tree_nodes_sum[i] = np.sum(tree_nodes_counter[:i])
        total_nodes = np.sum(tree_nodes_counter)
        left_ids = np.zeros(total_nodes, dtype=np.int32)
        right_ids = np.zeros(total_nodes, dtype=np.int32)
        features = np.zeros(total_nodes, dtype=np.int32)
        thres_or_leaf = np.ones(total_nodes, dtype=np.float32)
        
        tree = self.wrapper.module.TREE()
        node = self.wrapper.module.TREE_NODE()
        index = 0 
        for i in xrange (self.n_estimators):
            displacement = np.sum(tree_nodes_counter[:i])
            #print(displacement)
            tree = self.get_wrapped_tree(i)
            #test_start_time = time.time()
            for j in xrange (tree_nodes_counter[i]):
                index = displacement+j
                self.wrapper.module.get_tree_node_extern(tree, j, node)
                left_ids[index] = node.left_id
                right_ids[index] = node.right_id
                features[index] = node.feature
                thres_or_leaf[index] = node.thres_or_leaf
            #test_end_time = time.time()
            #print("Initiation time is %f:" % (test_end_time - test_start_time))
        self.left_ids = left_ids
        self.right_ids = right_ids
        self.features = features
        self.thres_or_leafs = thres_or_leaf
        self.tree_nodes_sum = tree_nodes_sum
            


    def tree_as_arrays(self,index):
        tree = self.get_wrapped_tree(index)
        n_nodes = tree.node_counter
        #print(n_nodes)
        nodes = []
        for i in xrange(n_nodes):
            node = self.wrapper.module.TREE_NODE()
            self.wrapper.module.get_tree_node_extern(tree, i, node)
            nodes.append(node)
        left_ids = np.zeros(n_nodes, dtype=np.int32)
        right_ids = np.zeros(n_nodes, dtype=np.int32)
        features = np.zeros(n_nodes, dtype=np.int32)
        thres_or_leaf = np.ones(n_nodes, dtype=np.float32)
        leaf_criterion = np.zeros(n_nodes, dtype=np.int32)

        #cuda_start = time.time()
        for i in range(n_nodes):
            left_ids[i] = nodes[i].left_id
            right_ids[i] = nodes[i].right_id
            features[i] = nodes[i].feature
            thres_or_leaf[i] = nodes[i].thres_or_leaf
            leaf_criterion[i] = nodes[i].leaf_criterion
        #cuda_end = time.time()
        #print("cuda time:      %f" % (cuda_end - cuda_start))
        return left_ids, right_ids, features, thres_or_leaf, leaf_criterion



    def get_leaves_ids(self, X, n_jobs=1, indices=None, verbose=0):
        
        if X.dtype != self.numpy_dtype_float:
            X = X.astype(self.numpy_dtype_float)

        if indices is None: 
            indices = np.empty((0, 0), dtype=np.int32)
            preds = np.zeros(X.shape[0] * self.n_estimators, dtype=self.numpy_dtype_float)
            self.wrapper.params.prediction_type = 1
            self.wrapper.params.verbosity_level = verbose
            self.wrapper.module.predict_extern(X, preds, indices, self.wrapper.params, self.wrapper.forest)
        else:

            indices = np.array(indices).astype(dtype=np.int32)
            if indices.ndim == 1:
                indices = indices.reshape((1, len(indices)))            
            
            preds = np.zeros(self.n_estimators * indices.shape[1], dtype=self.numpy_dtype_float)
            self.wrapper.params.prediction_type = 1
            self.wrapper.params.verbosity_level = verbose
            
            self.wrapper.module.predict_extern(X, preds, indices, self.wrapper.params, self.wrapper.forest)            
            
        return preds
    
    def print_parameters(self):
        """
        """
        
        self.wrapper.module.print_parameters_extern(self.wrapper.params)

    def get_n_nodes(self, tree_index):
        
        tree = self.wrapper.module.TREE() 
        self.wrapper.module.get_tree_extern(tree, tree_index, self.wrapper.forest)
        n_nodes = tree.node_counter
        
        return n_nodes
    
    def get_wrapped_tree(self, index):

        tree = self.wrapper.module.TREE() 
        self.wrapper.module.get_tree_extern(tree, index, self.wrapper.forest)
        
        return tree        
        
    def get_tree(self, index):
        
        try:
            import networkx as nx
        except Exception as e:
            raise Exception("Module 'networkx' is required to export the tree structure: %s" % str(e))
        
        tree = self.get_wrapped_tree(index)
        n_nodes = tree.node_counter
        
        nodes = []
        for i in xrange(n_nodes):
            # i is also the node_id (stored consecutively)
            node = self.wrapper.module.TREE_NODE()
            self.wrapper.module.get_tree_node_extern(tree, i, node)
            nodes.append(node)
        
        G = nx.Graph()
        for i in xrange(len(nodes)):
            G.add_node(i)
            
        for i in xrange(len(nodes)):
            G.node[i]['node_id'] = i
            if nodes[i].left_id == 0 and nodes[i].right_id == 0:
                G.node[i]['is_leaf'] = True
            else:
                G.node[i]['is_leaf'] = False
            G.node[i]['leaf_criterion'] = int(nodes[i].leaf_criterion)

        for i in xrange(len(nodes)):
            if nodes[i].left_id != 0:
                G.add_edge(i, nodes[i].left_id)
            if nodes[i].right_id != 0:
                G.add_edge(i, nodes[i].right_id)

        return G

    def draw_tree(self, index, node_stats=None, ax=None, figsize=(200,20), fname="tree.pdf", with_labels=False, edge_width=1.0, edges_alpha=1.0, arrows=False, alpha=0.5, node_size=1000):
        """
        """
        
        tree = self.get_tree(index)
        draw_single_tree(tree, 
                         node_stats=node_stats,
                         ax=ax,
                         figsize=figsize,
                         fname=fname,
                         with_labels=with_labels,
                         arrows=arrows,
                         edge_width=edge_width,
                         edges_alpha=edges_alpha,
                         alpha=alpha,
                         node_size=node_size, 
                         )
                    
    def attach_subtree(self, index, leaf_id, subtree, subtree_index):
        """ Replaces the leaf with id leaf_id 
        with the subtree provided
        """
        
        wrapped_subtree = subtree.get_wrapped_tree(subtree_index)
        self.wrapper.module.attach_tree_extern(index, self.wrapper.forest, wrapped_subtree, int(leaf_id))        
    
    def save(self, fname):
        """
        Saves the model to a file.
        
        Parameters
        ----------
        fname : str
            the filename of the model
        """
        
        d = os.path.dirname(fname)
        if not os.path.exists(d):
            os.makedirs(d)            
        
        try:
            
            # protocol=0 (readable), protocol=1 (python2.3 and 
            # backward), protocol=2 (binary, new python versions
            filehandler_model = open(fname, 'wb') 
            pickle.dump(self, filehandler_model, protocol=2)
            filehandler_model.close()
            
        except Exception, e:
            
            raise Exception("Error while saving model to " + unicode(fname) + u":" + unicode(e))
                
    @staticmethod
    def load(fname):
        """
        Loads a model from disk.
          
        Parameters
        ----------
        filename_model : str
            The filename of the model
  
        Returns
        -------
        Model: the loaded model
        """
  
        try:
          
            filehandler_model = open(fname, 'rb')
            new_model = pickle.load(filehandler_model)
            filehandler_model.close()
            
            return new_model
          
        except Exception, e:
              
            raise Exception("Error while loading model from " + unicode(fname) + u":" + unicode(e))