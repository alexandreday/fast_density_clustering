'''
Created on Jan 19, 2017

@author: Alexandre Day

    Purpose:
        Compute robustness of clustering for indivial data point
        
'''

import numpy as np
import pickle
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn import datasets
import time
from density_cluster import DCluster

def main():

    n_true_center=15
    n_sample=10000
    X,y=datasets.make_blobs(n_sample,2,n_true_center,random_state=0)    
    
    dcluster=DCluster()
    idx_sub=np.sort(np.random.choice(range(n_sample), round(0.9*n_sample), replace=False))
    
    X_sub=X[idx_sub]
    
    cluster_label_X_sub,idx_centers,rho,delta,kd_tree=dcluster.fit(X_sub)
    
    """cluster_members is a dictionary from cluster (labelled from [0,n_center[ ) to the absolute idx of it's member points 
    """
    
    cluster_label_X=np.full(n_sample,-1,dtype=np.int32)

    cluster_label_X[idx_sub]=cluster_label_X_sub
    
    cluster_members=group_by_label(cluster_label_X_sub,idx_sub)
    
    #print(NH_Cluster(idx_sub[np.random.randint(0,5000,5)],cluster_label_X,X,kd_tree,idx_sub))
    
def NH_Cluster(a_IDX,cluster_label_X,X,kd_tree,idx_sub):
    """
    Purpose:
        Determines the the neighborhood for data points identified
        by their absolute index in a_IDX array.
    Return:
        List of neighborhood (in absolute index) for each data point in a_IDX.
    """
    
    n_data=len(a_IDX)
    _,nn_r_IDX=kd_tree.query(list(X[a_IDX]), k=40)
    nn_a_IDX=idx_sub[nn_r_IDX]
            
    cluster_label_a_IDX=cluster_label_X[a_IDX].reshape(-1,1)
    
    cluster_label_nn_a_IDX=cluster_label_X[nn_a_IDX]

    tmp=(cluster_label_nn_a_IDX == cluster_label_a_IDX)

    return [nn_a_IDX[i][t] for i,t in zip(range(n_data),tmp)]
    
    
    
    
    
    
    
    #return cluster_members
    
    
    
    
    
    
    
    
    

def group_by_label(cluster_label,idx_sub):
    """
    Purpose:
        Given the cluster labels found for every data point, return a 
        dictionnary from cluster label to absolute indices in that cluster
        The absolute indices are the indices of the data in the whole dataset
        
    """
    sort_idx = np.argsort(cluster_label)
    a_sorted = cluster_label[sort_idx]
    unq_first = np.concatenate(([True], a_sorted[1:] != a_sorted[:-1]))
    unq_items = a_sorted[unq_first]
    unq_count = np.diff(np.nonzero(unq_first)[0])
    groups=np.split(sort_idx, np.cumsum(unq_count))
    
    return dict(zip(unq_items,[np.sort(idx_sub[g]) for g in groups]))

#def NH_Cluster(a_IDX,cluster_members,):
#===============================================================================
#     
#     idx_sub=np.random.shuffle(range(n_sample))[:round(0.9*n_sample)]
#     
#     
#     
# 
#     nn_dist,nn_list=tree.query(list(X), k=40)
#===============================================================================







#===============================================================================
# 
# 
# 
# def NH(a_label,cluster,tree): # a_label refers to absolute label
#===============================================================================




    
if __name__=="__main__":
    main()

