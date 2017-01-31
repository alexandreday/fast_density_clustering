'''
Created on Jan 16, 2017

@author: Alexandre Day

    Purpose:
        Code for performing robust density clustering 
        
'''

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.neighbors import KernelDensity
from sklearn import datasets
import plotting
import time
from sklearn.model_selection import train_test_split
from numpy.random import random
import sys,os
from special_datasets import gaussian_mixture
from tsne.tsne import TSNE # download from https://github.com/alexandreday/tsne_visual
import pickle

def main():
    """
    Example on a gaussian mixture with n=15 centers in 2 dimension with 100000 data points
    """
    #tsne=TSNE(n_components=2,n_iter=5000,angle=0.5)    
    n_true_center=10
    
    X,y=datasets.make_blobs(10000,2,n_true_center,random_state=0)
    #===========================================================================
    # X,y=gaussian_mixture(n_sample=10000,n_center=n_true_center,sigma_range=[0.25,0.5,1.25],pop_range=[0.1,0.02,0.1,0.1,0.3,0.1,0.08,0.02,0.08,0.1],
    #                      random_state=10
    #                      )
    # 
    #===========================================================================
    
    
    #plotting.scatter_w_label(X[:,0],X[:,1],y)
    #exit()
    #exit()
    #Xred=np.fromfile('result.dat').reshape(-1,2)
    #Xred=X
    #Xred=tsne.fit_transform(X)
    #exit()
    
    dcluster=DCluster(bandwidth='auto',perplexity=40.,NH_size=40)
    cluster_label,idx_centers,rho,delta,kde_tree=dcluster.fit(X)    
    plotting.summary(idx_centers,cluster_label,rho,n_true_center,X,y)
    
class DCluster:
    """ Density clustering via kernel density modelling
    
       
    Parameters
    ----------
    
    n_components : int, optional (default: 2)
        Dimension of the embedded space.

    perplexity : int, optional (default: 40)
        The perplexity is related to the number of nearest neighbors that
        is used in other manifold learning algorithms. Larger datasets
        usually require a larger perplexity. Consider selecting a value
        between 20 and 100. Perplexity can be thought as the effective number
        of neighbors.
    
    random_state: int, optional (default: 0)
        Random number for seeding random number generator
    
    verbose: int, optional (default: 1)
        Set to 0 if you don't want to see print to screen.
    
    bandwidth: str, optional (default: 'auto')
        If you want the bandwidth to be set automatically or want to set it yourself.
        Valid options = {'auto' | 'manual'}
        
    bandwidth_value: float, required if bandwidth='manual'
    
    """
    
    def __init__(self,perplexity=40.0,test_size=0.1,
                 random_state=0,verbose=1,NH_size=40,
                 bandwidth='auto',bandwidth_value=None):
        self.perplexity=round(perplexity)
        self.test_size=test_size
        self.random_state=random_state
        self.verbose=verbose
        self.bandwidth_value=None
        self.NH_size=NH_size
        if bandwidth is not 'auto': # Need to implement something else here
            assert bandwidth is 'manual'
            self.bandwidth_value=bandwidth_value
    
    def fit(self,X):
        if self.verbose==0:
            blockPrint()
            
        n_sample=X.shape[0]
        print("--> Starting clustering with n=%i samples..."%n_sample)
        start=time.time()
        
        # Find optimal bandwidth

        X_train, X_test, _,_ = train_test_split(X,range(X.shape[0]),test_size=0.5, random_state=0)
        
        if self.bandwidth_value is None:
            print("--> Finding optimal bandwidth ...")
            bandwidthCV=find_optimal_bandwidth(X,X_train,X_test)
        else:
            bandwidthCV=self.bandwidth_value
        
        print("--> Using bandwidth = %.3f"%bandwidthCV)
    
        print("--> Computing density ...")
        rho,kde=compute_density(X,bandwidth=bandwidthCV)
    
        print("--> Finding centers ...")
        delta,nn_delta,idx_centers,density_graph=compute_delta(X,rho,kde.tree_,cutoff=self.NH_size)
        
        print("--> Assigning labels ...")
        cluster_label=assign_cluster(idx_centers,nn_delta,density_graph)
        
        print("--> Done in %.3f s"%(time.time()-start))
        
        print("--> Found %i centers ! ..."%idx_centers.shape[0])
        
        enablePrint()
        
        return cluster_label,idx_centers,rho,delta,kde.tree_
    
 
def bandwidth_estimate(X):
    """
    Purpose:
        Gives a rough estimate of the optimal bandwidth ... this is a bit wonky ... needs some more justification ... 
    """
    
    assert X.shape[1]==2
    n_sample=X.shape[0]
    x25,x75=np.percentile(X[:,0],25),np.percentile(X[:,0],75)
    y25,y75=np.percentile(X[:,1],25),np.percentile(X[:,1],75)
    Aeff=(y75-y25)*(x75-x25)
    
    dist_uniform=np.sqrt(Aeff/(n_sample*0.5))
    
    kde=KernelDensity(bandwidth=0.3, algorithm='kd_tree', kernel='gaussian', metric='euclidean', atol=0.5, rtol=0.05, breadth_first=True, leaf_size=40)
    kde.fit(X)
    nn_dist,_=kde.tree_.query(list(X), k=40)
    return dist_uniform/np.mean(nn_dist[:,1:]) # --- need to understand this better !
  
def log_likelihood_test_set(bandwidth,X_train,X_test):
    """
    Purpose:
        Fit the kde model on the training set given some bandwidth and evaluates the log-likelihood of the test set
    """
    
    kde=KernelDensity(bandwidth=bandwidth, algorithm='kd_tree', kernel='gaussian', metric='euclidean', atol=0.0000005, rtol=0.000005, breadth_first=True, leaf_size=40)
    kde.fit(X_train)
    return -kde.score(X_test)

def find_optimal_bandwidth(X,X_train,X_test):
    """
    Purpose:
        Given a training and a test set, finds the optimal bandwidth in a gaussian kernel density model
    """
    from scipy import optimize
    hi=bandwidth_estimate(X)
    #===========================================================================
    # d=[]
    # for h in np.arange(0.05,5.0,0.05):
    #     vv=[h,log_likelihood_test_set(h,X_train,X_test)]
    #     print(vv)
    #     d.append(vv)
    # d=np.array(d)
    # plt.scatter(d[:,0],d[:,1])
    # plt.show()
    # exit()
    #===========================================================================
    args=(X_train,X_test)
    options={'maxiter':25,'disp':False}
    res=optimize.minimize(log_likelihood_test_set,hi, args=args,method='L-BFGS-B', bounds=[(0.01,5)], tol=0.001, options=options)
    
    return res.x

def compute_density(X,bandwidth=1.0):
    """
    Purpose:
        Given an array of data, computes the local density of every point using kernel density estimation
        
    Return: array, shape(n_sample,1)

    """
    kde=KernelDensity(bandwidth=bandwidth, algorithm='kd_tree', kernel='gaussian', metric='euclidean', atol=0.000005, rtol=0.00005, breadth_first=True, leaf_size=40)
    kde.fit(X)
    
    return kde.score_samples(X),kde
    
def compute_delta(X,rho,tree,cutoff=40):
    """
    Purpose:
        Computes distance to nearest-neighbor with higher density
    """
    n_sample,n_feature=X.shape
    
    maxdist=np.linalg.norm([np.max(X[:,i])-np.min(X[:,i]) for i in range(n_feature)])
    
    nn_dist,nn_list=tree.query(list(X), k=cutoff)
    delta=maxdist*np.ones(n_sample,dtype=np.float)
    nn_delta=np.ones(n_sample,dtype=np.int)
    
    density_graph=[[] for i in range(n_sample)] # store incoming leaves
    
    for i in range(n_sample):
        idx=index_greater(rho[nn_list[i]])
        if idx is not None:
            density_graph[nn_list[i,idx[0]]].append(i)
            nn_delta[i]=nn_list[i,idx[0]]
            delta[i]=nn_dist[i,idx[0]]
        else:
            nn_delta[i]=-1
    
    idx_centers=np.array(range(n_sample))[delta > 0.99*maxdist]
    
    return delta,nn_delta,idx_centers,density_graph

def index_greater(array):
    """
    Purpose:
        Fast compiled function for finding first item in an array that has a value greater than the first element in that array
        If no element is found, returns None
    """
    item=array[0]
    for idx, val in np.ndenumerate(array):
        if val > item:
            return idx


def assign_cluster_deep(root,cluster_label,density_graph,label):
    """
    Purpose:
        Recursive function for assigning labels for a tree graph.
        Stopping condition is met when the root is empty (i.e. a leaf has been reached)
        
    """
    
    if not root:  # then must be a leaf !
        return
    else:
        for child in root:
            cluster_label[child]=label
            assign_cluster_deep(density_graph[child],cluster_label,density_graph,label)

def assign_cluster(idx_centers,nn_delta,density_graph):
    """ 
    Purpose:
        Given the cluster centers and the local gradients (nn_delta) assign to every
        point a cluster label
    """
    
    n_center=idx_centers.shape[0]
    n_sample=nn_delta.shape[0]
    cluster_label=-1*np.ones(n_sample,dtype=np.int)
    
    for c,label in zip(idx_centers,range(n_center)):
        cluster_label[c]=label
        assign_cluster_deep(density_graph[c],cluster_label,density_graph,label)    
    return cluster_label    

def blockPrint():
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    sys.stdout = sys.__stdout__


if __name__=="__main__":
    main()

