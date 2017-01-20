'''
Created on Jan 16, 2017

@author: Alexandre Day
'''

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

def set_latex():
    import latex
    
def density_map(x,y,z,
                xlabel=None,ylabel=None,zlabel=None,label=None,
                centers=None,
                out_file=None,title=None,show=True,cmap='coolwarm',remove_tick=False):
    """ 
    Purpose:
        Produces a 2D intensity map
    """
    palette=np.array(sns.color_palette('hls', 10))
    
    fontsize=15

    if label is not None:
        plt.scatter(x,y,c=z,cmap=cmap,alpha=1.0,rasterized=True,label=label)
    else:
        plt.scatter(x,y,c=z,cmap=cmap,alpha=1.0,rasterized=True)
    
    cb=plt.colorbar()
    
    if remove_tick:
        plt.tick_params(labelbottom='off',labelleft='off')
    
    if xlabel is not None:
        plt.xlabel(xlabel,fontsize=fontsize)
    if ylabel is not None:
        plt.ylabel(ylabel,fontsize=fontsize)
    if zlabel is not None:
        cb.set_label(label=zlabel,labelpad=10)
    if title is not None:
        plt.title(title,fontsize=fontsize)
    if label is not None:
        plt.legend(loc='best')
        
    if centers is not None:
        plt.scatter(centers[:,0],centers[:,1],s=200,marker='*',c=palette[3])
    
    if out_file is not None:
        plt.savefig(out_file)
    if show:
        plt.show()

def summary(idx_centers,cluster_label,rho,n_true_center,X,y=None):
    
    n_center=idx_centers.shape[0]
    palette=sns.color_palette('Paired',n_center+10)
    plt.figure(1)

    if y is not None:
        plt.subplot(221,aspect='equal')
        plt.title('True labels')
        print("--> First plotting true clustered labels, then inferred labels, then density map ... ")
        for i in range(n_true_center):
            pos=(y==i)
            plt.scatter(X[pos,0],X[pos,1],c=palette[i],rasterized=True)
        
    plt.subplot(222,aspect='equal')
    for i in range(n_center):
        pos=(cluster_label==i)
        plt.scatter(X[pos,0],X[pos,1],c=palette[i],rasterized=True)
 
    plt.title('Inferred labels')
    
    plt.subplot(223,aspect='equal')
    density_map(X[:,0], X[:,1],rho,centers=X[idx_centers])
    plt.clf()

