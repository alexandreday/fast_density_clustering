'''
Created on Jan 16, 2017

@author: Alexandre Day
'''

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patheffects as PathEffects
from .mycolors import COLOR_PALETTE
from .fdc import FDC

def set_nice_font(size = 18):
    font = {'family' : 'serif', 'size'   : size}
    plt.rc('font', **font)

def set_latex():
    import latex
    
def density_map(X,z,
                xlabel=None,ylabel=None,zlabel=None,label=None,
                centers=None,
                psize=20,
                out_file=None,title=None,show=True,cmap='coolwarm',
                remove_tick=False,
                use_perc=False,
                rasterized = True,
                fontsize = 15
                ):
    """Plots a 2D density map given x,y coordinates and an intensity z for
    every data point

    Parameters
    ----------
    X : array-like, shape=[n_samples,2]
        Input points.
    z : array-like, shape=[n_samples]
        Density at every point

    Returns
    -------
    None
    """
    x, y = X[:,0], X[:,1]
    
    fontsize = fontsize

    if use_perc :
        n_sample = len(x)
        outlier_window = int(0.05 * n_sample)

        argz = np.argsort(z)
        bot_outliers = argz[:outlier_window]
        top_outliers = argz[-outlier_window:]
        typical = argz[outlier_window:-outlier_window]

        # plot typical
        plt.scatter(x[typical],y[typical],c=z[typical],cmap=cmap,s=psize, alpha=1.0,rasterized=rasterized)
        cb=plt.colorbar()
        # plot bot outliers (black !)
        plt.scatter(x[bot_outliers],y[bot_outliers],c='black',s=psize,alpha=1.0,rasterized=rasterized)
        # plot top outliers (green !)
        plt.scatter(x[top_outliers],y[top_outliers],c='#36DA36',s=psize,alpha=1.0,rasterized=rasterized)

    else:
        if label is not None:
            plt.scatter(x,y,c=z,cmap=cmap,s=psize,alpha=1.0,rasterized=rasterized,label=label)
        else:
            plt.scatter(x,y,c=z,cmap=cmap,s=psize,alpha=1.0,rasterized=rasterized)
    
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
        plt.scatter(centers[:,0],centers[:,1], c='lightgreen', marker='*',s=200, edgecolor='black',linewidths=0.5)
    
    if out_file is not None:
        plt.savefig(out_file)
    if show:
        plt.show()

def scatter_w_label(x,y,z,psize=20):
    n_unique_label=np.unique(z).shape[0]
    for i in range(n_unique_label):
        pos=(z==i)
        plt.scatter(x[pos],y[pos],s=psize,c=contrast_colors(i), rasterized=True)
    plt.show()


def plot_true_label(X, palette, y=None, fontsize = 15, psize = 20):
    plt.title('True labels', fontsize=fontsize)
    print("--> Plotting summary: True clustered labels, inferred labels and density map ")
    if y is None:
        plt.scatter(X[:,0],X[:,1],c=palette[0],rasterized=True)
    else:
        y_unique = np.unique(y)
        for i, yu in enumerate(y_unique):
            pos=(y==yu)
            plt.scatter(X[pos,0],X[pos,1], s=psize, c=palette[i],rasterized=True)

def plot_inferred_label(ax, X, idx_centers, cluster_label, palette, psize = 20, delta = None, delta_show = True, fontsize=15):
    
    n_center = len(idx_centers)

    for i in range(n_center):
        pos=(cluster_label==i)
        plt.scatter(X[pos,0],X[pos,1],c=palette[i], s=psize, rasterized=True)
     
    centers = X[idx_centers]
    for xy, i in zip(centers, range(n_center)) :
        # Position of each label.
        txt = ax.annotate(str(i),xy,
        xytext=(0,0), textcoords='offset points',
        fontsize=20,horizontalalignment='center', verticalalignment='center'
        )
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
    
    xmin,xmax = plt.xlim()
    ymin,ymax = plt.ylim()
    dx = xmax - xmin
    dy = ymax - ymin

    if delta is not None: # displaying delta parameter
        if delta_show:
            txt = ax.annotate("$\delta=%.2f$"%delta,[xmin+0.15*dx,ymin+0.05*dy], xytext=(0,0), textcoords='offset points',
            fontsize=20,horizontalalignment='center', verticalalignment='center')
            txt.set_path_effects([
                PathEffects.Stroke(linewidth=5, foreground="w"),
                PathEffects.Normal()])
    
    plt.title('Inferred labels',fontsize=fontsize)
    plt.tight_layout()


def summary(idx_centers, cluster_label, rho, X, delta = None, delta_show = True, y=None, psize=20, savefile=None, show=False,
plot_to_show = None
):
    """ Summary plots : original labels (if available), inferred labels and density map used for clustering """

    fontsize=15
    n_sample=X.shape[0]
    n_center=idx_centers.shape[0]
    palette=COLOR_PALETTE()
    
    plt.figure(1,figsize=(22,10))

    plt.subplot(131)
    plot_true_label(X, palette, y=y,fontsize=fontsize, psize = psize)

    ax = plt.subplot(132)
    plot_inferred_label(ax, X, idx_centers, cluster_label, palette, psize =psize, delta = delta, delta_show = delta_show, fontsize=fontsize)

    plt.subplot(133)
    density_map(X,rho,centers=X[idx_centers],title='Density map', psize=psize, show=False)

    if savefile:
        plt.savefig(savefile)
    if show is True:
        plt.show()

    plt.clf()

def summary_model(model, delta=None, show=True, savefile = None, delta_show = True):
    """ Summary figure passing in only an FDC object (model), noise can be specified via the delta parameter """
    
    if delta is None:
        delta_ = model.noise_threshold
        idx_centers = model.idx_centers
        cluster_label = model.cluster_label
    else:
        pos = np.argmin(np.abs(np.array(model.noise_range)-delta))
        delta_ = model.noise_range[pos]
        idx_centers = model.hierarchy[pos]['idx_centers']
        cluster_label = model.hierarchy[pos]['cluster_labels']

    rho = model.rho
    X = model.X
    summary(idx_centers, cluster_label, rho, X, delta = delta_, show=show, savefile=savefile, delta_show=delta_show)

def inferred_label(model, delta=None, show=True, savefile = None, delta_show = True, fontsize =15, psize = 20):

    if delta is None:
        delta_ = model.noise_range[-1]
        idx_centers = model.idx_centers
        cluster_label = model.cluster_label
    else:
        pos = np.argmin(np.abs(np.array(model.noise_range)-delta))
        delta_ = model.noise_range[pos]
        idx_centers = model.hierarchy[pos]['idx_centers']
        cluster_label = model.hierarchy[pos]['cluster_labels']

    rho = model.rho
    X = model.X

    n_sample=X.shape[0]
    n_center=idx_centers.shape[0]
    palette=COLOR_PALETTE()
    
    plt.figure(1,figsize=(10,10))
    ax = plt.subplot(111)
    plot_inferred_label(ax, X, idx_centers, cluster_label, palette, psize = psize, delta = delta_, delta_show = delta_show, fontsize=fontsize)

    if savefile is not None:
        plt.savefig(savefile)

    if show is True:
        plt.show()

    plt.clf()

def cluster_w_label(X, model:FDC,
                xlabel=None,ylabel=None,zlabel=None,label=None, 
                psize=20,
                out_file=None,title=None,
                show=True,
                remove_tick=False,
                rasterize=False,
                fontsize = 15
                ):

    """Plots the data point X colored with the clustering assignment found by FDC class

    """
    from matplotlib.offsetbox import AnchoredText

    fontsize = fontsize

    n_center = len(model.idx_centers)
    cluster_label = model.cluster_label
    idx_centers = model.idx_centers
    palette=COLOR_PALETTE()

    ax = plt.subplot(111)
    
    for i in range(n_center):
        pos = (cluster_label==i)
        plt.scatter(X[pos,0], X[pos,1], c=palette[i], s=psize, rasterized=rasterize)
    
    centers = X[idx_centers]
    for xy, i in zip(centers, range(n_center)) :
        # Position of each label.
        txt = ax.annotate(str(i),xy,
        xytext=(0,0), textcoords='offset points',
        fontsize=20, horizontalalignment='center', verticalalignment='center'
        )
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])

    if remove_tick:
        plt.tick_params(labelbottom='off',labelleft='off')
    if label is not None:
        anchored_text = AnchoredText(label, loc=2)
        ax.add_artist(anchored_text)
        #axupdate_frame(bbox, fontsize=None)

    if xlabel is not None:
        plt.xlabel(xlabel,fontsize=fontsize)
    if ylabel is not None:
        plt.ylabel(ylabel,fontsize=fontsize)
    if zlabel is not None:
        cb.set_label(label=zlabel,labelpad=10)
    if title is not None:
        plt.title(title,fontsize=fontsize)

    plt.tight_layout(pad = 1.2)

    if out_file is not None:
        plt.savefig(out_file)
    if show:
        plt.show()
    

def summary_v2(idx_centers, cluster_label, rho, X, n_true_center=1, y=None, psize=20, savefile=None, show=False):
    """ Summary plots w/o density map """

    fontsize=15
    n_sample=X.shape[0]
    n_center=idx_centers.shape[0]
    palette=COLOR_PALETTE()

    '''plt.figure(1,figsize=(10,10))

    plt.subplot(131)
    plt.title('True labels',fontsize=fontsize)
    print("--> Plotting summary: True clustered labels, inferred labels and density map ")
    if y is None:
        plt.scatter(X[:,0],X[:,1],c=palette[0],rasterized=True)
    else:
        for i in range(n_true_center):
            pos=(y==i)
            plt.scatter(X[pos,0],X[pos,1], s=psize,c=palette[i],rasterized=True)
    '''        
    ax = plt.subplot(111)
    for i in range(n_center):
        pos=(cluster_label==i)
        plt.scatter(X[pos,0],X[pos,1],c=palette[i], s=psize, rasterized=True)
     
    centers = X[idx_centers]
    for xy, i in zip(centers, range(n_center)) :
        # Position of each label.
        txt = ax.annotate(str(i),xy,
        xytext=(0,0), textcoords='offset points',
        fontsize=20,horizontalalignment='center', verticalalignment='center'
        )
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])

    #plt.title('Inferred labels',fontsize=fontsize)
    #plt.tight_layout()
    #plt.subplot(133)

    #density_map(X,rho,centers=X[idx_centers],title='Density map', psize=psize, show=False)

    if savefile:
        plt.savefig(savefile)
    if show is True:
        plt.show()

    plt.clf()


def dendrogram(model, show=True, savefile=None):
    from scipy.cluster.hierarchy import dendrogram as scipydendro
    from .hierarchy import compute_linkage_matrix
    
    fontsize=15

    Z = compute_linkage_matrix(model)
    scipydendro(Z)

    plt.ylim(0, 1.2 * model.max_noise)

    plt.xlabel('cluster $\#$',fontsize=fontsize)
    plt.ylabel('$\delta$',fontsize=fontsize)
    plt.title('Clustering hierarchy',fontsize=fontsize)
    plt.tight_layout()

    if show is True:
        plt.show()
    if savefile :
        plt.savefig(savefile)

    plt.clf()