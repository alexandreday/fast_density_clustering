'''
Created on Jan 16, 2017

@author: Alexandre Day
'''

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patheffects as PathEffects
from .mycolors import COLOR_PALETTE
from .fdc import FDC
import math

def set_nice_font(size = 18, usetex=False):
    font = {'family' : 'serif', 'size'   : size}
    plt.rc('font', **font)
    if usetex is True:
        plt.rc('text', usetex=True)
    
def density_map(X, z,
                xlabel=None, ylabel=None, zlabel=None, label=None,
                centers=None,
                psize=20,
                out_file=None, title=None, show=True, cmap='coolwarm',
                remove_tick=False,
                use_perc=False,
                rasterized = True,
                fontsize = 15,
                vmax = None,
                vmin = None
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
            plt.scatter(x,y,c=z,cmap=cmap,s=psize,alpha=1.0,rasterized=rasterized,label=label,vmax=vmax,vmin=vmin)
        else:
            plt.scatter(x,y,c=z,cmap=cmap,s=psize,alpha=1.0,rasterized=rasterized, vmax=vmax,vmin=vmin)
    
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

def scatter_w_label(x, y, z, psize=20, label = None):

    unique_z=np.sort(np.unique(z.flatten()))
    mycol = COLOR_PALETTE()

    plt.subplots(figsize=(8,6))

    for i, zval in enumerate(unique_z):
        pos=(z.flatten()==zval)
        if label is not None:
            plt.scatter(x[pos],y[pos],s=psize,c=mycol[i], label=label[i], rasterized=True)
        else:
            plt.scatter(x[pos],y[pos],s=psize,c=mycol[i], rasterized=True)
    
    if label is not None:
        plt.legend(loc='best',fontsize=12)

    plt.tight_layout()
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

def plot_inferred_label(ax, X, idx_centers, cluster_label, palette, psize = 20, eta = None, eta_show = True, fontsize=15):
    
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

    if eta is not None: # displaying eta parameter
        if eta_show:
            txt = ax.annotate("$\eta=%.2f$"%eta,[xmin+0.15*dx,ymin+0.05*dy], xytext=(0,0), textcoords='offset points',
            fontsize=20,horizontalalignment='center', verticalalignment='center')
            txt.set_path_effects([
                PathEffects.Stroke(linewidth=5, foreground="w"),
                PathEffects.Normal()])
    
    plt.title('Inferred labels',fontsize=fontsize)
    plt.tight_layout()


def summary(idx_centers, cluster_label, rho, X, eta = None, eta_show = True, y=None, psize=20, savefile=None, show=False,
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
    plot_inferred_label(ax, X, idx_centers, cluster_label, palette, psize =psize, eta = eta, eta_show = eta_show, fontsize=fontsize)

    plt.subplot(133)
    density_map(X,rho,centers=X[idx_centers],title='Density map', psize=psize, show=False)

    plt.tight_layout()
    if savefile:
        plt.savefig(savefile, dpi=300)
    if show is True:
        plt.show()
        plt.clf()

def summary_model(model, eta=None, ytrue = None, show=True, savefile = None, eta_show = True):
    """ Summary figure passing in only an FDC object (model), noise can be specified via the eta parameter """
    
    if eta is None:
        eta_ = model.eta
        idx_centers = model.idx_centers
        cluster_label = model.cluster_label
    else:
        pos = np.argmin(np.abs(np.array(model.noise_range)-eta))
        eta_ = model.noise_range[pos]
        idx_centers = model.hierarchy[pos]['idx_centers']
        cluster_label = model.hierarchy[pos]['cluster_labels']

    rho = model.rho
    X = model.X
    summary(idx_centers, cluster_label, rho, X, y=ytrue, eta = eta_, show=show, savefile=savefile, eta_show=eta_show)

def inferred_label(model, eta=None, show=True, savefile = None, eta_show = True, fontsize =15, psize = 20):

    if eta is None:
        eta_ = model.noise_range[-1]
        idx_centers = model.idx_centers
        cluster_label = model.cluster_label
    else:
        pos = np.argmin(np.abs(np.array(model.noise_range)-eta))
        eta_ = model.noise_range[pos]
        idx_centers = model.hierarchy[pos]['idx_centers']
        cluster_label = model.hierarchy[pos]['cluster_labels']

    rho = model.rho
    X = model.X

    n_sample=X.shape[0]
    n_center=idx_centers.shape[0]
    palette=COLOR_PALETTE()
    
    plt.figure(1,figsize=(10,10))
    ax = plt.subplot(111)
    plot_inferred_label(ax, X, idx_centers, cluster_label, palette, psize = psize, eta = eta_, eta_show = eta_show, fontsize=fontsize)

    if savefile is not None:
        plt.savefig(savefile)

    if show is True:
        plt.show()

    plt.clf()

def cluster_w_label(X, y, Xcluster=None, show=True, savefile = None, fontsize =15, psize = 20, title=None, w_label = True, figsize=None,
     dpi=200, alpha=0.7, edgecolors=None, cp_style=1, w_legend=False, outlier=True):

    
    if figsize is not None:
        plt.figure(figsize=figsize)
    y_unique_ = np.unique(y)
    
    palette = COLOR_PALETTE(style=cp_style)
    idx_centers = []
    ax = plt.subplot(111)
    all_idx = np.arange(len(X))
    
    if outlier is True:
        y_unique = y_unique_[y_unique_ > -1]
    else:
        y_unique = y_unique_
    n_center = len(y_unique)

    for i, yu in enumerate(y_unique):
        pos=(y==yu)
        Xsub = X[pos]
        plt.scatter(Xsub[:,0],Xsub[:,1],c=palette[i], s=psize, rasterized=True, alpha=alpha, edgecolors=edgecolors, label = yu)
        
        if Xcluster is not None:
            Xmean = Xcluster[i]
        else:
            Xmean = np.mean(Xsub, axis=0)
        #Xmean = np.mean(Xsub,axis=0)
        idx_centers.append(all_idx[pos][np.argmin(np.linalg.norm(Xsub - Xmean, axis=1))])

    if outlier is True:
        color_out = {-3 : '#ff0050', -2 : '#9eff49', -1 : '#89f9ff'}
        for yi in [-3, -2, -1]:
            pos = (y == yi)
            if np.count_nonzero(pos) > 0:
                Xsub = X[pos]
                plt.scatter(Xsub[:,0],Xsub[:,1],c=color_out[yi], s=psize, rasterized=True, alpha=alpha, marker="2",edgecolors=edgecolors, label = yi)
            

    if w_label is True:
        centers = X[idx_centers]
        for xy, i in zip(centers, y_unique) :
            # Position of each label.
            txt = ax.annotate(str(i),xy,
            xytext=(0,0), textcoords='offset points',
            fontsize=fontsize,horizontalalignment='center', verticalalignment='center'
            )
            txt.set_path_effects([
                PathEffects.Stroke(linewidth=5, foreground="w"),
                PathEffects.Normal()])
        
    
    xmin,xmax = plt.xlim()
    ymin,ymax = plt.ylim()
    dx = xmax - xmin
    dy = ymax - ymin
    plt.xticks([])
    plt.yticks([])
    
    if title is not None:
        plt.title(title,fontsize=fontsize)
    if w_legend is True:
        plt.legend(loc='best')

    plt.tight_layout()
    if savefile is not None:
        if dpi is None:
            plt.savefig(savefile)
        else:
            plt.savefig(savefile,dpi=dpi)

    if show is True:
        plt.show()
        plt.clf()
        plt.close()

    return ax
    
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
    plt.ylabel('$\eta$',fontsize=fontsize)
    plt.title('Clustering hierarchy',fontsize=fontsize)
    plt.tight_layout()

    if show is True:
        plt.show()
    if savefile is not None:
        plt.savefig(savefile)

    plt.clf()

def viSNE(X_2D, X_original, markers, show=True, savefig=None, col_index = None, col_wrap = 4, downsample = None):
    import pandas as pd
    """Plots intensity on top of data.

    Parameters
    ------------

    X_2D : coordinates of the data points (2d points)   

    X_original : original marker intensity

    makers : list of str, list of names of the markers (for showing as titles)

    savefig : str, if u want to save figure, should be the name of the output file

    downsample : int, number of data points in a random sample of the original data

    show : bool, if u want to see the plots

    """

    X = X_2D
    if col_index is not None:
        z_df = pd.DataFrame(X_original[:,col_index], columns=[markers[i] for i in col_index])
    else:
        z_df = pd.DataFrame(X_original, columns=markers)

    facegrid(X[:,0], X[:,1], z_df, show=show, savefig=savefig, downsample = downsample, col_wrap=col_wrap)


def facegrid(x, y, z_df, col_wrap = 4, downsample = None, show=True, savefig = None):

    n_sample = x.shape[0]
    if downsample is not None:
        random_sub = np.random.choice(np.arange(0, n_sample, dtype=int), downsample, replace=False)
        xnew = x[random_sub]
        ynew = y[random_sub]
        znew = z_df.iloc[random_sub]
    else:
        xnew = x
        ynew = y
        znew = z_df
   
    n_plot = z_df.shape[1]

    assert len(x) == len(y) and len(x) == len(z_df)

    n_row = math.ceil(n_plot / col_wrap)
    
    xfig = 12
    yfig = 8
    xper_graph = xfig/col_wrap
    yper_graph = yfig/n_row

    if n_row >= col_wrap:
        xper_graph = yper_graph
    else:
        yper_graph = xper_graph

    plt.figure(figsize=(xper_graph*col_wrap,yper_graph*n_row))

    col_names = z_df.columns.values

    for i in range(n_plot):
        ax = plt.subplot(n_row, col_wrap, i+1)
        my_scatter(xnew, ynew, znew.iloc[:,i].as_matrix(), ax)
        ax.set_title(col_names[i])
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()

    if show is True:
        plt.show()

    if savefig is True:
        plt.savefig(savefig)

def my_scatter(x, y, z, ax):

    cmap = plt.get_cmap('coolwarm')
    
    argz = np.argsort(z)

    #zmin, zmax = z[argz[0]], z[argz[-1]]
    #ztmp = (z+zmin)*(1./zmax)

    n_sample = len(x)
    bot_5 = round(n_sample*0.05)
    top_5 = round(n_sample*0.95)
    mid = argz[bot_5:top_5]
    bot = argz[:bot_5]
    top = argz[top_5:]

    x_mid = x[mid]
    y_mid = y[mid]
    z_mid = z[mid]

    x_bot = x[bot]
    y_bot = y[bot]
    z_bot = z[bot]

    x_top = x[top]
    y_top = y[top]
    z_top = z[top]

    ax.scatter(x_mid, y_mid, c = z_mid, cmap = cmap, s=6)
    ax.scatter(x_bot, y_bot, c = "purple", s=4)
    ax.scatter(x_top, y_top, c = "#00FF00",s=4)

def select_data(X, y, X_original = None, option = None, loop=False, kwargs=None):
    from .widget import Highlighter
    # Taking selection from the user, will plot an histogram of the underlying data (default)
    # Other options are {mnist, etc. etc.}

    if loop is True:
        n_repeat = 10

    if option == 'mnist':

        for _ in range(n_repeat):
            if kwargs is not None:
                ax = cluster_w_label(X, y, show=False, **kwargs)
            else:
                ax = cluster_w_label(X, y, show=False)

            highlighter = Highlighter(ax, X[:,0], X[:,1])
            selected_regions = highlighter.mask
            plt.close()
            X_sub = X_original[selected_regions]
            n_plot = min([len(X_sub), 16])
            #print(xcluster.shape)

            rpos = np.random.choice(np.arange(len(X_sub)), size=n_plot)
            #print(rpos)
            fig, ax = plt.subplots(4,4,figsize=(8,8))
            count = 0
            for i in range(4):
                for j in range(4):
                    count+=1
                    if count > n_plot:
                        break
                    ax[i,j].imshow(X_sub[rpos[4*i+j]].reshape(28,28),cmap="Greys")
                    ax[i,j].set_xticks([])
                    ax[i,j].set_yticks([])
                
            plt.tight_layout()
            plt.show()
            plt.clf()

