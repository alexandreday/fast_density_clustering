'''
Created on Jan 16, 2017

@author: Alexandre Day
'''

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patheffects as PathEffects
import seaborn as sns

sns.set_style("whitegrid")
global contrast_colors

contrast_colors=["#1CE6FF", "#FF34FF", "#FF4A46",
 "#008941", "#006FA6", "#A30059", "#0000A6", "#63FFAC", 
 "#B79762", "#004D43", "#8FB0FF", "#997D87","#5A0007", "#809693", 
 "#1B4400", "#4FC601", "#3B5DFF", "#4A3B53", "#FF2F80","#61615A", "#BA0900",
  "#6B7900", "#00C2A0", "#FFAA92", "#FF90C9", "#B903AA", "#D16100","#DDEFFF",
   "#000035", "#7B4F4B", "#A1C299", "#300018", "#0AA6D8", "#013349", "#00846F",
   "#372101", "#FFB500", "#C2FFED", "#A079BF", "#CC0744", "#C0B9B2", "#C2FF99",
    "#001E09","#00489C", "#6F0062", "#0CBD66", "#EEC3FF", "#456D75", "#B77B68",
     "#7A87A1", "#788D66","#885578", "#FAD09F", "#FF8A9A", "#D157A0", "#BEC459",
      "#456648", "#0086ED", "#886F4C","#34362D", "#B4A8BD", "#00A6AA", "#452C2C",
       "#636375", "#A3C8C9", "#FF913F", "#938A81","#575329", "#00FECF", "#B05B6F",
        "#8CD0FF", "#3B9700", "#04F757", "#C8A1A1", "#1E6E00","#7900D7", "#A77500",
         "#6367A9", "#A05837", "#6B002C", "#772600", "#D790FF", "#9B9700","#549E79",
          "#FFF69F", "#201625", "#72418F","#BC23FF","#99ADC0","#3A2465”,”#922329",
          "#5B4534", "#FDE8DC", "#404E55", "#0089A3", "#CB7E98", "#A4E804", "#324E72", "#6A3A4C","#7A4900"]


def set_latex():
    import latex
    
def density_map(X,z,
                xlabel=None,ylabel=None,zlabel=None,label=None,
                centers=None,
                psize=20,
                out_file=None,title=None,show=True,cmap='coolwarm',remove_tick=False,
                use_perc=False):
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
    palette = np.array(sns.color_palette('hls', 10))
    
    fontsize = 15

    if use_perc :
        n_sample = len(x)
        outlier_window = int(0.05 * n_sample)

        argz = np.argsort(z)
        bot_outliers = argz[:outlier_window]
        top_outliers = argz[-outlier_window:]
        typical = argz[outlier_window:-outlier_window]

        # plot typical
        plt.scatter(x[typical],y[typical],c=z[typical],cmap=cmap,s=psize,alpha=1.0,rasterized=True)
        cb=plt.colorbar()
        # plot bot outliers (black !)
        plt.scatter(x[bot_outliers],y[bot_outliers],c='black',s=psize,alpha=1.0,rasterized=True)
        # plot top outliers (green !)
        plt.scatter(x[top_outliers],y[top_outliers],c='#36DA36',s=psize,alpha=1.0,rasterized=True)

    else:
        if label is not None:
            plt.scatter(x,y,c=z,cmap=cmap,s=psize,alpha=1.0,rasterized=True,label=label)
        else:
            plt.scatter(x,y,c=z,cmap=cmap,s=psize,alpha=1.0,rasterized=True)
    
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

def scatter_w_label(x,y,z,psize=20):
    n_unique_label=np.unique(z).shape[0]
    palette=sns.color_palette('Paired',n_unique_label+10)
    for i in range(n_unique_label):
        pos=(z==i)
        plt.scatter(x[pos],y[pos],s=psize,c=palette[i],rasterized=True)
    plt.show()

def summary(idx_centers, cluster_label, rho, X, n_true_center=1, y=None, psize=20, savefile=None, show=False):

    fontsize=15
    n_sample=X.shape[0]
    n_center=idx_centers.shape[0]
    palette=contrast_colors
    #palette=sns.color_palette('Paired',n_center+10)
    

    plt.figure(1,figsize=(20,10))

    plt.subplot(131)
    plt.title('True labels',fontsize=fontsize)
    print("--> Plotting summary: True clustered labels, inferred labels and density map ")
    if y is None:
        plt.scatter(X[:,0],X[:,1],c=palette[0],rasterized=True)
    else:
        for i in range(n_true_center):
            pos=(y==i)
            plt.scatter(X[pos,0],X[pos,1], s=psize,c=palette[i],rasterized=True)
            
    ax = plt.subplot(132)
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

    plt.title('Inferred labels',fontsize=fontsize)
    plt.tight_layout()
    plt.subplot(133)
    density_map(X,rho,centers=X[idx_centers],title='Density map', psize=psize, show=False)

    if savefile:
        plt.savefig(savefile)
    if show is True:
        plt.show()

    plt.clf()

def summary_v2(idx_centers, cluster_label, rho, X, n_true_center=1, y=None, psize=20, savefile=None, show=False):

    fontsize=15
    n_sample=X.shape[0]
    n_center=idx_centers.shape[0]
    palette=contrast_colors
    #palette=sns.color_palette('Paired',n_center+10)
    

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




def build_dendrogram(hierarchy, noise_range):
    """Constructs the linkage matrix for plotting using the scipy.hierarchy function

    Parameters
    ----------
    hierarchy : list of dictionaries, length = number of coarse graining steps
        First element of the list is the dictionary specifying the clusters at the finest scale
        Further elements of the list are coarsed grained.
    noise_range : array-like, length = number of coarse graining steps
        The value of the noise parameter at every scale
    
    Returns
    -------
    Z : array-like, shape=(n_coarse_grain,4) ; see scipy for more info 
    Linkage matrix for plotting dendrogram
    
    """

    Z = []
    initial_idx_centers = list(hierarchy[0]['idx_centers'])
    dict_center_relative = {}
    for idx in initial_idx_centers:
        dict_center_relative[idx] = -1
    
    depth = len(hierarchy)
    n_init_centers = len(initial_idx_centers)
    merge_count = 0
    member_count_dict = {}

    for d in range(depth-1):

        pre_idx_centers = hierarchy[d]['idx_centers']
        cur_idx_centers = hierarchy[d+1]['idx_centers']

        pre_cluster_labels = hierarchy[d]['cluster_labels']
        cur_cluster_labels = hierarchy[d+1]['cluster_labels']
        
        for idx in pre_idx_centers :
            if idx not in cur_idx_centers : # means it's been merged

                i = cur_cluster_labels[idx]
                new_idx = cur_idx_centers[i] # pic -> new_pic 
                z = [-1,-1,-1,-1] # linkage list

                if (dict_center_relative[idx] == -1) & (dict_center_relative[new_idx] == -1): # both have not been merged yet
                    z[0] = initial_idx_centers.index(idx)
                    z[1] = initial_idx_centers.index(new_idx)
                    z[2] = noise_range[d+1]
                    z[3] = 2
                elif (dict_center_relative[idx] == -1) & (dict_center_relative[new_idx] != -1):
                    z[0] = initial_idx_centers.index(idx)
                    z[1] = dict_center_relative[new_idx]
                    z[2] = noise_range[d+1]
                    z[3] = 1 + member_count_dict[ z[1] ]
                elif (dict_center_relative[idx] != -1) & (dict_center_relative[new_idx] == -1):
                    z[0] = dict_center_relative[idx]
                    z[1] = initial_idx_centers.index(new_idx)         # ~ new point
                    z[2] = noise_range[d+1]
                    z[3] = 1 + member_count_dict[ z[0] ]
                else:
                    z[0] = dict_center_relative[idx]
                    z[1] = dict_center_relative[new_idx]
                    z[2] = noise_range[d+1]
                    z[3] = member_count_dict[ z[0] ] + member_count_dict[ z[1] ]

                new_cluster_idx = merge_count + n_init_centers
                dict_center_relative[idx] = new_cluster_idx
                dict_center_relative[new_idx] = new_cluster_idx

                member_count_dict[new_cluster_idx] = z[3]
                merge_count += 1

                Z.append(z)
    
    return Z


def dendrogram(model, show=True, savefile=None):
    from scipy.cluster.hierarchy import dendrogram
    fontsize=15

    hierarchy = model.hierarchy
    noise_range = model.noise_range
    Z = build_dendrogram(hierarchy, noise_range)
    dendrogram(Z)

    plt.ylim(0,1.2 * model.max_noise)

    plt.xlabel('cluster \#',fontsize=fontsize)
    plt.ylabel('$\delta$',fontsize=fontsize)
    plt.title('Clustering hierarchy',fontsize=fontsize)
    plt.tight_layout()

    if show is True:
        plt.show()
    if savefile :
        plt.savefig(savefile)

    plt.clf()