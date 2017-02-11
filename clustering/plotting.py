'''
Created on Jan 16, 2017

@author: Alexandre Day
'''

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
global contrast_colors

contrast_colors=["#1CE6FF", "#FF34FF", "#FF4A46",
 "#008941", "#006FA6", "#A30059","#FFDBE5", "#7A4900", "#0000A6", "#63FFAC", 
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
          "#5B4534", "#FDE8DC", "#404E55", "#0089A3", "#CB7E98", "#A4E804", "#324E72", "#6A3A4C"
]


def set_latex():
    import latex
    
def density_map(x,y,z,
                xlabel=None,ylabel=None,zlabel=None,label=None,
                centers=None,
                psize=20,
                out_file=None,title=None,show=True,cmap='coolwarm',remove_tick=False):
    """ 
    Purpose:
        Produces a 2D intensity map
    """
    palette=np.array(sns.color_palette('hls', 10))
    
    fontsize=15

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

def summary(idx_centers, cluster_label, rho, n_true_center, X, y=None, psize=20, savefile=False, show=False):
    
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
            
    plt.subplot(132)
    for i in range(n_center):
        pos=(cluster_label==i)
        plt.scatter(X[pos,0],X[pos,1],c=palette[i], s=psize, rasterized=True)
 
    plt.title('Inferred labels',fontsize=fontsize)
    plt.tight_layout()
    plt.subplot(133)
    density_map(X[:,0], X[:,1],rho,centers=X[idx_centers],title='Density map', psize=psize, show=False)

    if savefile is True:
        plt.savefig(savefile)
    if show is True:
        plt.show()

    plt.clf()

