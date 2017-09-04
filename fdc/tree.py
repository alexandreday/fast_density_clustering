from fdc import FDC
import numpy as np
from sklearn.datasets import make_blobs
from fdc import plotting
import pickle
from scipy.cluster.hierarchy import dendrogram as scipydendro
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import to_tree
from fdc.hierarchy import compute_linkage_matrix

class TreeNode:

    def __init__(self, id = -1, child = [], scale = -1):
        self.child = child # has to be list of TreeNode
        self.scale = scale
        self.id = id

    def __repr__(self):
        return ("Node: [%s] @ s = %.3f" % (self.id,self.scale))

    def is_leaf(self):
        return len(self.child) == 0

    def get_child(self):
        return self.child

    def get_scale(self):
        return self.scale

    def get_id(self):
        return self.id

    def add_child(self, treenode):
        self.child.append(treenode)

    def get_rev_child(self):
        child = self.child[:]
        child.reverse()
        return child 

def main():

    #X,y = make_blobs(n_samples=10000, centers = 20, random_state=0)
    
    
    f = open('/Users/robertday/Dropbox/Work/Project_PHD/Immunology/analysis/scripts/tsne/tsne.pkl','rb')
    #print('Final score is : ', model_tsne.KLscore_[-1])
    [score, X] = pickle.load(f)
    #plt.scatter(X[:,0],X[:,1])

    #model = FDC(noise_threshold=0.0, nh_size = 40, test_ratio_size = 0.2, xtol = 0.001, atol=0.0000005, rtol=0.0000005)
    #model.fit(X)
    #model.coarse_grain(X,0.0,3.5,0.01)

    #fopen = open('model_2.pkl','wb')
    #pickle.dump(model,fopen)
    #exit()

    fopen = open('model_2.pkl','rb')
    model = pickle.load(fopen)
    model.X = X
    #plotting.dendrogram(model)
    #plotting.summary_model(model, delta = 1.5)
    print(len(model.hierarchy[0]['idx_centers']))
    
    root, node_dict, mergers  = build_tree(model)  ## --- starting from root , perform classification with soft-max ?
    #print(mergers) # classify each mergers !
    print(mergers[0])
    print(find_idx_cluster_in_root(model, node_dict[mergers[0][0][5]]))

    #print(find_idx_cluster_in_root(model, root.get_child()[0]))

    ########### --------- ################# ----------------- ##########

    merger_to_mathematica(mergers, out_file = 'test.txt')
    exit()

def sample_root_labels(root, model):
    childs = root.get_child()
    n_sample = len(model.X)
    y = -1*np.ones(n_sample,dtype=np.int)
    y_init = model.hierarchy[0]['cluster_labels']

    for i, c in enumerate(childs):
        init_c = find_idx_cluster_in_root(model, c)
        for ic in init_c:
            y[y_init == ic] = i # assigns arbitray label, just set by ordering of the childrens.

    return y
    

def find_mergers(hierarchy , noise_range):

    """ Determines the list of merges that are made during the coarse-graining """
    
    n_depth = len(noise_range)
    n_initial_cluster = len(hierarchy[0]['idx_centers'])
    initial_labels = hierarchy[0]['cluster_labels']
    n_pre_cluster = n_initial_cluster

    current_merge_idx = n_initial_cluster
    n_merge = 0
    merging_dict = {}

    merger_record = []

    for i in range(n_initial_cluster):
        merging_dict[i] = -1

    for i, d in enumerate(noise_range[1:]):

        n_cluster = len(hierarchy[i+1]['idx_centers'])

        if n_pre_cluster != n_cluster: # merger(s) have occured
            for j in range(n_cluster):
                elements_mask = (hierarchy[i+1]['cluster_labels'] == j)
                content = hierarchy[i]['cluster_labels'][elements_mask]
                tmp = np.unique(content)
                if len(tmp) > 1 :
                    tmp_u = np.unique(initial_labels[elements_mask])
                    mapped_u = []
                    for k in tmp_u:
                        mapped_k = apply_map(merging_dict, k)

                        mapped_u.append(mapped_k)
                        
                    mapped_u = np.unique(mapped_u)
                    for e in mapped_u:
                        merging_dict[e] = current_merge_idx

                    merger_record.append([list(mapped_u), current_merge_idx, d])
                    merging_dict[current_merge_idx] = -1

                    current_merge_idx +=1
    
    # merge remaining ----
    mapped_u = []
    for k,v in merging_dict.items():
        if v == -1:
            mapped_u.append(k)
    merger_record.append([mapped_u, current_merge_idx, 1.5*(merger_record[-1][2])])

    return merger_record

def build_tree(model):
    """Given hierachy, builds a tree of the clusterings. The nodes are class objects define in the class TreeNode

    Parameters
    ---------
    model : object from the FDC class
        contains the fitted hierarchy produced via the coarse_graining() method
    
    Returns
    ---------
    tuple = (root, node_dict, mergers)

    root : TreeNode class object
        root of the tree
    node_dict : dictionary of TreeNode objects. 
        Objects are stored by their merger ID
    mergers :
        list of nodes being merged with the corresponding scale of merging
    """

    mergers = find_mergers(model.hierarchy , model.noise_range)
    mergers.reverse()
    
    node_dict = {}

    m = mergers[0]

    child_list = [TreeNode(id = mc, child = [], scale = -1) for mc in m[0]]
    root = TreeNode(id = m[1], child = child_list, scale=m[2])
    
    node_dict[root.get_id()] = root

    for c in root.get_child():
        node_dict[c.get_id()] = c

    for m in mergers[1:]:
        child_list = [TreeNode(id = mc, child = [], scale = -1) for mc in m[0]]
        node_dict[m[1]].child = child_list
        node_dict[m[1]].scale = m[2]
        for c in child_list:
            node_dict[c.get_id()] = c

    return root, node_dict, mergers 

def apply_map(mapdict, k):
    old_idx = k
    while True:
        new_idx = mapdict[old_idx]
        if new_idx == -1:
            break
        old_idx = new_idx
    return old_idx


def float_equal(a,b,eps = 1e-6):
    if abs(a-b) < 1e-6:
        return True
    return False


def get_scale(Z, c_1, c_2):
    for z in Z:
        if (z[0],z[1]) == (c_1,c_2) or (z[0],z[1]) == (c_2,c_1):
            return z[2]
    return -1
        
def breath_first_search(root):
    
    stack = [root]
    node_list = []
    # breath-first search
    while stack:
        current_node = stack[0]
        stack = stack[1:]
        node_list.append(current_node.get_id())

        if not current_node.is_leaf():
            for node in current_node.get_child():
                stack.append(node)

    return node_list

def breath_first_search_w_scale(root, Z):
    scale = find_scale(root.get_id(), Z)

    node_list = breath_first_search(root)
    node_list_scale = []
    for n in node_list:
        if float_equal(find_scale(n.get_id(),Z),scale):
            node_list_scale.append(n)

    return node_list_scale

def find_scale(id, Z):
    n_merge = len(Z)
    n_init_c = np.max(Z[:,:2]) + 2 - n_merge # + 2 since u start from 0 and u don't count the root
    if id < n_init_c :
        return  0
    else:
        return Z[int(id-n_init_c),2]

def find_leaves(cluster_n, Z, n_init_cluster): # find the leafs of a cluster a given scale
    idx = cluster_n - n_init_cluster - 1
    return Z[idx]
    
def find_idx_cluster_in_root(model, root):
    node_list = np.array(breath_first_search(root))
    n_initial_cluster = len(model.hierarchy[0]['idx_centers'])
    return np.sort(node_list[node_list < n_initial_cluster])


def merger_to_mathematica(mergers, out_file = None):
    """ Creates a graph output for ez plotting in mathematica """

    pointers = []
    for m in mergers:
        for m0 in m[0]:
            pointers.append("%i -> %i"%(m0,m[1]))

    if out_file is not None:
        with open(out_file,'w') as f:
            for p in pointers:
                f.write(p)
                if p != pointers[-1]: 
                    f.write(',')
        f.close()
if __name__=="__main__":
    main()