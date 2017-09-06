from .fdc import FDC
import classify
import numpy as np
import pickle
from scipy.cluster.hierarchy import dendrogram as scipydendro
from scipy.cluster.hierarchy import to_tree
from .hierarchy import compute_linkage_matrix

class TreeNode:

    def __init__(self, id = -1, child = [], scale = -1):
        self.child = child # has to be list of TreeNode
        self.scale = scale
        self.id = id

    def __repr__(self):
        return ("Node: [%s] @ s = %.3f" % (self.id,self.scale))

    def is_leaf(self):
        return len(self.child) == 0

    def get_child(self, id = None):
        if id is None:
            return self.child
        else:
            for c in self.child:
                if c.get_id() == id:
                    return c

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


def identify_robust_merge(model, X, n_average = 10, score_threshold = 0.5):
    """Starting from the root, goes down the tree and evaluates which clustering node are robust
    It returns a list of the nodes for which their corresponding clusters are well defined according to 
    a logistic regression and a score_threshold given by the user
    """

    root, node_dict, mergers  = build_tree(model)  # Extracts all the information from model and outputs a tree
    
    stack = [root]
    node_list = []
    result = {}

    # breath-first search from the root
    while stack:
        current_node = stack[0]
        stack = stack[1:]
        if current_node.is_leaf():
            score = 1.0
        else:
            result = score_merge(current_node, model, X, n_average = n_average)
            score = result['mean_score']
        
        if score > score_threshold: # --- search stops if the node is not statistically signicant (threshold)
            node_list.append([current_node.get_id(), score, result]) # result contains the info for the gates

            if not current_node.is_leaf():
                for node in current_node.get_child():
                    stack.append(node)
        else:
            print("node ", current_node.get_id(), " below threshold")

    result = {'robust_node': node_list, 'root': root, 'node_dict': node_dict, 'mergers': mergers}
    
    if len(node_list) == 0:
        node_list = [[root.get_id(), -1]]
    
    return result

def find_robust_labelling(model, X, n_average = 10, score_threshold = 0.5):
    """ Finds the merges that are statistically significant (i.e. greater than the score_threshold)
    and relabels the data accordingly
    """

    result = identify_robust_merge(model, X, n_average = n_average, score_threshold = score_threshold)
    
    node_info_list = result['robust_node']
    root = result['root']
    node_dict = result['node_dict']
    mergers = result['mergers']
    
    merger_to_mathematica(mergers, out_graph_file="graph.txt") # for visualizing, just run the attached mathematica file
    
    robust_terminal_node = [] # we want to remove ancestors and only keep the finest scales possible
    node_list = []
    for n in node_info_list:
        node_list.append(n[0])

    for node_id in node_list:
        node = node_dict[node_id]
        if node.is_leaf():
            robust_terminal_node.append(node_id)
        else:
            for c in node.get_child():
                if c.get_id() not in node_list:
                    robust_terminal_node.append(c.get_id())
    
    ###################
    ###################
    # RELABELLING DATA !
    ###################
    ###################

    cluster_n = 0
    n_sample = len(model.X)
    y_robust = -1*np.ones(n_sample,dtype=np.int)
    y_original = model.hierarchy[0]['cluster_labels']

    for node_id in robust_terminal_node:
        node = node_dict[node_id]
        y_node = classification_labels(node, model)
        if node.is_leaf():
            pos = (y_node == 0)
            y_robust[pos] = cluster_n
            cluster_n +=1
        else:
            n_unique = len(node.get_child())
            for i in range(n_unique):
                pos = (y_node == i)
                y_robust[pos] = cluster_n
                cluster_n +=1
    
    new_idx_centers = []
    
    all_idx = np.arange(0,model.X.shape[0],dtype=int)
    for i in range(cluster_n):
        pos_i = (y_robust == i)
        max_rho = np.argmax(model.rho[y_robust == i])
        idx_i = all_idx[pos_i][max_rho]
        new_idx_centers.append(idx_i)

    return y_robust, robust_terminal_node, node_list, np.array(new_idx_centers,dtype=int)

def check_all_merge(model, X, n_average = 10):
    root, node_dict, mergers  = build_tree(model)  ## ---> starting from root , perform classification with soft-max <---
    #mergers.reverse() # starts from the coarse grain scale
    #print(mergers)
    for merger in mergers :
        node_id = merger[1]
        node = node_dict[node_id]

        y = classification_labels(node, model)
        pos_subset =  (y != -1)
        Xsubset = X[pos_subset]
        ysubset = y[pos_subset]

        results = classify.fit_logit(Xsubset, ysubset, n_average = n_average, C = 1.0) # contains the information about the gates 
        merger.append(results['mean_score']) ## ---> adding classification score to nodes

        merger_to_mathematica(mergers, out_graph_file="graph.txt", out_score_file = 'score.txt') # for visualizing, just run the attached mathematica file

    return root, node_dict, mergers, results 

def score_merge(root, model, X, n_average = 10):
    """ Using a logistic regression multi-class classifier, determines the merges that are statistically
    signicant based on a CV prediction score. Returns a robust clustering in the original space.
    """
    #print(root.get_id())

    y = classification_labels(root, model)

    pos_subset =  (y != -1)
    Xsubset = X[pos_subset] # original space coordinates
    ysubset = y[pos_subset] # labels

    results = classify.fit_logit(Xsubset, ysubset, n_average = n_average, C = 1.0)

    return results
  
def classification_labels(root, model):
    """ Returns a list of labels for the original data according to the classification
    given at root. root is a TreeNode object which contains childrens. Each children (and the data it contains)
    is assigned an arbitrary integer label. Data points not contained in that node are labelled as -1.

    Parameters
    -------
    root : TreeNode

    model : FDC object

    Returns
    --------
    1D array of labels

    """

    childs = root.get_child()
    n_sample = len(model.X)
    y = -1*np.ones(n_sample,dtype=np.int)
    y_init = model.hierarchy[0]['cluster_labels']

    if root.is_leaf():
        init_c = root.get_id()
        y[y_init == init_c] = 0
    else:
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

def merger_to_mathematica(mergers, out_graph_file = None, out_score_file = None):
    """ Creates a graph output for ez plotting in mathematica """

    pointers = []
    for m in mergers:
        for m0 in m[0]:
            pointers.append("%i -> %i"%(m0,m[1]))

    if out_graph_file is not None:
        with open(out_graph_file,'w') as f:
            for p in pointers:
                f.write(p)
                if p != pointers[-1]: 
                    f.write(',')
        f.close()

    if out_score_file is not None:
        with open(out_score_file,'w') as f:
            for m in mergers:
                f.write("%i -> %.4f"%(m[1],m[-1]))
                if m != mergers[-1]:
                    f.write(',')
        f.close()
