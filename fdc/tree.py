from .fdc import FDC
from . import classify
import numpy as np
import pickle
from scipy.cluster.hierarchy import dendrogram as scipydendro
from scipy.cluster.hierarchy import to_tree
from .hierarchy import compute_linkage_matrix

class TreeNode:

    def __init__(self, id = -1, parent = None, child = [], scale = -1):
        self.child = child # has to be list of TreeNode
        self.scale = scale
        self.parent = parent
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

class TreeStructure:
    """ Contains all the hierachy and information concerning the clustering
    """
    def __init__(self, root = None, shallow_copy = None):
        self.root = root
        self.node_dict = None
        self.mergers = None
        self.robust_node = None
        self.new_cluster_label = None
        self.robust_terminal_node = None #list of the terminal robust nodes
        self.robust_clf_node = None # full information about classification is recorded here, keys of dict are the classifying nodes 
        self.all_clf_node = None # calculated when checking all nodes !
        self.all_robust_node = None # list of all nodes in the robust tree (classifying nodes and leaf nodes)
        self.cluster_to_node_id = None # dictionary mapping cluster labels (displayed on plot) with node id

        self.new_idx_centers = None
        self.tree_constructed = False

    def build_tree(self, model):
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
        if self.tree_constructed is True:
            return

        mergers = find_mergers(model.hierarchy , model.noise_range)
        mergers.reverse()
        
        node_dict = {}
        
        m = mergers[0]
        root = TreeNode(id = m[1], scale = m[2])
        node_dict[root.get_id()] = root
        
        for m in mergers:
            for mc in m[0]:
                c_node = TreeNode(id = mc, parent = node_dict[m[1]], child = [], scale = -1)
                node_dict[m[1]].add_child(c_node)
                node_dict[c_node.get_id()] = c_node
            node_dict[m[1]].scale = m[2]

        self.root = root
        self.node_dict = node_dict
        self.mergers = mergers
        self.tree_constructed = True

    def identify_robust_merge(self, model, X, n_average = 10, score_threshold = 0.5):
        """Starting from the root, goes down the tree and evaluates which clustering node are robust
        It returns a list of the nodes for which their corresponding clusters are well defined according to 
        a logistic regression and a score_threshold given by the user

        Will write information in the following objects :

        self.robust_terminal_node (list) # 
        self.robust_clf_node (dict) # full info
        """

        self.build_tree(model)  # Extracts all the information from model and outputs a tree    

        root, node_dict, mergers = self.root, self.node_dict, self.mergers

        print("---> 2 top layers")
        print("---> Root :", root)
        print("---> Root's childs :", root.get_child())

        stack = [root]
        node_list = []
        result = {}

        self.robust_terminal_node = [] #list of the terminal robust nodes
        self.robust_clf_node = {} # dictionary of the nodes where a partition is made (non-leaf nodes)


        if self.all_clf_node is not None: # meaning, the nodes have already been checked for classification 
            for k, v in self.all_clf_node.items():
                if v['mean_score'] > score_threshold:
                    self.robust_clf_node[k] = v

            for k, v in self.robust_clf_node.items(): # identify terminal nodes (leaves)
                c_node = node_dict[k]
                for child in c_node.child:
                    id_c = child.get_id() 
                    if id_c not in self.robust_clf_node.keys():
                        self.robust_terminal_node.append(id_c)
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        if self.all_clf_node is None:

            while stack:
                current_node = stack[0]
                stack = stack[1:]
                if not current_node.is_leaf():
                    result_classify = score_merge(current_node, model, X, n_average = n_average)
                    score = result_classify['mean_score']
                    
                    if score > score_threshold: # --- search stops if the node is not statistically signicant (threshold)
                        print("[tree.py] : robust node ", current_node.get_id()," %.4f > %.4f"%(score, score_threshold))
                        self.robust_clf_node[current_node.get_id()] = result_classify
                        
                        for node in current_node.get_child():
                            stack.append(node)
                    else:
                        print("[tree.py] : reject node ", current_node.get_id()," %.4f < %.4f"%(score, score_threshold))
                        self.robust_terminal_node.append(current_node.get_id())
                else:
                    self.robust_terminal_node.append(current_node.get_id())
        
            if len(node_list) == 0:
                assert False, "check this case, not sure what it means""
                #node_list = [[root.get_id(), -1]]
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # Listing all nodes in the robust tree ...

        for k, _ in self.robust_clf_node.items():
            all_robust_node.add(k)
            current_node = node_dict[k]
            for c in current_node.child:
                all_robust_node.add(c.get_id())

        self.all_robust_node = list(all_robust_node)
        
    
    def find_robust_labelling(self, model, X, n_average = 10, score_threshold = 0.5):
        """ Finds the merges that are statistically significant (i.e. greater than the score_threshold)
        and relabels the data accordingly
        """

        self.identify_robust_merge(model, X, n_average = n_average, score_threshold = score_threshold)
        
        root = self.root
        node_dict = self.node_dict
        mergers = self.mergers
        robust_terminal_node = self.robust_terminal_node
        
        ###################
        ###################
        # RELABELLING DATA !
        ###################
        ###################

        cluster_n = len(robust_terminal_node)
        n_sample = len(model.X)
        y_robust = -1*np.ones(n_sample,dtype=np.int)
        y_original = model.hierarchy[0]['cluster_labels']
        cluster_to_node_id = {}

        y_node = classification_labels([node_dict[i] for i in robust_terminal_node], model)
        for i, v in enumerate(robust_terminal_node):
            node_id = v
            pos = (y_node == i)
            y_robust[pos] = i
            cluster_to_node_id[i] = v

        if len(robust_terminal_node) == 0:
            y_robust *= 0 # only one coloring !
        
        new_idx_centers = []
        all_idx = np.arange(0,model.X.shape[0],dtype=int)

        for i in range(cluster_n):
            pos_i = (y_robust == i)
            max_rho = np.argmax(model.rho[y_robust == i])
            idx_i = all_idx[pos_i][max_rho]
            new_idx_centers.append(idx_i)

        self.new_cluster_label = y_robust
        self.robust_terminal_node = robust_terminal_node
        self.new_idx_centers = np.array(new_idx_centers,dtype=int)
        self.cluster_to_node_id = cluster_to_node_id
        self.node_to_cluster_id = {v: k for k, v in self.cluster_to_node_id.items()}

        return self

    def check_all_merge(self, model, X, n_average = 10):
        """ Goes over all classification nodes and evaluates classification scores """ 

        self.build_tree(model)
        self.all_clf_node = {}
        
        for merger in mergers : # don't need to go through the whole hierachy, since we're checking everything
            
            node_id = merger[1]
            node = self.node_dict[node_id]

            score_merge(current_node, model, X, n_average = n_average)
            result_classify = score_merge(current_node, model, X, n_average = n_average)
            self.all_clf_node[node_id] = result_classify

    def predict(self, X):
        """ Given find_robust_labelling was performed, new data from X can be classified using self.robust_clf_node
        returns the terminal "cluster" label (not the node !)
        """
        #uprint(self.robust_clf_node)
        y_pred = -1 * np.ones(X.shape[0], dtype=int)
        terminal_nodes = set(self.robust_terminal_node)
        node_to_cluster = self.node_to_cluster_id

        for i, x in enumerate(X):
            current_clf_node = self.root
            while True:
                if current_clf_node.get_id() in terminal_nodes: # robust terminal node reached !
                    y_pred[i] = node_to_cluster[current_clf_node.get_id()]
                    break
                
                child_list = current_clf_node.child
                info = self.robust_clf_node[current_clf_node.get_id()]
                W,b,mu,inv_std = info['coeff'], info['intercept'], info['mean_xtrain'], info['inv_std_xtrain']
                y = self.classify_point(inv_std*(x-mu), W, b)
                #print('child_pos =', y)
                current_clf_node = child_list[y]
        
        return y_pred

    def classify_point(self, x, W, b):
        """ Given weight matrix and intercept (bias), classifies point x 
        w.r.t to a linear classifier """

        n_class = len(b)
        
        if n_class == 1: # binary classification
            f = (np.dot(x,W[0]) + b)[0]
            if f > 0.:
                return 1
            else:
                return 0
        else:
            score_per_class = []
            for i, w in enumerate(W):
                score_per_class.append(np.dot(x,w)+b[i])
            #print(score_per_class)
            return np.argmax(score_per_class)

    def write_result_mathematica(self, model, marker) : # graph should be a dict of list
        """ 
        -> Saves results in .txt files, which are easily read with a Mathematica
        script for ez plotting ...
        """
        if self.robust_clf_node is None :
            assert False, "Model not yet fitted !"

        self.gate_dict = self.find_full_gate(model)

        my_graph = {}
        my_graph_score = {}
        for e,v in self.robust_clf_node.items():
            my_graph[e] = []
            my_graph_score[e] = v['mean_score']
            for c in self.node_dict[e].child:
                my_graph[e].append(c.get_id())
        
        self.graph = my_graph
        self.graph_score = my_graph_score

        self.write_graph_mathematica()
        self.write_graph_score_mathematica()
        self.write_gate_mathematica(self.gate_dict, marker)
        self.write_cluster_label_mathematica()

        
    def write_graph_mathematica(self, out_file = "graph.txt"):
        """ Writes graph in mathematica readable format """
        f = open(out_file,'w')
        my_string_list = []
        for node_id, node_childs in self.graph.items(): # v is a list
            for child in node_childs :
                my_string_list.append("%i -> %i"%(node_id, child))
        f.write(",".join(my_string_list))
        f.close()

    def write_graph_score_mathematica(self, out_file = "graph_score.txt"):
        """ Writes scores of classification for every division node """
        f = open(out_file, 'w')
        string_list = []
        for k, v in self.graph_score.items():
            string_list.append('%i -> % .5f'%(k,v))
        f.write(','.join(string_list))
        f.close()

    def write_gate_mathematica(self, gate_dict, marker, out_file = "gate.txt"):
        """ Writes most important gates for discriminating data in a classification """
        f = open(out_file, 'w')
        string_list = []
        for k, g in gate_dict.items():
            string_list.append("{%i -> %i, \"%s\"}"%(k[0],k[1],str_gate(marker[g[0][0]],g[1][0])))
        f.write("{")
        f.write(','.join(string_list))
        f.write("}")
        f.close()

    def write_cluster_label_mathematica(self, out_file = "n_to_c.txt"): # cton is a dictionary of clusters to node id
        """ Node id to cluster labels """
        f = open(out_file, 'w')
        string_list = []
        
        for k, v in self.cluster_to_node_id.items():
            string_list.append("{%i -> %i}"%(v,k))
        f.write("<|")
        f.write(','.join(string_list))
        f.write("|>")
        f.close()

    def find_gate(self, node_id):
        """ Return the most important gates, sorted by amplitude. One set of gate per category (class)
        """
        import copy
        clf_info = self.robust_clf_node[node_id]
        n_class = len(self.node_dict[node_id].child)

        weights = clf_info['coeff'] # weights should be sorted by amplitude for now, those are the most important for the scoring function
        gate_array = []

        for i, w in enumerate(weights):
            argsort_w = np.argsort(np.abs(w))[::-1] # ordering (largest to smallest) -> need also to get the signs
            sign = np.sign(w[argsort_w])
            gate_array.append([argsort_w, sign])

        if n_class == 2: # for binary classfication the first class as a negative score ... for all other cases the classes have positive scores
            gate_array.append(copy.deepcopy(gate_array[-1]))
            gate_array[0][1]*=-1
            
        return gate_array
    
    def find_full_gate(self, model):
        """ Determines the most relevant gates which specify each partition 
        """
        gate_dict = {} # (tuple to gate ... (clf_node, child_node) -> gate (ordered in magnitude))

        for node_id, info in self.robust_clf_node.items():

            childs = self.node_dict[node_id].child
            gates = self.find_gate(node_id)

            for c, g in zip(childs, gates):
                gate_dict[(node_id, c.get_id())] = g # storing gate info 

        return gate_dict

def score_merge(root, model, X, n_average = 10):
    """ Using a logistic regression multi-class classifier, determines the merges that are statistically
    signicant based on a CV prediction score. Returns a robust clustering in the original space.
    """
    
    y = classification_labels(root.get_child(), model)

    pos_subset =  (y != -1)
    Xsubset = X[pos_subset] # original space coordinates
    ysubset = y[pos_subset] # labels

    results = classify.fit_logit(Xsubset, ysubset, n_average = n_average, C = 1.0)
    
    return results
  
def classification_labels(node_list, model):
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
    
    n_sample = len(model.X)
    y = -1*np.ones(n_sample,dtype=np.int)
    y_init = model.hierarchy[0]['cluster_labels']

    for i, node in enumerate(node_list):
        init_c = find_idx_cluster_in_root(model, node)
        for ic in init_c:
            y[y_init == ic] = i

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



########################################################################################
########################################################################################
################################# UTILITY FUNCTIONS ####################################
########################################################################################
########################################################################################

def str_gate(marker, sign):
    if sign < 0. :
        return marker+"-"
    else:
        return marker+"+"

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
