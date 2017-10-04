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
        self.ignore_root = True

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

    def node_items(self): # breath-first ordering
        """ Returns a list of the nodes using a breath-first search
        """
        stack = [self.root]
        list_nodes = []
        while stack:
            current_node = stack[0]
            list_nodes.append(current_node)
            for c in current_node.child:
                stack.append(c)
            stack = stack[1:]
        
        return list_nodes

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
        
        self.robust_terminal_node = [] #list of the terminal robust nodes
        self.robust_clf_node = {} # dictionary of the nodes where a partition is made (non-leaf nodes)

        if self.all_clf_node is not None: # meaning, the nodes have already been checked for classification
            
            res = self.all_clf_node[root.get_id()]

            if res['mean_score'] > score_threshold :
                self.robust_clf_node[root.get_id()] = res

            for current_node in self.node_items() :
                for c in current_node.child :
                    if c.get_id() in self.all_clf_node.keys() :
                        res = self.all_clf_node[c.get_id()]
                        if res['mean_score'] > score_threshold :
                            self.robust_clf_node[c.get_id()] = res

            for k, v in self.robust_clf_node.items() : # identify terminal nodes (leaves)
                c_node = node_dict[k]
                for child in c_node.child :
                    id_c = child.get_id() 
                    if id_c not in self.robust_clf_node.keys() :
                        self.robust_terminal_node.append(id_c)
        
        else: # focus on this loop .........
            
            # add root first 
            result_classify = score_merge(self.root, model, X, n_average = n_average)
            score = result_classify['mean_score']

            if self.ignore_root is True:
                print("[tree.py] : root is ignored, #  %i \t score = %.4f"%(self.root.get_id(),score))
                self.robust_clf_node[self.root.get_id()] = result_classify
            else:
                if score > score_threshold: # --- search stops if the node is not statistically signicant (threshold)
                    print("[tree.py] : root is robust #  %i \t score = %.4f"%(self.root.get_id(),score))
                    self.robust_clf_node[self.root.get_id()] = result_classify
                else:
                    print("[tree.py] : root is not robust #  %i \t score = %.4f"%(self.root.get_id(),score))

            for current_node in self.node_items()[1:]:
                if current_node.parent.get_id() in self.robust_clf_node.keys():
                    if not current_node.is_leaf():
                        
                        result_classify = score_merge(current_node, model, X, n_average = n_average)
                        score = result_classify['mean_score']
                        
                        if score > score_threshold: # --- search stops if the node is not statistically signicant (threshold)
                            print("[tree.py] : robust node #  %i \t score = %.4f"%(current_node.get_id(),score))
                            self.robust_clf_node[current_node.get_id()] = result_classify

                        else:
                            print("[tree.py] : reject node #  %i \t score = %.4f"%(current_node.get_id(),score))
                            self.robust_terminal_node.append(current_node.get_id())
                    else: # implies it's parent was robust, and is a leaf node 
                        self.robust_terminal_node.append(current_node.get_id())
        
        #self.robust_clf_node is a dict containing all the classification info for further queries

        # updating robust_clf_node ... 
        self.probability_tree = {}

        for node_id, classify_results in self.robust_clf_node.items():
            current_node = self.node_dict[node_id]
            for i, c in enumerate(current_node.child):
                self.probability_tree[(node_id, c.get_id())] = classify_results['mean_score_cluster'][i]
        
        #~~~~~~~~~~~~~~~> recompute tree with combinatorial factors 

        #self.combinatorial_tree() :

        for node_id, node_clf in self.robust_clf_node.items(): 
            if self.node_dict[node_id].child[0].get_id() in self.robust_terminal_node: # iterate over terminal robust clf nodes !
                probability_set = [node_clf['mean_score']]
                n = node_id
                while n != self.root.get_id():
                    parent = self.node_dict[n].parent
                    probability_set.append(self.probability_tree[(parent.get_id(), n)])
                    n = parent.get_id()
        
        ########### ----------> LEFT IT HERE " NOW COMPUTE COMBINATORIAL FACTORS ....... <---------
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # Listing all nodes in the robust tree ...==
        all_robust_node = set([])

        for k, _ in self.robust_clf_node.items():
            all_robust_node.add(k)
            current_node = node_dict[k]
            for c in current_node.child:
                all_robust_node.add(c.get_id())

        self.all_robust_node = list(all_robust_node)

        
    def find_robust_labelling(self, model, X, n_average = 10, score_threshold = 0.5):
        """ Finds the merges that are statistically significant (i.e. greater than the score_threshold)
        and relabels the data accordingly

        Parameters
        ------

        model : fdc object
            Contains the coarse graining information

        X  : array, shape = (n_sample, n_marker)
            Contains the data in the original space
        
        n_average : int
            Number of folds in the cross validation

        score_threshold : float
            Classification score threshold
        
        Returns
        ---------
        self : TreeStructure() object

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
        
        for merger in self.mergers : # don't need to go through the whole hierarchy, since we're checking everything
            node_id = merger[1]
            result_classify = score_merge(self.node_dict[node_id], model, X, n_average = n_average)
            print("[tree.py] : ", node_id, "accuracy : %.3f"%result_classify['mean_score'], "sample_size : %i"%result_classify['n_sample'], sep='\t')

            self.all_clf_node[node_id] = result_classify

        return self
    
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
        gate_weights = []

        for i, w in enumerate(weights):
            argsort_w = np.argsort(np.abs(w))[::-1] # ordering (largest to smallest) -> need also to get the signs
            sign = np.sign(w[argsort_w])
            gate_array.append([argsort_w, sign])
            gate_weights.append(w)

        if n_class == 2: # for binary classfication the first class (0) has a negative score ... for all other cases the classes have positive scores
            gate_array.append(copy.deepcopy(gate_array[-1]))
            gate_array[0][1] *= -1
            gate_weights.append(copy.deepcopy(w))
            gate_weights[0] *= -1
        
        gate_weights = np.array(gate_weights)

        return gate_array, gate_weights

    def print_clf_weight(self, markers=None, file='weight.txt'):

        weight_summary = {}
        for node_id, info in self.robust_clf_node.items():
            node = self.node_dict[node_id]
            _, gate_weights = self.find_gate(node_id)
            for i,c in enumerate(node.child):
                weight_summary[(node_id, c.get_id())] = gate_weights[i]
        
        fout = open(file, 'w')

        fout.write('n1\tn2\t')
        n_feature = len(gate_weights[0])
        if markers is not None:
            for m in markers:
                fout.write(m+'\t')
        else:
            for i in range(n_feature):
                fout.write(str(i)+'\t')
        fout.write('\n')

        for k, v in weight_summary.items():
            k0 = k[0]
            k1 = k[1]
            if k[0] in self.node_to_cluster_id.keys():
                k0 = self.node_to_cluster_id[k[0]]
            if k[1] in self.node_to_cluster_id.keys():
                k1 = self.node_to_cluster_id[k[1]]
            fout.write('%i\t%i\t'%(k0,k1))

            for w in v:
                fout.write('%.3f\t'%w)
            fout.write('\n')
    
    
    def find_full_gate(self, model):
        """ Determines the most relevant gates which specify each partition 
        """
        gate_dict = {} # (tuple to gate ... (clf_node, child_node) -> gate (ordered in magnitude))

        for node_id, info in self.robust_clf_node.items():

            childs = self.node_dict[node_id].child
            gates, _ = self.find_gate(node_id)

            for c, g in zip(childs, gates):
                gate_dict[(node_id, c.get_id())] = g # storing gate info
        
        return gate_dict

    def describe_clusters(self, X_standard, cluster_label = None, marker = None, perc = 0.05):
        """ Checks the composition of each clusters in terms of outliers (define by top and bottom perc)

        Parameters
        --------------
        X_standard : array, shape = (n_sample, n_marker)
            Data array with raw marker expression 
        cluster_label : optional, array, shape = n_sample
            Cluster labels for each data point. If none, just uses the labels infered by the Tree
        marker : optional, list of str, len(list) = n_marker
            Marker labels. If not specified will use marker_0, marker_1, etc.
        perc : optional, float
            The percentage of most and least expressed data points for a marker that you consider outliers
        
        Return
        -------------
        df_pos, df_neg : tuple of pandas.DataFrame
            dataframes with row index as markers and columns as cluster labels. An additional row also
            indicates the size of each cluster as a fraction of the total sample.

        """

        if cluster_label is None:
            cluster_label = self.new_cluster_label
            
        label_to_idx = {} # cluster label to data index
        unique_label = np.unique(cluster_label)
        n_sample, n_marker = X_standard.shape

        if marker is None:
            marker = ['marker_%i'%i for i in range(n_marker)]

        assert n_sample == len(X_standard)
        n_perc = int(round(0.05*n_sample))

        for ul in unique_label:
            label_to_idx[ul] = np.where(cluster_label == ul)[0]

        idx_top = []
        idx_bot = []

        for m in range(n_marker):
            asort = np.argsort(X_standard[:,m])
            idx_bot.append(asort[:n_perc]) # botoom most expressed markers
            idx_top.append(asort[-n_perc:]) # top most expressed markers

        cluster_positive_composition = {}
        cluster_negative_composition = {}

        for label, idx in label_to_idx.items():
            # count percentage of saturated markers in a given cluster ...
            # compare that to randomly distributed (size_of_cluster/n_sample)*n_perc
            cluster_positive_composition[label] = []
            cluster_negative_composition[label] = []
            for m in range(n_marker):

                ratio_pos = len(set(idx_top[m]).intersection(set(idx)))/len(idx_top[m])
                ratio_neg = len(set(idx_bot[m]).intersection(set(idx)))/len(idx_bot[m])

                cluster_positive_composition[label].append(ratio_pos)
                cluster_negative_composition[label].append(ratio_neg)

        df_pos = pd.DataFrame(cluster_positive_composition, index = marker)
        df_neg = pd.DataFrame(cluster_negative_composition, index = marker)
        
        cluster_ratio_size = np.array([len(label_to_idx[ul])/n_sample for ul in unique_label])
        df_cluster_ratio_size = pd.DataFrame(cluster_ratio_size.reshape(1,-1), index = ['Cluster_ratio'], columns = label_to_idx.keys())

        # data frame, shape = (n_marker + 1, n_cluster) with index labels [cluster_ratio, marker_1, marker_2 ...]
        df_pos_new = df_cluster_ratio_size.append(df_pos)
        df_neg_new = df_cluster_ratio_size.append(df_neg)

        return df_pos_new, df_neg_new

    def score_merge(root, model, X, n_average = 10):
        """ Using a logistic regression multi-class classifier, determines the merges that are statistically
        signicant based on a CV prediction score. Returns a robust clustering in the original space.

        Returns
        ---------

        classifier_info : dict
            Keys of dict are ['mean_score', 'mean_score_cluster', 
            'var_score_cluster', 'coeff', 
            'intercept', 'clf', 'mean_xtrain',
            'inv_std_xtrain', 'n_sample']

        """
        
        y = classification_labels(root.get_child(), model)

        pos_subset =  (y != -1)
        Xsubset = X[pos_subset] # original space coordinates
        ysubset = y[pos_subset] # labels

        return classify.fit_logit(Xsubset, ysubset, n_average = n_average, C = 1.0)

    def classification_labels(node_list, model):
        """ Returns a list of labels for the original data according to the classification
        given at root. root is a TreeNode object which contains childrens. Each children (and the data it contains)
        is assigned an arbitrary integer label. Data points not contained in that node are labelled as -1.

        Parameters
        -------
        node_list : list of nodes 

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
    return np.sort(node_list[node_list < n_initial_cluster]) # subset of initial clusters contained in the subtree starting at tree.

