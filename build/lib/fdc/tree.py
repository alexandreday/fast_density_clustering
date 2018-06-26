from .fdc import FDC
from .classify import CLF
import numpy as np
import pickle
from scipy.cluster.hierarchy import dendrogram as scipydendroed
from scipy.cluster.hierarchy import to_tree
from .hierarchy import compute_linkage_matrix
import copy
from collections import OrderedDict as OD
from collections import Counter

class TREENODE:

    def __init__(self, id_ = -1, parent = None, child = None, scale = -1):
        if child is None:
            self.child = [] # has to be list of TreeNode
        else:
            self.child = child
        self.scale = scale
        self.parent = parent
        self.id_ = id_

    def __repr__(self):
        return ("Node: [%s] @ s = %.3f" % (self.id_,self.scale)) 

    def is_leaf(self):
        return len(self.child) == 0

    def get_child(self, id_ = None):
        if id_ is None:
            return self.child
        else:
            for c in self.child:
                if c.get_id() == id_:
                    return c

    def get_child_id(self):
        if len(self.child) == 0:
            return []
        else:
            return [c.id_ for c in self.child]

    def get_scale(self):
        return self.scale

    def get_id(self):
        return self.id_

    def add_child(self, treenode):
        self.child.append(treenode)
    
    def remove_child(self, treenode):
        self.child.remove(treenode)
        
    def get_rev_child(self):
        child = self.child[:]
        child.reverse()
        return child 
    
class TREE:
    """ Contains all the hierachy and information concerning the clustering """
    
    def __init__(self, n_average = 50, cv_score = 0., min_size = 50, test_size_ratio = 0.5, ignore_root = False):
        """" Tree model to deal with hiearchical clustering stored in FDC().hierarchy
        
        Parameters
        -----------
        n_average: int, optional (default = 10)
            number of classifiers that will be trained on different random partitioned. the score of each
            node of the tree corresponds to the average score. the prediction is obtained using a majority vote.

        cv_score: float, optional (default = 0.0)
            if the average cross-validation score for a node is above this value, it is called a robust node
            should be starting with cv_score = 0, then sweeping up, using bottom-up clustering

        min_size: int, optional (default = 50)
            minimal size of a cluster

        test_size_ratio: float (default = 0.5)
            size of the test set to be used when cross-validating
        
        ignore_root: bool (default = False)
            wether or not to ignore the root score when checking for robust nodes.
            the root is special since in many density hierarchies, the top most clusters will never be merged together
            thus the root as a node will split into multiple (more than 2) high-level clusters, whereas all other nodes will typically 
            split into 2 clusters.

        Return
        ------------
        self: TREE object
    
        """

        self.node_dict = None
        self.mergers = None
        self.new_cluster_label = None
        self.robust_terminal_node = None #list of the terminal robust nodes
        self.robust_clf_node = None # full information about classification is recorded here, keys of dict are the classifying nodes id
        self.all_robust_node = None # list of all nodes in the robust tree (classifying nodes and leaf nodes)
        self.cluster_to_node_id = None # dictionary mapping cluster labels (displayed on plot) with node id

        self.new_idx_centers = None
        self.tree_constructed = False

        # ------------------> Classifier information
        self.ignore_root = ignore_root
        self.n_average = n_average
        self.cv_score = cv_score
        self.min_size = min_size
        self.test_size_ratio = test_size_ratio

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
        
        mergers = find_mergers(model.hierarchy, model.noise_range)
        mergers.reverse()
        m = mergers[0]

        self.node_dict = OD()
        
        self.root = TREENODE(id_ = m[1], scale = m[2])
        self.node_dict[self.root.get_id()] = self.root
            
        for m in mergers:
            for mc in m[0]:
                c_node = TREENODE(id_ = mc, parent = self.node_dict[m[1]], child = [], scale = -1)
                self.node_dict[m[1]].add_child(c_node)
                self.node_dict[c_node.get_id()] = c_node
            self.node_dict[m[1]].scale = m[2]

        self.mergers = mergers
        self.tree_constructed = True
    
    def merge_nodes(self, node_list, target_node):
        for node in node_list:
            parent = node.parent
            target_node.add_child(node) # transfering node
            parent.remove_child(node)

    def node_items(self): # breath-first ordering
        """ Returns the full list of nodes below the root
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

    def identify_robust_merge(self, model, X):
        """Starting from the root, goes down the tree and evaluates which clustering nodes are robust.
        Each node in the tree corresponds to a partitioning of the a subset of the data. For each 
        node one computes a cross-validation score on a downsampled dataset in order to compare nodes.

        The nodes that are robust and are terminal (meaning no robust nodes exist below them) are
        stored in two attributes: 

        self.robust_terminal_node (list) # list of the terminal node indices (which can then be accessed by self.node_dict)
        self.robust_clf_node (dict) # dictionary of classifiers (CLF objects from classify.py) - keys are node indices ; 

        """

        self.build_tree(model)  # Extracts all the information from model and outputs a tree    
    
        root, node_dict, mergers = self.root, self.node_dict, self.mergers

        print("[tree.py] : Printing two top-most layers")
        print("[tree.py] : Root :", root)
        print("[tree.py] : Root's childs :", root.get_child())

        self.compute_robust_node(model, X)

        # Listing all nodes in the robust tree ...

        all_robust_node = set([])

        for k, _ in self.robust_clf_node.items():
            all_robust_node.add(k)
            current_node = node_dict[k]
            for c in current_node.child:
                all_robust_node.add(c.get_id())

        self.all_robust_node = list(all_robust_node)

    def compute_robust_node(self, model, X):
        """ Start from the root, computes the classification score at every branch in the tree
        and stops if classication score is below a certain threshold.

        Results are stored in:
            self.robust_clf_node : dictionary of node id to classification information (weights, biases, scores, etc.)
            self.robust_terminal_node : list of terminal nodes id, whose parents are robust classifiers.
        """
        if self.robust_clf_node is None:
            self.robust_terminal_node = [] #list of the terminal robust nodes
            self.robust_clf_node = OD() # dictionary of the nodes where a partition is made (non-leaf nodes)
        
        else:    
            ######## once the tree has been fully fitted, will just perform bottom-up merges based on desired score
            while True:
                found_merge = False
                for node_id in self.robust_terminal_node:
                    p = self.node_dict[node_id].parent
                    if p is None: # root is reached !
                        break
                    clf = self.robust_clf_node[p.get_id()]
                    if clf.cv_score - clf.cv_score_std < self.cv_score: # remove that node
                        # 2 cases here:
                        #   1. node is merged with another leaf node
                        #   2. node is merged with a subtree (which can create big instabilities !)
                        case = 1
                        for c in p.get_child_id():
                            if c not in self.robust_terminal_node:
                                case = 2
                                break
                        
                        if case == 1:
                            self.robust_clf_node.pop(p.get_id())
                            #print('popped')
                            sub_node_list = breath_first_search(p)[1:]
                            for n in sub_node_list: # need to clear out the full subtree
                                if n in self.robust_clf_node.keys():
                                    self.robust_clf_node.pop(n)
                                if n in self.robust_terminal_node:
                                    self.robust_terminal_node.remove(n)
                                found_merge = True
                            self.robust_terminal_node.append(p.get_id()) # parent now becomes the terminal node
                            break

                if found_merge is False:
                    break
            return

        if self.root.get_id() in self.robust_clf_node.keys():
            clf = self.robust_clf_node[self.root.get_id()]
        else:
            clf = self.classify_node(self.root.get_child(), model, X)

        min_cv_score = self.cv_score
        clf_score = clf.cv_score
        std_score = clf.cv_score_std
    
        if self.ignore_root is True:
            print("[tree.py] : root is ignored, #  %i \t score = %.4f"%(self.root.get_id(), clf_score))
            self.robust_clf_node[self.root.get_id()] = clf
        else:
            if clf_score-std_score > min_cv_score: # --- search stops if the node is not statistically signicant (threshold)
                node_info(self.root, clf_score, std_score, min_cv_score)
                self.robust_clf_node[self.root.get_id()] = clf
            else:
                print("[tree.py] : root not robust #  %i \t score = %.4f"%(self.root.get_id(),clf_score))

        for current_node in self.node_items()[1:]:
            if current_node.parent.get_id() in self.robust_clf_node.keys():
                if not current_node.is_leaf():
                    
                    if current_node.get_id() in self.robust_clf_node.keys():
                        clf = self.robust_clf_node[current_node.get_id()]
                    else:
                        clf = self.classify_node(current_node.get_child(), model, X)

                    clf_score = clf.cv_score
                    std_score = clf.cv_score_std
                    
                    if clf_score-std_score > min_cv_score: # --- search stops if the node is not statistically signicant (threshold)
                        node_info(current_node, clf_score, std_score, min_cv_score)
                        self.robust_clf_node[current_node.get_id()] = clf

                    else:
                        node_info(current_node, clf_score, std_score, min_cv_score)
                        self.robust_terminal_node.append(current_node.get_id())
                else: # implies it's parent was robust, and is a leaf node 
                    self.robust_terminal_node.append(current_node.get_id())
        
    def fit(self, model, X, cv_score = None):
        """ Finds the merges that are statistically significant (i.e. greater than the cv_score)
        and relabels the data accordingly
        
        Trick here: first use a low threshold (will compute the tree down to it's lowest components)
        Then one can just iterate quickly over score threshold ...

        Parameters
        ------

        model : FDC object
            Contains the coarse graining information

        X  : array, shape = (n_sample, n_marker)
            Contains the data in the original space
        
        n_average : int
            Number of folds in the cross validation

        cv_score : float
            Classification score threshold
        
        Returns
        ---------
        self : TREE() object

        """

        n_average = self.n_average
        
        if cv_score is None:
            cv_score = self.cv_score
        else:
            self.cv_score = cv_score
        
        if cv_score > 1.0 or cv_score < 0.0:
            assert False, "** cv_score must be between 0.0 and 1.0 **"

        print('[tree.py] : fitting with cv_score = %.4f'%self.cv_score)
        self.identify_robust_merge(model, X) # fitting all the classifiers
        robust_terminal_node = self.robust_terminal_node # this is a list
        
        root = self.root
        node_dict = self.node_dict
        mergers = self.mergers
        
        ##### Below : relabelling data to output final labels according to classifiers

        cluster_n = len(robust_terminal_node)

        n_sample = len(model.X)
        y_robust = -1*np.ones(n_sample,dtype=np.int)
        y_original = model.hierarchy[0]['cluster_labels']
        cluster_to_node_id = OD()

        # here all terminal nodes are given a label, in the same order they are stored.
        y_node = classification_labels([node_dict[i] for i in robust_terminal_node], model)
        
        assert np.count_nonzero(y_node == -1) == 0, "Wrong labelling or ROOT is not robust ... !"

        for i, node_id in enumerate(robust_terminal_node):
            pos = (y_node == i)
            y_robust[pos] = i
            cluster_to_node_id[i] = node_id
        #print(cluster_n)
        #print(Counter(y_node))
        
        if len(robust_terminal_node) == 0:
            y_robust *= 0 # only one coloring 
        
        new_idx_centers = []
        all_idx = np.arange(0, model.X.shape[0], dtype=int)

        for i in range(cluster_n):
            pos_i = (y_robust == i)
            max_rho = np.argmax(model.rho[y_robust == i])
            idx_i = all_idx[pos_i][max_rho]
            new_idx_centers.append(idx_i)

        self.new_cluster_label = y_robust
        self.new_idx_centers = np.array(new_idx_centers,dtype=int)
        self.cluster_to_node_id = cluster_to_node_id
        self.node_to_cluster_id = OD({v: k for k, v in self.cluster_to_node_id.items()})
        
        print("\n")
        print("[tree.py] : -----------> VALIDATION SCORING INFORMATION < -----------------")
        print("[tree.py] : ", "{0:<15s}{1:<15s}{2:<15s}{3:<15s}{4:<15s}{5:15s}".format("Terminal node", "Parent node", "Displayed node", "Cv score", "Cv +-","Effective CV"))
        for n in robust_terminal_node:
            p = self.node_dict[n].parent
            if p is not None:
                p_id = p.get_id()
                cv_score = self.robust_clf_node[p_id].cv_score
                cv_score_std = self.robust_clf_node[p_id].cv_score_std
                print("[tree.py] : ", "{0:<15d}{1:<15d}{2:<15d}{3:<15.4f}{4:<15.5f}{5:<15.4f}".format(n,p_id,self.node_to_cluster_id[n],
                cv_score, cv_score_std, cv_score - cv_score_std))
            else:
                print("[tree.py] : root reached -> one cluster remaining")
                
        return self
    
    def predict(self, X):
        """ Uses the root classifiers to perform a hierarchical classification of the nodes !
        need to do recursive classification ... 
        
        """
        terminal_nodes = set(self.robust_terminal_node)
        node_to_cluster = self.node_to_cluster_id
        y_pred = -1*np.ones(len(X))
    
        for i, x in enumerate(X):
            if i% 1000 == 0:
                print(i)
            current_clf_node = self.root # recursively go down the tree, starting from root
            current_id = current_clf_node.get_id()
            while True:
                if current_clf_node.get_id() in terminal_nodes:
                    y_pred[i] = node_to_cluster[current_id] # reached the leaf node
                    break
                else:
                    y_branch = self.robust_clf_node[current_id].predict([x])[0]
                    
                child_list = current_clf_node.child
                current_clf_node = child_list[y_branch] # go down one layer
                current_id = current_clf_node.get_id()
            
        return y_pred

    def save(self, name=None):
        """ Saves current model to specified path 'name' """
        if name is None:
            name = self.make_file_name()
        fopen = open(name,'wb')
        pickle.dump(self,fopen)
        fopen.close()
        
    def load(self, name=None):
        if name is None:
            name = self.make_file_name()

        self.__dict__.update(pickle.load(open(name,'rb')).__dict__)
        return self

    def make_file_name(self):
        t_name = "clf_tree.pkl"
        return t_name

    def write_result_mathematica(self, model, marker) : # graph should be a dict of list
        """ 
        -> Saves results in .txt files, which are easily read with a Mathematica
        script for ez plotting ...
        """
        if self.robust_clf_node is None :
            assert False, "Model not yet fitted !"

        self.gate_dict = self.find_full_gate(model)

        my_graph = OD()
        my_graph_score = OD()
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

    def print_mapping(self):
        print("Mapping of terminal nodes to plotted labels:") 
        [print(k, " -> ", v) for k,v in OD(self.node_to_cluster_id).items()]

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
            
        label_to_idx = OD() # cluster label to data index
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

        cluster_positive_composition = OD()
        cluster_negative_composition = OD()

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

##############################################
###############################################

    def classify_node(self, node_list, model, X, C=1.0):
        """ Trains a classifier on the childs of "root" and returns a classifier for these types.

        Important attributes are (for CLF object):

            self.scaler_list -> [mu, std]

            self.cv_score -> mean cv score

            self.mean_train_score -> mean train score

            self.clf_list -> list of sklearn classifiers (for taking majority vote)
        
        Returns
        ---------
        CLF object (from classify.py). Object has similar syntax to sklearn's classifier syntax

        """
        ## ok need to down sample somewhere here
        min_size = self.min_size
        test_size_ratio = self.test_size_ratio
        n_average = self.n_average

        y = classification_labels(node_list, model)

        if len(np.unique(y)) == 1:
            return CLF(clf_type='trivial')

        pos_subset =  (y != -1)
        Xsubset = X[pos_subset] # original space coordinates
        ysubset = y[pos_subset] # labels

        count = Counter(ysubset)
        for v in count.values():
            if v < min_size: # cluster should be merged, it is considered too small
                fake_clf = CLF()
                fake_clf.cv_score = -1.
                fake_clf.cv_score_std = -1.
                return fake_clf

        return CLF(clf_type='svm', n_average=n_average).fit(Xsubset, ysubset)

def classification_labels(node_list, model):
    """ Returns a list of labels for the original data according to the classification
    given at root. root is a TreeNode object which contains childrens. Each children (and the data it contains)
    is assigned an arbitrary integer label. Data points not contained in that node are labelled as -1.

    Parameters
    -------
    node_list : list of nodes, these should be the child of a parent node for instance.

    model : FDC object

    Returns
    --------
    1D array of labels
    """
    
    n_sample = len(model.X)
    y = -1*np.ones(n_sample,dtype=np.int)
    y_init = model.hierarchy[0]['cluster_labels'] # full set of labels at smallest scale ... 

    for i, node in enumerate(node_list): # all data points contained a node take label i
        init_c = find_idx_cluster_in_root(model, node)
        for ic in init_c:
            # relabelling here according to merger
            y[y_init == ic] = i 

    return y


def find_mergers(hierarchy, noise_range):    

    """ Determines the list of merges that are made during the coarse-graining """
    
    n_depth = len(noise_range)
    n_initial_cluster = len(hierarchy[0]['idx_centers'])
    initial_labels = hierarchy[0]['cluster_labels']
    n_pre_cluster = n_initial_cluster

    current_merge_idx = n_initial_cluster
    n_merge = 0
    merging_dict = OD()

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
    for k, v in merging_dict.items():
        if v == -1:
            mapped_u.append(k)
    
    # adding top row !!
    if len(merger_record) == 0:
        merger_record.append([mapped_u, current_merge_idx, 1.0])
    elif len(mapped_u) > 1:
        merger_record.append([mapped_u, current_merge_idx, 1.5*(merger_record[-1][2])])

    return merger_record

########################################################################################
########################################################################################
################################# UTILITY FUNCTIONS ####################################
########################################################################################
########################################################################################
def node_info(node, cv_score, std_score, min_score):
    if cv_score > min_score:
        print("[tree.py] : {0:<15s}{1:<10d}{2:<10s}{3:<7.4f}{4:5s}{5:6.5f}".format("robust node #",node.get_id(),"score =",cv_score,"\t+-",std_score))
    else:
        print("[tree.py] : {0:<15s}{1:<10d}{2:<10s}{3:<7.4f}{4:5s}{5:6.5f}".format("reject node #",node.get_id(),"score =",cv_score,"\t+-",std_score))


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
    """
    Returns
    -------
    node_list : list of node id contained in root 
    """

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
    
def find_idx_cluster_in_root(model, node):
    """ Finds the original (noise_threshold = init) clusters contains in the node
    Returns the index of the terminal nodes contained in node.
    """
    node_list = np.array(breath_first_search(node)) #list of terminal nodes contained in node. 
    n_initial_cluster = len(model.hierarchy[0]['idx_centers']) # map out what is going on here .
    # recall that the cluster labelling is done following the dendrogram convention (see scipy)
    return np.sort(node_list[node_list < n_initial_cluster]) # subset of initial clusters contained in the subtree starting at node