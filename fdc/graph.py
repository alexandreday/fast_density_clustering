from .fdc import FDC
from .classify import CLF
import numpy as np
from collections import Counter

class DGRAPH:
    """ Check for neighbors """
    def __init__(self, n_average = 50, cv_score = 0., min_size = 50, test_size_ratio = 0.5):
        self.n_average = n_average
        self.cv_score_threshold = cv_score
        self.min_size = min_size
        self.test_size_ratio = test_size_ratio

    def fit(self, model:FDC, X):
        #self.nh_graph = model.nh_graph
        #model = FDC()

        idx_centers = model.idx_centers
        cluster_label = model.cluster_label

        self.graph = {} # adjacency "matrix" -> indexed by tuple
        self.nn_list = {} # list of neighboring clusters

        for idx in idx_centers:
            nh_idx = model.find_NH_tree_search(idx, -10, cluster_label)
            nh_cluster = np.unique(cluster_label[nh_idx])
            current_label = cluster_label[idx]
        
            if len(nh_cluster) > 1:
                tmp = list(nh_cluster)
                tmp.remove(current_label)
                self.nn_list[current_label] = tmp
            else: # well isolated blob !!
                self.nn_list[current_label] = []
        
        self.fit_all_clf(model, X)
    
    def fit_all_clf(self, model:FDC, X):

        for center_label, nn_list in self.nn_list.items():
            for nn in nn_list:
                idx_tuple = (center_label, nn)
                idx_tuple_reverse = (nn, center_label)
                if idx_tuple in self.graph.keys():
                    self.graph[idx_tuple_reverse] = self.graph[idx_tuple]
                elif idx_tuple_reverse in self.graph.keys():
                    self.graph[idx_tuple] = self.graph[idx_tuple_reverse]
                else: # hasn't been computed yet
                    clf = self.classify_edge(idx_tuple, model, X)
                    edge_info(idx_tuple, clf.cv_score, clf.cv_score_std, self.cv_score_threshold)

    def classify_edge(self, edge_tuple, model, X, C=1.0):
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

        y = np.copy(model.cluster_label)
        y[(y != edge_tuple[0]) & (y != edge_tuple[1])] = -1

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

        return CLF(clf_type='svm', n_average=n_average, C=C, down_sample=min_size).fit(Xsubset, ysubset)

def edge_info(edge_tuple, cv_score, std_score, min_score):
    edge_str = "%i -- %i"%(edge_tuple[0],edge_tuple[1])
    if cv_score > min_score:
        print("[graph.py] : {0:<15s}{1:<10s}{2:<10s}{3:<7.4f}{4:5s}{5:6.5f}".format("robust edge ",edge_str,"score =",cv_score,"\t+-",std_score))
    else:
        print("[graph.py] : {0:<15s}{1:<10s}{2:<10s}{3:<7.4f}{4:5s}{5:6.5f}".format("reject edge ",edge_str,"score =",cv_score,"\t+-",std_score))