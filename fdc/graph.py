from .fdc import FDC
from .classify import CLF
import numpy as np
from copy import deepcopy
from collections import Counter, OrderedDict
from matplotlib import pyplot as plt
import pickle

class DGRAPH:
    """ Check for neighbors """
    def __init__(self, n_average = 10, cv_score = 0., test_size_ratio = 0.8, clf_type='svm', clf_args=None):
        self.n_average = n_average
        self.cv_score_threshold = cv_score
        self.test_size_ratio = test_size_ratio
        self.clf_type = clf_type
        self.clf_args = clf_args
        self.cluster_label = None
        self.edge_score = OrderedDict()
        self.fout = open('out.txt','a')

    def fit(self, model:FDC, X):
        self.find_nn_list(model) # still need to fit a density map !
        self.fit_all_clf(model, X)
        return self
    
    def find_nn_list(self, model:FDC):
        self.idx_centers = model.idx_centers
        self.rho_idx_centers = model.rho[self.idx_centers]
        cluster_label = model.cluster_label

        self.cluster_label = np.copy(cluster_label)
        self.init_n_cluster = len(np.unique(cluster_label))
        self.current_n_merge = 0


        self.graph = OrderedDict() # adjacency "matrix" -> indexed by tuple
        self.nn_list = OrderedDict() # list of neighboring clusters

        for idx in self.idx_centers:
            nh_idx = model.find_NH_tree_search(idx, -10, cluster_label)
            nh_cluster = np.unique(cluster_label[nh_idx])
            current_label = cluster_label[idx]
        
            if len(nh_cluster) > 1:
                tmp = list(nh_cluster)
                tmp.remove(current_label)
                self.nn_list[current_label] = set(tmp)
            else: # well isolated blob !!
                self.nn_list[current_label] = set([])
        
        # symmetrizing neighborhoods (sometimes this is necessary)

        #print(self.nn_list)
        for k, nnls in self.nn_list.items():
            for e in nnls:
                if k not in self.nn_list[e]:
                    self.nn_list[e].add(k)

    def fit_all_clf(self, model:FDC, X):
        """ Fit clf on all graph edges """

        for center_label, nn_list in self.nn_list.items():
            for nn in nn_list:
                idx_tuple = (center_label, nn)
                idx_tuple_reverse = (nn, center_label)
                if idx_tuple in self.graph.keys():
                    self.graph[idx_tuple_reverse] = self.graph[idx_tuple]
                elif idx_tuple_reverse in self.graph.keys():
                    self.graph[idx_tuple] = self.graph[idx_tuple_reverse]
                else: # hasn't been computed yet
                    clf = self.classify_edge(idx_tuple, X)
                    self.edge_score[idx_tuple] = [clf.cv_score, clf.cv_score_std]
                    edge_info(idx_tuple, clf.cv_score, clf.cv_score_std, self.cv_score_threshold)
                    self.graph[idx_tuple] = clf
    
    def merge_edge(self, X, edge_tuple):
        """ relabels data according to merging, and recomputing new classifiers for new edges """
        
        idx_1, idx_2 = edge_tuple
        pos_1 = (self.cluster_label == idx_1)
        pos_2 = (self.cluster_label == idx_2)
        new_cluster_label = self.init_n_cluster + self.current_n_merge
        
        self.cluster_label[pos_1] = self.init_n_cluster + self.current_n_merge # updating labels !
        self.cluster_label[pos_2] = self.init_n_cluster + self.current_n_merge # updating labels !
        self.current_n_merge += 1

        # recompute classifiers for merged edge
        new_idx = []
        idx_to_del = set([]) # avoids duplicates
        for e in self.nn_list[idx_1]:
            idx_to_del.add((e, idx_1))
            idx_to_del.add((idx_1, e))
            new_idx.append(e)

        for e in self.nn_list[idx_2]:
            idx_to_del.add((e, idx_2))
            idx_to_del.add((idx_2, e))
            new_idx.append(e)
        
        new_nn_to_add = set([])
        for k, v in self.nn_list.items():
            if idx_1 in v:
                v.remove(idx_1)
                v.add(new_cluster_label)
                new_nn_to_add.add(k)
            if idx_2 in v:
                v.remove(idx_2)
                v.add(new_cluster_label)
                new_nn_to_add.add(k)
        
        self.nn_list[new_cluster_label] = set(new_nn_to_add)

        if idx_1 in self.nn_list.keys():
            del self.nn_list[idx_1]
        if idx_2 in self.nn_list.keys():
            del self.nn_list[idx_2]
        
        for k,v in self.nn_list.items():
            if idx_1 in v:
                v.remove(idx_1)
            if idx_2 in v:
                v.remove(idx_2)

        ########################################
        #########################################        

        new_idx.remove(idx_1)
        new_idx.remove(idx_2)
    
        for idxd in idx_to_del:
            del self.graph[idxd]
        
        new_idx_set = set([])
        for ni in new_idx:
            new_idx_set.add((new_cluster_label, ni))

        for idx_tuple in new_idx_set:
            clf = self.classify_edge(idx_tuple, X)
            self.edge_score[idx_tuple] = [clf.cv_score, clf.cv_score_std]
            edge_info(idx_tuple, clf.cv_score, clf.cv_score_std, self.cv_score_threshold)
            self.graph[idx_tuple] = clf
            idx_tuple_reverse = (idx_tuple[1], idx_tuple[0])
            self.graph[idx_tuple_reverse] = self.graph[idx_tuple]
        
        k0_update = []
        k1_update = []
        for k, v in self.graph.items():
            if (k[0] == idx_1) or (k[0] == idx_2): # old index still present !
                k0_update.append(k)        
            elif (k[1] == idx_1) or (k[1] == idx_2):
                k1_update.append(k)
        
        for k0 in k0_update:
            self.graph[(new_cluster_label, k0[1])] = self.graph.pop(k0)
        for k1 in k1_update:
            self.graph[(k1[0], new_cluster_label)] = self.graph.pop(k1)

    def merge_until_robust(self, X, cv_robust):
        self.history = []
    
        while True:
            all_robust = True
            worst_effect_cv = 10
            worst_edge = -1
            for edge, clf in self.graph.items():
                effect_cv = clf.cv_score - clf.cv_score_std
                if effect_cv < worst_effect_cv:
                    worst_effect_cv = effect_cv
                    worst_edge = edge
                if effect_cv < cv_robust:
                    all_robust = False
            
            if all_robust is False:
                n_cluster = self.init_n_cluster - self.current_n_merge - 1
                current_label = self.init_n_cluster + self.current_n_merge - 1

                merge_info(worst_edge[0], worst_edge[1], worst_effect_cv, current_label, n_cluster)
                
                # info before the merge -> this score goes with these labels
                self.history.append([worst_effect_cv, np.copy(self.cluster_label),np.copy(self.idx_centers), deepcopy(self.nn_list)])
                
                pos_idx0 = (self.cluster_label[self.idx_centers] == worst_edge[0])
                pos_idx1 = (self.cluster_label[self.idx_centers] == worst_edge[1])
                
                rho_0 = self.rho_idx_centers[pos_idx0]
                rho_1 = self.rho_idx_centers[pos_idx1]

                if rho_0 > rho_1:
                    tmp_idx = self.idx_centers[pos_idx0]
                    tmp_rho = rho_0
                else:
                    tmp_idx = self.idx_centers[pos_idx1]
                    tmp_rho = rho_1

                self.idx_centers[pos_idx0] = -20
                self.idx_centers[pos_idx1] = -20

                pos_del = self.idx_centers > -1

                # new "center" should go to end of list
                tmp_idx_center_array = np.zeros(len(self.idx_centers)-1,dtype=int)
                tmp_idx_center_array[:-1] = self.idx_centers[pos_del]
                tmp_idx_center_array[-1] = tmp_idx
                self.idx_centers = tmp_idx_center_array

                tmp_rho_array = np.zeros(len(self.rho_idx_centers)-1,dtype=float)
                tmp_rho_array[:-1] = self.rho_idx_centers[pos_del]
                tmp_rho_array[-1] = tmp_rho
                self.rho_idx_centers = tmp_rho_array
                
                self.merge_edge(X, worst_edge)
        
            else:
                break

        if len(self.idx_centers) == 1:
            self.history.append([1.0, np.copy(self.cluster_label),np.copy(self.idx_centers), deepcopy(self.nn_list)])
        else:
            self.history.append([worst_effect_cv, np.copy(self.cluster_label),np.copy(self.idx_centers), deepcopy(self.nn_list)])

    def classify_edge(self, edge_tuple, X, C=1.0):
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
        test_size_ratio = self.test_size_ratio
        n_average = self.n_average

        y = np.copy(self.cluster_label)
        y[(y != edge_tuple[0]) & (y != edge_tuple[1])] = -1

        pos_subset =  (y != -1)
        Xsubset = X[pos_subset] # original space coordinates
        ysubset = y[pos_subset] # labels
        count = Counter(ysubset)

        ''' for v in count.values():
            if v < min_size: # cluster should be merged, it is considered too small
                fake_clf = CLF()
                fake_clf.cv_score = -1.
                fake_clf.cv_score_std = -1.
                return fake_clf '''
        n_sample = len(ysubset)
        return CLF(clf_type=self.clf_type, n_average=n_average, test_size=self.test_size_ratio,clf_args=self.clf_args).fit(Xsubset, ysubset)
    
    def merging_history(self):
        """ Returns the merging history (starting from low cv score and merging iteratively) 
        format is a list with elements of the form [score, y_pred, idx_centers]
        score should be increasing for further elements in the list
        """
        return self.history

    def get_cluster_label(self, n_cluster = None):
        """return dict with keys {cv, y, idx_centers, nn_List} """
        if n_cluster is None:
            tmp = np.array(self.edge_score.values())
            return {'cv':np.min(tmp[:,0]-tmp[:,1]), 'y':self.cluster_label, 'idx_centers':self.idx_centers, 'nn_list':self.nn_list}
        else:
            for s, y, idx, nnlist in self.history:
                if len(idx) == n_cluster:
                    return {'cv':s, 'y':y, 'idx_centers':idx, 'nn_list':nnlist}
                    #return s,y,idx, nnlist
        assert False, 'number of cluster chosen incompatible with merging, no such number achieved'
    
    def cluster_label_standard(self, y=None):
        " instead of using self.cluster_label, relabels cluster so that labels start from 0 and so on"
        if y is None:
            ytmp = self.cluster_label
        else:
            ytmp =y
        new_y = np.zeros(len(ytmp),dtype=int)
        for i, yu in enumerate(np.unique(ytmp)):
            new_y[yu == ytmp]=i
        return new_y

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
    
    def plot_decision_graph(self):
        decision_graph(self.merging_history())

    def plot_density_graph(self, Xss, n_cluster=None, label=None, name = 'graph.pdf', dpi=100, c="#fc4f30", radius=0.1, fontsize=20):
        from lattice import draw_graph # internal package

        if n_cluster is not None:
            data = self.get_cluster_label(n_cluster)
            y, idx_centers, nn_list = data['y'], data['idx_centers'], data['nn_list']
        else:
            idx_centers = self.idx_centers
            nn_list = self.nn_list
        
        xcenter = Xss[idx_centers]
        # careful here with ordering of idx centers ...
        # does not follow the nn_list order !
        n_cluster=len(idx_centers)
        n_cluster = len(nn_list)
        node_label = list(nn_list.keys())
        #print(nn_list)
        order = {node_label[i]:i for i in range(len(node_label))}
        #print('Order for labels:',order)

        A = np.zeros((n_cluster,n_cluster),dtype=int)
        for i, kv in enumerate(nn_list.items()):
            for j, k2 in enumerate(kv[1]):
    
                idx1 = order[kv[0]]
                idx2 = order[k2]
                A[idx1, idx2] = 1
        
        #xcenter =  xcenter[::-1]
        draw_graph(xcenter, A, label=label, savefig=name,radius=radius, dpi=dpi, cnode=c, fontsize=fontsize, figsize=(6,6))

def edge_info(edge_tuple, cv_score, std_score, min_score):
    edge_str = "{0:5<d}{1:4<s}{2:5<d}".format(edge_tuple[0]," -- ",edge_tuple[1])
    if cv_score > min_score:
        out = "[graph.py] : {0:<15s}{1:<15s}{2:<15s}{3:<7.4f}{4:<16s}{5:>6.5f}".format("robust edge ",edge_str,"score =",cv_score,"\t+-",std_score)
    else:
        out = "[graph.py] : {0:<15s}{1:<15s}{2:<15s}{3:<7.4f}{4:<16s}{5:>6.5f}".format("reject edge ",edge_str,"score =",cv_score,"\t+-",std_score)
    print(out)
    self.fout.write(out)

    

def merge_info(c1, c2, score, new_c, n_cluster):
    edge_str = "{0:5<d}{1:4<s}{2:5<d}".format(c1," -- ",c2)
    out = "[graph.py] : {0:<15s}{1:<15s}{2:<15s}{3:<7.4f}{4:<16s}{5:>6d}{6:>15s}{7:>5d}".format("merge edge ",edge_str,"score - std =",score,
    "\tnew label ->",new_c,'n_cluster=',n_cluster)
    print(out)
    self.fout.write(out)


def decision_graph(merging_hist):
    plt.rc('text', usetex=True)
    score = []
    n_cluster = []
    for s, ypred, idx in merging_hist:
        score.append(s)
        n_cluster.append(len(idx))
    
    a= plt.plot(n_cluster[1:],np.diff(score), c="#30a2da",label='$\\leftarrow$',zorder=1)
    plt.scatter(n_cluster[1:],np.diff(score), c="#30a2da", edgecolors='k',zorder=2)
    plt.ylabel('$\Delta s = s(N_c+1)-s(N_c)$')
    plt.xlabel('$N_c$, $\#$ of clusters')
    ax = plt.twinx()
    b=plt.plot(n_cluster, score, c="#fc4f30",label='$\\rightarrow$',zorder=1)
    plt.scatter(n_cluster, score,  c="#fc4f30", edgecolors='k',zorder=2)
    ax.set_ylabel('cross-validation score $s(N_c)$')
    plt.title('Decision Graph\n CV score difference (left), cv score(right) vs. $\#$ of clusters')
    lns = a+b
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc='best')
    plt.show()