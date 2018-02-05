from .fdc import FDC

class DGRAPH:
    """ Check for neighbors """
    def __init__(self, n_average = 50, cv_score = 0., min_size = 50, test_size_ratio = 0.5):
        self.n_average = n_average
        self.cv_score_threshold = cv_score
        self.min_size = min_size
        self.test_size_ratio = test_size_ratio

    def fit(self, model, X):
        self.nh_graph = model.nh_graph
        #model = FDC()

        idx_centers = model.idx_centers
        cluster_label = model.cluster_label
        
        for idx in idx_centers:
            nh_idx = model.find_NH_tree_search(idx, 10, cluster_label)
            nh_cluster = np.unique(nh_idx)
            print(idx, nh_cluster)
        
