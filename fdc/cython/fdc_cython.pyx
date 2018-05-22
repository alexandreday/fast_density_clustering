def find_NH_tree_search(
        long idx,
        double eta, 
        long[:] cluster_label,
        double[:] rho,
        long[:,:] nn_list,
        long search_size,

    ):

    cdef long[:] new_leaves = nn_list[idx, :search_size] # view
    is_NH_ = np.zeros(len(nn_list),dtype=np.int) 
    cdef long[:] is_NH = is_NH_ # view 
    cdef long i, n_sample=len(is_NH)

    for i in range(n_sample):
        if 

    is_NH[new_leaves[rho[new_leaves] > eta]] = 1

    current_label = cluster_label[idx]

    # ideally here we cythonize what's below... this is highly ineficient ...
    while True:

        update = False
        leaves=np.hstack(new_leaves)
        new_leaves=[]

        y_leave = cluster_label[leaves]
        leaves_cluster = leaves[y_leave == current_label]        
        nn_leaf = nn_list[leaves_cluster]

        for i in range(1, self.search_size):
            res = nn_leaf[is_NH[nn_leaf[:,i]] == 0, i]
            pos = np.where(rho[res] > eta)[0]

            if len(pos) > 0: update=True
            
            is_NH[res[pos]] = 1
            new_leaves.append(res[pos])

        if update is False:
            break

    return np.where(is_NH == 1)[0]