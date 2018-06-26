import numpy as np

def build_dendrogram(hierarchy, noise_range):
    
    """Constructs the linkage matrix for plotting using the scipy.hierarchy function

    Parameters
    ----------
    hierarchy : list of dictionaries, length of list = number of coarse graining steps
        First element of the list is the dictionary specifying the clusters at the finest scale
        Further elements of the list are coarsed grained. Each element of the list are for different delta values
    noise_range : array-like, length = number of coarse graining steps
        The value of the noise parameter at every scale/step
    
    Returns
    -------
    Z : array-like, shape=(n_coarse_grain,4) ; see scipy for more info 
    Linkage matrix for plotting dendrogram
    
    """

    Z = []
    initial_idx_centers = list(hierarchy[0]['idx_centers'])
    dict_center_relative = {}
    for idx in initial_idx_centers:
        dict_center_relative[idx] = -1
    
    depth = len(hierarchy)
    n_init_centers = len(initial_idx_centers)
    merge_count = 0
    member_count_dict = {}

    for d in range(depth-1):

        pre_idx_centers = hierarchy[d]['idx_centers']
        cur_idx_centers = hierarchy[d+1]['idx_centers']

        pre_cluster_labels = hierarchy[d]['cluster_labels']
        cur_cluster_labels = hierarchy[d+1]['cluster_labels']
        
        for idx in pre_idx_centers :
            if idx not in cur_idx_centers : # means it's been merged

                i = cur_cluster_labels[idx]
                new_idx = cur_idx_centers[i] # pic -> new_pic 
                z = [-1,-1,-1,-1] # linkage list

                if (dict_center_relative[idx] == -1) & (dict_center_relative[new_idx] == -1): # both have not been merged yet
                    z[0] = initial_idx_centers.index(idx)
                    z[1] = initial_idx_centers.index(new_idx)
                    z[2] = noise_range[d+1]
                    z[3] = 2
                elif (dict_center_relative[idx] == -1) & (dict_center_relative[new_idx] != -1):
                    z[0] = initial_idx_centers.index(idx)
                    z[1] = dict_center_relative[new_idx]
                    z[2] = noise_range[d+1]
                    z[3] = 1 + member_count_dict[ z[1] ]
                elif (dict_center_relative[idx] != -1) & (dict_center_relative[new_idx] == -1):
                    z[0] = dict_center_relative[idx]
                    z[1] = initial_idx_centers.index(new_idx)         # ~ new point
                    z[2] = noise_range[d+1]
                    z[3] = 1 + member_count_dict[ z[0] ]
                else:
                    z[0] = dict_center_relative[idx]
                    z[1] = dict_center_relative[new_idx]
                    z[2] = noise_range[d+1]
                    z[3] = member_count_dict[ z[0] ] + member_count_dict[ z[1] ]

                new_cluster_idx = merge_count + n_init_centers
                dict_center_relative[idx] = new_cluster_idx
                dict_center_relative[new_idx] = new_cluster_idx

                member_count_dict[new_cluster_idx] = z[3]
                merge_count += 1

                Z.append(z)
    
    return Z


def compute_linkage_matrix(model):
    from fdc import FDC
    """Wrapper for constructing the final linkage matrix using build_dendrogram function

    Parameters
    ---------------

    model : FDC class object 
        Contains the coarse graining information determined by fitting and coarse graining
        with the data

    Return
    -------
    Z : linkage matrix - (n_coarse_grain, 4)
        From scipy's definition : "An (n−1)(n−1) by 4 matrix Z is 
        returned. At the ii-th iteration, clusters with indices Z[i, 0]
        and Z[i, 1] are combined to form cluster n+in+i. 
        A cluster with an index less than nn corresponds to one of
        the nn original observations. The distance between clusters
        Z[i, 0] and Z[i, 1] is given by Z[i, 2]. The fourth value 
        Z[i, 3] represents the number of original observations in 
        the newly formed cluster."

    """
    from copy import deepcopy
    
    assert type(model) == type(FDC()), 'wrong type !'
    
    hierarchy = deepcopy(model.hierarchy)
    noise_range = deepcopy(model.noise_range)
    
    # ---- PADDING trick --- for plotting purposes  ... 
    n_elem = len(hierarchy[-1]['cluster_labels'])
    terminal_cluster = hierarchy[-1]['idx_centers'][0]
    hierarchy.append({'idx_centers': [terminal_cluster], 'cluster_labels' : np.zeros(n_elem,dtype=int)})
    noise_range.append(1.5*model.max_noise)

    # -------------------------------------------
    # -------------------------------------------

    Z = build_dendrogram(hierarchy, noise_range)

    return Z