import numpy as np

def match_labels(y1, y2):
    y1_u = np.unique(y1)
    y2_u = np.unique(y2)

    element_1 = {l1 : set(list(np.where(y1 == l1)[0])) for l1 in y1_u}
    element_2 = {l2 : set(list(np.where(y2 == l2)[0])) for l2 in y2_u}

    map_l1_to_l2 = {}

    for l1 in y1_u:
        e1 = element_1[l1]
        min_dist = 0.
        match = -1
        for l2 in y2_u :
            e2 = element_2[l2]
            dist = len(e1.intersection(e2))/len(e1.union(e2))
            if dist > min_dist:
                min_dist = dist
                match = l2
        map_l1_to_l2[l1] = match
    
    return map_l1_to_l2


    




