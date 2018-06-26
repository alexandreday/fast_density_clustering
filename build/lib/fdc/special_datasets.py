'''
Created on Jan 20, 2017

@author: Alexandre Day
'''

import numpy as np

def gaussian_mixture(n_sample=1000,n_center=2,sigma_range=[1.0],pop_range=[0.1,0.9],random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
        
    assert abs(np.sum(pop_range)-1.0)<0.0001
    assert n_center==len(pop_range)
    n_sigma=len(sigma_range)
    sigma_range=np.array(sigma_range)
    boundx=10
    boundy=10
     
    X=np.empty((n_sample,2),dtype=np.float)
    y=np.empty(n_sample,dtype=np.int)
    total_sample=0
     
    for c in range(n_center):
        n_sample_c=int(round(n_sample*pop_range[c])-0.5)
        #print(n_sample_c)
        sig_1,sig_2=sigma_range[np.random.randint(0,n_sigma,2)]
        bound=np.sqrt(sig_1*sig_2)
        sig_12=np.random.uniform()*2*bound-bound
        xcenter=np.random.uniform()*2*boundx-boundx
        ycenter=np.random.uniform()*2*boundy-boundy
        
        C = np.array([[sig_1, sig_12], [sig_12, sig_2]])
        
        X[total_sample:total_sample+n_sample_c] = np.random.multivariate_normal([xcenter,ycenter], C, n_sample_c)
        y[total_sample:total_sample+n_sample_c]=np.full(n_sample_c,c,dtype=np.int)
        total_sample+=n_sample_c

    return X,y