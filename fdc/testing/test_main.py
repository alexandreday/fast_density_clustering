def main():

    '''
        Example for gaussian mixture
    '''
    from matplotlib import pyplot as plt
    import sys
    #import plotting # custom plotting functions
    from fdc.special_datasets import gaussian_mixture
    import fdc.plotting as plotting
    from sklearn import datasets
    import pickle
    from fdc import FDC

    n_true_center = 10
    #X,y=datasets.make_blobs(10000, 2, n_true_center, random_state=1984)
    #np.save("data.txt",X)
    
    #exit()
    X,y = gaussian_mixture(n_sample=10000, n_center = n_true_center, sigma_range = [0.25,0.5,1.25],
                            pop_range = [0.1,0.02,0.1,0.1,0.3,0.1,0.08,0.02,0.08,0.1])
                            #random_state = 0)

    model = FDC(nh_size = 40, noise_threshold=0.3)  
    model.fit(X) # Fitting X -> computing density maps and graphs
    
    idx_centers = model.idx_centers
    #print(idx_centers)
    cluster_label = model.cluster_label
    model.coarse_grain(X,0.01,0.2,0.01, compute_hierarchy=True)
    f=open('hier.pkl','wb')
    pickle.dump(model.hierarchy,f)

    #hierarchy

    #print(cluster_label)
    #exit()
    rho = model.rho

    plotting.summary(idx_centers, cluster_label, rho, X, n_true_center=n_true_center, y=y, show=True)

    plotting.dendrogram(model, show=True)

    #print("[fdc] Saving in result.dat with format [idx_centers, cluster_label, rho, n_true_center, X, y, delta]")
    #with open("result.dat", "wb") as f:
    #    pickle.dump([idx_centers, cluster_label, rho, n_true_center, X, y, delta],f)

if __name__=="__main__":
    main()