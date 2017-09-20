from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np

def gates(W, n_g=2):
    """ Return the most important gates, sorted by amplitude. One set of get per category (class)
    The number of gates returns is set by n_g
    """
    g_list = []
    for i, w in enumerate(W):
        wsort = np.sort(np.abs(w))
        best_amp = wsort[-n_g:]
        best_gates = []
        for ba in best_amp:
            pos = np.argmin(np.abs(np.abs(w)-ba))
            best_gates.append(pos)
            if w[pos] < 0: # stores position and sign of the gate for now
                best_gates[-1] *= -1
        g_list.append(best_gates)

    return g_list

def fit_logit(X, y, n_average = 10, C = 1.0, n_iter_max = 100):
    """ Multivariate logistic regression model (softmax) for classification with labels specified in model.cluster_label
    and coordinates given by X. 

    Parameters
    ----------
    X : array, shape = (n_sample, n_features)
        Data set
    n_average : int (optional, default= 10)
        Number of folds for CV (different random partitions) for fitting the weights
    y: list, shape = (n_sample) 
        Contains the cluster label for every data point
    C : float (optional, default=1.0) 
        Inverse of regularization parameter strength

    Returns
    ----------
    results : dict
        Contains all the classifier info; 
        Keys: ['mean_score', 'mean_score_cluster', 'var_score_cluster', 'coeff', 'intercept', 'clf', 'mean_xtrain', 'inv_std_xtrain', 'n_sample']
    
    """

    W_list = []
    b_list = []
    total_score_list = []
    accuracy_sample = {} 
    clf_list = []
    n_unique = len(np.unique(y)) # need to standardize the data ... 
    n_sample = X.shape[0]

    for i in range(n_unique):
        accuracy_sample[i] = []

    for _ in range(n_average):
        
        ytrain, ytest, xtrain, xtest = train_test_split(y, X, test_size=0.2)
        mu, inv_sigma = np.mean(xtrain, axis=0), 1./np.std(xtrain, axis = 0)
        xtrain = (xtrain - mu)*inv_sigma

        unique_ytrain = np.unique(ytrain)

        logreg = LogisticRegression(penalty = 'l2', C=C, solver = 'lbfgs', multi_class='multinomial', class_weight='balanced', max_iter=n_iter_max)
        #logreg = LogisticRegression(penalty = 'l1', C=C, solver = 'saga', multi_class='multinomial', class_weight='balanced', max_iter=n_iter_max)
        logreg.fit(xtrain, ytrain)
        clf_list.append(logreg)

        W_list.append(logreg.coef_)
        b_list.append(logreg.intercept_)

        total_score = logreg.score(inv_sigma*(xtest-mu), ytest) # predict on test set
        total_score_list.append(total_score)

        ypred = logreg.predict(xtest) # check performand per 

        min_accuracy = 1.01
        unique_ytest = np.unique(ytest)
       
        for c in unique_ytest:
            ypred_c = ypred[ytest == c]
            #exit()
            TP = np.count_nonzero(ypred_c == c) # number of good answers !
            accuracy = TP/len(ypred_c)*1.0
            if accuracy < min_accuracy:
                min_accuracy = accuracy
            accuracy_sample[c].append(accuracy)
    
    mean_accuracy = {}
    std_accuracy = {}
    for cluster, sample in accuracy_sample.items():
        mean_accuracy[cluster] = np.mean(sample)
        std_accuracy[cluster] = np.std(sample)

    best = np.argmax(total_score_list)

    #print('Score for %i = '%, total_score_list[best])

    results = {
        'mean_score' : np.mean(total_score_list),
        'mean_score_cluster' :  mean_accuracy,
        'var_score_cluster' : std_accuracy,
        'coeff' : W_list[best],
        'intercept' : b_list[best],
        'clf': clf_list[best],
        'mean_xtrain' : mu,
        'inv_std_xtrain' : inv_sigma,
        'n_sample': n_sample # number of samples used for training -> !
    }
    
    return results

