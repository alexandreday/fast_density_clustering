from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from collections import Counter

def most_common(lst):
    return max(set(lst), key=lst.count)

def fit_logit(X, y, n_average = 10, C = 1.0, n_iter_max = 100):
    """ Multivariate logistic regression model (softmax) for classification with labels specified in y
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
    n_iter_max : int (optional, defaul=1.0)
        Number of iterations for fitting the classifier

    Returns
    ----------
    results : dict
        Contains all the classifier info; 
        Keys: ['mean_score', 'mean_score_cluster', 'var_score_cluster', 'coeff', 'intercept', 'mean_xtrain', 'inv_std_xtrain', 'n_sample']
    
    """

    W_list = []
    b_list = []
    total_score_list = []
    accuracy_sample = {} 
    unique_y = np.unique(y)
    n_unique = len(unique_y) # need to standardize the data ... 
    n_sample = X.shape[0]
    zero_eps = 1e-6 

    for i in range(n_unique):
        accuracy_sample[i] = []

    for _ in range(n_average):
        
        ytrain, ytest, xtrain, xtest = train_test_split(y, X, test_size=0.2)
    
        std = np.std(xtrain, axis = 0)
        std[std < zero_eps] = 1.0 # get rid of zero variance data.
        mu, inv_sigma = np.mean(xtrain, axis=0), 1./std

        xtrain = (xtrain - mu)*inv_sigma

        unique_ytrain = np.unique(ytrain)

        logreg = LogisticRegression(penalty = 'l2', C=C, solver = 'lbfgs', multi_class='multinomial', class_weight='balanced', max_iter=n_iter_max)
        #logreg = LogisticRegression(penalty = 'l1', C=C, solver = 'saga', multi_class='multinomial', class_weight='balanced', max_iter=n_iter_max)
        logreg.fit(xtrain, ytrain)

        W_list.append(logreg.coef_)
        b_list.append(logreg.intercept_)

        total_score = logreg.score(inv_sigma*(xtest-mu), ytest) # predict on test set
        total_score_list.append(total_score)

        ypred = logreg.predict(inv_sigma*(xtest-mu)) # check performance per cluster
        

        #-----------
        #dlsfklsd
        for c in unique_y:
            if c in ytest:
                idx_y = (ytest == c)
                y_sub = ypred[idx_y] 
                s = np.sum(y_sub == c)/np.sum(idx_y) # number of correctly predicted
                accuracy_sample[c].append(s)
            else:
                accuracy_sample[c].append(-1) # MAY trigger a bug here, need to fix small clusters classification ... !

    mean_accuracy = {}
    std_accuracy = {}

    score_sort = np.argsort(total_score_list)
    half_sample = int(n_average*0.5)
    best_half_idx = score_sort[-half_sample:] # average over best half of the results, but with different partitions !

    for cluster, sample in accuracy_sample.items():
        mean_accuracy[cluster] = np.mean(np.array(sample)[best_half_idx])  # sample_accuracy over best ones
        std_accuracy[cluster] = np.std(np.array(sample)[best_half_idx])    # std_accuracy over best ones

    W_mean = np.mean([W_list[i] for i in best_half_idx], axis=0)
    b_mean = np.mean([b_list[i] for i in best_half_idx], axis=0)

    # Do a majority vote over folds 
    # 


    results = { # this contains the weight matrix and the intercept to reconstruct the classifier.
        'mean_score' : np.mean(total_score_list),
        'mean_score_cluster' :  mean_accuracy, # mean accuracy for each cluster
        'var_score_cluster' : std_accuracy, # std of accuracy for each cluster
        'coeff' : W_mean,       # this averaged weights over best performing classifiers
        'intercept' : b_mean,   #b_list[best],
        'mean_xtrain' : mu,
        'inv_std_xtrain' : inv_sigma,
        'n_sample': n_sample # number of samples used for training (xtrain.shape[0] + xtest.shape[0])
    }
    
    return results

class CLF:

    def __init__(self, clf_type, **kwargs):
        """ Implements a classifier for hierarchical clustering

        Parameters:
        ----------
        clf_type : str
            Type of cluster, either 'svm' or 'logreg'
        kwarg : optional arguments for SVM (see SVM definition below for name of keyword arguments)
        """
        self.clf_type = clf_type
        self.kwargs = kwargs
        self.trained = False
        self.cv_score = 1.0
    
    def fit(self, X, y):
        """Fits classifier

        Important attributes are:

        self.scaler_list -> [mu, std]
        self.cv_score -> mean cv score
        self.mean_train_score -> mean train score
        self.clf_list -> list of sklearn classifiers (for taking majority vote)

        Returns
        -----------
        CLF object (self)
        """
        
        self.trained = True

        if self.clf_type == 'svm':
            return self.fit_SVM(X, y, **self.kwargs)
        else:
            return fit_logit(X, y, **self.kwargs)
        
    def predict(self, X):
        """ Returns labels for X (-1, 1)
        """
        if self.clf_type == 'trivial':
            self._n_sample = len(X)
            return np.zeros(len(X))

        if self.trained is False:
            assert False, "Must train model first !"

        # col is clf, row are different data points
        n_clf = len(self.clf_list)
        vote = []

        for i in range(n_clf):
            clf = self.clf_list[i]
            mu, inv_sigma = self.scaler_list[i]
            xstandard = inv_sigma*(X-mu)

            vote.append(clf.predict(inv_sigma*(X-mu)))

        vote = np.vstack(vote).T
        # row are data, col are clf
    
        y_pred = []
        for x_vote in vote:
            y_pred.append(most_common(list(x_vote)))

        return np.array(y_pred).reshape(-1,1)

    ''' def prob_predict(self, X):
        """ Makes a prediction but also returns the probability for that prediction (based on the vote)
        Returns an array of shape (-1, 2)
        """
        if self.trained is False:
            assert False, "Must train model first !"

        n_clf = len(self.clf_list)

        vote = []
        for i in range(n_clf):
            clf = self.clf_list[i]
            mu, inv_sigma = self.prob_predict[i]
            vote.append(clf.predict(inv_sigma*(X-mu)))

        vote = np.hstack(vote)
        y_pred = []

        for x_vote in vote:
            count = Counter(x_vote)
            p = count.values / n_clf
            pos_max = np.argmax(p)
            y_pred.append([count.keys[pos_max], p[pos_max]])

        return np.array(y_pred).reshape(-1, 2) '''

    def fit_SVM(self, X, y, n_average = 'auto', C = 1.0, n_iter_max = 100, test_size = 0.5):
        #### ----------
        # -> 
        # ->
        #### ----------

        predict_score = []
        training_score = []
        clf_list = []
        xtrain_scaler_list = []

        n_sample = X.shape[0]
        zero_eps = 1e-6

        for _ in range(n_average):
            
            ytrain, ytest, xtrain, xtest = train_test_split(y, X, test_size=test_size)
        
            std = np.std(xtrain, axis = 0)    
            std[std < zero_eps] = 1.0 # get rid of zero variance data.
            mu, inv_sigma = np.mean(xtrain, axis=0), 1./std

            xtrain = (xtrain - mu)*inv_sigma
            xtest = (xtest - mu)*inv_sigma

            clf = SVC(C=C, class_weight='balanced')

            clf.fit(xtrain, ytrain)

            t_score = clf.score(xtrain, ytrain) # predict on test set
            training_score.append(t_score)
        
            p_score = clf.score(xtest, ytest) # predict on test set
            predict_score.append(p_score)

            clf_list.append(clf)
            xtrain_scaler_list.append([mu,inv_sigma])

        self.scaler_list = xtrain_scaler_list # scaling transformations (zero mean, unit std)
        self.cv_score = np.mean(predict_score)  
        self.mean_train_score = np.mean(training_score)
        self.clf_list = clf_list # classifier list for majority voting !
        self._n_sample = len(y)

        return self