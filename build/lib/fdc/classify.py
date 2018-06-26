from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from collections import Counter

def most_common(lst):
    return max(set(lst), key=lst.count)

class CLF:
    """ Implements a classifier for hierarchical clustering

    Parameters:
    ----------
    clf_type : str
        Type of cluster, either 'svm' or 'logreg'
    kwarg : optional arguments for SVM (see SVM definition below for name of keyword arguments)
    
    """

    def __init__(self, clf_type='svm', n_average=10, test_size = 0.8, clf_args=None):
        self.clf_type = clf_type
        self.n_average = n_average
        self.test_size = test_size

        self.clf_args = clf_args

        self.trained = False
        self.cv_score = 1.0

    def fit(self, X, y):
        """ Fit clf to data.

        Parameters
        ------------
        X: array, shape = (n_sample, n_feature)
            your data

        y: array, shape = (n_sample, 1)
            your labels
        
        Other parameters
        ------------
        self.n_average : int
            number of classifiers to train (will then take majority vote)
        
        self.test_size: float
            ratio of test size (between 0 and 1). 

        Return
        -------
        self, CLF object

        """
        #### ----------
        #         #### ----------

        self.trained = True
        
        if self.clf_type == 'svm':
            if self.clf_args is not None:
                clf = SVC(**self.clf_args)
            else:
                clf = SVC()

        elif self.clf_type == 'rf':
            if self.clf_args is not None:
                clf = RandomForestClassifier(**self.clf_args)
                #clf = RandomForestClassifier(**self.clf_args, warm_start=True)
            else:
                clf = RandomForestClassifier()

        n_average = self.n_average
    
        predict_score = []
        training_score = []
        clf_list = []
        xtrain_scaler_list = []

        n_sample = X.shape[0]
        zero_eps = 1e-6

        y_unique = np.unique(y) # different labels
        assert len(y_unique)>1, "Cluster provided only has a unique label, can't classify !"

        n_sample = X.shape[0]
        idx = np.arange(n_sample)
        yu_pos = {yu : idx[(y == yu)] for yu in y_unique}
        n_class = len(y_unique)
        import time 

        dt=0.0

        import pickle

        for _ in range(n_average):
            while True:
                ytrain, ytest, xtrain, xtest = train_test_split(y, X, test_size=self.test_size)
                if len(np.unique(ytrain)) > 1: # could create a bug otherwise
                    break

            #print("train size, test size:", len(ytrain),len(ytest),sep='\t')
            
            std = np.std(xtrain, axis = 0)    
            std[std < zero_eps] = 1.0 # get rid of zero variance data.
            mu, inv_sigma = np.mean(xtrain, axis=0), 1./std

            xtrain = (xtrain - mu)*inv_sigma # zscoring the data 
            xtest = (xtest - mu)*inv_sigma
            pickle.dump([xtrain, ytrain], open('test.pkl','wb'))
            s=time.time()
            print(len(xtrain))
            clf.fit(xtrain, ytrain)
            dt += (time.time() - s)

            t_score = clf.score(xtrain, ytrain) # predict on test set
            training_score.append(t_score)
        
            p_score = clf.score(xtest, ytest) # predict on test set
            #print(t_score,'\t',p_score)
            predict_score.append(p_score)

            clf_list.append(clf)
            xtrain_scaler_list.append([mu,inv_sigma])
        print("TRAINING ONLY\t",dt)

        self.scaler_list = xtrain_scaler_list # scaling transformations (zero mean, unit std)
        self.cv_score = np.mean(predict_score)
        self.cv_score_std = np.std(predict_score)  
        self.mean_train_score = np.mean(training_score)
        self.std_train_score = np.std(training_score)
        self.clf_list = clf_list # classifier list for majority voting !
        self._n_sample = len(y)

        return self


    def predict(self, X, option='fast'):
        """Returns labels for X (-1, 1)"""
        if option is 'fast':
            mu, inv_sigma = self.scaler_list[0]
            return self.clf_list[0].predict(inv_sigma*(X-mu))

        if self.clf_type == 'trivial':
            self._n_sample = len(X)
            return np.zeros(len(X))

        assert self.trained is True, "Must train model first !" 

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
        for x_vote in vote: # majority voting here !
            y_pred.append(most_common(list(x_vote)))

        return np.array(y_pred)#.reshape(-1,1)

    def score(self, X, y):
        y_pred = self.predict(X).flatten()
        return np.count_nonzero(y_pred == y)/len(y)