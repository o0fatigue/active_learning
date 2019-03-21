import os
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize


NUM_CAND = 10


############################## Base Model ##############################
class ActiveLearning():
    """
    This is a base class for active learning.
    You can just inherit this class to create an new active learning algorithm, by:
    1. overide the method "init_model()" to initiate the model with a machine learning model.
    2. overide the method "get_next()" to define a selecting policy (or so called querying function) to pick up the next sample.
    
    Attribute "self.metrics" should include all needed metrics, and the array of each metric will be used to plot the curve.
    """
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.unlabeled_msk = pd.Series([True] * X_train.shape[0])
        self.model = None
        self.ordered_idx = []

        self.metrics = {}
        
        self.metrics['pos_coverage'] = []
        self.metrics['labeled_prop'] = []
        self.metrics['unlabeled_prop'] = []
        self.metrics['unlabeled_pred_prop'] = []
        
        self.metrics['acc'] = []
        self.metrics['f1'] = []
        self.metrics['recall'] = []
        self.metrics['precision'] = []
        
        self.metrics['current_acc'] = []
        self.metrics['current_f1'] = []
        self.metrics['current_recall'] = []
        self.metrics['current_precision'] = []
        
        self.metrics['unlabeled_acc'] = []
        self.metrics['unlabeled_f1'] = []
        self.metrics['unlabeled_recall'] = []
        self.metrics['unlabeled_precision'] = []
        
        self.metrics['r_precision'] = []
        self.metrics['probas'] = []
        self.metrics['true_probas'] = []
        self.metrics['false_probas'] = []
        self.metrics['recent_prop'] = []

    def init_model(self):
        pass

    def train(self):
        self.init_model()
        try:
            X_train, X_test, y_train, y_test = train_test_split(self.X_train[self.unlabeled_msk==False].values, self.y_train[self.unlabeled_msk==False].values, test_size=0.2)
            self.model.fit(X_train, y_train)
        except:
            X_train = self.X_train[self.unlabeled_msk==False].values
            y_train = self.y_train[self.unlabeled_msk==False].values
            X_test = self.X_test
            y_test = self.y_test
            self.model.fit(X_train, y_train)

        # O: oracle
        # C: current
        # L: labeled
        # U: unlabeled
        # R: recent

        total_pos = self.y_train[self.y_train==1].shape[0]
        y_labeled = self.y_train[self.unlabeled_msk==False].values
        X_unlabeled = self.X_train[self.unlabeled_msk].values
        y_unlabeled = self.y_train[self.unlabeled_msk].values
        
        U_y_pred = self.model.predict(X_unlabeled)
        y_pred = self.model.predict(self.X_test)
        C_y_pred = self.model.predict(X_test)
        
        acc = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        
        C_f1 = f1_score(y_test, C_y_pred)
        C_acc = accuracy_score(y_test, C_y_pred)
        C_recall = recall_score(y_test, C_y_pred)
        C_precision = precision_score(y_test, C_y_pred)
        
        U_f1 = f1_score(y_unlabeled, U_y_pred)
        U_acc = accuracy_score(y_unlabeled, U_y_pred)
        U_recall = recall_score(y_unlabeled, U_y_pred)
        U_precision = precision_score(y_unlabeled, U_y_pred)
    
        pos_cov = y_labeled[y_labeled==1].shape[0] / total_pos
        L_prop = y_labeled[y_labeled==1].shape[0] / y_labeled.shape[0]
        try:
            y_R = self.y_train[self.unlabeled_msk==False].loc[self.ordered_idx[:-11:-1]]
            R_prop = (y_R[y_R==1].shape[0] / y_R.shape[0]) if y_R.shape[0] != 0 else 0
        except:
            R_prop = 0

        U_prop = y_unlabeled[y_unlabeled==1].shape[0] / y_unlabeled.shape[0]
        U_pred_prop = U_y_pred[U_y_pred==1].shape[0] / U_y_pred.shape[0]
        
        R = y_unlabeled[y_unlabeled==1].shape[0]
        dist = self.model.decision_function(X_unlabeled)
        proba = (dist - dist.min()) / (dist.max() - dist.min())
        try:
            T_dist = self.model.decision_function(X_unlabeled[y_unlabeled==1])
            T_proba = (T_dist - dist.min()) / (dist.max() - dist.min())
        except:
            T_proba = np.array([0])
        try:
            F_dist = self.model.decision_function(X_unlabeled[y_unlabeled==0])
            F_proba = (F_dist - dist.min()) / (dist.max() - dist.min())
        except:
            F_proba = np.array([0])
        msk = np.argsort(proba)[::-1][:R]
        r = np.sum(y_unlabeled[msk])
        r_precision = (r + y_labeled[y_labeled==1].shape[0]) / (R + y_labeled[y_labeled==1].shape[0])
        
        
        self.metrics['acc'].append(acc)
        self.metrics['f1'].append(f1)
        self.metrics['recall'].append(recall)
        self.metrics['precision'].append(precision)
        
        self.metrics['current_f1'].append(C_f1)
        self.metrics['current_acc'].append(C_acc)
        self.metrics['current_recall'].append(C_recall)
        self.metrics['current_precision'].append(C_precision)
        
        self.metrics['unlabeled_f1'].append(U_f1)
        self.metrics['unlabeled_acc'].append(U_acc)
        self.metrics['unlabeled_recall'].append(U_recall)
        self.metrics['unlabeled_precision'].append(U_precision)
        
        self.metrics['pos_coverage'].append(pos_cov)
        self.metrics['labeled_prop'].append(L_prop)
        self.metrics['unlabeled_prop'].append(U_prop)
        self.metrics['unlabeled_pred_prop'].append(U_pred_prop)
        
        self.metrics['r_precision'].append(r_precision)
        self.metrics['true_probas'].append(T_proba)
        self.metrics['false_probas'].append(F_proba)
        self.metrics['probas'].append(proba)
        self.metrics['recent_prop'].append(R_prop)

    def get_next(self):
        pass

    def push(self, idx):
        self.unlabeled_msk.loc[idx] = False
        self.ordered_idx.append(idx)

    def init_label(self, size=2):
        indices = np.random.choice(self.y_train[self.y_train==1].index, size//2, replace=False)
        self.unlabeled_msk.loc[indices] = False
        indices = np.random.choice(self.y_train[self.y_train==0].index, size//2, replace=False)
        self.unlabeled_msk.loc[indices] = False


class RandomSVM(ActiveLearning):
    def __init__(self, X_train, y_train, X_test, y_test):
        super().__init__(X_train, y_train, X_test, y_test)
        self.model = LinearSVC()

    def init_model(self):
        y_labeled = self.y_train[self.unlabeled_msk==False]
        self.model.__init__(class_weight='balanced')

    def get_next(self):
        idx = np.random.choice(self.X_train.index[self.unlabeled_msk], 1)[0]
        return idx


class UncertaintySVM(RandomSVM):
    def __init__(self, X_train, y_train, X_test, y_test):
        super().__init__(X_train, y_train, X_test, y_test)

    def get_next(self):
        dist = pd.Series(np.abs(self.model.decision_function(self.X_train)))
        idx = dist[self.unlabeled_msk].idxmin()
        return idx


class EnumerateSVM(RandomSVM):
    def __init__(self, X_train, y_train, X_test, y_test):
        super().__init__(X_train, y_train, X_test, y_test)

    def get_next(self):
        dist = pd.Series(self.model.decision_function(self.X_train))
        idx = dist[self.unlabeled_msk].idxmax()
        return idx
    

class RandomSwitchEnumerateSVM(RandomSVM):
    def __init__(self, X_train, y_train, X_test, y_test):
        super().__init__(X_train, y_train, X_test, y_test)

    def get_next(self):
        if len(self.ordered_idx) > self.X_train.shape[0] * 0.1:
            dist = pd.Series(self.model.decision_function(self.X_train))
            idx = dist[self.unlabeled_msk].idxmax()
        else:
            idx = np.random.choice(self.X_train.index[self.unlabeled_msk], 1)[0]
        
        return idx

# the classes below is not used in DEIM paper, only used in my master's thesis

############################## Basic ##############################
class CountSVM(RandomSVM):
    def __init__(self, X_train, y_train, X_test, y_test):
        super().__init__(X_train, y_train, X_test, y_test)
        self.countsum = np.sum((X_train > 0), axis=1)

    def get_next(self):
        idx = self.countsum[self.unlabeled_msk].idxmax()
        return idx


class TfidfSVM(RandomSVM):
    def __init__(self, X_train, y_train, X_test, y_test):
        super().__init__(X_train, y_train, X_test, y_test)
        self.top20sum = self.X_train.sort_index(axis=1).loc[:, -20:].sum(axis=1)

    def get_next(self):
        idx = self.top20sum[self.unlabeled_msk].idxmax()
        return idx


class TfidfNotShownSVM(RandomSVM):
    def __init__(self, X_train, y_train, X_test, y_test):
        super().__init__(X_train, y_train, X_test, y_test)
        self.feature = X_train.copy()

    def get_next(self):
        topsum = self.feature.sort_index(axis=1).loc[:, -20:].sum(axis=1)
        idx = topsum[self.unlabeled_msk].idxmax()
        msk = self.feature.loc[idx, :] > 0
        cols = np.arange(self.feature.shape[1])[msk]
        self.feature.loc[:, cols] /= 2
        return idx

class Doc2VecSVM(RandomSVM):
    def __init__(self, X_train, y_train, X_test, y_test, norm, docmat):
        super().__init__(X_train, y_train, X_test, y_test)
        # print(kwargs['docmat'].shape, self.unlabeled_msk.shape)
        self.doc2vec_norms = pd.DataFrame(np.linalg.norm(docmat, ord=norm, axis=1))

    def get_next(self):
        idx = self.doc2vec_norms[self.unlabeled_msk].idxmax()
        return idx


############################## Candidates Uncertatinty Sampling ##############################
class CandidateSVMwithTfidf(UncertaintySVM):
    def __init__(self, X_train, y_train, X_test, y_test):
        super().__init__(X_train, y_train, X_test, y_test)
        self.top20sum = self.X_train.sort_index(axis=1).loc[:, -20:].sum(axis=1)

    def get_next(self):
        dist = pd.Series(np.abs(self.model.decision_function(self.X_train)))
        candidates_idx = dist[self.unlabeled_msk].nsmallest(NUM_CAND).index
        idx = self.top20sum.loc[candidates_idx].idxmax()
        return idx


class CandidateSVMwithCount(UncertaintySVM):
    def __init__(self, X_train, y_train, X_test, y_test):
        super().__init__(X_train, y_train, X_test, y_test)
        self.countsum = np.sum((X_train > 0), axis=1)

    def get_next(self):
        dist = pd.Series(np.abs(self.model.decision_function(self.X_train)))
        candidates_idx = dist[self.unlabeled_msk].nsmallest(NUM_CAND).index
        idx = self.countsum.loc[candidates_idx].idxmax()
        return idx


class CandidateSVMwithDoc2Vec(UncertaintySVM):
    def __init__(self, X_train, y_train, X_test, y_test, norm, docmat):
        super().__init__(X_train, y_train, X_test, y_test)
        self.doc2vec_norms = pd.DataFrame(np.linalg.norm(docmat, ord=norm, axis=1))

    def get_next(self):
        dist = pd.Series(np.abs(self.model.decision_function(self.X_train)))
        candidates_idx = dist[self.unlabeled_msk].nsmallest(NUM_CAND).index
        idx = self.doc2vec_norms[self.unlabeled_msk].loc[candidates_idx].idxmax()
        return idx


############################## Candidates Enumerate ##############################
class EnumerateSVMwithTfidf(EnumerateSVM):
    def __init__(self, X_train, y_train, X_test, y_test):
        super().__init__(X_train, y_train, X_test, y_test)
        self.top20sum = self.X_train.sort_index(axis=1).loc[:, -20:].sum(axis=1)

    def get_next(self):
        dist = pd.Series(self.model.decision_function(self.X_train))
        candidates_idx = dist[self.unlabeled_msk].nlargest(NUM_CAND).index
        idx = self.top20sum.loc[candidates_idx].idxmax()
        return idx


class EnumerateSVMwithCount(EnumerateSVM):
    def __init__(self, X_train, y_train, X_test, y_test):
        super().__init__(X_train, y_train, X_test, y_test)
        self.countsum = np.sum((X_train > 0), axis=1)

    def get_next(self):
        dist = pd.Series(self.model.decision_function(self.X_train))
        candidates_idx = dist[self.unlabeled_msk].nlargest(NUM_CAND).index
        idx = self.countsum.loc[candidates_idx].idxmax()
        return idx


class EnumerateSVMwithDoc2Vec(EnumerateSVM):
    def __init__(self, X_train, y_train, X_test, y_test, norm, docmat):
        super().__init__(X_train, y_train, X_test, y_test)
        self.doc2vec_norms = pd.DataFrame(np.linalg.norm(docmat, ord=norm, axis=1))

    def get_next(self):
        dist = pd.Series(self.model.decision_function(self.X_train))
        candidates_idx = dist[self.unlabeled_msk].nlargest(NUM_CAND).index
        idx = self.doc2vec_norms[self.unlabeled_msk].loc[candidates_idx].idxmax()
        return idx
