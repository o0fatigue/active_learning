import os
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import MeCab

from ActiveLearning import *


def experiment(model_name, X, y, num_iter=-1, k_fold=5, **kwargs):
    metrics = []
    kf = KFold(n_splits=k_fold)

    print(model_name, end=': ')
    for j, (train_index, test_index) in enumerate(kf.split(X)):
        time_flag = time.time()
        X_train = X[train_index]
        X_test = X[test_index]
        y_train = y[train_index]
        y_test = y[test_index]
        
        if num_iter == -1:
            # "- 10" is because last several samples sometimes give unexpected error
            num_iter = train_index.shape[0] - 10
        
        # "+ 2" is because the model should be initated with one sample from each class
        X_train = pd.DataFrame(X_train)[:num_iter + 2]
        X_test = pd.DataFrame(X_test)
        y_train = pd.Series(y_train)[:num_iter + 2]
        y_test = pd.Series(y_test)

        if 'Doc2Vec' in model_name:
            docmat_train = kwargs['docmat'][train_index][:num_iter + 2]
            docmat_test = kwargs['docmat'][test_index]
            model = eval(model_name)(X_train, y_train, X_test, y_test, norm=2, docmat=docmat_train)
        else:
            model = eval(model_name)(X_train, y_train, X_test, y_test)

        model.init_label()

        for i in range(num_iter):
            model.train()
            idx = model.get_next()
            model.push(idx)

        metrics.append(model.metrics)
        print(f'{j + 1}:{time.time() - time_flag:.2f}[sec]', end=' ')

    print()
    return metrics

# if raw text is japanese, use this tokenizer
def tokenize(text):
    mecab = MeCab.Tagger('mecabrc')
    result = mecab.parse(text)
    token = []
    for row in result.split('\n'):
        if not row or row == 'EOS' :
            continue
        temp = row.split('\t')
        word = temp[0]
        vect = temp[1].split(',')
        if vect[6] != '*':
            word = vect[6]
        if vect[0] not in ['記号', '助詞', '助動詞'] :
            token.append(word)

    return token
