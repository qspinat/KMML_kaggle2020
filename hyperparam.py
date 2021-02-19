#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 11:07:03 2021

@author: quentin
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score, GridSearchCV, train_test_split
from sklearn.metrics import f1_score, accuracy_score

from kernels import spectrum_kernel
from kernel_methods import KRR, KLR, SVM, cross_val_SVM

#%% load data

X = pd.concat((pd.read_csv("Data/Xtr0.csv"),pd.read_csv("Data/Xtr1.csv"),pd.read_csv("Data/Xtr2.csv")))['seq'].values
y = pd.concat((pd.read_csv("Data/Ytr0.csv"),pd.read_csv("Data/Ytr1.csv"),pd.read_csv("Data/Ytr2.csv")))['Bound'].values

#X_train=X_train[:1000]
#Y_train=Y_train[:1000]

#%%

liste_C = [0.01,0.1,1,10,100]
scores = [f1_score, accuracy_score]

n_splits=3
skf = StratifiedKFold(n_splits=n_splits, random_state=0, shuffle=True)

best_params={f1_score.__name__:-1, accuracy_score.__name__:-1}

for score in scores:
    print("# Tuning hyper-parameters for %s" % score.__name__)
    print()
    best_score=-1
    for C in liste_C:
        val_score = 0
        for train_index, test_index in skf.split(X, y):  
            clf = cross_val_SVM(C=C,kernel=spectrum_kernel(k=6),n_splits=4)
            clf.fit(X[train_index], y[train_index])
            
            y_pred = clf.predict(X[test_index])
            y_pred = (y_pred > 0.5)*1

            val_score += score(y[test_index],y_pred)
        val_score/=n_splits
        if best_score<val_score:
            best_score=val_score
            best_params[score.__name__]=C
            
        print("Parameter C={} : {}={}".format(C,score.__name__,val_score))
    print()
