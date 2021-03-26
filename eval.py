#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from kernels import spectrum_kernel, mismatch_kernel, concat_kernel
from kernel_methods import KRR, SVM, cross_clf
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold


#%% load data

X = pd.concat((pd.read_csv("Data/Xtr0.csv"),pd.read_csv("Data/Xtr1.csv"),pd.read_csv("Data/Xtr2.csv")))['seq'].values
Y = pd.concat((pd.read_csv("Data/Ytr0.csv"),pd.read_csv("Data/Ytr1.csv"),pd.read_csv("Data/Ytr2.csv")))['Bound'].values
Y = 2*Y-1

X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size=1/6, random_state=42, stratify=Y)

#X_train=X_train[:1000]
#Y_train=Y_train[:1000]

#%% train on the whole dataset

clf = KRR(C=50000,kernel=mismatch_kernel(k=9,m=1))
clf.fit(X,Y)

#%% test data

X_val = pd.concat((pd.read_csv("Data/Xte0.csv"),pd.read_csv("Data/Xte1.csv"),pd.read_csv("Data/Xte2.csv")))['seq'].values
#Y_test = pd.concat((pd.read_csv("Data/Yte0.csv"),pd.read_csv("Data/Yte1.csv"),pd.read_csv("Data/Yte2.csv")))['Bound'].values

#%% predict test set

y_pred = clf.predict(X_val)
y_pred = (y_pred>0)*2-1
y_pred=((y_pred+1)//2).astype(np.int)


#%% save

df = pd.DataFrame(np.vstack((np.arange(y_pred.shape[0]),y_pred)).T,columns=['Id','Bound'])

df.to_csv('Eval/eval.csv',index=False)