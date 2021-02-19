#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 17:58:00 2021

@author: quentin
"""

import numpy as np
import pandas as pd
from kernels import spectrum_kernel
from kernel_methods import KRR, KLR
from sklearn.metrics import f1_score, accuracy_score

#%% load data

X_train = pd.concat((pd.read_csv("Data/Xtr0.csv"),pd.read_csv("Data/Xtr1.csv"),pd.read_csv("Data/Xtr2.csv")))['seq'].values
Y_train = pd.concat((pd.read_csv("Data/Ytr0.csv"),pd.read_csv("Data/Ytr1.csv"),pd.read_csv("Data/Ytr2.csv")))['Bound'].values

X_train=X_train[:2000]
Y_train=Y_train[:2000]

#%% Kernel Ridge Regression

krr = KRR(C=10,kernel=spectrum_kernel(k=5))
krr.fit(X_train,Y_train)


#%% predict training set KRR

y_pred = krr.predict(X_train)
y_pred = (y_pred > 0.5)*1

print("f1 :",f1_score(Y_train,y_pred))
print("accuracy :", accuracy_score(Y_train,y_pred))

#%% Kernel LR

klr = KLR(C=10,kernel=spectrum_kernel(k=5))
klr.fit(X_train,Y_train,it=2)


#%% predict training set KLR

y_pred = klr.predict(X_train)
y_pred = (y_pred > 0.5)*1

print("f1 :",f1_score(Y_train,y_pred))
print("accuracy :", accuracy_score(Y_train,y_pred))

#%% test data

X_test = pd.concat((pd.read_csv("Data/Xte0.csv"),pd.read_csv("Data/Xte1.csv"),pd.read_csv("Data/Xte2.csv")))['seq'].values
#Y_test = pd.concat((pd.read_csv("Data/Yte0.csv"),pd.read_csv("Data/Yte1.csv"),pd.read_csv("Data/Yte2.csv")))['Bound'].values

#%% predict test set

y_pred = krr.predict(X_test)
y_pred = (y_pred > 0.5)*1

#%% save

df = pd.DataFrame(np.vstack((np.arange(y_pred.shape[0]),y_pred)).T,columns=['Id','Bound'])

df.to_csv('Eval/eval.csv',index=False)