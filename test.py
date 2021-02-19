#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 17:58:00 2021

@author: quentin
"""

import numpy as np
import pandas as pd
from kernels import spectrum_kernel
from kernel_methods import KRR, KLR, SVM, cross_val_SVM
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold


#%% load data

X = pd.concat((pd.read_csv("Data/Xtr0.csv"),pd.read_csv("Data/Xtr1.csv"),pd.read_csv("Data/Xtr2.csv")))['seq'].values
Y = pd.concat((pd.read_csv("Data/Ytr0.csv"),pd.read_csv("Data/Ytr1.csv"),pd.read_csv("Data/Ytr2.csv")))['Bound'].values

X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size=1/6, random_state=42, stratify=Y)

#X_train=X_train[:1000]
#Y_train=Y_train[:1000]

#%% Kernel Ridge Regression

krr = KRR(C=10,kernel=spectrum_kernel(k=5))
krr.fit(X_train,Y_train)


#%% predict training set KRR

y_pred = krr.predict(X_train)
y_pred = (y_pred > 0.5)*1

print("On training set:")
print("f1 :",f1_score(Y_train,y_pred))
print("accuracy :", accuracy_score(Y_train,y_pred))
print()

y_pred = krr.predict(X_test)
y_pred = (y_pred > 0.5)*1

print("On test set:")
print("f1 :",f1_score(Y_test,y_pred))
print("accuracy :", accuracy_score(Y_test,y_pred))

#%% SVM

svm = SVM(C=0.005,kernel=spectrum_kernel(k=6))
svm.fit(X_train,Y_train)


#%% predict training set SVM

y_pred = svm.predict(X_train)
y_pred = (y_pred > 0.5)*1

print("On training set:")
print("f1 :",f1_score(Y_train,y_pred))
print("accuracy :", accuracy_score(Y_train,y_pred))

y_pred = svm.predict(X_test)
y_pred = (y_pred > 0.5)*1

print("On test set:")
print("f1 :",f1_score(Y_test,y_pred))
print("accuracy :", accuracy_score(Y_test,y_pred))


#%% cross_validated SVM

cross_svm = cross_val_SVM(C=0.01,kernel=spectrum_kernel(k=6),n_splits=4)
cross_svm.fit(X_train,Y_train)


#%% predict training set SVM

y_pred = cross_svm.predict(X_train)
y_pred = (y_pred > 0.5)*1

print("On training set:")
print("f1 :",f1_score(Y_train,y_pred))
print("accuracy :", accuracy_score(Y_train,y_pred))

y_pred = cross_svm.predict(X_test)
y_pred = (y_pred > 0.5)*1

print("On test set:")
print("f1 :",f1_score(Y_test,y_pred))
print("accuracy :", accuracy_score(Y_test,y_pred))

#%% test data

X_val = pd.concat((pd.read_csv("Data/Xte0.csv"),pd.read_csv("Data/Xte1.csv"),pd.read_csv("Data/Xte2.csv")))['seq'].values
#Y_test = pd.concat((pd.read_csv("Data/Yte0.csv"),pd.read_csv("Data/Yte1.csv"),pd.read_csv("Data/Yte2.csv")))['Bound'].values

#%% predict test set

y_pred = svm.predict(X_val)
y_pred = (y_pred > 0.5)*1


#%% save

df = pd.DataFrame(np.vstack((np.arange(y_pred.shape[0]),y_pred)).T,columns=['Id','Bound'])

df.to_csv('Eval/eval.csv',index=False)