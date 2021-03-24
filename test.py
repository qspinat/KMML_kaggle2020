#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 17:58:00 2021

@author: quentin
"""

import numpy as np
import pandas as pd
from kernels import spectrum_kernel, mismatch_kernel, concat_kernel
from kernel_methods import KRR, KLR, SVM, cross_clf
from data_augmentation import data_augmentation
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold


#%% load data

X = pd.concat((pd.read_csv("Data/Xtr0.csv"),pd.read_csv("Data/Xtr1.csv"),pd.read_csv("Data/Xtr2.csv")))['seq'].values
Y = pd.concat((pd.read_csv("Data/Ytr0.csv"),pd.read_csv("Data/Ytr1.csv"),pd.read_csv("Data/Ytr2.csv")))['Bound'].values
Y = 2*Y-1

X, Y = data_augmentation(X, Y)

X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size=1/6, random_state=42, stratify=Y)

#X_train=X_train[:1000]
#Y_train=Y_train[:1000]

#%% Kernel Ridge Regression

krr = KRR(C=150,kernel=spectrum_kernel(k=9))
krr.fit(X_train,Y_train)

#%% Kernel Ridge Regression

krr = KRR(C=50000,kernel=mismatch_kernel(k=9,m=1))
krr.fit(X_train,Y_train)

#%% Kernel Ridge Regression

kernels=[spectrum_kernel(6),spectrum_kernel(9)]
weights=[1,1]

krr = KRR(C=1000,kernel=concat_kernel(kernels,weights))
krr.fit(X_train,Y_train)


#%% predict training set KRR

y_pred = krr.predict(X_train)
y_pred = np.sign(y_pred)

print("On training set:")
print("f1 :",f1_score(Y_train,y_pred))
print("accuracy :", accuracy_score(Y_train,y_pred))
print()

y_pred = krr.predict(X_test)
y_pred = (y_pred>0)*2-1

print("On test set:")
print("f1 :",f1_score(Y_test,y_pred))
print("accuracy :", accuracy_score(Y_test,y_pred))

#%% SVM

svm = SVM(C=0.008,kernel=spectrum_kernel(k=10))
svm.fit(X_train,Y_train)


#%% SVM

svm = SVM(C=0.001,kernel=mismatch_kernel(k=9,m=1))
svm.fit(X_train,Y_train)

#%% SVM

kernels=[spectrum_kernel(5),spectrum_kernel(9)]
weights=[1,1]

svm = SVM(C=0.005,kernel=concat_kernel(kernels,weights))
svm.fit(X_train,Y_train)

#%% predict training set SVM

y_pred = svm.predict(X_train)
y_pred = np.sign(y_pred)

print("On training set:")
print("f1 :",f1_score(Y_train,y_pred))
print("accuracy :", accuracy_score(Y_train,y_pred))

y_pred = svm.predict(X_test)
y_pred = (y_pred>0)*2-1

print("On test set:")
print("f1 :",f1_score(Y_test,y_pred))
print("accuracy :", accuracy_score(Y_test,y_pred))


#%% cross_validated KRR

cross_krr = cross_clf(C=1000,kernel=spectrum_kernel(k=10),clf=KRR,n_splits=5)
cross_krr.fit(X_train,Y_train)


#%% predict training set cross KRR

y_pred = cross_krr.predict(X_train)
y_pred = np.sign(y_pred)

print("On training set:")
print("f1 :",f1_score(Y_train,y_pred))
print("accuracy :", accuracy_score(Y_train,y_pred))

y_pred = cross_krr.predict(X_test)
y_pred = np.sign(y_pred)

print("On test set:")
print("f1 :",f1_score(Y_test,y_pred))
print("accuracy :", accuracy_score(Y_test,y_pred))

#%% cross_validated SVM

cross_svm = cross_clf(C=0.008,kernel=spectrum_kernel(k=9),clf=SVM,n_splits=5)
cross_svm.fit(X_train,Y_train)


#%% predict training set SVM

y_pred = cross_svm.predict(X_train)
y_pred = np.sign(y_pred)

print("On training set:")
print("f1 :",f1_score(Y_train,y_pred))
print("accuracy :", accuracy_score(Y_train,y_pred))

y_pred = cross_svm.predict(X_test)
y_pred = np.sign(y_pred)

print("On test set:")
print("f1 :",f1_score(Y_test,y_pred))
print("accuracy :", accuracy_score(Y_test,y_pred))


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