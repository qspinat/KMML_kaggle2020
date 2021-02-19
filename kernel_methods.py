#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 16:41:42 2021

@author: quentin
"""

import numpy as np
import pandas as pd
import scipy.optimize as scopt
from numba import jit
from kernels import spectrum_kernel
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold

from qpsolvers import solve_qp
from cvxopt import matrix, solvers

#%% kernel ridge regression

class KRR:
    def __init__(self,C=1,kernel=spectrum_kernel(k=5)):
        """
        Parameters
        ----------
        C : float, optional
            regularization constant. The default is 1.
        Returns
        -------
        None.

        """
        self.C=C
        self.w=0
        self.kernel=kernel
        
    #@jit(nopython=True)
    def fit(self,X,Y):    
        """
        fits the KRR
        
        Parameters
        ----------
        X : 1-D array of strings
            ATCG inputs
        Y : 1-D array
            labels.

        Returns
        -------
        None.

        """
        n = X.shape[0]
        
        phi = self.kernel.phi(X)
        
        d = len(phi)
        #K = phi.dot(phi.T)
        
        if n<=d: # to do 
            self.w = phi.T.dot(np.linalg.inv(phi.dot(phi.T)+self.C*np.eye(Y.shape[0]))).dot(Y)
        else:
            self.w = np.linalg.inv(phi.T.dot(phi)+self.C*np.eye(phi.shape[0])).dot(phi.T).dot(Y)
        return
    
    def predict(self,x):
        """
        predict value from input and training set

        Parameters
        ----------
        x : string
            input.

        Returns
        -------
        out : float
            prediction.

        """
        phi = self.kernel.phi(x)
        out = self.w.dot(phi.T)
        return out
    
#%% kernel logistic regression

class KLR:
    def __init__(self,C=1,kernel=spectrum_kernel(k=5)):
        """
        Parameters
        ----------
        C : float, optional
            regularization constant. The default is 1.
        Returns
        -------
        None.

        """
        self.C=C
        self.w=0
        self.kernel=kernel
        
    #@jit(nopython=True)
    def fit(self,X,Y,it=10):    
        """
        fits the KRR
        
        Parameters
        ----------
        X : 1-D array of strings
            ATCG inputs
        Y : 1-D array
            labels.
        it : int
            number of iterations

        Returns
        -------
        None.

        """
        n = X.shape[0]
        alpha = np.zeros(n) #np.random.randn(n)
                
        phi = self.kernel.phi(X)
        
        d = len(phi)
        K = phi.dot(phi.T)
        
        for i in tqdm(range(it)):
            #print(i)
            m = K.dot(alpha)
            print("m :",m.min(),m.max())
            #print("m:",m.shape)
            P = -1/(1+np.exp(Y*m))
            print("P :",P.min(),P.max())
            #print("P :",P.shape)
            W = 1/(1+np.exp(Y*m))/(1+np.exp(-Y*m))
            print("W :",W.min(),W.max())
            #print("W :",W.shape)
            z = m + Y*(1+np.exp(-Y*m))
            print("z :",z.min(),z.max())
            #print("z :",z.shape)
            
            alpha = np.diag(np.sqrt(W)).dot( np.linalg.inv( np.diag(np.sqrt(W)).dot(K.dot(np.diag(np.sqrt(W))))+self.C*np.eye(z.shape[0]) ).dot( np.diag(np.sqrt(W)).dot(z) ) )
            #print("alpha :",alpha.shape)
            
        self.w = alpha.dot(phi)
        #print("w :",self.w.shape)
        return
    
    def predict(self,x):
        """
        predict value from input and training set

        Parameters
        ----------
        x : string
            input.

        Returns
        -------
        out : float
            prediction.

        """
        phi = self.kernel.phi(x)
        out = self.w.dot(phi.T)
        return out
    
    
#%% SVM

class SVM:
    def __init__(self,C=1,kernel=spectrum_kernel(k=5)):
        """
        Parameters
        ----------
        C : float, optional
            regularization constant. The default is 1.
        Returns
        -------
        None.

        """
        self.C=C
        self.w=0
        self.kernel=kernel
        
    def fit(self,X,Y):    
        """
        fits the KRR
        
        Parameters
        ----------
        X : 1-D array of strings
            ATCG inputs
        Y : 1-D array
            labels.
        it : int
            number of iterations

        Returns
        -------
        None.

        """
        n = X.shape[0]
        alpha = np.zeros(n) #np.random.randn(n)
                
        phi = self.kernel.phi(X)
        
        K = phi.dot(phi.T)
        
        #alpha = solve_qp(P=2*K.astype(np.double), q=-2*Y.astype(np.double), lb=np.zeros(n,dtype=np.double), G=np.diag(Y).astype(np.double), h=self.C*np.ones(n,dtype=np.double))
        
        P = matrix(2*K.astype(np.float))
        q = matrix(-2*Y.astype(np.float))
        G = matrix(np.concatenate((np.diag(Y),-np.diag(Y))).astype(np.float))
        h = matrix(np.concatenate((self.C*np.ones(n),np.zeros(n))))       
        
        alpha = np.squeeze(solvers.qp(P=P, q=q, G=G, h=h, show_progress=False)['x'])
        
        self.w = alpha.dot(phi)
        
        return
    
    def predict(self,x):
        """
        predict value from input and training set

        Parameters
        ----------
        x : string
            input.

        Returns
        -------
        out : float
            prediction.

        """
        phi = self.kernel.phi(x)
        out = self.w.dot(phi.T)
        return out
    
#%%
    
class cross_val_SVM:
    def __init__(self,C=1,kernel=spectrum_kernel(k=5),n_splits=3,random_state=42):
        """
        Parameters
        ----------
        C : float, optional
            regularization constant. The default is 1.
        n_splits : int, optional
            number of cross validation folds. The default is 3.
        Returns
        -------
        None.

        """
        self.C=C
        self.w=0
        self.kernel=kernel
        self.n_splits=n_splits
        self.random_state=random_state
        
    def fit(self,X,Y):    
        """
        fits the KRR
        
        Parameters
        ----------
        X : 1-D array of strings
            ATCG inputs
        Y : 1-D array
            labels.
        it : int
            number of iterations

        Returns
        -------
        None.

        """
        skf = StratifiedKFold(n_splits=self.n_splits, random_state=0, shuffle=True)
        
        clf = SVM(kernel=self.kernel,C=self.C)
        
        for train_index, test_index in skf.split(X, Y):  
            clf.fit(X[train_index], Y[train_index])
            self.w+=clf.w
        self.w/=self.n_splits
        
        return
    
    def predict(self,x):
        """
        predict value from input and training set

        Parameters
        ----------
        x : string
            input.

        Returns
        -------
        out : float
            prediction.

        """
        phi = self.kernel.phi(x)
        out = self.w.dot(phi.T)
        return out