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
import scipy.linalg
import scipy.sparse as sparse

from qpsolvers import solve_qp
from cvxopt import matrix, spmatrix, solvers

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
        self.alpha=0
        self.phi_train=0
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
        self.phi_train = phi
        
        d = phi.shape[1]
        #print(n,d)
        #K = phi.dot(phi.T)
        
        if n<=d: # to do 
            #self.w = phi.T.dot(sparse.linalg.inv(phi.dot(phi.T)+self.C*sparse.eye(Y.shape[0]))).dot(Y)
            self.alpha = sparse.csr_matrix(np.linalg.inv(phi.dot(phi.T)+self.C*np.eye(Y.shape[0])).dot(Y).T)
            self.alpha = self.alpha.multiply(np.abs(self.alpha)>10e-8)
            self.w=None
            #self.w = np.array(phi.T.dot(np.linalg.inv(phi.dot(phi.T)+self.C*np.eye(Y.shape[0])).dot(Y).T)).squeeze()
        else:
            #self.w = sparse.linalg.inv(phi.T.dot(phi)+self.C*sparse.eye(phi.shape[0],dtype=np.int32)).dot(phi.T).dot(Y)
            self.w = np.array(np.linalg.inv(phi.T.dot(phi)+self.C*np.eye(phi.shape[0],dtype=np.int)).dot(phi.T).dot(Y)).squeeze()
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
        if self.w != None:
            phi = self.kernel.phi(x)
            out = np.array(phi.dot(self.w.T).T)
        else :
            phi = self.kernel.phi(x)
            out = phi.dot(self.phi_train.T.dot(self.alpha))
            out = out.toarray().squeeze()
        return out
    
#%% kernel logistic regression ## DOES NOT WORK

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
        
        d = phi.shape[1]
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
        self.alpha=0
        self.phi_train=0
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
                
        phi = self.kernel.phi(X)
        self.phi_train = phi
        
        K = phi.dot(phi.T).toarray()
        
        #alpha = solve_qp(P=2*K.astype(np.double), q=-2*Y.astype(np.double), lb=np.zeros(n,dtype=np.double), G=np.diag(Y).astype(np.double), h=self.C*np.ones(n,dtype=np.double))
        
        P = matrix(2*K.astype(np.float))
        q = matrix(-2*Y.astype(np.float))
        G = matrix(np.concatenate((np.diag(Y),-np.diag(Y))).astype(np.float))
        h = matrix(np.concatenate((self.C*np.ones(n),np.zeros(n))))       
        
        #print("solving QP problem")
        
        self.alpha = sparse.csr_matrix(solvers.qp(P=P, q=q, G=G, h=h, show_progress=True)['x'])
        self.alpha = self.alpha.multiply(np.abs(self.alpha)>(self.C*10e-6))
                        
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
        out = phi.dot(self.phi_train.T.dot(self.alpha))
        out = out.toarray().squeeze()
        return out
    
#%% cross validation

class cross_clf:
    def __init__(self,C=1,kernel=spectrum_kernel(k=5),clf=SVM,n_splits=4,random_state=42):
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
        self.kernel=kernel
        self.clfs = [clf(C=C,kernel=kernel) for i in range(n_splits)]
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
                
        i=0
        for train_index, test_index in skf.split(X, Y):  
            self.clfs[i].fit(X[train_index], Y[train_index])
            i+=1
            
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
        out = self.clfs[0].predict(x)
        for i in range(1,self.n_splits):
            out+=self.clfs[i].predict(x)
        return out/self.n_splits