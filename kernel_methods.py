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
from preprocess import spectrum_kernel

#%% kernel ridge regression

class KRR:
    def __init__(self,C=1,kernel=spectrum_kernel(k=10)):
        """
        Parameters
        ----------
        C : float, optional
            regularization constant. The default is 1.
        Returns
        -------
        None.

        """
        self.C=1
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
    def __init__(self,C=1,kernel=spectrum_kernel(k=10)):
        """
        Parameters
        ----------
        C : float, optional
            regularization constant. The default is 1.
        Returns
        -------
        None.

        """
        self.C=1
        self.kernel=kernel
        self.alpha=0
        
    #@jit(nopython=True)
    def fit(self,X,Y,it=100):    
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
        self.alpha = np.zeros(n)
        
        K = np.zeros((X.shape[0],X.shape[0]),dtype=np.int)
        for i in range(n):
            K[i,i] = self.kernel(X[i],X[i])
            for j in range(i+1,n):
                K[i,j] = self.kernel(X[i],X[j])
                K[j,i] = self.kernel(X[i],X[j])
        
        for i in range(it):
            m = K.dot(self.alpha)
            P = -1/(1+np.exp(Y.dot(m)))
            W = 1/(1+np.exp(m))/(1+np.exp(-m))
            z = m - P.dot(Y)/W
            
            self.alpha = np.diag(np.sqrt(W))*np.linalg.inv(np.diag(np.diag(np.sqrt(W))).dot(K.dot(np.diag(np.sqrt(W))))+self.C*np.eye(z.shape[0])).dot(np.diag(np.sqrt(W)).dot(z))
        return
    
    def predict(self,x,X):
        """
        predict value from input and training set

        Parameters
        ----------
        x : string
            input.
        X : array of strings
            training set.

        Returns
        -------
        out : float
            prediction.

        """
        n = self.alpha.shape[0]
        out = 0
        for i in range(n):
            out += self.alpha[i]*self.kernel(X[i], x)
        return out
    