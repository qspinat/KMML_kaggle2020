#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 16:03:06 2021

@author: quentin
"""

#%% importations

import numpy as np
import pandas as pd
from numba import jit 
from itertools import product, combinations
from scipy.special import comb
import scipy.sparse as sparse

#%% spectrum kernel

# @jit(nopython=True)
# def phi_spectrum(X,k):
#     """
#     Parameters
#     ----------
#     X : string
#     ATCG string
#     k : int, optional
#     length of consecutive features considered. The default is 5.

#     Returns
#     -------
#     out : array of int
#     spectrum kernel value  
#     """
#     values = {'A':0,'T':1,'C':2,'G':3}
    
#     # if type(X)==str:
    
#     #     out = np.zeros(4**k).astype(np.int32)
        
#     #     for i in range(len(X)-k+1):
#     #         u = X[i:i+k]
#     #         ind = values[u[0]]
#     #         for j in range(1,k):
#     #             ind += values[u[j]]*4**j
#     #         out[ind]+=1
                
    
#     out = np.zeros((len(X),4**k)).astype(np.int32)
#     for l in range(len(X)):
#         for i in range(len(X[0])-k+1):
#             u = X[l][i:i+k]
#             ind = values[u[0]]
#             for j in range(1,k):
#                 ind += values[u[j]]*4**j
#             out[l,ind]+=1         
#     return out



class spectrum_kernel:
    def __init__(self,k=5):
        self.k=k
        
    #@jit(nopython=True)
    def phi(self,X):
        """
        Parameters
        ----------
        X : string
            ATCG string
        k : int, optional
            length of consecutive features considered. The default is 5.
    
        Returns
        -------
        out : array of int
              spectrum kernel value  
        """
        values = {'A':0,'T':1,'C':2,'G':3}
        
        if type(X)==str:
        
            #out = np.zeros(4**self.k).astype(np.int32)
            #out = sparse.csr_matrix((1,4**self.k),dtype=np.int32)
            out = sparse.lil_matrix((1,4**self.k),dtype=np.int32)
            
            
            for i in range(len(X)-self.k+1):
                u = X[i:i+self.k]
                ind = values[u[0]]
                for j in range(1,self.k):
                    ind += values[u[j]]*4**j
                out[ind]+=1
                
        else:
            #out = np.zeros((len(X),4**self.k)).astype(np.int32)
            #out = sparse.csr_matrix((len(X),4**self.k),dtype=np.int32)
            out = sparse.lil_matrix((len(X),4**self.k),dtype=np.int32)
            for l in range(len(X)):
                for i in range(len(X[0])-self.k+1):
                    u = X[l][i:i+self.k]
                    ind = values[u[0]]
                    for j in range(1,self.k):
                        ind += values[u[j]]*4**j
                    out[l,ind]+=1         
        return sparse.csc_matrix(out)

    #@jit(nopython=True)
    def __call__(self,X1,X2):
        """
        Parameters
        ----------
        X1 : string
            ATCG string
        X2 : string
            ATCG string
        k : int, optional
            length of consecutive features considered. The default is 5.
    
        Returns
        -------
        out : array of int
              spectrum kernel value  
        """
        
        phi1 = self.phi(X1)
        phi2 = self.phi(X2)
            
        return phi1.dot(phi2.T)

#%% mismatch kernel

class mismatch_kernel:
    def __init__(self, k=10, m=1):
        self.k = k
        self.m = m
        self.values = {'A':0,'T':1,'C':2,'G':3}
        self.tab = 4 ** np.arange(k)
        
        self.mis = np.zeros((int(comb(k, m)) * 4 ** m, k))
        shift = np.array(list(product([0,1,2,3], repeat=m)))
        for i, idx in enumerate(combinations(range(k), m)):
            self.mis[i*4**m:(i+1)*4**m, idx] = shift
        # self.mis = np.unique(self.mis, axis=0)
        
    def phi(self, X):
        """
        Parameters
        ----------
        X : string
            ATCG string
        k : int, optional
            length of consecutive features considered. The default is 5.
        m : int, optional
            number of mismatch allowed. The default is 2.
        Returns
        -------
        out : array of int
            mismatch kernel value
        """
        if type(X) == str:
            out = sparse.lil_matrix((1, 4**self.k), dtype=np.int32)
            X = list(map(lambda x: self.values[x], list(X))) + [0] * (self.k-len(X)%self.k)
            for i in range(len(X)-self.k+1):
                u = X[i:i+self.k]
                ind = ((u + self.mis) % 4) * self.tab
                ind = ind.sum(1)
                for j in ind:
                    out[0, j] += 1
                    
        else:
            out = sparse.lil_matrix((len(X), 4**self.k), dtype=np.int32)
            for l in range(len(X)):
                print(l)
                XX = list(map(lambda x: self.values[x], list(X[l]))) + [0] * (self.k-len(X[l])%self.k)
                for i in range(len(X[l])-self.k+1):
                    u = XX[i:i+self.k]
                    ind = ((u + self.mis) % 4) * self.tab
                    ind = ind.sum(1)
                    for j in ind:
                        out[l, j] += 1
                        
        return out.tocsc()
        
    def __call__(self,X1,X2):
        phi1 = self.phi(X1)
        phi2 = self.phi(X2)
        return phi1.dot(phi2.T)

#%% substring kernel

class substring_kernel:
    def __init__(self,k=5,lbd=0.5):
        self.k=k
        self.lbd=lbd
        
    #@jit(nopython=True)
    def phi(self,X):
        """
        Parameters
        ----------
        X : string
            ATCG string
        k : int, optional
            length of consecutive features considered. The default is 5.
        lbd : float, optional
            decay when gaps. Must be between 0 and 1 The default is 0.5
    
        Returns
        -------
        out : array of int
              spectrum kernel value  
        """
        values = {'A':0,'T':1,'C':2,'G':3}
        
        if type(X)==str:
        
            out = np.zeros(4**self.k).astype(np.int32)
            
            for i in range(len(X)-self.k+1):
                u = X[i:i+self.k]
                ind = values[u[0]]
                for j in range(1,self.k):
                    ind += values[u[j]]*4**j
                out[ind]+=1
                
        else:
            out = np.zeros((len(X),4**self.k)).astype(np.int32)
            for l in range(len(X)):
                for i in range(len(X[0])-self.k+1):
                    u = X[l][i:i+self.k]
                    ind = values[u[0]]
                    for j in range(1,self.k):
                        ind += values[u[j]]*4**j
                    out[l,ind]+=1         
        return out

    #@jit(nopython=True)
    def __call__(self,X1,X2):
        """
        Parameters
        ----------
        X1 : string
            ATCG string
        X2 : string
            ATCG string
        k : int, optional
            length of consecutive features considered. The default is 5.
    
        Returns
        -------
        out : array of int
              spectrum kernel value  
        """
        
        phi1 = self.phi(X1).astype(np.int32)
        phi2 = self.phi(X2).astype(np.int32)
            
        return np.dot(phi1,phi2.T)