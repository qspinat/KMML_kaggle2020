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

#%% spectrum kernel

#@jit(nopython=True)
def phi_spectrum_kernel(X,k=10):
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
    
    out = np.zeros(4**k).astype(np.int8)
    
    for i in range(len(X)-k+1):
        u = X[i:i+k]
        ind = values[u[0]]
        for j in range(1,k):
            ind += values[u[j]]*4**j
        out[ind]+=1
        
    return out

#@jit(nopython=True)
def spectrum_kernel(X1,X2,k=10):
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
    phi1 = phi_spectrum_kernel(X1,k=k)
    phi2 = phi_spectrum_kernel(X2,k=k)
    
    return np.dot(phi1,phi2)

#%% mismatch kernel

#@jit(nopython=True)
def phi_mismatch_kernel(X,k=10,m=2): ## TO DO
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
    values = {'A':0,'T':1,'C':2,'G':3}
    
    out = np.zeros(4**k).astype(np.int8)
    
    for i in range(len(X)-k+1):
        u = X[i:i+k]
        ind = values[u[0]]
        for j in range(1,k):
            ind += values[u[j]]*4**j            
                
        out[ind]+=1
        
    return out

#%% substring kernel

