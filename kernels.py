import numpy as np
from itertools import product, combinations
from scipy.special import comb
import scipy.sparse as sparse
from tqdm import tqdm


#%% spectrum kernel
class spectrum_kernel:
    def __init__(self, k=5):
        self.k = k
        self.values = {'A':0,'T':1,'C':2,'G':3}
    
    def phi(self, X):
        values = {'A':0,'T':1,'C':2,'G':3}
        
        if type(X)==str:
            out = sparse.lil_matrix((1,4**self.k),dtype=np.int32)
            
            for i in range(len(X)-self.k+1):
                u = X[i:i+self.k]
                ind = values[u[0]]
                for j in range(1,self.k):
                    ind += values[u[j]]*4**j
                out[ind]+=1
                
        else:
            out = sparse.lil_matrix((len(X),4**self.k),dtype=np.int32)
            for l in tqdm(range(len(X))):
                for i in range(len(X[0])-self.k+1):
                    u = X[l][i:i+self.k]
                    ind = values[u[0]]
                    for j in range(1,self.k):
                        ind += values[u[j]]*4**j
                    out[l,ind]+=1         
        return sparse.csc_matrix(out)
        
    def __call__(self, X1, X2):
        phi1 = self.phi(X1)
        phi2 = self.phi(X2)
            
        return phi1.dot(phi2.T)

#%% mismatch kernel

class mismatch_kernel:
    def __init__(self, k=10, m=1):
        self.k = k
        self.m = m
        self.values = {'A':0,'T':1,'C':2,'G':3}
        self.tab = 4 ** np.arange(k,dtype=int)
        
        self.mis = np.zeros((int(comb(k, m)) * 4 ** m, k),dtype=int)
        shift = np.array(list(product([0,1,2,3], repeat=m)),dtype=int)
        for i, idx in enumerate(combinations(range(k), m)):
            self.mis[i*4**m:(i+1)*4**m, idx] = shift
        
    def phi(self, X):
        if type(X) == str:
            out = sparse.lil_matrix((1, 4**self.k), dtype=int)
            X = list(map(lambda x: self.values[x], list(X))) + [0] * (self.k-len(X)%self.k)
            for i in range(len(X)-self.k+1):
                u = X[i:i+self.k]
                ind = ((u + self.mis) % 4) * self.tab
                ind = ind.sum(1)
                for j in ind:
                    out[0, j] += 1
                    
        else:
            out = sparse.lil_matrix((len(X), 4**self.k), dtype=int)
            for l in tqdm(range(len(X))):
                XX = list(map(lambda x: self.values[x], list(X[l]))) + [0] * (self.k-len(X[l])%self.k)
                for i in range(len(X[l])-self.k+1):
                    u = XX[i:i+self.k]
                    ind = ((u + self.mis) % 4) * self.tab
                    ind = ind.sum(1)
                    for j in ind:
                        out[l, j] += 1
                        
        return out.tocsc()
        
    def __call__(self, X1, X2):
        phi1 = self.phi(X1)
        phi2 = self.phi(X2)
        return phi1.dot(phi2.T)

#%% Concatenation

class concat_kernel:
    def __init__(self, kernels=[spectrum_kernel(k=5), spectrum_kernel(k=10)], weights=[1, 1]):
        self.kernels=kernels
        self.weights=weights
        
    def phi(self, X):
        out = self.kernels[0].phi(X)*self.weights[0]
        
        for i in range(1, len(self.kernels)):
            out = sparse.hstack([out, self.kernels[i].phi(X)*self.weights[i]])
        
        return out
    
    def __call__(self, X1, X2):
        phi1 = self.phi(X1)
        phi2 = self.phi(X2)
        return phi1.dot(phi2.T)
