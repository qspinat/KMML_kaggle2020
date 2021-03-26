import numpy as np
from kernels import spectrum_kernel
from sklearn.model_selection import StratifiedKFold
import scipy.sparse as sparse

from qpsolvers import solve_qp
from cvxopt import matrix, spmatrix, solvers


#%% kernel ridge regression
class KRR:
    def __init__(self, C=1, kernel=spectrum_kernel(k=5)):
        """
        Parameters
        ----------
        C : float, optional
            regularization constant. The default is 1.
        Returns
        -------
        None
        """
        self.C=C
        self.alpha=0
        self.phi_train=0
        self.w=0
        self.kernel=kernel
        
    def fit(self, X, Y):    
        """
        Fits the KRR
        
        Parameters
        ----------
        X : 1-D array of strings
            ATCG inputs
        Y : 1-D array
            labels.

        Returns
        -------
        None
        """
        n = X.shape[0]
        
        phi = self.kernel.phi(X)
        self.phi_train = phi
        
        d = phi.shape[1]
        
        if n<=d: 
            self.alpha = sparse.csr_matrix(np.linalg.inv(phi.dot(phi.T)+self.C*np.eye(Y.shape[0])).dot(Y).T)
            self.alpha = self.alpha.multiply(np.abs(self.alpha)>10e-8)
            self.w=None
        else:
            self.w = np.array(np.linalg.inv(phi.T.dot(phi)+self.C*np.eye(phi.shape[0],dtype=np.int)).dot(phi.T).dot(Y)).squeeze()
        return
    
    def predict(self, x):
        """
        Predicts value from input and training set

        Parameters
        ----------
        x : string
            input.

        Returns
        -------
        out : float
            prediction
        """
        if self.w != None:
            phi = self.kernel.phi(x)
            out = np.array(phi.dot(self.w.T).T)
        else:
            phi = self.kernel.phi(x)
            out = phi.dot(self.phi_train.T.dot(self.alpha))
            out = out.toarray().squeeze()
        return out

    
#%% SVM
class SVM:
    def __init__(self, C=1, kernel=spectrum_kernel(k=5)):
        """
        Parameters
        ----------
        C : float, optional
            regularization constant. The default is 1.
        """
        self.C=C
        self.alpha=0
        self.phi_train=0
        self.kernel=kernel
        
    def fit(self, X, Y):
        n = X.shape[0]
        
        phi = self.kernel.phi(X)
        self.phi_train = phi
        
        K = phi.dot(phi.T).toarray()
        
        P = matrix(2*K.astype(np.float))
        q = matrix(-2*Y.astype(np.float))
        G = matrix(np.concatenate((np.diag(Y),-np.diag(Y))).astype(np.float))
        h = matrix(np.concatenate((self.C*np.ones(n),np.zeros(n))))       
        
        self.alpha = sparse.csr_matrix(solvers.qp(P=P, q=q, G=G, h=h, show_progress=True)['x'])
        self.alpha = self.alpha.multiply(np.abs(self.alpha)>(self.C*1e-6))
        return
    
    def predict(self, x):
        phi = self.kernel.phi(x)
        out = phi.dot(self.phi_train.T.dot(self.alpha))
        out = out.toarray().squeeze()
        return out
    
    
#%% cross validation
class cross_clf:
    def __init__(self, C=1, kernel=spectrum_kernel(k=5), clf=SVM, n_splits=4, random_state=42):
        """
        Parameters
        ----------
        C : float, optional
            regularization constant. The default is 1.
        n_splits : int, optional
            number of cross validation folds. The default is 3.
        """
        self.C=C
        self.kernel=kernel
        self.clfs = [clf(C=C,kernel=kernel) for i in range(n_splits)]
        self.n_splits=n_splits
        self.random_state=random_state
        
    def fit(self, X, Y):
        skf = StratifiedKFold(n_splits=self.n_splits, random_state=0, shuffle=True)
        
        i=0
        for train_index, test_index in skf.split(X, Y):  
            self.clfs[i].fit(X[train_index], Y[train_index])
            i+=1
            
        return
    
    def predict(self, x):
        out = self.clfs[0].predict(x)
        for i in range(1,self.n_splits):
            out+=self.clfs[i].predict(x)
        return out/self.n_splits
