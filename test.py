#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 17:58:00 2021

@author: quentin
"""

import numpy as np
import pandas as pd
from preprocess import spectrum_kernel
from kernel_methods import KRR, KLR

#%% load data

X_train = pd.concat((pd.read_csv("Data/Xtr0.csv"),pd.read_csv("Data/Xtr1.csv"),pd.read_csv("Data/Xtr2.csv")))['seq'].values
Y_train = pd.concat((pd.read_csv("Data/Ytr0.csv"),pd.read_csv("Data/Ytr1.csv"),pd.read_csv("Data/Ytr2.csv")))['Bound'].values

#%% Kernel Ridge Regression

krr = KRR(C=1,kernel=spectrum_kernel)
krr.fit(X_train,Y_train)