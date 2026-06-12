# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 12:42:04 2026

@author: GKoinig
"""

from SKLEARN_PIPELINES import Shallow_NN, SVM_RBF

paths = ["PE.mat", "PP.mat", "PET.mat"]

clf = Shallow_NN()
clf.train(paths, name="PE_PP_PET_SNN.joblib")

svm = SVM_RBF(C=10.0)
svm.train(paths, name="PE_PP_PET_SVM_RBF.joblib")