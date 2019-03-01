# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 21:51:19 2019

@author: m_ana
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt

with open('data','rb') as f:
    fb,ACAresultrain,cMTrain,c1,aca,mapT, textons,acaResul2train,aca2,c,cMtrain2 = pickle.load(f)

plt.figure()
cMTrain=cMTrain.astype('float')/cMTrain.sum(axis=1)[:,np.newaxis]
plt.imshow(cMTrain,interpolation='nearest',cmap='cool') 
plt.colorbar()
plt.show()
plt.suptitle('Confussion Matrix KNN on Train Set',size=16)

plt.figure()
c1=c1.astype('float')/c1.sum(axis=1)[:,np.newaxis]
plt.imshow(c1,interpolation='nearest',cmap='cool') 
plt.colorbar()
plt.show()
plt.suptitle('Confussion Matrix KNN on Test Set',size=16)




plt.figure()
c=c.astype('float')/c.sum(axis=1)[:,np.newaxis]
plt.imshow(c,interpolation='nearest',cmap='cool') 
plt.colorbar()
plt.show()
plt.suptitle('Confussion Matrix Random Forest on Test Set',size=16)


plt.figure()
cMtrain=cMtrain2.astype('float')/cMtrain2.sum(axis=1)[:,np.newaxis]
plt.imshow(cMtrain2,interpolation='nearest',cmap='cool') 
plt.colorbar()
plt.show()
plt.suptitle('Confussion Matrix Random Forest on Train Set',size=16)

plt.figure()
plt.imshow(textons,cmap='cool')
plt.suptitle('Textons Map',size=16)
plt.show()
