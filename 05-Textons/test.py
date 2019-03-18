#!/usr/bin/ipython3
import pickle
import numpy as np
import matplotlib.pyplot as plt

try:    
    import random
except ImportError:   
    subprocess.call(['pip3','install','random'])
    import random

with open('data','rb') as f:
    fb,ACAresultrain,cMTrain,c1,aca,mapT, textons,acaResul2train,aca2,c,cMtrain2, tmapTrain,histTrain,tmapTest,histTest,imagesTrain,labelsTrain,imagesTest,labelTest,result2 = pickle.load(f)


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
plt.imshow(textons,cmap='hot')
plt.suptitle('Textons Map',size=16)
plt.show()
