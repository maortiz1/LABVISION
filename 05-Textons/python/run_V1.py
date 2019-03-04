#!/usr/bin/ipython3
#import ipdb 
import sys
sys.path.append('lib/python')
import requests    
import pickle
import os
import cv2
import scipy.io
import random 
import numpy as np
from subprocess import call
import imutils
import tarfile
import matplotlib.pyplot as plt
import os
from skimage import io
from skimage import color
from skimage.transform import resize
from fbRun import fbRun
from computeTextons import computeTextons
from assignTextons import assignTextons
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import chi2_kernel
from datetime import datetime
from sklearn.metrics import confusion_matrix

from computeTextons import computeTextons
import cifar10
#Apply filterbank to sample image
from fbRun import fbRun
import numpy as np
#filter bank creation
from fbCreate import fbCreate

start=datetime.now()

# if it is already downloaded, then do not do it again
if not os.path.isdir(os.path.join(os.getcwd(),'cifar-10-batches-py')):
    url='https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    r=requests.get(url,allow_redirects=True)
    open('cifar-10-python.tar.gz','wb').write(r.content)
    tar=tarfile.open("cifar-10-python.tar.gz","r")
    tar.extractall()
    tar.close
    

dictTrain = cifar10.load_cifar10(meta='cifar-10-batches-py', mode=1)
dictTest = cifar10.load_cifar10(meta='cifar-10-batches-py',mode='test')
numTrain=0.01#porcentaje de imagenes que se van a tomar para el train

imagesTrain,labelsTrain = cifar10.get_data(dictTrain,sliced=numTrain)
imagesTest,labelTest = cifar10.get_data(dictTest,sliced=0.01)


# Obtain concatenated matrices of random pixels



n=100; #number of train and test images
sz=32 #size of squared images
numel=sz*sz #number of selected pixels per image
train=np.array([]).reshape(sz,0)
test=np.array([]).reshape(sz,0)



for j in range(0,len(imagesTrain)):
    train=np.hstack((train,imagesTrain[j,::,::]))

for j in range(0,len(imagesTest)):
    test=np.hstack((test,imagesTest[j,::,::]))
    
    
k = 16

fb = fbCreate(support=2, startSigma=0.6) # fbCreate(**kwargs, vis=True) for visualization

filterResponses = fbRun(fb,train)

#Computer textons from filter
mapT, textons = computeTextons(filterResponses, k)

#Load more images
def histc(X, bins):
    import numpy as np
    map_to_bins = np.digitize(X,bins)
    r = np.zeros(bins.shape)
    for i in map_to_bins:
        r[i-1] += 1
    return np.array(r)
tmapTrain=[]
histTrain=[]
#Calculate texton representation with current texton dictionary
for j in range(0,len(imagesTrain)):
    actTmap=assignTextons(fbRun(fb,imagesTrain[j,:,:]),textons.transpose())
    tmapTrain.append(actTmap)
    histTrain.append(histc(actTmap.flatten(), np.arange(k)))
    
tmapTest = []
histTest=[]
for j in range (0,len(imagesTest)):
    actTmap=assignTextons(fbRun(fb,imagesTest[j,:,:]),textons.transpose())
    tmapTest.append(actTmap)
    histTest.append(histc(actTmap.flatten(), np.arange(k)))
#Metric functions
#Chi-squared distance metric
def distchi(x,y):
	import numpy as np
	np.seterr(divide='ignore', invalid='ignore')
	d=np.sum(((x-y)**2)/(x+y)) 
	return d
#Intersection kernel metric
def inter(x,y):
        import numpy as np
        min=np.minimum(x,y)
        d=1-np.sum(min)
        return d
print("clasificando")
#KNN - con chi2
strKNN = datetime.now()
kn=KNeighborsClassifier(n_neighbors=50,metric=distchi)
kn=kn.fit(histTrain,labelsTrain)
result=kn.predict(histTrain)
aca= accuracy_score(labelsTrain,result)
c=confusion_matrix(labelsTrain,result)
endt = datetime.now() -strKNN
secondsKNN=endt.total_seconds()
#RandomForest

strF=datetime.now()
clf = RandomForestClassifier(n_estimators=30, max_features=0.5)
clf.fit(histTrain,labelsTrain)
result2=clf.predict(histTest)
aca2= accuracy_score(labelTest,result2)
endF=datetime.now() - strF
secondsFo=endF.total_seconds()

endT=datetime.now() - start
secod = endT.total_seconds()
notaexp='K = 16 Rf 30 y KNN 50'
print(aca)
print(aca2)
now=datetime.now()
File = open('%i-%i-%i-%i.txt'%(now.month, now.day, now.hour ,now.minute),'a+')
File.write('ACA KNN: '+str(aca)+'\n')
File.write('ACA RF: '+ str(aca2)+'\n')
File.write('Time: '+str(secod)+'\n')  
File.write('Time KNN: '+ str(secondsKNN)+'\n')   
File.write('Time Random Forests: '+ str(secondsFo)+'\n')
File.write('Number of Train Images: ' + str(len(imagesTrain))+'\n')
File.write('Number of Test Images: ' + str(len(imagesTest))+'\n')
File.write('Note: '+notaexp+'\n')
File.close()