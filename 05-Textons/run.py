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

now=datetime.now()
File = open('%i-%i-%i-%i.txt'%(now.month, now.day, now.hour ,now.minute),'a+')    
    
def histc(X, bins):
    import numpy as np
    map_to_bins = np.digitize(X,bins)
    r = np.zeros(bins.shape)
    for i in map_to_bins:
        r[i-1] += 1
    return np.array(r)
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
    

# if it is already downloaded, then do not do it again
sinit= datetime.now()
if not os.path.isdir(os.path.join(os.getcwd(),'cifar-10-batches-py')):
    url='https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    r=requests.get(url,allow_redirects=True)
    open('cifar-10-python.tar.gz','wb').write(r.content)
    tar=tarfile.open("cifar-10-python.tar.gz","r")
    tar.extractall()
    tar.close
    ow()
    
endt=datetime.now()-sinit
File.write('Time dowloading database= '+str(endt.total_seconds())+'\n')

start = datetime.now()

dictTrain = cifar10.load_cifar10(meta='cifar-10-batches-py', mode=1)
dictTest = cifar10.load_cifar10(meta='cifar-10-batches-py',mode='test')
numTrain=0.1#porcentaje de imagenes que se van a tomar para el train

imagesTrain,labelsTrain = cifar10.get_data(dictTrain,sliced=numTrain)
imagesTest,labelTest = cifar10.get_data(dictTest,sliced=1)
print('Label  and Data already loaded')

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



k = 144
sinit=datetime.now();
#fb = fbCreate(support=1, startSigma=0.6) # fbCreate(**kwargs, vis=True) for visualization
fb = fbCreate(support=1, startSigma=0.6)



filterResponses = fbRun(fb,train)
print('Filter responses created')
#Computer textons from filter
mapT, textons = computeTextons(filterResponses, k)
print('Computed textones done')
#Load more images
endt=datetime.now()-sinit
File.write('Time computing time of textons='+str(endt.total_seconds())+'\n')
tmapTrain=[]
histTrain=[]
#Calculate texton representation with current texton dictionary
sinit=datetime.now()
for j in range(0,len(imagesTrain)):
    actTmap=assignTextons(fbRun(fb,imagesTrain[j,:,:]),textons.transpose())
    tmapTrain.append(actTmap)
    histTrain.append(histc(actTmap.flatten(), np.arange(k)))
    print('Image %d'%(j)+' from train already textones assign')
    
tmapTest = []
histTest=[]
for j in range (0,len(imagesTest)):
    actTmap=assignTextons(fbRun(fb,imagesTest[j,:,:]),textons.transpose())
    tmapTest.append(actTmap)
    histTest.append(histc(actTmap.flatten(), np.arange(k)))
    print('Image %d'%(j)+' from test already textones assign')

endt=datetime.now()-sinit
File.write('Time Assign Textons= '+ str(endt.total_seconds())+'\n')
print("clasificando")
#KNN - con chi2


print('KNN starting')
strKNN = datetime.now()
kn=KNeighborsClassifier(n_neighbors=100,metric=distchi)
kn=kn.fit(histTrain,labelsTrain)
resultrain=kn.predict(histTrain)
ACAresultrain=accuracy_score(labelsTrain,resultrain)
cMTrain=confusion_matrix(labelsTrain,resultrain)
print(ACAresultrain)
File.write('KNN Confussion Train\n')
File.write(str(cMTrain))
File.write('\n')
File.write('ACA KNN Train\n')
File.write(str(ACAresultrain))
File.write('\n')
result=kn.predict(histTest)
aca= accuracy_score(labelTest,result)
c1=confusion_matrix(labelTest,result)
print(c1)
File.write('\n')
File.write('Confussion Test KNN\n')
File.write(str(c1)+'\n')
endt = datetime.now() -strKNN
secondsKNN=endt.total_seconds()
print('KNN ending')
#RandomForest
print('RF starting')
strF=datetime.now()
clf = RandomForestClassifier(n_estimators=170, max_features=0.2,max_depth=None,min_samples_split=2,bootstrap=False)
clf.fit(histTrain,labelsTrain)
result2=clf.predict(histTest)
resul2train=clf.predict(histTrain)
acaResul2train=accuracy_score(labelsTrain,resul2train)
File.write('ACA random forests train= '+str(acaResul2train)+'\n')
cMtrain2=confusion_matrix(labelsTrain,resul2train)
File.write('Confussion Matrix Train Random Forest=\n')
File.write(str(cMtrain2)+'\n')
aca2 = accuracy_score(labelTest,result2)
endF=datetime.now() - strF
secondsFo=endF.total_seconds()
c=confusion_matrix(labelTest,result2)
print(c)
File.write('CONFUSSION RF \n')
File.write(str(c)+'\n')
print('RF ending')

endT=datetime.now() - start
secod = endT.total_seconds()
notaexp='tamano de los filtros'
print('KNN: '+str(aca))
print('RF: '+ str(aca2))

File.write('ACA KNN: '+str(aca)+'\n')
File.write('ACA RF: '+ str(aca2)+'\n')
File.write('Time: '+str(secod)+'\n')  
File.write('Time KNN: '+ str(secondsKNN)+'\n')   
File.write('Time Random Forests: '+ str(secondsFo)+'\n')
File.write('Number of Train Images: ' + str(len(imagesTrain))+'\n')
File.write('Number of Test Images: ' + str(len(imagesTest))+'\n')
File.write('Note: '+notaexp+'\n')
File.write('END\n\n') 
File.close()


with open('data', 'wb') as f:
    pickle.dump([fb,ACAresultrain,cMTrain,c1,aca,mapT, textons,acaResul2train,aca2,c,cMtrain2, tmapTrain,histTrain,tmapTest,histTest,imagesTrain,labelsTrain,imagesTest,labelTest,result2],f)
