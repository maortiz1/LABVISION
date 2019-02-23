#!/usr/bin/ipython3
#import ipdb 
import sys
sys.path.append('lib/python')
import requests    
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

start=datetime.now()

# if it is already downloaded, then do not do it again
if not os.path.isdir(os.path.join(os.getcwd(),'cifar-10-batches-py')):
    url='https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    r=requests.get(url,allow_redirects=True)
    open('cifar-10-python.tar.gz','wb').write(r.content)
    tar=tarfile.open("cifar-10-python.tar.gz","r")
    tar.extractall()
    tar.close
    
# Mari segun cifar asi se crea el dicccionario pero no lo alcance a probar
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
    
#filter bank creation
from fbCreate import fbCreate
fb = fbCreate(support=2, startSigma=0.6)
# Obtain concatenated matrices of random pixels

folders=os.listdir("ruta de las imagenes")
folders.sort()
del folders[0]
n=100; #number of train and test images
sz=32 #size of squared images
numel=sz*sz #number of selected pixels per image
train=np.array([]).reshape(sz,0)
test=np.array([]).reshape(sz,0)
label=[]

for i in folders:
    path=os.path.join('ruta de las imagenes',i)
    ran_files=random.sample(os.listdir(path),n*2)
    ims_selec=np.array([]).reshape(sz,0)
    for j in ran_files:
        im=io.imread(os.path.join('ruta de las imagenes',i,j))
        mn=np.size(im)
        im=np.reshape(im,(mn,1))
        img_pix=random.sample(range(1,mn),numel)
        selec_pix=np.reshape(im[img_pix],[sz, sz])
        ims_selec=np.hstack([ims_selec, selec_pix])
    train=np.hstack([train, ims_selec[:,0:n*sz]])
    test=np.hstack([test,ims_selec[:,n*sz:]])
    label=np.hstack([label, [i[1:3]]*n])