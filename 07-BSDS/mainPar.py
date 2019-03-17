#!/usr/bin/ipython3
import os
import requests 
import tarfile
import cv2
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
import ipdb
from Segment import segmentByClustering 
from joblib import Parallel,delayed
# download dataset
if not os.path.isdir(os.path.join(os.getcwd(),'BSR')):
    url='http://bcv001.uniandes.edu.co/BSDS500FastBench.tar.gz'
    r=requests.get(url,allow_redirects=True)
    open('BSDS500FastBench.tar.gz','wb').write(r.content)
    tar=tarfile.open("BSDS500FastBench.tar.gz","r")
    tar.extractall()
    tar.close

# go through database and apply segmentation method
filenames=os.listdir("BSR/BSDS500/data/images/test/")
#ipdb.set_trace()
K = range(2,411,8)# K numbers to segmentate
#for i in filenames:
i = "100075.jpg"
os.mkdir(os.path.join('BSR','kmeanshsv411'))

    


def trypal(filenam):
  filenam2,ext =os.path.splitext(filenam) 
  if ext=='.jpg':
     temp=cv2.imread(os.path.join("BSR/BSDS500/data/images/test/", filenam))          
     concat = np.empty((1,np.size(K)), dtype=object)
     count = 0
     for j in K:   
        seg = segmentByClustering(rgbImage=temp, colorSpace='hsv', clusteringMethod='kmeans', numberOfClusters=j)
        #ipdb.set_trace()   
        concat[0,count] = seg
        count = count+1
        print('K=%d , method=kmeanhsv, img=%s '%(j,filenam))
     filenam2,ext =os.path.splitext(filenam)  
     
     sio.savemat(os.path.join('BSR','kmeanshsv411','%s.mat'%(filenam2)), {'segs':concat})
     print('%s.mat'%(filenam2))


Parallel(n_jobs=10) (delayed(trypal)(filenam) for filenam in filenames)