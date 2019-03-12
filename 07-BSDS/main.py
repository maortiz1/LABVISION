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
# download dataset
if not os.path.isdir(os.path.join(os.getcwd(),'BSR')):
    url='http://bcv001.uniandes.edu.co/BSDS500FastBench.tar.gz'
    r=requests.get(url,allow_redirects=True)
    open('BSDS500FastBench.tar.gz','wb').write(r.content)
    tar=tarfile.open("BSDS500FastBench.tar.gz","r")
    tar.extractall()
    tar.close

# go through database and apply segmentation method
filenames=os.listdir("BSR/BSDS500/data/images/train/")
#ipdb.set_trace()
K = np.array([2,3]) # K numbers to segmentate
#for i in filenames:
i = "100075.jpg"
count = 0
concat = np.empty((1,np.size(K)), dtype=object)
temp=cv2.imread(os.path.join("BSR/BSDS500/data/images/train/", i))
for j in K:   
   seg = segmentByClustering(rgbImage=temp, colorSpace='rgb', clusteringMethod='kmeans', numberOfClusters=j)
   #ipdb.set_trace()   
   concat[0,count] = seg
   count = count+1


sio.savemat('i.mat', {'segs':concat})

