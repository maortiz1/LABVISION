# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 18:47:29 2019

@author: m_ana
"""

import numpy as np

def metricJaccard(groundtruth,segmentation):
    unqseg = np.unique(segmentation)
    unqground = np.unique(groundtruth)
    mat=np.zeros([len(unqseg),len(unqground)])
    mat2=np.zeros([len(unqseg),len(unqground)])
    numpixelsground=[];
    numpixelsseg=[];

    
    for segV,k in zip(unqseg,range(0,len(unqseg))):
        
        ac=np.zeros(groundtruth.shape)
        ac[segmentation==segV]=1       
        numpixelsseg.append(np.sum(ac))
        
        for groundV,j in zip(unqground,range(0,len(unqground))):
            
            ac2=np.zeros(groundtruth.shape)
            ac2[groundtruth==groundV]=1
            if k==0:
                numpixelsground.append(np.sum(ac2))
   
            join = np.count_nonzero(np.logical_and(ac2,ac)==True)
            lolor= np.count_nonzero(np.logical_or(ac2,ac)==True)
            
            mat2[k][j]=join
            mat[k][j]=join/lolor      
  

    maxvalue=np.amax(mat,axis=0)
    
    jacc = np.mean(maxvalue)
    
    return [mat,jacc]
    
    
def groundtruth(img_file):
    import scipy.io as sio
    import imageio
    img = imageio.imread(img_file)
    gt=sio.loadmat(img_file.replace('jpg', 'mat'))
    segm=gt['groundTruth'][0,1][0][0]['Segmentation']
    imshow(img, segm, title='Groundtruth')
    return segm



import os
import imageio as im
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
pathImg=os.path.join('BSDS_small','train')
img= im.imread(os.path.join(pathImg,'12003.jpg'))
height = np.size(img, 0)
width = np.size(img, 1)
feat = img.reshape(height*width,3)
kmeans = KMeans(n_clusters=8).fit_predict(feat)
segmentation = np.reshape(kmeans,(height,width))
ground=groundtruth(os.path.join(pathImg,'12003.jpg'))


mat,jacc=metricJaccard(ground,segmentation)
