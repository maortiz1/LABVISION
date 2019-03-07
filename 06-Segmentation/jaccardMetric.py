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
    


