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
    
    
    
    for segV,k in zip(unqseg,range(0,len(unqseg))):
    for groundV,j in zip(unqground,range(0,len(unqground))):
        ac=np.zeros(seg.shape)
        ac[seg==segV]=1        
        ac2=np.zeros(ground.shape)
        ac2[ground==groundV]=1
        join = np.count_nonzero(np.logical_and(ac2,ac)==True)
        lolor= np.count_nonzero(np.logical_or(ac2,ac)==True)
        
        mat[k][j]=join/lolor        
        print('['+str(segV)+','+str(groundV)+'] = '+ str(join/lolor))
