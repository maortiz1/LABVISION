# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 22:34:45 2019

@author: m_ana
"""
import sys 
import os
import numpy as np
import subprocess
import math
import time
import pickle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
try:    
    import random
except ImportError:   
    subprocess.call(['pip3','install','random'])
    import random

with open('data','rb') as f:
    fb,ACAresultrain,cMTrain,c1,aca,mapT, textons,acaResul2train,aca2,c,cMtrain2, tmapTrain,histTrain,tmapTest,histTest,imagesTrain,labelsTrain,imagesTest,labelTest,result2 = pickle.load(f)


N=10;

fig=plt.figure(figsize=(10,8))
outer=gridspec.GridSpec(2,5,wspace=0.2,hspace=0.1)



for i in range(10):

    inner=gridspec.GridSpecFromSubplotSpec(2,1,subplot_spec=outer[i],wspace=0.1,hspace=0.1)

    j=random.randint(0,1000)
    ax=plt.Subplot(fig,inner[0])
    ax2=plt.Subplot(fig,inner[1])
    img=imagesTest[j]
    s='Imagen Original Label:' +str(labelTest[j]) 
    ax.imshow(img)
    ax.set_title(s)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    s2='Classification Label: ' + str(result2[j]);
    ax2.imshow(tmapTest[j])
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    ax2.set_title(s2)
    fig.add_subplot(ax)
    fig.add_subplot(ax2)
    

 
    