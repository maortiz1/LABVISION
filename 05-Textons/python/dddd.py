# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 21:38:27 2019

@author: m_ana
"""
import matplotlib.pyplot as plt
from fbCreate import fbCreate
import math
fb = fbCreate(support=1, startSigma=0.6)
#fb2 = fbCreate(support=2, startSigma=0.5)
#fb3 = fbCreate(support=3, startSigma=0.5)

fig,axes = plt.subplots(4,int(len(fb)/2))
allf=[]
for fi,x in zip(fb[0:int(len(fb)/2)],range(0,int(len(fb)/2))):
    for fd,nm in zip(fi,range(0,2)):
        allf.append(fd)        
        axes[nm,x].imshow(fd,cmap='gray')
        axes[nm,x].get_xaxis().set_visible(False)
        axes[nm,x].get_yaxis().set_visible(False)
        
for fi,x in zip(fb[int(len(fb)/2)-1:-1],range(0,int(len(fb)/2))):
    print('x='+str(x))
    for fd,nm in zip(fi,range(2,5)):
        print(nm)
        print(math.floor(x/4))
        allf.append(fd)        
        
        axes[nm,x].imshow(fd,cmap='gray')
        axes[nm,x].get_xaxis().set_visible(False)
        axes[nm,x].get_yaxis().set_visible(False)
        
fig.suptitle('Filter Bank',fontsize=50)
figmana=plt.get_current_fig_manager()
figmana.window.showMaximized()



