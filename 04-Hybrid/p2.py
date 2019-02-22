# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 11:49:21 2019

@author: m_ana
"""

import zipfile
#import ipdb 
import requests    
import tarfile
import os
import cv2
import scipy.io
from random import *
import numpy as np
import imutils

from subprocess import call
import timeit
import pickle
#import ipdb
import matplotlib.pyplot as plt
from PIL import ImageFont, ImageDraw, Image
from xlrd import open_workbook
# if it is already downloaded, then do not do it again
if not os.path.isdir(os.path.join(os.getcwd(),'img')):
   url='https://drive.google.com/uc?export=download&id=1vFxqw2UQGdwRb4V1cLZSE7MZXT3IMA5r'
   r=requests.get(url,allow_redirects=True)
   open('img.zip','wb').write(r.content) 
   zip_ref = zipfile.ZipFile('img.zip', 'r')
   zip_ref.extractall()
   zip_ref.close()
   
low=cv2.imread(os.path.join("img/", "junior.jpg"))
side1=cv2.resize(low,(2048,2048))
side1 = cv2.cvtColor(side1, cv2.COLOR_BGR2RGB)

high=cv2.imread(os.path.join("img/", "garfield.jpg"))
side2=cv2.resize(high,(2048,2048))
side2 = cv2.cvtColor(side2, cv2.COLOR_BGR2RGB) 

#genetarteLaplacianPyramids
side1Copy= side1.copy()
side2Copy= side2.copy()

AG=[side1Copy]
BG=[side2Copy]

LpAD=[]
LpBD=[]


w = 2048
k=7;
sizes=[(w,w)]
factor=2;
for i in range(0,k):
     imgA = cv2.GaussianBlur(AG[i],(15,15),20)
     resImgA = imutils.resize(imgA,width=int(w/factor))
#     imgA = cv2.GaussianBlur(resImgA,(11,11),5)
     res2ImgA=imutils.resize(imgA, width=w)
     
     AG.append(resImgA)
     lap = cv2.subtract(AG[i],imgA)
     LpAD.append(lap)
     
     imgB = cv2.GaussianBlur(BG[i],(15,15),20)
     resImgB = imutils.resize(imgB,width=int(w/factor))
#     imgB = cv2.GaussianBlur(resImgB,(11,11),5)
     res2ImgB=imutils.resize(imgB, width=w)
     BG.append(resImgB)
     lap = cv2.subtract(BG[i],imgB)
     LpBD.append(lap)

     w = int(w/factor)
     sizes.append((w,w))
     
joinH=[]    
jpinR=[]
#joinHalves

for lpbd,lpad,size,ag,bg in zip(LpAD,LpBD,sizes,AG,BG):
    h=int(size[1]/2)
    h1=lpbd[:,0:h]
    h2=lpad[:,h:]
    jh=np.hstack((h1,h2))
    joinH.append(jh)
    h1=ag[:,0:h]
    h2=bg[:,h:]
    jh=np.hstack((h1,h2))
    jpinR.append(jh)
    

beg = jpinR[-1]
w= beg.shape[1]   

for j in range(6,0,-1):
    
    beg=cv2.add(beg,joinH[j])
    
    beg=imutils.resize(beg,width=(w)*factor)    
    beg = cv2.GaussianBlur(beg,(15,15),20)
    w=(w)*factor
    
    
plt.figure()
plt.imshow(beg)
plt.axis('off')
plt.show()
beg = cv2.cvtColor(beg, cv2.COLOR_BGR2RGB) 
cv2.imwrite('pru1.jpg',beg)


import numpy as np
import matplotlib.pyplot as plt

from skimage import data
from skimage.transform import pyramid_gaussian


image = AG[0]
rows, cols, dim = image.shape
#pyramid = tuple(pyramid_gaussian(image, downscale=2, multichannel=True))
pyramid=AG
composite_image = np.zeros((rows, cols + cols // 2, 3), dtype=np.double)

composite_image[:rows, :cols, :] = pyramid[0]

i_row = 0
for p in pyramid[1:]:
    n_rows, n_cols = p.shape[:2]
    composite_image[i_row:i_row + n_rows, cols:cols + n_cols] = p
    i_row += n_rows


cv2.imwrite('comp.JPEG',composite_image)

plt.imshow(composite_image)


