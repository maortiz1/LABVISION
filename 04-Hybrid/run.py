#!/usr/bin/python3

import zipfile
#import ipdb 
import requests    
import tarfile
import os
import cv2
import scipy.io
from random import *
import numpy as np
from imutils import build_montages
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
#ipdb.set_trace()

low=cv2.imread(os.path.join("img/", "garfield.jpg"))
lowc=cv2.resize(low,(512,512))
lowc = cv2.cvtColor(lowc, cv2.COLOR_BGR2RGB)
high=cv2.imread(os.path.join("img/", "junior.jpg"))
highc=cv2.resize(high,(512,512))
highc= cv2.cvtColor(highc, cv2.COLOR_BGR2RGB)



highblur = cv2.GaussianBlur(highc,(35,35),110)
highblur = cv2.subtract(highc,highblur)


lowblur = cv2.GaussianBlur(lowc,(35,35),50)
final = cv2.add(highblur,lowblur)
final=cv2.resize(final,(8192,8192))
cv2.imwrite('final.jpg',final)

plt.imshow(final)
plt.show()

from skimage import data
from skimage.transform import pyramid_gaussian
image = final
rows, cols, dim = image.shape
pyramid = tuple(pyramid_gaussian(image, downscale=2, multichannel=True,max_layer=-1))
#pyramid=jpinR
composite_image = np.zeros((rows, cols + cols // 2, 3), dtype=np.double)

composite_image[:rows, :cols, :] = pyramid[0]

i_row = 0
for p,k in zip(pyramid[1:],range(1,len(pyramid))):
    n_rows, n_cols = p.shape[:2]
    composite_image[i_row:i_row + n_rows, cols:cols + n_cols] = p
    i_row += n_rows
    
#    
#composite_image= cv2.cvtColor(composite_image, cv2.COLOR_BGR2RGB)
#cv2.imwrite('piramidet.jpg',composite_image)

fig, ax = plt.subplots()
ax.imshow(composite_image)
plt.show()
plt.axis('off')
#AG=[final]
#w=final.shape[1]
#factor=2
#for j in range(0,10):
#    name='%d.jpg'%(j)
##    imgA = cv2.GaussianBlur(AG[j],(51,51),115)
#    resImgA = cv2.resize(AG[j],(int(w/factor),int(w/factor)))
#    cv2.imwrite(name,resImgA)
#    AG.append(resImgA)
#    w=resImgA.shape[1]


