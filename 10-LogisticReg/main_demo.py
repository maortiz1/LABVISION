#!/usr/bin/ipython3

from skimage import color
from skimage import io
import zipfile
import ipdb 
import requests    
import tarfile
import os
import cv2

if not os.path.isdir(os.path.join(os.getcwd(),'demo')):
    url='https://drive.google.com/uc?export=download&id=16TGyOoqyV8huJqHhenw7loSc8O0FcOEV'
    r=requests.get(url,allow_redirects=True)
    open('demo.zip','wb').write(r.content)
    tar=tarfile.open("fdemo.zip","r")
    tar.extractall()
    tar.close
    
filenames=os.listdir("demo/")
test = []
for i in filenames:
   temp=cv2.imread(os.path.join("demo/", i))
   #the files are too big. It is necessary to resize
   temp = color.rgb2gray (temp)
   imCrop=cv2.resize(temp,(300,300))
   test.append(imCrop)
   


