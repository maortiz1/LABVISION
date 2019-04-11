#!/usr/bin/ipython3
import requests
import numpy as np
import tarfile
import zipfile
import os
import cv2
from skimage import color
from matplotlib import pyplot as plt

if not os.path.isdir(os.path.join(os.getcwd(),'Emotions_test')):
    url='http://bcv001.uniandes.edu.co/Emotions_test.zip'
    r=requests.get(url,allow_redirects=True)
    open('Emotions_test.zip','wb').write(r.content) 
    zip_ref = zipfile.ZipFile('Emotions_test.zip', 'r')
    zip_ref.extractall()
    zip_ref.close()

filenames=os.listdir("Emotions_test/")
    
for ix in filenames:
    img = cv2.imread(os.path.join("Emotions_test/", ix))
    imgtest1 = img
    imgtest = color.rgb2gray(imgtest1) 
    imgtest = np.array(imgtest, dtype='uint8')
    facecascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
   
    faces = facecascade.detectMultiScale(imgtest, scaleFactor=5, minNeighbors=5)
 
    print('Total number of Faces found',len(faces))
    
    for (x, y, w, h) in faces:
        face_detect = cv2.rectangle(imgtest, (x, y), (x+w, y+h), (255, 0, 255), 2)
        roi_gray = imgtest[y:y+h, x:x+w]
        roi_color = imgtest[y:y+h, x:x+w]        
        plt.imshow(face_detect)
        print("Please press any key to continue")
        plt.waitforbuttonpress(0)
