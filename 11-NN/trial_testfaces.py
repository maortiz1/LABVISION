#!/usr/bin/ipython3
import requests
import numpy as np
import tarfile
import zipfile
import os
import cv2
from skimage import color
from matplotlib import pyplot as plt
import face_recognition

if not os.path.isdir(os.path.join(os.getcwd(),'Emotions_test')):
    url='http://bcv001.uniandes.edu.co/Emotions_test.zip'
    r=requests.get(url,allow_redirects=True)
    open('Emotions_test.zip','wb').write(r.content) 
    zip_ref = zipfile.ZipFile('Emotions_test.zip', 'r')
    zip_ref.extractall()
    zip_ref.close()

filenames=os.listdir("Emotions_test/")
    
for ix in filenames:
#    img = cv2.imread(os.path.join("Emotions_test/", ix))
#    imgtest1 = img
#    imgtest = color.rgb2gray(imgtest1) 
#    imgtest = np.array(imgtest, dtype='uint8')
    imgtest= face_recognition.load_image_file(os.path.join("Emotions_test/", ix))
    facelocations =face_recognition.face_locations(imgtest);
    print(facelocations)
    faces = len(facelocations)
 
    print('Total number of Faces found: ',(faces))
    
    for facelocation in facelocations:
        top, right, bottom, left = facelocation
        face_detect = imgtest[top:bottom, left:right]
        roi_gray = imgtest[top:bottom, left:right]
        roi_color = imgtest[top:bottom, left:right]        
        plt.imshow(face_detect)
        print("Please press any key to continue")
        plt.waitforbuttonpress(0)
