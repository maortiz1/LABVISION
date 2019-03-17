#!/usr/bin/ipython3
import imageio as im
import Segment
import os
import imageio
import matplotlib.pyplot as plt
import scipy.io as sio

img_file=os.path.join('BSR','BSDS500','data','images','test','101027.jpg')
img_file2=os.path.join('BSR','BSDS500','data','images','test','223060.jpg')
img_file3=os.path.join('BSR','BSDS500','data','images','test','228076.jpg')

img1 = imageio.imread(img_file)
gt=sio.loadmat((img_file.replace('jpg', 'mat')).replace('images','groundTruth'))    
segm=gt['groundTruth'][0,1][0][0]['Segmentation']
segm12=gt['groundTruth'][0,4][0][0]['Segmentation']    

img2 = imageio.imread(img_file2)
gt=sio.loadmat((img_file2.replace('jpg', 'mat')).replace('images','groundTruth'))    
segm2=gt['groundTruth'][0,1][0][0]['Segmentation']
segm21=gt['groundTruth'][0,4][0][0]['Segmentation']

img3 = imageio.imread(img_file3)
gt=sio.loadmat((img_file3.replace('jpg', 'mat')).replace('images','groundTruth'))    
segm3=gt['groundTruth'][0,1][0][0]['Segmentation']
segm31=gt['groundTruth'][0,4][0][0]['Segmentation']

plt.subplot(231)
plt.imshow(img1, cmap=plt.get_cmap('gray'))
plt.imshow(segm, cmap=plt.get_cmap('rainbow'), alpha=0.5)
plt.title('Groundtruth 1 101027.jpg')
plt.axis('off')

plt.subplot(234)
plt.imshow(img1,cmap=plt.get_cmap('gray'))
plt.imshow(segm12, cmap=plt.get_cmap('rainbow'), alpha=0.5)
plt.title('Groundtruth 2 101027.jpg')
plt.axis('off')


plt.subplot(232)
plt.imshow(img2, cmap=plt.get_cmap('gray'))
plt.imshow(segm2, cmap=plt.get_cmap('rainbow'), alpha=0.5)
plt.title('Groundtruth 223060.jpg')
plt.axis('off')

plt.subplot(235)
plt.imshow(img2,cmap=plt.get_cmap('gray'))
plt.imshow(segm21, cmap=plt.get_cmap('rainbow'), alpha=0.5)
plt.title('Groundtruth 2 223060.jpg')
plt.axis('off')

plt.subplot(233)
plt.imshow(img3, cmap=plt.get_cmap('gray'))
plt.imshow(segm3, cmap=plt.get_cmap('rainbow'), alpha=0.5)

plt.title('Groundtruth 228076.jpg')
plt.axis('off')

plt.subplot(236)
plt.imshow(img3,cmap=plt.get_cmap('gray'))
plt.imshow(segm31, cmap=plt.get_cmap('rainbow'), alpha=0.5)
plt.title('Groundtruth 2 228076.jpg')
plt.axis('off')

plt.show()


plt.figure()
plt.subplot(131)
plt.imshow(img1)
plt.title('101027.jpg')


plt.subplot(132)
plt.imshow(img2)
plt.title('223060.jpg')

plt.subplot(133)
plt.imshow(img3)
plt.title('228076.jpg')
plt.show()


