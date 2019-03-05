#!/usr/bin/ipython3
import ipdb 
def imshow(img, seg, title='Image'):
    import matplotlib.pyplot as plt
    plt.imshow(img, cmap=plt.get_cmap('gray'))
    plt.imshow(seg, cmap=plt.get_cmap('rainbow'), alpha=0.5)
    cb = plt.colorbar()
    cb.set_ticks(range(seg.max()+1))
    plt.title(title)
    plt.axis('off')
    plt.show()
    
def groundtruth(img_file):
    import scipy.io as sio
    img = imageio.imread(img_file)
    gt=sio.loadmat(img_file.replace('jpg', 'mat'))
    segm=gt['groundTruth'][0,5][0][0]['Segmentation']
    imshow(img, segm, title='Groundtruth')
    
def check_dataset(folder):
    import os
    import zipfile
    import requests  
    if not os.path.isdir(folder):
     url='http://157.253.196.67/BSDS_small.zip'
     r=requests.get(url,allow_redirects=True)
     open('BSDS_small.zip','wb').write(r.content) 
     zip_ref = zipfile.ZipFile('BSDS_small.zip', 'r')
     zip_ref.extractall()
     zip_ref.close()  

if __name__ == '__main__':
    import argparse
    import imageio
    import os
    import cv2
    from Segment import segmentByClustering 
    parser = argparse.ArgumentParser()

    parser.add_argument('--color', type=str, default='rgb', choices=['rgb', 'lab', 'hsv', 'rgb+xy', 'lab+xy', 'hsv+xy']) # If you use more please add them to this list.
    parser.add_argument('--k', type=int, default=4)
    parser.add_argument('--method', type=str, default='watershed', choices=['kmeans', 'gmm', 'hierarchical', 'watershed'])
    parser.add_argument('--img_file', type=str, required=True)
	
    opts = parser.parse_args()
    check_dataset(opts.img_file.split('/')[0])
    import numpy as np
    img = imageio.imread(os.path.join("BSDS_small/train/", opts.img_file))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    seg = segmentByClustering(rgbImage=img, colorSpace=opts.color, clusteringMethod=opts.method, numberOfClusters=opts.k)
#    ipdb.set_trace()
    imshow(img, seg, title='Prediction')
    groundtruth(opts.img_file)
