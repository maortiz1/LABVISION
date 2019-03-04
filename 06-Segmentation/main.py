#!/usr/bin/ipython3

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
    if not os.path.isdir(folder):
     url='http://157.253.196.67/BSDS_small.zip'
     r=requests.get(url,allow_redirects=True)
     open('BSDS_small.zip','wb').write(r.content) 
     zip_ref = zipfile.ZipFile('BSDS_small.zip', 'r')
     zip_ref.extractall()
     zip_ref.close()
     
def segmentByClustering( rgbImage, colorSpace, clusteringMethod, numberOfClusters):
    import pandas as pd
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import LabelEncoder
    from sklearn.preprocessing import MinMaxScaler
    import seaborn as sns
    import matplotlib.pyplot as plt   
    from skimage import io, color
    import cv2
    
    # normalizing function
    def debugImg(rawData):
      toShow = np.zeros((rawData.shape), dtype=np.uint8)
      cv2.normalize(rawData, toShow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
      cv2.imwrite('img', toShow)
    
    #resize if it is hierarchical
    if clusteringMethod == "hierarchical"       
      rgbImage = cv2.resize(rgbImage, (0,0), fx=0.5, fy=0.5) 
      height = np.size(rgbImage, 0)
      width = np.size(rgbImage, 1)
    else
      height = np.size(rgbImage, 0)
      width = np.size(rgbImage, 1)
    
    #change to the specified color space
    if colorSpace == "lab"
      img = color.rgb2lab(rgbImage)    
      debugImg(img) 
    elif colorSpace == "hsv"
       img = color.rgb2hsv(rgbImage) 
       debugImg(img) 
    elif colorSpace == "rgb+xy"
    elif colorSpace == "lab+xy"
    elif colorSpace == "hsv+xy"
    else
      img = rgbImage
    
    #proceed to the specified clustering method
    if clusteringMethod == "kmeans"
      kmeans = KMeans(n_clusters=numberOfClusters).fit(img)
    elif clusteringMethod == "gmm"
      gmm = mixture.GaussianMixture( n_components=params[numberOfClusters]).fit(img)
    elif clusteringMethod == "hierarchical"
    else
    
    
    
    return segmentation

if __name__ == '__main__':
    import argparse
    import imageio
    from Segment import segmentByClustering 
    parser = argparse.ArgumentParser()

    parser.add_argument('--color', type=str, default='rgb', choices=['rgb', 'lab', 'hsv', 'rgb+xy', 'lab+xy', 'hsv+xy']) # If you use more please add them to this list.
    parser.add_argument('--k', type=int, default=4)
    parser.add_argument('--method', type=str, default='watershed', choices=['kmeans', 'gmm', 'hierarchical', 'watershed'])
    parser.add_argument('--img_file', type=str, required=True)
	
    opts = parser.parse_args()

    check_dataset(opts.img_file.split('/')[0])

    img = imageio.imread(opts.img_file)
    seg = segmentByClustering(rgbImage=img, colorSpace=opts.color, clusteringMethod=opts.method, numberOfClusters=opts.k)
    imshow(img, seg, title='Prediction')
    groundtruth(opts.img_file)
