#!/usr/bin/ipython3
def segmentByClustering( rgbImage, colorSpace, clusteringMethod, numberOfClusters):
   #module importation
     import pandas as pd
     import numpy as np
     from sklearn.cluster import KMeans
     import matplotlib.pyplot as plt   
     from skimage import io, color
     import cv2
     from sklearn.cluster import AgglomerativeClustering
     
     # scale from 0 to 1 in colorspace
     #rgbImage = rgbImage / 255.0;
     
     # normalizing function
     def debugImg(rawData):
       toShow = np.zeros((rawData.shape), dtype=np.uint8)
       cv2.normalize(rawData, toShow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
       cv2.imwrite('img', toShow)
     
     #resize if it is hierarchical
     if clusteringMethod=='hierarchical':       
       rgbImage = cv2.resize(rgbImage, (0,0), fx=0.5, fy=0.5) 
       height = np.size(rgbImage, 0)
       width = np.size(rgbImage, 1)
     else:
       height = np.size(rgbImage, 0)
       width = np.size(rgbImage, 1)
     
     #change to the specified color space
     if colorSpace == "lab":
       img = color.rgb2lab(rgbImage)    
       debugImg(img) 
     elif colorSpace == "hsv":
        img = color.rgb2hsv(rgbImage) 
        debugImg(img) 
   #  elif colorSpace == "rgb+xy"
   #  elif colorSpace == "lab+xy"
   #  elif colorSpace == "hsv+xy"
     else:
       img = rgbImage
     
     #proceed to the specified clustering method
     if clusteringMethod == "kmeans":
       feat = img.reshape(height*width,3)
       kmeans = KMeans(n_clusters=numberOfClusters).fit_predict(feat)
       segmentation = np.reshape(kmeans,(height,width))
       print(segmentation)
       plt.imshow(segmentation, cmap=plt.get_cmap('jet'))
       plt.show()
     elif clusteringMethod == "gmm":
       from sklearn import mixture
       feat = img.reshape(height*width,3)
       gmm = mixture.GaussianMixture(n_components=numberOfClusters).fit_predict(feat)
       segmentation = np.reshape(gmm,(height,width))
       print(segmentation)
       plt.imshow(segmentation, cmap=plt.get_cmap('jet'))
       plt.show()
     elif clusteringMethod == "hierarchical":
       feat = img.reshape(height*width,3)
       clustering = AgglomerativeClustering(n_clusters=numberOfClusters).fit_predict(feat)
       segmentation = np.reshape(clustering,(height,width))
       print(segmentation)
       plt.imshow(segmentation, cmap=plt.get_cmap('jet'))
       plt.show()
     else:
       gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
       ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
       # noise removal
       kernel = np.ones((3,3),np.uint8)
       opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
       
       # sure background area
       sure_bg = cv2.dilate(opening,kernel,iterations=3)
       
       # Finding sure foreground area
       dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
       ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
       
       # Finding unknown region
       sure_fg = np.uint8(sure_fg)
       unknown = cv2.subtract(sure_bg,sure_fg)
       # Marker labelling
       ret, markers = cv2.connectedComponents(sure_fg)
       
       # Add one to all labels so that sure background is not 0, but 1
       markers = markers+1
       
       # Now, mark the region of unknown with zero
       markers[unknown==255] = 0
       water = cv2.watershed(img,markers)
       segmentation = water 
       print(segmentation)
       plt.imshow(segmentation, cmap=plt.get_cmap('jet'))
       plt.show()   
       return segmentation