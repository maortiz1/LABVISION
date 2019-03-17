#!/usr/bin/ipython3
import ipdb
def segmentByClustering( rgbImage, colorSpace, clusteringMethod, numberOfClusters):
   #module importation
     import pandas as pd
     import numpy as np
     from sklearn.cluster import KMeans
     import matplotlib.pyplot as plt   
     from skimage import io, color
     import cv2
     import ipdb
     from sklearn.cluster import AgglomerativeClustering
     xyimg=[]
     # normalizing function
     def debugImg(rawData):
       toShow = np.zeros((rawData.shape), dtype=np.uint8)
       cv2.normalize(rawData, toShow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
#       cv2.imwrite('img.jpg', toShow)
     def xy(img):
       height = np.size(img, 0)
       width = np.size(img, 1)
       mat=np.zeros((height,width,2))
       mat[::,::,1]=(mat[::,::,1]+np.arange(width)[np.newaxis,:])/width       
       mat[::,::,0]=(mat[::,::,0]+np.arange(height)[:,np.newaxis])/height
       return mat
   
    
     def merge(img,xy):
         im=np.sum(img,axis=-1)
         xysum=np.sum(xy,axis=-1)
         fin=np.add(im,xysum)/5
         return fin
     #resize if it is hierarchical
     if clusteringMethod=='hierarchical':       
#       rgbImage = cv2.resize(rgbImage, (0,0), fx=0.5, fy=0.5) 
       height = np.size(rgbImage, 0)
       width = np.size(rgbImage, 1)
     else:
      # ipdb.set_trace()
       height = np.size(rgbImage, 0)
       width = np.size(rgbImage, 1)
     img=rgbImage
     #change to the specified color space
     if colorSpace == "lab":
      img_lab = color.rgb2lab(rgbImage)    
      debugImg(img_lab)

      img =img_lab
     elif colorSpace == "hsv":
      img_hsv = color.rgb2hsv(rgbImage)    
      debugImg(img_hsv)
      img =img_hsv
     elif colorSpace == "rgb+xy":
      r = rgbImage[:,:,0]
      g = rgbImage[:,:,1]
      b = rgbImage[:,:,2]
      xyimg=xy(rgbImage)
      
     elif colorSpace == "lab+xy":
      img_lab = color.rgb2lab(rgbImage)    
      debugImg(img_lab)
      img = img_lab
      xyimg=xy(img_lab)
     elif colorSpace == "hsv+xy":
      img_hsv = color.rgb2hsv(rgbImage)    
      debugImg(img_hsv)
      img=img_hsv
      xyimg=xy(img)
     else:
       img = rgbImage
#       img = color.rgb2gray(img)
     # preparation to classifiers
     
     
     
     
     #proceed to the specified clustering method
     f=img
#     img=merge(f,xyimg)

     debugImg(img)

     if clusteringMethod == "kmeans":
       feat = img.reshape(height*width,3)
       kmeans = KMeans(n_clusters=numberOfClusters).fit_predict(feat)
       segmentation = np.reshape(kmeans,(height,width))

     elif clusteringMethod == "gmm":
       from sklearn import mixture
       feat = img.reshape(height*width,3)
       gmm = mixture.GaussianMixture(n_components=numberOfClusters).fit_predict(feat)
       segmentation = np.reshape(gmm,(height,width))

     elif clusteringMethod == "hierarchical":
       feat = img.reshape(height*width,1)
       clustering = AgglomerativeClustering(n_clusters=numberOfClusters).fit_predict(feat)
       segmentation = (np.reshape(clustering,(height,width)))

#       segmentation=cv2.resize(segmentation, None, fx = 2, fy = 2, interpolation = cv2.INTER_CUBIC)  
     else:
        from skimage import morphology
        from skimage import feature
        import skimage
        img = color.rgb2gray(img)
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        # Compute gradient magnitude
        grad_magn = np.sqrt(sobelx**2 + sobely**2)
        debugImg(grad_magn)
        
        import matplotlib.pyplot as plt


        imagenW=grad_magn
        
        imagenW=grad_magn

        found=100000
        minimum=found
        while(minimum!=numberOfClusters): 
            imagenW=morphology.h_maxima(grad_magn,found)
            _, labeled_fg = cv2.connectedComponents(imagenW.astype(np.uint8))
            print(len(np.unique(labeled_fg)))
            labels = morphology.watershed(grad_magn, labeled_fg)
            
            found=found-1;
            minimum=len(np.unique(labels))
            print(minimum)
 
            
            

#        plt.figure()
#        plt.imshow(labeled_fg)
#        print(labeled_fg)
        segmentation = labels
        
 
     return segmentation