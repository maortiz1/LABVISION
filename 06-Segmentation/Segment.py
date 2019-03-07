#!/usr/bin/ipython3
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
     print (type(rgbImage))
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
       rgbImage = cv2.resize(rgbImage, (0,0), fx=0.5, fy=0.5) 
       height = np.size(rgbImage, 0)
       width = np.size(rgbImage, 1)
     else:
       height = np.size(rgbImage, 0)
       width = np.size(rgbImage, 1)
     img=rgbImage
     #change to the specified color space
     if colorSpace == "lab":
      img_lab = color.rgb2lab(rgbImage)    
      debugImg(img_lab)
#      l = img_lab[:,:,0]
#      a = img_lab[:,:,1]
#      b = img_lab[:,:,2]
#      sum = l+a+b
#      sum = sum/3
      img =img_lab
     elif colorSpace == "hsv":
      img_hsv = color.rgb2hsv(rgbImage)    
      debugImg(img_hsv)
#      h = img_hsv[:,:,0]
#      s = img_hsv[:,:,1]
#      v = img_hsv[:,:,2]
#      sum = h+s+v
#      sum = sum/3
#      img = sum
      img =img_hsv
     elif colorSpace == "rgb+xy":
      r = rgbImage[:,:,0]
      g = rgbImage[:,:,1]
      b = rgbImage[:,:,2]
#      img_xyz = color.rgb2xyz(rgbImage)
#      x = img_xyz[:,:,0]
#      y = img_xyz[:,:,1]
      xyimg=xy(rgbImage)
      #img = np.concatenate((r,g,b, x, y), axis=0)
#      sum = r+g+b+x+y
#      sum = sum/5
#      img = sum
      
     elif colorSpace == "lab+xy":
      img_lab = color.rgb2lab(rgbImage)    
      debugImg(img_lab)
#      l = img_lab[:,:,0]
#      a = img_lab[:,:,1]
#      b = img_lab[:,:,2]
#      img_xyz = color.lab2xyz(img_lab)
#      x = img_xyz[:,:,0]
#      y = img_xyz[:,:,1]
#      sum = l+a+b+x+y
#      sum = sum/5
#      #img = np.concatenate((l,a,b, x, y), axis=0)
      img = img_lab
      xyimg=xy(img_lab)
     elif colorSpace == "hsv+xy":
      img_hsv = color.rgb2hsv(rgbImage)    
      debugImg(img_hsv)
#      h = img_hsv[:,:,0]
#      s = img_hsv[:,:,1]
#      v = img_hsv[:,:,2]
#      img_xyz = color.hsv2xyz(img_hsv)
#      x = img_xyz[:,:,0]
#      y = img_xyz[:,:,1]
#      #img = np.concatenate((h,s,v, x, y), axis=0)
#      sum = h+s+v+x+y
#      sum = sum/5
#      img = sum
      img=img_hsv
      xyimg=xy(img)
     else:
       img = rgbImage
       img = color.rgb2gray(img)
     # preparation to classifiers
     
     
     
     
     #proceed to the specified clustering method
     f=img
     img=merge(f,xyimg)
     if clusteringMethod == "kmeans":
       feat = img.reshape(height*width,1)
       kmeans = KMeans(n_clusters=numberOfClusters).fit_predict(feat)
       segmentation = np.reshape(kmeans,(height,width))

     elif clusteringMethod == "gmm":
       from sklearn import mixture
       feat = img.reshape(height*width,1)
       gmm = mixture.GaussianMixture(n_components=numberOfClusters).fit_predict(feat)
       segmentation = np.reshape(gmm,(height,width))

     elif clusteringMethod == "hierarchical":
       feat = img.reshape(height*width,1)
       clustering = AgglomerativeClustering(n_clusters=numberOfClusters).fit_predict(feat)
       segmentation = np.reshape(clustering,(height,width))
      
     else:
        from skimage import morphology
        from skimage import feature
        import skimage
#        img = color.rgb2gray(img)
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        # Compute gradient magnitude
        grad_magn = np.sqrt(sobelx**2 + sobely**2)
        # Put it in [0, 255] value range
        grad_magn = 255*(grad_magn - np.min(grad_magn)) / (np.max(grad_magn) - np.min(grad_magn))
        #ipdb.set_trace()
#        selem = morphology.disk(5)
#        opened = morphology.opening(img, selem)
#        eroded = morphology.erosion(img, selem)
#        opening_recon = morphology.reconstruction(seed=eroded, mask=img, method='dilation')
#        closed_opening = morphology.closing(opened, selem)
#        dilated_recon_dilation = morphology.dilation(opening_recon, selem)
#        recon_erosion_recon_dilation = morphology.reconstruction(dilated_recon_dilation,        opening_recon,method='erosion').astype(np.uint8)
        
        def imregionalmax(img, ksize=3):
           filterkernel = np.ones((ksize, ksize)) # 8-connectivity
           reg_max_loc = feature.peak_local_max(img,footprint=filterkernel, indices=False, exclude_border=0)
           return reg_max_loc.astype(np.uint8)
         # Mari aqui es donde se imponen los minimos
#        foreground_1 = imregionalmax(recon_erosion_recon_dilation, ksize=65)
#        fg_superimposed_1 = img.copy()
#        fg_superimposed_1[foreground_1 == 1] = 255
        
        imagenW = grad_magn.copy()
        m=np.amax(grad_magn)
        
        mi=np.amin(grad_magn)
        posi=np.arange(mi,m)
        print(posi)
        if numberOfClusters>len(posi):
            imagenW[img<posi[-1]]=0
            
            
        else:         
            imagenW[img<posi[-k]]=0
        
        
#        fg_superimposed_1[img]
        _, labeled_fg = cv2.connectedComponents(imagenW.astype(np.uint8))
        labels = morphology.watershed(grad_magn, labeled_fg)
#        superimposed = img.copy()
#        watershed_boundaries = skimage.segmentation.find_boundaries(labels)
#        superimposed[watershed_boundaries] = 255
#        superimposed[foreground_1] = 255
        segmentation = labels
        
 
     return segmentation