
#!/usr/bin/ipython3
"""
Created on Tue Mar  5 12:49:03 2019

@author: m_ana
"""


from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
import os
import imageio as im
import numpy as np
pathImg=os.path.join('BSDS_small','train')
img= im.imread(os.path.join(pathImg,'12003.jpg'))
              
height = np.size(img, 0)
width = np.size(img, 1)
feat = img.reshape(height*width,3)  
kmeans = KMeans()

visualizer = KElbowVisualizer(kmeans, k=(2,20))


visualizer.fit(feat)    # Fit the data to the visualizer
visualizer.poof()    # Draw/show/poof the data

