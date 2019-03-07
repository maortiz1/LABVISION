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
    import imageio
    img = imageio.imread(img_file)
    gt=sio.loadmat(img_file.replace('jpg', 'mat'))
    segm=gt['groundTruth'][0,1][0][0]['Segmentation']
    imshow(img, segm, title='Groundtruth')
    return segm
    
import imageio as im
import os
import Segment
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
#import opencv as cv2

pathImg=os.path.join('BSDS_small','train')
img1= im.imread(os.path.join(pathImg,'12003.jpg'))

seg = Segment.segmentByClustering(rgbImage=img1, colorSpace='hsv', clusteringMethod='gmm', numberOfClusters=4)



plt.figure()
imshow(img1,seg,title='preduction')
plt.figure()
ground = groundtruth(os.path.join(pathImg,'12003.jpg'))




unqseg = np.unique(seg)
unqground = np.unique(ground)

mat=np.zeros([len(unqseg),len(unqground)])


for segV,k in zip(unqseg,range(0,len(unqseg))):
    for groundV,j in zip(unqground,range(0,len(unqground))):
        ac=np.zeros(seg.shape)
        ac[seg==segV]=1        
        ac2=np.zeros(ground.shape)
        ac2[ground==groundV]=1
        join = np.count_nonzero(np.logical_and(ac2,ac)==True)
        lolor= np.count_nonzero(np.logical_or(ac2,ac)==True)
        
        mat[k][j]=join/lolor        
        print('['+str(segV)+','+str(groundV)+'] = '+ str(join/lolor))

#mat=mat.astype('float')/mat.sum(axis=1)[:,np.newaxis]
        
plt.figure()
plt.imshow(mat,cmap='Pastel1',interpolation='nearest')
plt.title('Matrix')

plt.colorbar()
plt.show()



pathImg=os.path.join('BSDS_small','train')
img1= im.imread(os.path.join(pathImg,'12003.jpg'))

seg = Segment.segmentByClustering(rgbImage=img1, colorSpace='hsv+xy', clusteringMethod='gmm', numberOfClusters=4)


