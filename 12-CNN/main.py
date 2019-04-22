#!/usr/bin/ipython3

# Modules needed
import ipdb
import numpy as np
import pandas as pd  
import tarfile
import zipfile
import os
import requests
from skimage import color
import urllib
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import os
from glob import glob
from PIL import Image

# Downloading dataset
URL='https://www.dropbox.com/s/pcyp3a2an1shj5c/celeba-dataset.zip?dl=1'
print('It will be proceed to download the database')
#checking if databse is already downloaded
if not(os.path.exists('celeba-dataset.zip')):
    urllib.request.urlretrieve(URL, "celeba-dataset.zip") 
    print('The database had been download')
else: 
    print('The file celeba-dataset.zip already exists')
    
print('It will be proceed to decompress de database')
#checking if database is already decompress

if not(os.path.exists('celeba-dataset')):
     zips = zipfile.ZipFile('celeba-dataset.zip','r')
     zips.extractall('celeba-dataset')
     zips.close()
if not(os.path.exists('celeba-dataset/img_align_celeba')):
     zip1 = zipfile.ZipFile('celeba-dataset/img_align_celeba.zip','r')
     print('Unzipping part1')
     zip1.extractall('celeba-dataset/img_align_celeba')
     zip1.close()
     
# Preprocessing     
#"/media/user_home2/vision/lmunar10/LABVISION/12-CNN/celeba-dataset/"
print(os.listdir("celeba-dataset/"))
data = pd.read_csv("celeba-dataset/list_attr_celeba.csv")
print(data.head())
#print(data.info())

# Loading and resizing images
#base_skin_dir = "celeba-dataset/" 
#imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x
                   #  for x in glob(os.path.join(base_skin_dir, '*', '*.jpg'))}
#data['path'] = data['image_id'].map(imageid_path_dict.get)
#from PIL import ImageFile
#ImageFile.LOAD_TRUNCATED_IMAGES = True
#data['image'] = data['path'].map(lambda x: np.asarray(Image.open(x).resize((150,150))))
#print(data.head())

# Se eliminan las columnas que no se van a usar para entrenar
data.drop(["5_o_Clock_Shadow"],axis=1,inplace = True)
data.drop(["Arched_Eyebrows"],axis=1,inplace = True)
data.drop(["Attractive"],axis=1,inplace = True)
data.drop(["Bags_Under_Eyes"],axis=1,inplace = True)
data.drop(["Bald"],axis=1,inplace = True)
data.drop(["Big_Lips"],axis=1,inplace = True)
data.drop(["Big_Nose"],axis=1,inplace = True)
data.drop(["Blurry"],axis=1,inplace = True)
data.drop(["Bushy_Eyebrows"],axis=1,inplace = True)
data.drop(["Chubby"],axis=1,inplace = True)
data.drop(["Double_Chin"],axis=1,inplace = True)
data.drop(["Goatee"],axis=1,inplace = True)
data.drop(["Heavy_Makeup"],axis=1,inplace = True)
data.drop(["High_Cheekbones"],axis=1,inplace = True)
data.drop(["Mouth_Slightly_Open"],axis=1,inplace = True)
data.drop(["Mustache"],axis=1,inplace = True)
data.drop(["Narrow_Eyes"],axis=1,inplace = True)
data.drop(["No_Beard"],axis=1,inplace = True)
data.drop(["Oval_Face"],axis=1,inplace = True)
data.drop(["Pointy_Nose"],axis=1,inplace = True)
data.drop(["Receding_Hairline"],axis=1,inplace = True)
data.drop(["Rosy_Cheeks"],axis=1,inplace = True)
data.drop(["Sideburns"],axis=1,inplace = True)
data.drop(["Straight_Hair"],axis=1,inplace = True)
data.drop(["Wavy_Hair"],axis=1,inplace = True)
data.drop(["Wearing_Earrings"],axis=1,inplace = True)
data.drop(["Wearing_Hat"],axis=1,inplace = True)
data.drop(["Wearing_Necktie"],axis=1,inplace = True)
data.drop(["Wearing_Lipstick"],axis=1,inplace = True)
data.drop(["Wearing_Necklace"],axis=1,inplace = True)
print(data.head())
#xdata = data['image']
#data.drop(["image"],axis=1,inplace = True)
#ydata = data
#print(ydata.head())