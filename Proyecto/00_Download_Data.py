#!/usr/bin/ipython3
import numpy as np
import tarfile
import zipfile
import os
import requests
from skimage import color
import urllib
#matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from glob import glob
import seaborn as sns
from PIL import Image
np.random.seed(123)
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import itertools
import keras
from keras.utils.np_utils import to_categorical
# Downloading Dataset
URL='https://www.dropbox.com/s/9ybbu805dwh8258/skin-cancer-mnist-ham10000.zip?dl=1'
print('It will be proceed to download the database')
#checking if databse is already downloaded
if not(os.path.exists('skin-cancer-mnist-ham10000.zip')):
    urllib.request.urlretrieve(URL, "skin-cancer-mnist-ham10000.zip") 
    print('The database had been download')
else: 
    print('The file skin-cancer-mnist-ham10000.zip already exists')
    
print('It will be proceed to decompress de database')
#checkinf if database is already decompress
if not(os.path.exists('skin-cancer-mnist-ham10000')):
     zips = zipfile.ZipFile('skin-cancer-mnist-ham10000.zip','r')
     zips.extractall('skin-cancer-mnist-ham10000')
     zips.close()

#os.chdir("skin-cancer-mnist-ham10000/")
if not(os.path.exists('skin-cancer-mnist-ham10000/HAM10000_images_part_1')):
   zip1 = zipfile.ZipFile('HAM10000_images_part_1.zip','r')
   print('Unzipping part1')
   zip1.extractall('HAM10000_images_part_1')
   zip1.close()
if not(os.path.exists('skin-cancer-mnist-ham10000/HAM10000_images_part_2')):
   zip2 = zipfile.ZipFile('HAM10000_images_part_2.zip','r')
   print('Unzipping part2')
   zip2.extractall('HAM10000_images_part_2')
   zip2.close()
#os.chdir(..)
# Reading and preparation of the data
base_skin_dir = "/media/user_home2/vision/lmunar10/LABVISION/Proyecto/skin-cancer-mnist-ham10000/" # Mari esto es lo unico que no quedó eficiente porque no supe como poner para que quedara el path completo jaja no me funcionó ~/
#print (base_skin_dir)
#"/media/user_home2/vision/lmunar10/LABVISION/Proyecto/skin-cancer-mnist-ham10000/HAM10000_metadata.csv"
# Merging images from both folders HAM10000_images_part1.zip and HAM10000_images_part2.zip into one dictionary
imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x
                     for x in glob(os.path.join(base_skin_dir, '*', '*.jpg'))}
#print(imageid_path_dict)
# This dictionary is useful for displaying more human-friendly labels later on
lesion_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis-like lesions ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}
skin_df = pd.read_csv(os.path.join(base_skin_dir,'HAM10000_metadata.csv'))

# Creating New Columns for better readability

skin_df['path'] = skin_df['image_id'].map(imageid_path_dict.get)
skin_df['cell_type'] = skin_df['dx'].map(lesion_type_dict.get) 
skin_df['cell_type_idx'] = pd.Categorical(skin_df['cell_type']).codes

print(skin_df.head())

# Data cleaning. Checking for missing values in csv file
#print(skin_df.isnull().sum())
skin_df['age'].fillna((skin_df['age'].mean()), inplace=True) 
#print(skin_df.isnull().sum()) # no null files
#print(skin_df.dtypes) #check types in array

# Loading and resizing images
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
skin_df['image'] = skin_df['path'].map(lambda x: np.asarray(Image.open(x).resize((150,150))))
print(skin_df.head())

# Plot some example images

n_samples = 5
fig, m_axs = plt.subplots(7, n_samples, figsize = (4*n_samples, 3*7))
for n_axs, (type_name, type_rows) in zip(m_axs, 
   skin_df.sort_values(['cell_type']).groupby('cell_type')):
    n_axs[0].set_title(type_name)
    for c_ax, (_, c_row) in zip(n_axs, type_rows.sample(n_samples, random_state=1234).iterrows()):
        c_ax.imshow(c_row['image'])
        c_ax.axis('off')
fig.savefig('category_samples.png', dpi=300)

print('Checking the image size distribution')
skin_df['image'].map(lambda x: x.shape).value_counts()
features=skin_df.drop(columns=['cell_type_idx'],axis=1)
target=skin_df['cell_type_idx']

print('Dividing train y test set')

x_train_o, x_test_o, y_train_o, y_test_o = train_test_split(features, target, test_size=0.20,random_state=1234)

print(' Normalization')
x_train = np.asarray(x_train_o['image'].tolist())
x_test = np.asarray(x_test_o['image'].tolist())

x_train_mean = np.mean(x_train)
x_train_std = np.std(x_train)

x_test_mean = np.mean(x_test)
x_test_std = np.std(x_test)

x_train = (x_train - x_train_mean)/x_train_std
x_test = (x_test - x_test_mean)/x_test_std
print('Encode labels')
y_train = to_categorical(y_train_o, num_classes = 7)
y_test = to_categorical(y_test_o, num_classes = 7)

print('Separating validation set')
x_train, x_validate, y_train, y_validate = train_test_split(x_train, y_train, test_size = 0.1, random_state = 2)
print('Reshape 3D image')
x_train = x_train.reshape(x_train.shape[0], *(150, 150, 3))
x_test = x_test.reshape(x_test.shape[0], *(150, 150, 3))
x_validate = x_validate.reshape(x_validate.shape[0], *(150, 150, 3))