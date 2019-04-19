#!/usr/bin/python3

#Modules to use
import ipdb
import numpy as np
import pandas as pd  
import tarfile
import zipfile
import os
import requests
from skimage import color
import urllib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from glob import glob
import seaborn as sns
from PIL import Image
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import itertools
import keras
from keras.utils.np_utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler


# Function to plot confusion matrix    
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confusionmatrix.png', dpi=300)


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
#checking if database is already decompress
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
base_skin_dir = "/media/user_home2/vision/lmunar10/LABVISION/Proyecto/skin-cancer-mnist-ham10000/" # Mari esto es lo unico que no quedo eficiente porque no supe como poner para que quedara el path completo jaja no me funciono 
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

#n_samples = 5
#fig, m_axs = plt.subplots(7, n_samples, figsize = (4*n_samples, 3*7))
#for n_axs, (type_name, type_rows) in zip(m_axs, 
#   skin_df.sort_values(['cell_type']).groupby('cell_type')):
#    n_axs[0].set_title(type_name)
#    for c_ax, (_, c_row) in zip(n_axs, type_rows.sample(n_samples, random_state=1234).iterrows()):
#        c_ax.imshow(c_row['image'])
#        c_ax.axis('off')
#fig.savefig('category_samples.png', dpi=300)

print('Checking the image size distribution')
skin_df['image'].map(lambda x: x.shape).value_counts()
features=skin_df.drop(columns=['cell_type_idx'],axis=1)
#features=skin_df['image']
target=skin_df['cell_type_idx']

print('Dividing train y test set')
# x es data y y es la anotacion
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
# Volver un numero las clases 
#y_train = to_categorical(y_train_o, num_classes = 7)
#y_test = to_categorical(y_test_o, num_classes = 7)
#print(y_train[0])
y_train = pd.factorize(y_train_o)[0]
print(y_train)
print(y_train.shape)
y_test = pd.factorize(y_test_o)[0]
print('Separating validation set')
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.1, random_state = 2)
print('Reshape 3D image')
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)
x_val = x_val.reshape(x_val.shape[0], -1)

print('Number of observations in the training data:', len(x_train))
print('Number of observations in the validation data:',len(x_val))
print('Number of observations in the test data:',len(x_test))


print('Classifier RF')
print(x_train.shape)
print(y_train.shape)
# Training the classifier
RF = RandomForestClassifier(n_jobs=2, random_state=0)
RF.fit(x_train, y_train)
# Predict validation data
y_pred = RF.predict(x_val)
#print(RF.predict_proba(x_val)[0:10]) # view first 10 prediction
confusion_mtx = confusion_matrix(y_val,y_pred)
print(confusion_mtx)  
print(classification_report(y_val,y_pred))  
print(accuracy_score(y_val, y_pred))  
# plot the confusion matrix
plot_confusion_matrix(confusion_mtx, classes = range(7)) 