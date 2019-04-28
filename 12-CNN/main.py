#!/usr/bin/python3

# Modules needed for preprocessing
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
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import cv2

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

# Loading and resizing images
base_skin_dir = "celeba-dataset/img_align_celeba/img_align_celeba/"
#filenames=os.listdir(base_skin_dir)
#imgs=[]
#for filenam in filenames:
#  filenam2,ext =os.path.splitext(filenam) 
#  if ext=='.jpg':
#     temp=cv2.imread(os.path.join(base_skin_dir, filenam))
#     temp=cv2.resize(temp,(300,300))
#     imgs.append(temp)
#data['image'] = imgs
#print(data.head())

# Ya en data tengo las imagenes y las anotaciones que quiero. Ahora las voy a separar para poder entrenar la red
#xdata = data['image']
#data.drop(["image"],axis=1,inplace = True)
#data.drop(["image_id"],axis=1,inplace = True)
#ydata = data

#from sklearn.model_selection import train_test_split
#x_train_o, x_test_o, y_train, y_test = train_test_split(xdata,ydata,test_size = 0.3,random_state=42)

#x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.1, random_state = 0,stratify=y_train)



main_folder = '../celeba-dataset/'
images_folder = base_skin_dir

TRAINING_SAMPLES = 162770
VALIDATION_SAMPLES = 1986
TEST_SAMPLES = 19962
IMG_WIDTH = 178
IMG_HEIGHT = 218
BATCH_SIZE = 16
NUM_EPOCHS = 20

df_partition = pd.read_csv('celeba-dataset/list_eval_partition.csv')
print(df_partition.head())
print ('Distribution train val and test')
# display counter by partition
# 0 -> TRAINING
# 1 -> VALIDATION
# 2 -> TEST
print(df_partition['partition'].value_counts().sort_index())
data['partition'] = df_partition['partition']
print(data.head())

# Separar en train val y test. Cada set tiene un folder aparte

def load_reshape_img(fname):
    img = load_img(fname)
    x = img_to_array(img)/255.
    x = x.reshape((1,) + x.shape)

    return x

def generate_df(partition):
    '''
    partition
        0 -> train
        1 -> validation
        2 -> test
    
    '''
    
    df_ = data[(data['partition'] == partition)]
    # for Train and Validation
    if partition != 2:
        #ipdb.set_trace()
        x_ = np.array([load_reshape_img(images_folder+df_['image_id'][fname]) for fname in df_.index])

        x_ = x_.reshape(x_.shape[0], 218, 178, 3)
        print(x_.head())
        y_ = df_
        y_ = y_.drop(["image_id"],axis=1,inplace = True)
        y_ = y_.drop(["partition"],axis=1,inplace = True)
        print(y_.head())
    # for Test
    else:
        x_ = []
        y_ = []
        #df_.drop(["image_id"],axis=1,inplace = True)
        #df_.drop(["partition"],axis=1,inplace = True)
        for index, target in df_.iterrows():
            im = cv2.imread(images_folder +df_['image_id'][index])
            im = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (IMG_WIDTH, IMG_HEIGHT)).astype(np.float32) / 255.0
            im = np.expand_dims(im, axis =0)
            x_.append(im)
            y_.append(target)
           # print(x_)
            #print(y_)

    return x_, y_

# Train data
x_train, y_train = generate_df(0)
# val data
x_val, y_val = generate_df(1)

def get_data(batch_size):
    #transform_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    transform_train = transforms.Compose([transforms.ToTensor()])
    data_train = datasets.MNIST('data', train=True, transform = transform_train)
    train_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True)

    data_test = datasets.MNIST('data', train=False, transform = transform_train)
    test_loader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
