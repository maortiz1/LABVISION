#!/usr/bin/python3

# Modules needed for preprocessing
import ipdb
import numpy as np
import pandas as pd  
import tarfile
import zipfile
import os
import torch.nn as nn
import tqdm
import requests
from skimage import color
import urllib
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch
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
IMG_WIDTH = 224
IMG_HEIGHT = 224
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
       # print(x_.head())
        y_ = df_
        y_ = y_.drop(["image_id"],axis=1,inplace = True)
        y_ = y_.drop(["partition"],axis=1,inplace = True)
        #print(y_.head())
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

    x_=np.array(x_)
    x_=x_[:,np.newaxis]
    x_=torch.stack([torch.Tensor(i) for i in x_])
    
    y_=np.array(y_)
    y_=y_[:,np.newaxis]
    y_=torch.stack([torch.Tensor(i) for i in y_])
    return x_, y_




class Model(nn.Module):

    def __init__(self):

        super(Model,self).__init__()
        num_classes=10        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )     
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x   

def train(model,epochs):
   model.train()
   
  # Train data
   x_train, y_train = generate_df(0)
   # val data
   x_val, y_val = generate_df(1)
   
   train_dataset=train.utils.data.DataLoader(dataset=[x_train,y_train], batch_size=50,shuffle=True)
   optimizer = torch.optim.SGD(model.parameters(),lr=0.01,momentum=0.9,weight_decay=0.01)
   criterion = nn.BCELoss()
   
   for epoch in range(epochs):
     losser =[]
     train_precision =0
     print(enumerate(train_dataset))
     
     for batch_idx, (data,target) in tqdm(enumerate(train_dataset),total=len(train_dataset),desc="[TRAIN] Epoch{}".format(epoch)):
         data = data.to(device)
         t = target.type(torch.Tensor).squeeze(1).to(device)
         out = model(data)
         out = torch.sigmoid(out)
         loss = criterion(out,t)
         optimizer.zero_grad()
         loss.backward()
         optimizer.step()
         pred = torch.round(out)
         for k in range(len(pred)):
            for l in range(10):
               if pred[k][l]==target[k][l]:
                  train_precision +=1
         
         aux = loss.cpu()
         losser.append(aux.data.numpy())
     print("Loss: %0.3f"%(np.mean(losser)))
     print("Train Precicision",train_precision)   
#def test(model):





   

if __name__ == '__main__': 
   
   device=torch.device('cuda:1' if torch.cuda.is_available() else "cpu")
   model =Model().to(device)
   train(model,70)
   
    

    