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
from sklearn.model_selection import StratifiedShuffleSplit
from glob import glob
import seaborn as sns
from torch.utils import data
import torch
from PIL import Image

import pdb

torch.manual_seed(42)
np.random.seed(42)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    
    plt.figure
    


    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, round(cm[i, j],3),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    plt.savefig('confusionmatrix_prueba.png', dpi=300)




# Downloading Training Dataset
URL='https://www.dropbox.com/s/9ybbu805dwh8258/skin-cancer-mnist-ham10000.zip?dl=1'
print('It will be proceed to download the training database')
#checking if databse is already downloaded
if not(os.path.exists('skin-cancer-mnist-ham10000.zip')):
    urllib.request.urlretrieve(URL, "skin-cancer-mnist-ham10000.zip") 
    print('The training database had been download')
else: 
    print('The file skin-cancer-mnist-ham10000.zip already exists')
    
print('It will be proceed to decompress de training database')
#checking if database is already decompress
if not(os.path.exists('skin-cancer-mnist-ham10000')):
     zips = zipfile.ZipFile('skin-cancer-mnist-ham10000.zip','r')
     zips.extractall('skin-cancer-mnist-ham10000')
     zips.close()

#os.chdir("skin-cancer-mnist-ham10000/")
if not(os.path.exists('skin-cancer-mnist-ham10000/HAM10000_images_part_1')):
   zip1 = zipfile.ZipFile('skin-cancer-mnist-ham10000/HAM10000_images_part_1.zip','r')
   print('Unzipping part1')
   zip1.extractall('skin-cancer-mnist-ham10000/HAM10000_images_part_1')
   zip1.close()
if not(os.path.exists('skin-cancer-mnist-ham10000/HAM10000_images_part_2')):
   zip2 = zipfile.ZipFile('skin-cancer-mnist-ham10000/HAM10000_images_part_2.zip','r')
   print('Unzipping part2')
   zip2.extractall('skin-cancer-mnist-ham10000/HAM10000_images_part_2')
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
cate= pd.Categorical(skin_df['cell_type'])
skin_df['cell_type_idx'] = pd.Categorical(skin_df['cell_type']).codes   

skin_df[['cell_type_idx', 'cell_type']].sort_values('cell_type_idx').drop_duplicates()
print (skin_df['cell_type'].value_counts())
# Data cleaning. Checking for missing values in csv file
#print(skin_df.isnull().sum())
skin_df['age'].fillna((skin_df['age'].mean()), inplace=True) 
#print(skin_df.isnull().sum()) # no null files
#print(skin_df.dtypes) #check types in array
# Loading and resizing images
#from PIL import ImageFile
#ImageFile.LOAD_TRUNCATED_IMAGES = True
#skin_df['image'] = skin_df['path'].map(lambda x: np.asarray(Image.open(x).resize((150,150))))
#print(skin_df['path'])

#print('Checking the image size distribution')
#skin_df['image'].map(lambda x: x.shape).value_counts()
#skin_df.drop(columns=['cell_type_idx'],axis=1)
#features=skin_df['image']
target=skin_df['cell_type_idx']

print('Dividing train y test set')

# x es data y y es la anotacion
x_train_o, x_test_o, y_train_o, y_test_o = train_test_split(skin_df, target, test_size=0.20,random_state=0,stratify=target)
#print(target)
#print(y_train_o)
#print(y_test_o)

print(' Normalization')
#x_train = np.asarray(x_train_o['image'].tolist())
#x_test = np.asarray(x_test_o['image'].tolist())

#x_train_mean = np.mean(x_train)
#x_train_std = np.std(x_train)

#x_test_mean = np.mean(x_test)
#x_test_std = np.std(x_test)

#x_train = (x_train - x_train_mean)/x_train_std
#x_test = (x_test - x_test_mean)/x_test_std

print('Encode labels')
# Volver un numero las clases 
#y_train = to_categorical(y_train_o, num_classes = 7)
#y_test = to_categorical(y_test_o, num_classes = 7)
#print(y_train[0])
#y_train = pd.factorize(y_train_o)[0]
x_train=x_train_o
y_train=y_train_o
#print(y_train)
print(y_train.shape)
#y_test = pd.factorize(y_test_o)[0]
x_test = x_test_o
y_test=y_test_o
print('Separating validation set')
import collections

countertrain=(collections.Counter(y_train).values())
print(collections.Counter(y_train))
print(countertrain)
countertest=((collections.Counter(y_test).values()))
print(collections.Counter(y_test))
print(countertest)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.1, random_state = 0,stratify=y_train)
countertest=((collections.Counter(y_train).values()))
print(collections.Counter(y_train))
print(countertest)
countertest=((collections.Counter(y_val).values()))
print(collections.Counter(y_val))
print(countertest)

x_train = x_train.reset_index()
x_val = x_val.reset_index()
x_test = x_test.reset_index()

# CNN model
import torchvision.models as models
model_conv = models.resnet50(pretrained=True)
#print(model_conv)
#print(model_conv.fc)
num_ftrs = model_conv.fc.in_features
model_conv.fc = torch.nn.Linear(num_ftrs, 7)
#print(model_conv.fc)

# Define the device:
device = torch.device('cuda:2')

# Put the model on the device:
model = model_conv.to(device)

class Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, df, transform=None):
        'Initialization'
        self.df = df
        self.transform = transform

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.df)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Load data and get label
        #print(self.df['path'][index])
        X = Image.open(self.df['path'][index])
        y = torch.tensor(int(self.df['cell_type_idx'][index]))

        if self.transform:
            X = self.transform(X)

        return X, y

# Define the parameters for the dataloader
params = {'batch_size': 4,
          'shuffle': True,
          'num_workers': 0}

# define the transformation of the images.
import torchvision.transforms as trf
composed = trf.Compose([trf.RandomHorizontalFlip(), trf.RandomVerticalFlip(), trf.CenterCrop(256), trf.RandomCrop(224),  trf.ToTensor(),
                        trf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
                        
# Define the trainingsset using the table train_df and using our defined transitions (composed)
training_set = Dataset(x_train, transform=composed)
training_generator = data.DataLoader(training_set, **params)

# Same for the validation set:
validation_set = Dataset(x_val, transform=composed)
validation_generator = data.DataLoader(validation_set, **params)
#Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.0000001)
criterion = torch.nn.CrossEntropyLoss()

# Train

max_epochs = 300
trainings_error = []
validation_error = []
for epoch in range(max_epochs):
    print('epoch:', epoch)
    count_train = 0
    trainings_error_tmp = []
    model.train()
    for data_sample, y in training_generator:
        data_gpu = data_sample.to(device)
        y_gpu = y.to(device)
        output = model(data_gpu)
        err = criterion(output, y_gpu)
        err.backward()
        optimizer.step()
        trainings_error_tmp.append(err.item())
        count_train += 1
        if count_train >= 100:
            count_train = 0
            mean_trainings_error = np.mean(trainings_error_tmp)
            trainings_error.append(mean_trainings_error)
            print('trainings error:', mean_trainings_error)
            break
    with torch.set_grad_enabled(False):
        validation_error_tmp = []
        count_val = 0
        model.eval()
        for data_sample, y in validation_generator:
            data_gpu = data_sample.to(device)
            y_gpu = y.to(device)
            output = model(data_gpu)
            err = criterion(output, y_gpu)
            validation_error_tmp.append(err.item())
            count_val += 1
            if count_val >= 10:
                count_val = 0
                mean_val_error = np.mean(validation_error_tmp)
                validation_error.append(mean_val_error)
                print('validation error:', mean_val_error)
                break


#plt.plot(trainings_error, label = 'training error')
#plt.plot(validation_error, label = 'validation error')
#plt.legend()
#plt.show()


model.eval()
test_set = Dataset(x_test, transform=composed)
#test_generator = data.DataLoader(test_set, **params)
test_generator = data.SequentialSampler(test_set)

result_array = []
gt_array = []
for i in test_generator:
    data_sample, y = test_set.__getitem__(i)
    data_gpu = data_sample.unsqueeze(0).to(device)
    output = model(data_gpu)
    result = torch.argmax(output)
    result_array.append(result.item())
    gt_array.append(y.item())
    
correct_results = np.array(result_array)==np.array(gt_array)
sum_correct = np.sum(correct_results)
accuracy = sum_correct/test_generator.__len__()
print(accuracy)
#confusion_mtx = confusion_matrix(gt_array,result_array)
#plot_confusion_matrix(confusion_mtx, classes = range(7),title='Test Confusion Matrix',normalize=True) 
