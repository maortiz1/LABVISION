from PIL import Image

import os
from keras.preprocessing import image
from keras.models import Model, Sequential
from keras.layers import Activation, Dense, GlobalAveragePooling2D, BatchNormalization, Dropout, Conv2D, Conv2DTranspose, AveragePooling2D, MaxPooling2D, UpSampling2D, Input, Reshape
from keras import backend as K
from keras.optimizers import Nadam, Adam, SGD
from keras.metrics import categorical_accuracy, binary_accuracy
#from keras_contrib.losses import jaccard
import tensorflow as tf
import numpy as np
import pandas as pd
import glob
import PIL
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras



def jaccard_distance(y_true, y_pred, smooth=100):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.square(y_true), axis = -1) + K.sum(K.square(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac)

def iou(y_true, y_pred, smooth = 100):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.square(y_true), axis = -1) + K.sum(K.square(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return jac


# To read the images in numerical order
import re
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts
    
class DataLoader():
   def __init__(self,rootG,rootI):
     #self.batch_size=batch_size
     #self.image_size=image_size
     self.rootG=rootG
     self.rootI=rootI

     if os.path.isdir(rootG):
       self.groundtruthP= sorted(glob.glob(os.path.join(rootG,'*.png')),key=numericalSort)
       
     else:
       raise Exception('rootG debe ser un directorio')
     if os.path.isdir(rootI):
       self.imagesPts= sorted(glob.glob(os.path.join(rootI,'*.jpg')),key=numericalSort)
       
     else:
       raise Exception('rootI debe ser un directorio')
     
     print('Size of the dataset:  ',len(self.imagesPts))
     self.train_x, self.test_x, self.train_y, self.test_y =train_test_split(self.imagesPts,self.groundtruthP,
     test_size=0.20,random_state=42)
     self.train_x,self.val_x,self.train_y,self.val_y = train_test_split(self.train_x,self.train_y)
     print('Number of train images: ',len(self.train_x))
     print('Number of test images: ',len(self.test_x))
     print('Number of validation images: ', len(self.val_x))
     self.train_x=np.array([np.array(Image.open(fname).resize((256,192))) for fname in self.train_x])
     self.train_y = np.array([np.array(Image.open(fname).resize((256,192))) for fname in self.train_y])
     self.test_x = np.array([np.array(Image.open(fname).resize((256,192))) for fname in self.test_x])
     self.test_y = np.array([np.array(Image.open(fname).resize((256,192))) for fname in self.test_y])
     self.val_x = np.array([np.array(Image.open(fname).resize((256,192))) for fname in self.val_x])
     self.val_y = np.array([np.array(Image.open(fname).resize((256,192))) for fname in self.val_y])

     
rootG='ISIC2018_Task1_Training_GroundTruth'
rootI='ISIC2018_Task1-2_Training_Input'
data = DataLoader(rootG,rootI)

x = data.train_y[0]
print(x)
plt.imshow(x)
plt.show()

def unet(input_size = (192,256,3  )):
      img_input = Input(shape= (192, 256, 3))
      x = Conv2D(16, (3, 3), padding='same', name='conv1')(img_input)
      x = BatchNormalization(name='bn1')(x)
      x = Activation('relu')(x)
      x = Conv2D(32, (3, 3), padding='same', name='conv2')(x)
      x = BatchNormalization(name='bn2')(x)
      x = Activation('relu')(x)
      x = MaxPooling2D()(x)
      x = Conv2D(64, (3, 3), padding='same', name='conv3')(x)
      x = BatchNormalization(name='bn3')(x)
      x = Activation('relu')(x)
      x = Conv2D(64, (3, 3), padding='same', name='conv4')(x)
      x = BatchNormalization(name='bn4')(x)
      x = Activation('relu')(x)
      x = MaxPooling2D()(x)
      x = Conv2D(128, (3, 3), padding='same', name='conv5')(x)
      x = BatchNormalization(name='bn5')(x)
      x = Activation('relu')(x)
      x = Conv2D(128, (4, 4), padding='same', name='conv6')(x)
      x = BatchNormalization(name='bn6')(x)
      x = Activation('relu')(x)
      x = MaxPooling2D()(x)
      x = Conv2D(256, (3, 3), padding='same', name='conv7')(x)
      x = BatchNormalization(name='bn7')(x)
      x = Dropout(0.5)(x)
      x = Activation('relu')(x)
      x = Conv2D(256, (3, 3), padding='same', name='conv8')(x)
      x = BatchNormalization(name='bn8')(x)
      x = Activation('relu')(x)
      x = MaxPooling2D()(x)
      x = Conv2D(512, (3, 3), padding='same', name='conv9')(x)
      x = BatchNormalization(name='bn9')(x)
      x = Activation('relu')(x)
      x = Dense(1024, activation = 'relu', name='fc1')(x)
      x = Dense(1024, activation = 'relu', name='fc2')(x)
      
      # Deconvolution Layers (BatchNorm after non-linear activation)
      
      x = Conv2DTranspose(256, (3, 3), padding='same', name='deconv1')(x)
      x = BatchNormalization(name='bn19')(x)
      x = Activation('relu')(x)
      x = UpSampling2D()(x)
      x = Conv2DTranspose(256, (3, 3), padding='same', name='deconv2')(x)
      x = BatchNormalization(name='bn12')(x)
      x = Activation('relu')(x)
      x = Conv2DTranspose(128, (3, 3), padding='same', name='deconv3')(x)
      x = BatchNormalization(name='bn13')(x)
      x = Activation('relu')(x)
      x = UpSampling2D()(x)
      x = Conv2DTranspose(128, (4, 4), padding='same', name='deconv4')(x)
      x = BatchNormalization(name='bn14')(x)
      x = Activation('relu')(x)
      x = Conv2DTranspose(128, (3, 3), padding='same', name='deconv5')(x)
      x = BatchNormalization(name='bn15')(x)
      x = Activation('relu')(x)
      x = UpSampling2D()(x)
      x = Conv2DTranspose(64, (3, 3), padding='same', name='deconv6')(x)
      x = BatchNormalization(name='bn16')(x)
      x = Activation('relu')(x)
      x = Conv2DTranspose(32, (3, 3), padding='same', name='deconv7')(x)
      x = BatchNormalization(name='bn20')(x)
      x = Activation('relu')(x)
      x = UpSampling2D()(x)
      x = Conv2DTranspose(16, (3, 3), padding='same', name='deconv8')(x)
      x = BatchNormalization(name='bn17')(x)
      x = Dropout(0.5)(x)
      x = Activation('relu')(x)
      x = Conv2DTranspose(1, (3, 3), padding='same', name='deconv9')(x)
      x = BatchNormalization(name='bn18')(x)
      x = Activation('sigmoid')(x)
      pred = Reshape((192,256))(x)
      
      model = Model(inputs=img_input, outputs=pred)
  
  
      model.compile(optimizer= Adam(lr = 0.003), loss= [jaccard_distance], metrics=[iou])
      
      model.summary()
  
  
      return model

import tensorflow as tf
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    
from keras import backend as K
print(K.tensorflow_backend._get_available_gpus())

u_net = unet()
model_checkpoint = ModelCheckpoint('check_unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)

hist = u_net.fit(data.train_x,data.train_y,epochs=1000,batch_size=20,validation_data=(data.val_x,data.val_y),verbose=1,callbacks=[model_checkpoint])

u_net.save("model.h5")

accuracy = u_net.evaluate(x=test_x,y=test_y,batch_size=16)
   

