#!/usr/bin/ipython3

# read kaggle facial expression recognition challenge dataset (fer2013.csv)
# https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge
import numpy as np
import matplotlib.pyplot as plt
import os
import requests
import zipfile
import tarfile
import sklearn.metrics as metrics
import scipy.io as sio
import cv2
from skimage import color
from skimage import io
import pickle
import traceback

if not os.path.isdir(os.path.join(os.getcwd(),'fer2013')):
    url='https://drive.google.com/uc?export=download&id=1B9Lr_Q3mzu-H-DD2-i2SkTx0TndcyVvO'
    r=requests.get(url,allow_redirects=True)
    open('fer2013.tar.gz','wb').write(r.content)
    tar=tarfile.open("fer2013.tar.gz","r")
    tar.extractall()
    tar.close

os.chdir("fer2013/")


def sigmoid(x):
    return 1/(1+np.exp(-x))
def softmax(X):
   
    exps = np.exp(X)
    return exps / np.sum(exps)

def get_data():
    # angry, disgust, fear, happy, sad, surprise, neutral
    with open("fer2013.csv") as f:
        content = f.readlines()

    lines = np.array(content)
    num_of_instances = lines.size
    print("number of instances: ",num_of_instances)
    print("instance length: ",len(lines[1].split(",")[1].split(" ")))

    x_train, y_train,x_val, y_val ,x_test, y_test = [], [], [], [],[],[]

    for i in range(1,num_of_instances):
        emotion, img, usage = lines[i].split(",")
        pixels = np.array(img.split(" "), 'float32')
        #emotion = 1 if int(emotion)==3 else 0 # Only for happiness
        if 'Training' in usage:
            y_train.append(int(emotion)) # en y guardo las emociones (en este caso 0 o 1). groundtruth            
            x_train.append(pixels) # en x guardo las imagenes
        
        elif 'PublicTest' in usage:
            y_test.append(int(emotion))
            x_test.append(pixels)

    #------------------------------
    #data transformation for train and test sets
    x_train = np.array(x_train, 'float64')
    y_train = np.array(y_train, 'float64')
    x_test = np.array(x_test, 'float64')
    y_test = np.array(y_test, 'float64')

    x_train /= 255 #normalize inputs between [0, 1]
    x_test /= 255
    tempx = x_train
  #  print(tempx.shape,'tempx forma')
    x_train = tempx[np.arange(1,tempx.shape[0],2),::]
    x_val = tempx[np.arange(0,tempx.shape[0],2),::]
    tempy = y_train
   # print(tempy.shape,'tempy forma')
    y_train = tempy[np.arange(1,tempx.shape[0],2)]
    y_val = tempy[np.arange(0,tempx.shape[0],2)]  

    x_train = x_train.reshape(x_train.shape[0], 48, 48)
    x_val = x_val.reshape(x_val.shape[0], 48, 48)
    x_test = x_test.reshape(x_test.shape[0], 48, 48)
    y_train = y_train.reshape(y_train.shape[0], 1)
    y_val = y_val.reshape(y_val.shape[0], 1)
    y_test = y_test.reshape(y_test.shape[0], 1)
    
    #Dividir train en train y validation
   # x_val=x_train[np.arange(0,x_train.shape[0],2),::,::]
   # x_train=x_train[np.arange(1,x_train.shape[0],2),::,::]
   # y_val=x_val[np.arange(0,x_train.shape[0],2),1]
   # y_train=x_val[np.arange(1,x_train.shape[0],2),1]

    print(x_train.shape[0],'train size')
    print(x_train.shape[0], 'train samples')
    print(x_val.shape[0], 'validation samples')
    print(x_test.shape, 'test samples')

    #plt.hist(y_train, max(y_train)+1); plt.show()
   # return x_train,y_train,x_test,y_test	
    return x_train, y_train,x_val,y_val, x_test, y_test

class Model():
    def __init__(self):
        params = 48*48 # image reshape
        out = 7 # all emotions labels
        self.lr = 0.0001 # Change if you want
        self.W = np.random.randn(params, out)
        self.b = np.random.randn(out)

    def forward(self, image):
        image = image.reshape(image.shape[0], -1)
        out = np.dot(image, self.W) + self.b
        return out

    def compute_loss(self,X,y):
    
        m = y.shape[0]
       # print(m,' m')
        p = softmax(X)

        y=y.astype(int)
        

        log_likelihood = -np.log(p[range(m), y])
        loss = np.sum(log_likelihood) / m
        return loss

    def compute_gradient(self, image, pred, gt):
        image = image.reshape(image.shape[0], -1)
        W_grad = np.dot(image.T, pred-gt)/image.shape[0]      
        self.W -= W_grad*self.lr
        b_grad = np.sum(pred-gt)/image.shape[0]
        self.b -= b_grad*self.lr

def train(model):
    x_train, y_train,x_val,y_val,_,_ = get_data()
   # x_train,y_train,x_test,y_test=get_data()
    batch_size = 300 # Change if you want

    epochs = 2000 # Change if you want

    losstot = []
    lossTrain=[]
    lossVal=[]
    epochsVector=[]
    
    plt.ioff()
    lossAnt=float('Inf');
    fig=plt.figure()
    for i in range(epochs):
        loss = []
        
        for j in range(0,x_train.shape[0], batch_size):
            _x_train = x_train[j:j+batch_size]
            _y_train = y_train[j:j+batch_size]
            out = model.forward(_x_train)
            loss.append(model.compute_loss(out, _y_train))
            model.compute_gradient(_x_train, out, _y_train)
        out = model.forward(x_val)                
        loss_val = model.compute_loss(out, y_val)
        print('Epoch {:6d}: {:.5f} | val: {:.5f}'.format(i, np.array(loss).mean(), loss_val))
        lossVal.append(loss_val)
        lossTrain.append(np.array(loss).mean())
        epochsVector.append(i)
        plot(fig,epochsVector,lossVal,lossTrain)
        
        if loss_val>lossAnt and not(np.isnan(loss_val)):
            break;
        lossAnt=loss_val    
        
        
    return [epochsVector,lossVal,lossTrain]    


def plot(fig,epochsVector,lossVal,losstrain): # Add arguments
    plt.figure(fig.number)
    vis=True
    l1=plt.plot(epochsVector,lossVal,'r-')
    l2=plt.plot(epochsVector,losstrain,'b-')
    plt.xlabel('Model Complexity (epoch)')
    plt.ylabel('Error')
    plt.legend(('Validation','Train'))
    plt.draw()
    plt.savefig('epochsVsLossjjj.pdf')
    if vis:
      plt.show(block=False)
    fig.canvas.flush_events()
    # CODE HERE
    # Save a pdf figure with train and test losses
   #pass
    
def test(model):
    _, _,_,_, x_test, y_test = get_data()    
     
    # for j in range(0,x_test.shape[0]):
    #image = x_test[j,::,::]
    image = x_test
    print (image.shape,'size test')
    image = image.reshape(image.shape[0], -1)
   # print (image.shape)
    print (model.W.shape,'size W')
    out = np.dot(image, model.W) + model.b
    prob = softmax(out)
    prob = np.argmax(prob,axis=1)
    print(prob.shape,'size prob')
    prediction = []
    thrs= np.linspace(0.001,1,30)
    prec_vec=[]
    recal_vec=[]
    FMed_vec=[]
    CMat_vec=[]
    ACA_vec=[]
   
    print(prob)
    print(prob.shape)
    confM=metrics.confusion_matrix(y_test,prob)
    aca=metrics.accuracy_score(y_test,prob)
    CMat_vec.append(confM)
    ACA_vec.append(aca)
    
    
    #MaxFMed=np.amax(FMed_vec)  
    #index=np.argmax(FMed_vec)
    #print(MaxFMed,' Max F-Measure')
    #print(index, 'Max threshold')
    
    return prec_vec,recal_vec,FMed_vec,CMat_vec,ACA_vec
    

   # pass

def demo(model):
    if not os.path.isdir(os.path.join(os.getcwd(),'demo')):
        url='https://drive.google.com/uc?export=download&id=11VhP5pORYEiuUlmZ72S7K2rYlBZ5prWN'
        r=requests.get(url,allow_redirects=True)
        open('demo1.zip','wb').write(r.content) 
        zip_ref = zipfile.ZipFile('demo1.zip', 'r')
        zip_ref.extractall()
        zip_ref.close()
        
    filenames=os.listdir("demo1/")
    demo_test = []
    
    for i in filenames:
       temp=cv2.imread(os.path.join("demo1/", i))
       #the files are too big. It is necessary to resize
       temp = color.rgb2gray (temp)
       imCrop=cv2.resize(temp,(48,48))
       demo_test.append(imCrop)
    
 
    image = np.array(demo_test, 'float64')
    image /= 255

    image= image.reshape(image.shape[0],48,48)
    print(image.shape,'vector imagen')
    image = image.reshape(image.shape[0], -1)
    
    out = np.dot(image, model.W) + model.b
    prob = softmax(out)
    prob = np.argmax(prob,axis=1)
    figure=plt.figure() 
    
    for i in range(0,image.shape[0]):
      acImg= image[i,::].reshape(48,48)
      print(acImg.shape)
      clas_prob = prob[i]
      tag='Angry'
      if clas_prob ==1:
          tag='Disgust'
      elif clas_prob ==2:
          tag='Fear'
      elif clas_prob ==3:
          tag='Happy'
      elif clas_prob ==4:
          tag='Sad'
      elif clas_prob ==5:
          tag='Surprise'
      elif clas_prob ==6:
          tag='Neutral'
      plt.imshow(acImg,cmap='gray')
      plt.title(tag)
      plt.axis('off')
      plt.show(block=False)
      
      print("Please press any key to continue")
      plt.waitforbuttonpress(0)
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-t","--test",help="runs only test if model is saved",dest='test',action='store_true')
    parser.add_argument("-d","--demo",help="runs only demo if model is saved",dest='demo',action='store_true')
    parser.add_argument("-tr","--train",help="runs only train and val set to train model",dest='train',action='store_true')

    arguments = parser.parse_args()

        
    if arguments.test:    
      print('You chose test')  
      try:
        with open('dataemo','rb') as f:
          model = pickle.load(f)
          f.close()
        test(model)
      except: 
        print('No trained model found, model computation will proceed')
        model=Model()
        train(model)
        with open('dataemo', 'wb') as f:
          pickle.dump(model,f)  
          f.close()
        test(model)
        
    elif arguments.demo:
      print('You chose demo')
      try:
        with open('dataemo','rb') as f:
          model = pickle.load(f)
          f.close()
        demo(model)        
      except  Exception as err: 
        traceback.print_tb(err.__traceback__)
        print('No trained model found, model computation will proceed')
        model=Model()
        train(model)
        with open('dataemo', 'wb') as f:
          pickle.dump(model,f)  
          f.close()
        demo(model)
    elif arguments.train:
        print('Model will be trained')
        model=Model()
        train(model)
        with open('dataemo', 'wb') as f:
          pickle.dump(model,f)  
          f.close()
    
    else:
      model = Model()
      train(model)
      with open('dataemo', 'wb') as f:
        pickle.dump(model,f)     
      test(model)
      with open('dataemo', 'wb') as f:
        pickle.dump(model,f)      
              
    
