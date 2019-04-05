#!/usr/bin/ipython3

# read kaggle facial expression recognition challenge dataset (fer2013.csv)
# https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge
import numpy as np
import matplotlib.pyplot as plt
import os
import requests
import tarfile

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
        emotion = 1 if int(emotion)==3 else 0 # Only for happiness
        if 'Training' in usage:
            y_train.append(emotion) # en y guardo las emociones (en este caso 0 o 1). groundtruth            
            x_train.append(pixels) # en x guardo las imagenes
        
        elif 'PublicTest' in usage:
            y_test.append(emotion)
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
    print(tempx.shape,'tempx forma')
    x_train = tempx[np.arange(1,tempx.shape[0],2),::]
    x_val = tempx[np.arange(0,tempx.shape[0],2),::]
    tempy = y_train
    print(tempy.shape,'tempy forma')
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
        out = 1 # smile label
        self.lr = 0.00001 # Change if you want
        self.W = np.random.randn(params, out)
        self.b = np.random.randn(out)

    def forward(self, image):
        image = image.reshape(image.shape[0], -1)
        out = np.dot(image, self.W) + self.b
        return out

    def compute_loss(self, pred, gt):
        J = (-1/pred.shape[0]) * np.sum(np.multiply(gt, np.log(sigmoid(pred))) + np.multiply((1-gt), np.log(1 - sigmoid(pred))))
        return J

    def compute_gradient(self, image, pred, gt):
        print(image.shape,'img size bfo reshape')
        print(pred.shape,'img pred size')
        image = image.reshape(image.shape[0], -1)
       	print(image.shape,'img size aft reshape')
        W_grad = np.dot(image.T, pred-gt)/image.shape[0]
        print(W_grad.shape,'W_grad forma')
        print(self.W.shape,'w SHAPR')
        self.W -= W_grad*self.lr

        b_grad = np.sum(pred-gt)/image.shape[0]
        self.b -= b_grad*self.lr

def train(model):
    x_train, y_train,x_val,y_val,_,_ = get_data()
   # x_train,y_train,x_test,y_test=get_data()
    batch_size = 50 # Change if you want
    epochs = 10 # Change if you want
    losstot = []
    lossTrain=[]
    lossVal=[]
    epochsVector=[]
    
    plt.ioff()
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
        
    return [epochsVector,lossVal,lossTrain]    


def plot(fig,epochsVector,lossVal,losstrain): # Add arguments
    plt.figure(fig.number)
    vis=False
    l1=plt.plot(epochsVector,lossVal,'r-')
    l2=plt.plot(epochsVector,losstrain,'b-')
    plt.xlabel('Model Complexity (epoch)')
    plt.ylabel('Error')
    plt.legend([l1,l2],['Validation','Train'])
    plt.draw()
    plt.savefig('epochsVsLoss.pdf')
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
    prob = sigmoid(out)
    print(prob.shape,'size prob')
    prediction = []
    for pro in prob:
      if pro <= 0.5:
        prediction.append(0)
      elif pro >= 0.5:
        prediction.append(1)
    print (len(prediction))
    print(prediction[1])
    print(prediction[5])

   # pass

if __name__ == '__main__':
    model = Model()
    [epochsVector,lossVal,lossTrain] =train(model)
    test(model)

