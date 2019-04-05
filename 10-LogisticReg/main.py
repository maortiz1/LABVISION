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

    x_train, y_train, x_test, y_test = [], [], [], []

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
    #tempx = x_train
    #x_train = (tempx[np.arange(1,tempx.shape[0],2),::,::])
    #x_val = (tempxnp.arange(0,tempx.shape[0],2)],::,::)
    #tempy = y_train
    #y_train = (tempy[np.arange(1,tempx.shape[0],2),::,::])
    #y_val = (tempy[np.arange(0,tempx.shape[0],2),::,::])  

    x_train = x_train.reshape(x_train.shape[0], 48, 48)
    x_test = x_test.reshape(x_test.shape[0], 48, 48)
    y_train = y_train.reshape(y_train.shape[0], 1)
    y_test = y_test.reshape(y_test.shape[0], 1)
    
    #Dividir train en train y validation
   # x_val=x_train[np.arange(0,x_train.shape[0],2),::,::]
   # x_train=x_train[np.arange(1,x_train.shape[0],2),::,::]
   # y_val=x_val[np.arange(0,x_train.shape[0],2),1]
   # y_train=x_val[np.arange(1,x_train.shape[0],2),1]

    print(x_train.shape[0],'train size')
    print(x_train.shape[0], 'train samples')
    #print(x_val.shape[0], 'validation samples')
    print(x_test.shape[0], 'test samples')

    #plt.hist(y_train, max(y_train)+1); plt.show()

    return x_train, y_train, x_test, y_test

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
    x_train, y_train, x_test, y_test = get_data()
    batch_size = 100 # Change if you want
    epochs = 10000 # Change if you want
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
        out = model.forward(x_test)                
        loss_val = model.compute_loss(out, y_test)
        print('Epoch {:6d}: {:.5f} | test: {:.5f}'.format(i, np.array(loss).mean(), loss_val))
        lossVal.append(loss_val)
        lossTrain.append(np.array(loss).mean())
        epochsVector.append(i)
        plot(fig,epochsVector,lossVal,lossTrain)
        
    return [epochsVector,lossVal,lossTrain]    


def plot(fig,epochs,lossVal,losstrain): # Add arguments
    plt.figure(fig.number)
    vis=False
#    y =np.arange(epochs)
 #   x = losstot
    l1=plt.plot(epochsVector,lossVal,'r-')
    l2=plt.plot(epochsVector,losstrain,'b-')
    plt.xlabel('Model Complexity (epoch)')
    plt.ylabel('Error')
    plt.legend([l1,l2],['Validation','Error'])
    
    self.fig.savefig('epochsVsLoss.pdf')
    if vis:
      plt.show()
    plt.close()
    # CODE HERE
    # Save a pdf figure with train and test losses
   #pass
    
def test(model):
    # _, _, x_test, y_test = get_data()
    # YOU CODE HERE
    # Show some qualitative results and the total accuracy for the whole test set
    pass

if __name__ == '__main__':
    model = Model()
    [epochsVector,lossVal,lossTrain] =train(model)
    test(model)

