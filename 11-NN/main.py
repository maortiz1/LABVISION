#!/home/afromero/anaconda3/bin/python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import tqdm
import tarfile
import zipfile
import os
import requests
from skimage import color
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Descargar los dos dataset
# TRAIN
if not os.path.isdir(os.path.join(os.getcwd(),'fer2013')):
    url='https://drive.google.com/uc?export=download&id=1B9Lr_Q3mzu-H-DD2-i2SkTx0TndcyVvO'
    r=requests.get(url,allow_redirects=True)
    open('fer2013.tar.gz','wb').write(r.content)
    tar=tarfile.open("fer2013.tar.gz","r")
    tar.extractall()
    tar.close

#os.chdir("fer2013/")
# TEST
if not os.path.isdir(os.path.join(os.getcwd(),'Emotions_test')):
    url='http://bcv001.uniandes.edu.co/Emotions_test.zip'
    r=requests.get(url,allow_redirects=True)
    open('Emotions_test.zip','wb').write(r.content)
    zipi=zipfile.open("Emotions_test.zip","r")
    zipi.extractall()
    zipi.close

# Imprimir en pantalla los parametros del modelo entrenado
def print_network(model, name):
    num_params=0
    for p in model.parameters():
        num_params+=p.numel()
    print(name)
    print(model)
    print("The number of parameters {}".format(num_params))

# Declarar la red
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3) #Channels input: 1, c output: 63, filter of size 3
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3)       
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3)
        self.fc2 = nn.Linear(32, 10) # capa fully connected con 10 neuronas
        self.relu = nn.ReLU()
        print('done')
    def forward(self, x):
      
        x = F.max_pool2d(F.relu(self.conv1(x)), 2) #Perform a Maximum pooling operation over the nonlinear responses of the convolutional layer
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.Dropout(x, 0.25, training=self.training)#Try to control overfit on the network, by randomly excluding 25% of neurons on the last #layer during each iteration
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = F.Dropout(x, 0.25, training=self.training)
        x = x.view(-1, 32)
        x = self.fc(x)
        
        return x
        
    def training_params(self):
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0)
        self.Loss = nn.CrossEntropyLoss()
# Obtener los datos y preprocesamiento de las imagenes
# Importante
def get_data(batch_size):
    print('try')
    # Train 
    with open("fer2013.csv") as f:
        content = f.readlines()

    lines = np.array(content)
    num_of_instances = lines.size
    
    print("number of instances: ",num_of_instances)
    print("instance length: ",len(lines[1].split(",")[1].split(" ")))

    x_train_tot, y_train_tot, x_train, y_train = [],[],[],[]

    for i in range(1,num_of_instances):
        emotion, img, usage = lines[i].split(",")
        pixels = np.array(img.split(" "), 'float32')
        emotion = 1 if int(emotion)==3 else 0 # Only for happiness
        if 'Training' in usage:
            y_traintot.append(int(emotion)) # en y guardo las emociones (en este caso 0 o 1). groundtruth            
            x_traintot.append(pixels) # en x guardo las imagenes
    
    x_traintot = np.array(x_traintot, 'float64')
    y_traintot = np.array(y_traintot, 'float64')

    x_traintot /= 255 #normalize inputs between [0, 1]
    

    x_train = x_traintot[0:batchsize]
    ytrain = y_traintot[0:batchsize] 
    # Reshape a train
    x_train = x_train.reshape(x_train.shape[0], 48, 48)
    y_train = y_train.reshape(y_train.shape[0], 1)
    print(x_train.shape,'train size')
    print(x_train.shape[0], 'train samples')
    
    # Test preprocesamiento
    filenames=os.listdir("Emotions_test/")
    i=0;
    for ix in filenames:
      

      imgtest= face_recognition.load_image_file(os.path.join("Emotions_test/", ix))
      facelocations =face_recognition.face_locations(imgtest);
      print(facelocations)
      faces = len(facelocations)
    
    
      for facelocation in facelocations:
          top, right, bottom, left = facelocation
          face_detect = imgtest[top:bottom, left:right]
          gray=color.rgb2gray(face_detect)
          facereshape=gray.reshape(48,48)
          
      i=1+i
    
    
    
    
    
    #transform_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    transform_train = transforms.Compose([transforms.ToTensor()])
    data_train = datasets.MNIST('data', train=True, transform = transform_train)
    train_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True)

    data_test = datasets.MNIST('data', train=False, transform = transform_train)
    test_loader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader





def train(data_loader, model, epoch):
    model.train()
    loss_cum = []
    Acc = 0
    for batch_idx, (data,target) in tqdm.tqdm(enumerate(data_loader), total=len(data_loader), desc="Epoch: {}".format(epoch)):
        data = data.view(-1,784)
        data = data.to(device)
        target = target.float().to(device)

        output = model(data)
        model.optimizer.zero_grad()
        loss = model.Loss(output,target)
        #loss = F.mse_loss(output, target) #Practically the same
        loss.backward()
        model.optimizer.step()
        loss_cum.append(loss.item())
        Acc += torch.round(output.data.cpu()).squeeze(1).long().eq(target.data.cpu().long()).sum()
    
    print("Loss: %0.3f"%(np.array(loss_cum).mean()))
    print("Acc: %0.2f"%(float(Acc*100)/len(data_loader.dataset)))
    
if __name__=='__main__':
    epochs=20
    batch_size=1000
    train_loader = get_data(batch_size)

    model = Net()
    model.to(device)
    model.training_params()
    print_network(model, 'fc 2 layer non-linearity')    

    for epoch in range(epochs):
        train(train_loader, model, epoch)
