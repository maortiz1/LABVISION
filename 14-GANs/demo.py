
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML



import os
import requests
import urllib

URL='https://www.dropbox.com/s/bijwrvojafs2nwn/checkpoint.pth.tar?dl=1'
print('It will be proceed to download the model')
#checking if databse is already downloaded
if not(os.path.exists('checkpoint.pth.tar')):
  urllib.request.urlretrieve(URL, "checkpoint.pth.tar") 
  print('The model had been downloaded')
else: 
  print('The file  already exists')



# Set random seem for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
# Root directory for dataset
dataroot = "/media/user_home2/vision/data/CelebA"

# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 128

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 5

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

# We can use an image folder dataset the way we have it setup.
# Create the dataset
dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

# Create the generator
netG = Generator(ngpu).to(device)


resume = 'checkpoint.pth.tar'
print('Loading checkpoint from: '+resume)
print('Number of epochs training: '+str(torch.load(resume)['epoch']))
print('Number of iterations: '+str(torch.load(resume)['iteration']))
netG.load_state_dict(torch.load(resume)['model_state_dict'])
matplotlib.use('tkagg')
# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

netG = torch.load(resume)['model']
fake1 = netG(fixed_noise).detach().cpu()


fixed_noise = torch.randn(32, nz, 1, 1, device=device)

fake2 = netG(fixed_noise).detach().cpu()

img = vutils.make_grid(fake1, padding=2, normalize=True)
plt.imshow(np.transpose(img,(1,2,0)))
plt.show()



img = vutils.make_grid(fake2, padding=2, normalize=True)
plt.imshow(np.transpose(img,(1,2,0)))
plt.show()


fixed_noise = torch.randn(2, nz, 1, 1, device=device)

fake3 = netG(fixed_noise).detach().cpu()

img = vutils.make_grid(fake3, padding=2, normalize=True)
plt.imshow(np.transpose(img,(1,2,0)))
plt.show()
