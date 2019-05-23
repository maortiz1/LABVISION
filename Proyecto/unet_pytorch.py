import sys
import os
from optparse import OptionParser
import numpy as np
import glob
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim
from PIL import Image
from eval import eval_net
from unet import UNet
from utils import get_ids, split_ids, split_train_val, get_imgs_and_masks, batch
from sklearn.model_selection import train_test_split

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
     self.train_x=np.array([np.array(Image.open(fname)) for fname in self.train_x])
     self.train_y = np.array([np.array(Image.open(fname)) for fname in self.train_y])
     self.test_x = np.array([np.array(Image.open(fname)) for fname in self.test_x])
     self.test_y = np.array([np.array(Image.open(fname)) for fname in self.test_y])
     self.val_x = np.array([np.array(Image.open(fname)) for fname in self.val_x])
     self.val_y = np.array([np.array(Image.open(fname)) for fname in self.val_y])

     
rootG='ISIC2018_Task1_Training_GroundTruth'
rootI='ISIC2018_Task1-2_Training_Input'
data = DataLoader(rootG,rootI)


def train_net(net,
              epochs=5,
              batch_size=1,
              lr=0.1,
              val_percent=0.2,
              save_cp=True,
              gpu=True,
              img_scale=0.5):

#    dir_img = 'ISIC2018_Task1-2_Training_Input'
#   dir_mask = 'ISIC2018_Task1_Training_GroundTruth'
    dir_checkpoint = 'checkpoints/'

#    ids = get_ids(dir_img)
#    ids = split_ids(ids)

#    iddataset = split_train_val(ids, val_percent)

    print('''
    Starting training:
        Epochs: {}
        Batch size: {}
        Learning rate: {}
        Training size: {}
        Validation size: {}
        Checkpoints: {}
        CUDA: {}
    '''.format(epochs, batch_size, lr, len(data.train_x),
               len(data.val_x), str(save_cp), str(gpu)))

    N_train = len(data.train_x)

    optimizer = optim.SGD(net.parameters(),
                          lr=lr,
                          momentum=0.9,
                          weight_decay=0.0005)

    criterion = nn.BCELoss()

    for epoch in range(epochs):
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
        net.train()

        # reset the generators
        train = data.train_x
        mask_train = data.train_y
        val = data.val_x
        mask_val = data.val_y
        epoch_loss = 0

        for i, b in enumerate(batch(train, batch_size)):
            imgs = np.array([i[0] for i in b]).astype(np.float32)
            true_masks = np.array([i[1] for i in b])

            imgs = torch.from_numpy(imgs)
            true_masks = torch.from_numpy(true_masks)

            if gpu:
                imgs = imgs.cuda()
                true_masks = true_masks.cuda()

            masks_pred = net(imgs)
            masks_probs_flat = masks_pred.view(-1)

            true_masks_flat = true_masks.view(-1)

            loss = criterion(masks_probs_flat, true_masks_flat)
            epoch_loss += loss.item()

            print('{0:.4f} --- loss: {1:.6f}'.format(i * batch_size / N_train, loss.item()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Epoch finished ! Loss: {}'.format(epoch_loss / i))

        if 1:
            val_dice = eval_net(net, val, gpu)
            print('Validation Dice Coeff: {}'.format(val_dice))

        if save_cp:
            torch.save(net.state_dict(),
                       dir_checkpoint + 'CP{}.pth'.format(epoch + 1))
            print('Checkpoint {} saved !'.format(epoch + 1))



def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=5, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batchsize', default=10,
                      type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.1,
                      type='float', help='learning rate')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=False, help='use cuda')
    parser.add_option('-c', '--load', dest='load',
                      default=False, help='load file model')
    parser.add_option('-s', '--scale', dest='scale', type='float',
                      default=0.5, help='downscaling factor of the images')

    (options, args) = parser.parse_args()
    return options

if __name__ == '__main__':
    args = get_args()

    net = UNet(n_channels=3, n_classes=1)

    if args.load:
        net.load_state_dict(torch.load(args.load))
        print('Model loaded from {}'.format(args.load))

    if args.gpu:
        net.cuda()
        # cudnn.benchmark = True # faster convolutions, but more memory

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  gpu=args.gpu,
                  img_scale=args.scale)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)