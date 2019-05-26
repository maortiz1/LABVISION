import os
from PIL import Image

from predict import predict_img
from utils import rle_encode
from unet import UNet
import torch
import torch.nn.functional as F
from utils import get_ids, split_ids, split_train_val, get_imgs_and_masks, batch
from dice_loss import dice_coeff
from keras import backend as K
import torch
import torch.nn.functional as F
import sklearn.metrics as skm
import numpy as np
from dice_loss import dice_coeff
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import matplotlib.pyplot as plt


#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def test(net, dataset, gpu=True):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    tot = 0
    for i, b in enumerate(dataset):
        img = b[0]
        true_mask = b[1]

        img = torch.from_numpy(img).unsqueeze(0)
        true_mask = torch.from_numpy(true_mask).unsqueeze(0)
        
       
        if gpu:
            img = img.to(device)
            true_mask = true_mask.to(device)

        mask_pred = net(img)[0]

        mask_pred = (mask_pred > 0.5).float()
        

#        jac= jaccard_distance(true_mask,mask_pred)
        mask_pred=mask_pred.cpu().numpy()[:,:,:].squeeze()


        true_mask=true_mask.data.cpu().numpy().squeeze()/255
        jac = iou(true_mask,mask_pred).mean()

        print(i,':',jac)
        tot += jac
    return tot / (i + 1)
    
def jaccard_distance(y_true, y_pred, smooth=100):
    intersection = np.sum(np.abs(y_true * y_pred), axis=-1)
    sum_ = np.sum(np.square(y_true), axis = -1) + np.sum(np.square(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac)
    
def iou(y_true, y_pred, smooth = 100):
    intersection = np.sum(np.abs(y_true * y_pred), axis=-1)
    sum_ = np.sum(np.square(y_true), axis = -1) + np.sum(np.square(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return jac
    

if __name__ == '__main__':

    dir_img = 'ISIC2018_Task1-2_Training_Input/'
    dir_mask = 'ISIC2018_Task1_Training_GroundTruth/'

    ids = get_ids(dir_img)
    ids = split_ids(ids)

    iddataset = split_train_val(ids, val_percent=0.2)
    
    val = get_imgs_and_masks(iddataset['val'], dir_img, dir_mask, 0.5)


    net = UNet(3, 1).to(device)
    net.load_state_dict(torch.load('checkpoints/CP14.pth'))
    test(net, val, gpu = True)