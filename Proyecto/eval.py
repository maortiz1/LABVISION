  
import torch
import torch.nn.functional as F

from dice_loss import dice_coeff

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
def eval_net(net, dataset, gpu=False):
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

        tot += dice_coeff(mask_pred, true_mask).item()
    return tot / (i + 1)
    
