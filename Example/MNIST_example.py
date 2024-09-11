import os,sys,time,copy
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[-1] if sys.argv[-1].isdigit() else '0'

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from lib_CM import *


import warnings
warnings.filterwarnings('ignore')

from scipy.optimize import linear_sum_assignment
from sklearn.metrics import homogeneity_score as homog
from torch.utils.data import Subset
def accuracy(y_true, y_pred):
    assert y_pred.shape[0] == y_true.shape[0]
        
    D = int( max(y_pred.max(), y_true.max()) + 1 )
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.shape[0]):
        w[int(y_pred[i]), int(y_true[i])] += 1
    ind = np.vstack(linear_sum_assignment(w.max() - w)).T
    acc =  sum([w[i, j] for i, j in ind]) * 1.0 / np.prod(y_pred.shape)
    return acc

def a2s(array,p=3): 
    return str( ["{:.6f}".format(x) for x in array] )[1:-1].replace("'",'') 

def i2s(array,p=3): 
    return str( [str(x) for x in array] )[1:-1].replace("'",'') 


##########################################################################

class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape,  # extra comma

    def forward(self, x):
        return x.view(*self.shape)

class CV_CM(nn.Module):
  def __init__(self, LATENT):
    super().__init__()


    self.encoder = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),

        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),

        nn.Flatten(),

        nn.Linear(7 * 7 * 64, LATENT*2),
        # nn.Dropout(p=0.5),
        nn.ReLU(),

        nn.Linear(LATENT*2, LATENT)
    )

    self.decoder = nn.Sequential(
        nn.Linear(LATENT, 128),
        nn.ReLU(),

        nn.Linear(128, 7 * 7 * 64),

        View((-1,64,7,7)),

        nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        #
        nn.UpsamplingBilinear2d(scale_factor=2),
        #

        nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.UpsamplingBilinear2d(scale_factor=2),
        #

        nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1),
    )

    self.cm = Clustering_Module( LATENT, 5, False)

  def forward(self, x):
    z = self.encoder(x)
    tx = self.decoder(z)
    cm = self.cm(z) # NxC, NxK, NxC, KxC

    return tx, cm


######################################################################################


EPOCH = 100

"""
# Parameters for non-normalized Loss
BATCH = 500
ALPHA = 230
BETA = 5.
LBD = 1.
"""

# Parameters for normalized Loss
BATCH = 256
ALPHA = 1.1
BETA = 100.
LBD = .1

print( BATCH, ALPHA, BETA, LBD )


######################################################################################

torch.cuda.set_device(0)

def filter_by_labels(dataset, labels):
    indices = [i for i, (_, label) in enumerate(dataset) if label in labels]
    return Subset(dataset, indices)

# Define which digits to keep
allowed_labels = set(range(5))

# load data
train_dataset = datasets.MNIST(
    'mnist', 
    train=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,) )
    ]),
    download=False,
)
test_dataset = datasets.MNIST(
    'mnist', 
    train=False,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,) )
    ]),
    download=False,
)

train_dataset = filter_by_labels(train_dataset, allowed_labels)
test_dataset = filter_by_labels(test_dataset, allowed_labels)


train_loader = torch.utils.data.DataLoader(
    train_dataset,
    shuffle=True, 
    batch_size=BATCH, 
    num_workers=8, 
    drop_last=False
) #, pin_memory=True)

test_loader = torch.utils.data.DataLoader(
    test_dataset, 
    batch_size=max(BATCH,1024), 
    num_workers=8, 
    drop_last=False
) #, pin_memory=True)

    
# create model
model = CV_CM(2).cuda(0)

criterion_reconst = nn.MSELoss(reduction=('mean')).cuda(0)
criterion_cluster = Clustering_Module_Loss(
                        num_clusters=5, 
                        alpha=ALPHA, 
                        lbd=0,  # lbd == 0 
                        orth=False, # True ==> False 
                        normalize=True).cuda(0)

optim_params = model.parameters()
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3, 
    betas=(.9,.999), 
    eps=1e-3 
)


######################################################################################

def train():
    # switch to train mode
    model.train()
    
    for i, (images, _) in enumerate(train_loader):

        images = images.cuda(0, non_blocking=True)

        # compute output and loss
        tx, cm = model(x=images)
        loss_rc = criterion_reconst(images, tx)
        loss_cm = criterion_cluster(cm)
        loss = loss_rc * BETA + loss_cm
        
        # compute gradient and do GD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def evaluate(full=False):
    model.eval()

    pred,lbl,img = [],[],None
    for i, (images, labels) in enumerate(train_loader):
        if i == 0 or full:
            img = images.cuda(0, non_blocking=True)
            tx, cm = model(img)
            _,gamma,_,_ = cm
            pred += gamma.argmax(-1).detach().cpu().tolist()
            lbl += labels.cpu().tolist()
            break
    
    pred = np.array(pred)
    lbl = np.array(lbl).astype(int)
    
    rc_loss = criterion_reconst(img, tx).detach().cpu().numpy()
    cm_loss = criterion_cluster(cm, split=True).detach().cpu().numpy()
    
    c_lr =  optimizer.param_groups[0]["lr"]
        
    print('Epoch: [{:d}]\tlr: {:.6f}\taccuracy: {:.1f}\thomog: {:.1f}'.format( 
            epoch+1, 
            c_lr,
            accuracy( lbl, pred )*100,
            homog( lbl, pred )*100,
        ), flush=True, end='\t')
        
    print('Rec:{:.6f}'.format( rc_loss * BETA ), end='\t', flush=True)
    print('Loss:',a2s( cm_loss ), flush=True)
    
    return pred, lbl, cm_loss, rc_loss

def avg_epoch():
    weights = {}
    for k in model.state_dict():
        weights[k] = model.state_dict()[k].detach()
    # switch to train mode
    model.train()
    
    for i, (images, _) in enumerate(train_loader):

        images = images.cuda(0, non_blocking=True)

        # compute output and loss
        tx, cm = model(x=images)
        loss_rc = criterion_reconst(images, tx)
        loss_cm = criterion_cluster(cm)
        loss = loss_rc * BETA + loss_cm
        
        # compute gradient and do GD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        for k in weights:
            weights[k] = (weights[k]*i + model.state_dict()[k].detach()) / (i+1)
            
    model.load_state_dict(weights)

######################################################################################

for epoch in range(EPOCH):
    train()
    
    if (epoch)%10 == 0:
        print('.',end='\r')
        evaluate(full=False)
    if (epoch)%50 == 0:
        print( BATCH, ALPHA, BETA, LBD )

evaluate(full=False)
print('>>> End Training')
evaluate(full=True)
print('>>> Average Epoch')
avg_epoch()
evaluate(full=True)
