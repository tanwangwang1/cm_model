import os,sys,time,copy
#os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[-1] if sys.argv[-1].isdigit() else '0'

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F

from lib_CM import *


import warnings
warnings.filterwarnings('ignore')

from scipy.optimize import linear_sum_assignment
from sklearn.metrics import homogeneity_score as homog
from torch.utils.data import Subset
import argparse
from sklearn.metrics import silhouette_score,davies_bouldin_score, adjusted_rand_score, normalized_mutual_info_score, homogeneity_completeness_v_measure
BETA = 100.
def parse_arguments():
    """
    build and analyse the parameters of command
    return the parser(parameter object)
    """
    parser = argparse.ArgumentParser(description="Mnist_example")
    parser.add_argument("--alpha",'-a',type=float,default=1.1,help='set the alpha which must be more than 1.0')
    parser.add_argument("--centroids",'-c',type=int, default=5, help='set the amount of centroids')
    parser.add_argument("--new_alpha",'-n',type=float, default=0.1, help='set the added alpha [0.1,1], rename as new_alpha')
    parser.add_argument("--temperature", '-t', type=float, default=5, help='set the temperature of softmax')
    parser.add_argument("--save_csv",'-o', type=str, default='./',help='the savepath of csv')
    args = parser.parse_args()
    return args


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

    self.cm = Clustering_Module(LATENT, 5, False)

  def forward(self, x):
    z = self.encoder(x)
    tx = self.decoder(z)
    cm = self.cm(z) # NxC, NxK, NxC, KxC

    return tx, cm

######################################################################################

def train(
        model,
        dataloader,
        criterion_cluster,
        optimizer,
        criterion_reconst,
        device
        ):
    # switch to train mode
    model.train()
    
    for i, (images, _) in enumerate(dataloader):

        images = images.to(device)

        # compute output and loss
        tx, cm = model(x=images)
        loss_rc = criterion_reconst(images, tx)
        loss_cm = criterion_cluster(cm)
        loss = loss_rc * BETA + loss_cm
        
        # compute gradient and do GD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def evaluate(
            model,
            dataloader,
            criterion_cluster,
            optimizer,
            device,
            epoch,
            criterion_reconst,
            full=True,
            is_save=False, 
            alpha=None, 
            centroids=None, 
            new_alpha=None, 
            temp=None,
            cm_loss_list=None
            ):
    model.eval()

    pred,lbl,img = [],[],None
    X_all = []
    for i, (images, labels) in enumerate(dataloader):
        #import pdb;pdb.set_trace()
        X_all += images
        if i == 0 or full:
            img = images.to(device)
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
    #import pdb;pdb.set_trace()
#######################################################################################
    if is_save:
        import pdb;pdb.set_trace()
        X_all_np = np.array([x.flatten().numpy() for x in X_all])
        silhouette_score_ = round(silhouette_score(X_all_np, pred), 3)
        import pdb;pdb.set_trace()
        # davies_index = round(davies_bouldin_score(X_all, pred), 3)
        # acc = round(accuracy( lbl, pred )*100,1)
        # labels_true= np.array([tensor.numpy().flatten() for tensor in y_all])
        # adjusted_score = round(adjusted_rand_score(labels_true.flatten(), pred),3)
        # normalized_score  = round(normalized_mutual_info_score(labels_true.flatten(), pred), 3)
        # homogeneity_score_ = homogeneity_completeness_v_measure(labels_true.flatten(), pred)
        # homogeneity_score = round(homogeneity_score_[0], 3)
        # completeness_score = round(homogeneity_score_[1],3)
        # v_measure_score = round(homogeneity_score_[2],3)
    
    return pred, lbl, cm_loss, rc_loss

def avg_epoch(model,
              dataloader,
              criterion_cluster,
              optimizer,
              criterion_reconst,
              device):
    weights = {}
    for k in model.state_dict():
        weights[k] = model.state_dict()[k].detach()
    # switch to train mode
    model.train()
    
    for i, (images, _) in enumerate(dataloader):

        images = images.to(device)

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



def filter_by_labels(dataset, labels):
    indices = [i for i, (_, label) in enumerate(dataset) if label in labels]
    return Subset(dataset, indices)
def main(args):
    EPOCH = 100
    # Parameters for normalized Loss
    BATCH = 2048
    BETA = 100.
    LBD = .1

    print( BATCH, args.alpha, BETA, LBD )
    ######################################################################################
    #torch.cuda.set_device(0)



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
        shuffle=True, 
        num_workers=8, 
        drop_last=False
    ) #, pin_memory=True)
################################################################################
    # create model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = CV_CM(2).to(device)

    criterion_reconst = nn.MSELoss(reduction=('mean')).to(device)
    criterion_cluster = Clustering_Module_Loss(
                            num_clusters=args.centroids, 
                            alpha=args.alpha, 
                            lbd=0,  # lbd == 0 
                            orth=False, # True ==> False 
                            normalize=True).to(device)

    optim_params = model.parameters()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3, 
        betas=(.9,.999), 
        eps=1e-3 
    )

    
    ######################################################################################

    for epoch in range(EPOCH):
        train(model=model,
              dataloader=train_loader,
              criterion_cluster=criterion_cluster,
              optimizer=optimizer,
              criterion_reconst=criterion_reconst,
              device=device)
        
        if (epoch)%1 == 0:
            print('.',end='\r')
            pred,_,_,_ = evaluate(model=model,
                                dataloader=test_loader,
                                criterion_cluster=criterion_cluster,
                                optimizer=optimizer,
                                device=device,
                                epoch=epoch,
                                criterion_reconst = criterion_reconst,
                                full=False)
            if (epoch + 1)%10 == 0:
                with torch.no_grad(): 
                    print(pred)
                    freq = np.bincount( pred.astype(int), minlength=args.centroids ).astype(float) / args.temperature
                    freq = F.softmax(torch.tensor(freq), dim=-1)
                    freq = freq.numpy()
                    criterion_cluster.alpha = criterion_cluster.alpha*args.new_alpha + (torch.tensor(freq+1).float()*(1 - args.new_alpha)).to(device)
                    print(criterion_cluster.alpha)

    evaluate(model=model,
              dataloader=test_loader,
              criterion_cluster=criterion_cluster,
              optimizer=optimizer,
              epoch=EPOCH,
              criterion_reconst = criterion_reconst,
              device=device,
              full=False)
    print('>>> End Training')
    evaluate(model=model,
              dataloader=test_loader,
              criterion_cluster=criterion_cluster,
              optimizer=optimizer,
              criterion_reconst = criterion_reconst,
              device=device,
              full=True,
              epoch=EPOCH)
    print('>>> Average Epoch')
    avg_epoch(model=model,
              dataloader=test_loader,
              criterion_cluster=criterion_cluster,
              optimizer=optimizer,
              criterion_reconst = criterion_reconst,
              device=device)
    evaluate(
            model=model,
            dataloader=test_loader,
            criterion_cluster=criterion_cluster,
            optimizer=optimizer,
            criterion_reconst = criterion_reconst,
            device=device,
            epoch=EPOCH,
            full=True,
            is_save=True, 
            alpha=args.alpha, 
            centroids=args.centroids, 
            new_alpha=args.new_alpha, 
            temp=args.temperature,
            cm_loss_list=None
            )



if __name__=='__main__':
    args = parse_arguments()
    main(args)
