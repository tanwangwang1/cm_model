import os,sys,time,copy
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[-1] if sys.argv[-1].isdigit() else '0'

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms


from lib_CM import *


import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import homogeneity_score as homog
from sklearn.preprocessing import StandardScaler
from sklearn import cluster, datasets
from sklearn.model_selection import train_test_split
from custom_dataset import CustomDataset
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



######################################################################################


EPOCH = 1500

# Parameters for normalized Loss
BATCH = 256
ALPHA = 1.5  # modify
BETA = 100.
LBD = .1

print( BATCH, ALPHA, BETA, LBD )


######################################################################################
n_samples = 10000
X, y = datasets.make_blobs(n_samples=n_samples, random_state=170)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_dataset = CustomDataset(X_train, y_train)#transform=transform
test_dataset = CustomDataset(X_test, y_test)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8000, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    
# create model
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = Clustering_Module(2, 3, False).to(device)


criterion_reconst = nn.MSELoss(reduction=('mean')).to(device)
criterion_cluster = Clustering_Module_Loss(
                        num_clusters=3, 
                        alpha=ALPHA, 
                        lbd=0,
                        orth=False, #False
                        normalize=True,
                        device=device)

optim_params = model.parameters()
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3, 
    betas=(.9,.999), 
    eps=1e-3 
)


######################################################################################
def train(dataloader):

    model.train()

    for X, y in dataloader:
        X = X.float().to(device)
        cm = model(X)
        # loss_rc = criterion_reconst(X, tx)
        loss = criterion_cluster(cm)
        # compute gradient and do GD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
import matplotlib.pyplot as plt

def plot_predictions(pred, save_path=None):
    plt.figure(figsize=(10, 5))
    plt.scatter(pred, range(len(pred)), c=pred, cmap='viridis', marker='o', alpha=0.5)
    plt.xlabel('Predicted Class')
    plt.ylabel('Index')
    plt.title('Predictions')
    plt.colorbar(label='Class')
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def evaluate(dataloader,full=False):
    model.eval()
    pred,lbl,x = [],[],None
    for i, (X, y) in enumerate(dataloader):
        if i == 0 or full:
            x = X.float().to(device)

            cm = model(x)
            _,gamma,_,_ = cm
            pred += gamma.argmax(-1).detach().cpu().tolist()
            lbl += y.cpu().tolist()

            break

    pred = np.array(pred)
    lbl = np.array(lbl).astype(int)
    cm_loss = criterion_cluster(cm, split=True).detach().cpu().numpy()
    c_lr =  optimizer.param_groups[0]["lr"]
    print('Epoch: [{:d}]\tlr: {:.6f}\taccuracy: {:.1f}\thomog: {:.1f}'.format( 
            epoch+1, 
            c_lr,
            accuracy( lbl, pred )*100,
            homog( lbl, pred )*100,
        ), flush=True, end='\t')
        
    print('Loss:',a2s( cm_loss ), flush=True)
    import time 
    save_path = f"./blobs_test_a1.5_c3/plot_img_{time.time()}.png"
    plot_predictions(pred,save_path)
    return pred, lbl, cm_loss

def avg_epoch(dataloader):
    weights = {}
    for k in model.state_dict():
        weights[k] = model.state_dict()[k].detach()
    # switch to train mode
    model.train()
    i = 0
    for X, y in dataloader:
        X = X.float().to(device)
        cm = model(X)
        # loss_rc = criterion_reconst(X, tx)
        loss = criterion_cluster(cm)
        # compute gradient and do GD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        for k in weights:
            weights[k] = (weights[k]*i + model.state_dict()[k].detach()) / (i+1)
        i += 1
            
    model.load_state_dict(weights)

######################################################################################

for epoch in range(EPOCH):
    train(dataloader=train_loader)
    
    if (epoch)%10 == 0:
        print('.',end='\r')
        evaluate(dataloader=train_loader,full=False)
    if (epoch)%50 == 0:
        print( BATCH, ALPHA, BETA, LBD )




evaluate(dataloader=train_loader,full=False)
print('>>> End Training')
evaluate(dataloader=train_loader,full=True)
print('>>> Average Epoch')
avg_epoch(dataloader=train_loader)
evaluate(dataloader=train_loader,full=True)

