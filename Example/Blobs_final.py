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
import argparse
import matplotlib.pyplot as plt
import os

from itertools import cycle, islice

def parse_arguments():
    """
    build and analyse the parameters of command
    return the parser(parameter object)
    """
    parser = argparse.ArgumentParser(description="Blobs_example")
    parser.add_argument("--Alpha",'-a',type=float,default=1.1,required=True,help='set the Alpha which must be more than 1.0')
    parser.add_argument("--centroids",'-c',type=int, default=int, required=True, help='set the amount of centroids')
    parser.add_argument("--save_path_image",'-o', type=str, default='/home/matteo/thesis_2/clustering-Module-main/images',help='the savepath of plots')
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
####################################################################################
######################################################################################
def train(model,dataloader,criterion_cluster,optimizer,device,):

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

def evaluate(model,dataloader,criterion_cluster,optimizer,device, epoch,savepath,full=False,is_save=False, alpha=None, centroids=None):
    model.eval()
    pred,lbl,x = [],[],None
    X_all = []
    for i, (X, y) in enumerate(dataloader):
        X_all = X.tolist()
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
    save_path = savepath + f"/a{alpha}c{centroids}.png"
    if is_save:
        plot_predictions(pred,X_all,save_path, alpha, centroids)
    return pred, lbl, cm_loss

def avg_epoch(model,dataloader,criterion_cluster,optimizer,device):
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
# def plot_predictions(pred, save_path, alpha, centroids):
#     plt.figure(figsize=(10, 5))
#     #plt.scatter(pred, range(len(pred)), c=pred, cmap='viridis', marker='o', alpha=0.5)
#     plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])
#     plt.xlabel('Predicted Class')
#     plt.ylabel('Index')
#     plt.title(f'Alpha={alpha}, Centroids={centroids}')
#     plt.colorbar(label='Class')
#     if save_path:
#         plt.savefig(save_path)
#     else:
#         plt.show()

def plot_predictions(y_pred, X, save_path,alpha, centroids):
    colors = np.array(
        list(
            islice(
                cycle(
                    [
                        "#377eb8",
                        "#ff7f00",
                        "#4daf4a",
                        "#f781bf",
                        "#a65628",
                        "#984ea3",
                        "#999999",
                        "#e41a1c",
                        "#dede00",
                    ]
                ),
                int(max(y_pred) + 1),
            )
        )
    )
    colors = np.append(colors, ["#000000"])
    X_0 = [row[0] for row in X]
    X_1 = [row[1] for row in X]
    plt.scatter(X_0,X_1, s=10, color=colors[y_pred])
    # plt.xlim(-2.5, 2.5)
    # plt.ylim(-2.5, 2.5)
    plt.axis('auto')
    plt.xticks(())
    plt.yticks(())
    plt.title(f'Alpha={alpha}, Centroids={centroids}')
    # plt.text(
    #     0.99,
    #     0.01,
    #     (transform=plt.gca().transAxes,                       
    #     size=15,
    #     horizontalalignment="right",
    # )
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def main(args):
    if not os.path.exists(args.save_path_image):
        os.makedirs(args.save_path_image)
    EPOCH = 2000
    # Parameters for normalized Loss
    BATCH = 256
    ALPHA = args.Alpha  # modify
    BETA = 100.
    LBD = .1
    print( BATCH, ALPHA, BETA, LBD )
    ######################################################################################
    n_samples = 10000
    random_state = 195
    #X, y = datasets.make_blobs(n_samples=n_samples, centers=5, random_state=170) # random_state = 170, 180
    #X, y = datasets.make_blobs(n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=random_state)
    X, y = datasets.make_blobs(n_samples=n_samples, centers=5, random_state=random_state)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X = np.dot(X, transformation)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_dataset = CustomDataset(X_train, y_train)#transform=transform
    test_dataset = CustomDataset(X_test, y_test)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8000, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

        
    # create model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = Clustering_Module(2, args.centroids, False).to(device)
    criterion_cluster = Clustering_Module_Loss(
                            num_clusters=args.centroids, 
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

    for epoch in range(EPOCH):
        train(model=model,
              dataloader=train_loader,
              criterion_cluster=criterion_cluster,
              optimizer=optimizer,
              device=device)
        
        if (epoch)%100 == 0:
            print('.',end='\r')
            evaluate(model=model,
              dataloader=train_loader,
              criterion_cluster=criterion_cluster,
              optimizer=optimizer,
              device=device,
              epoch=epoch,
              savepath=args.save_path_image,
              full=False,
              is_save=False)
        if (epoch)%100 == 0:
            print( BATCH, ALPHA, BETA, LBD )
    # torch.save({
    #     'model_state_dict': model.state_dict(),
    #     'optimizer_state_dict': optimizer.state_dict(),
    #     'criterion_cluster': criterion_cluster.state_dict(),
    # }, 'model_checkpoint.pth')

    evaluate(model=model,
              dataloader=train_loader,
              criterion_cluster=criterion_cluster,
              optimizer=optimizer,
              epoch=EPOCH,
              savepath=args.save_path_image,
              device=device,
              full=False,
              is_save=False)
    print('>>> End Training')
    evaluate(model=model,
              dataloader=train_loader,
              criterion_cluster=criterion_cluster,
              optimizer=optimizer,
              device=device,
              savepath=args.save_path_image,
              full=True,
              epoch=EPOCH,
              is_save=False)
    print('>>> Average Epoch')
    avg_epoch(model=model,
              dataloader=train_loader,
              criterion_cluster=criterion_cluster,
              optimizer=optimizer,
              device=device)
    evaluate(model=model,
              dataloader=train_loader,
              criterion_cluster=criterion_cluster,
              optimizer=optimizer,
              device=device,
              epoch=EPOCH,
              savepath=args.save_path_image,
              full=True,
              is_save=True,
              alpha=args.Alpha,
              centroids=args.centroids)



if __name__=='__main__':
    args = parse_arguments()
    main(args)




