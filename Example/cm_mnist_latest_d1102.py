import os,sys,time,copy
#os.environ["CUDA_VISIBLE_DEVICES"] = '0'#sys.argv[-1] if sys.argv[-1].isdigit() else '0'

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F

from lib_CM import *
from cv_cm import *

import warnings
warnings.filterwarnings('ignore')

from scipy.optimize import linear_sum_assignment
from sklearn.metrics import homogeneity_score as homog
from torch.utils.data import Subset
import argparse
from sklearn.metrics import silhouette_score,davies_bouldin_score, adjusted_rand_score, normalized_mutual_info_score, homogeneity_score, completeness_score, v_measure_score,homogeneity_completeness_v_measure

import umap
import matplotlib.pyplot as plt
import seaborn as sns


BETA = 200.
def parse_arguments():
    """
    build and analyse the parameters of command
    return the parser(parameter object)
    """
    parser = argparse.ArgumentParser(description="Mnist_example")
    parser.add_argument("--alpha",'-a',type=float,default=1.04,help='set the alpha which must be more than 1.0')
    parser.add_argument("--centroids",'-c',type=int, default=20, help='set the amount of centroids')
    parser.add_argument("--new_alpha",'-n',type=float, default=.1, help='set the added alpha [0.1,1], rename as new_alpha')
    parser.add_argument("--temperature", '-t', type=float, default=20, help='set the temperature of softmax')
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
    cm_loss_list = []
    for i, (images, _) in enumerate(dataloader):

        images = images.to(device)

        # compute output and loss
        tx, cm = model(x=images)
        loss_rc = criterion_reconst(images, tx)
        loss_cm = criterion_cluster(cm)
        loss = loss_rc * BETA + loss_cm
        loss_rc_weighted = loss_rc * BETA
        total_loss = loss_rc_weighted + loss_cm
        cm_loss_list.append({
            'total_loss': total_loss.detach().item(),
            'reconstruction_loss': loss_rc.detach().item(),
            'weighted_reconstruction_loss': loss_rc_weighted.detach().item(),
            'clustering_loss': loss_cm.detach().item(),
        })
        # compute gradient and do GD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return cm_loss_list
def evaluate_umap(
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
            new_alpha=None, 
            temp=None,
            cm_loss_list=None
            ):
    model.eval()

    pred,lbl,img = [],[],None
    all_gamma = []
    all_lbl = []
    all_pred = []
    for i, (images, labels) in enumerate(dataloader):
        #import pdb;pdb.set_trace()
        
        img = images.to(device)
        tx, cm = model(img)
        _,gamma,_,_ = cm
        pred += gamma.argmax(-1).detach().cpu().tolist()
        lbl += labels.cpu().tolist()
        all_gamma.append(gamma.detach().cpu().numpy())
        all_pred += gamma.argmax(-1).detach().cpu().tolist()
        all_lbl.extend(labels.cpu().tolist())

    all_gamma_np = np.concatenate(all_gamma, axis=0)
    all_lbl_np = np.array(all_lbl)
    all_pred_np = np.array(all_pred)
    pred = np.array(pred)
    lbl = np.array(lbl).astype(int)
    
    rc_loss = criterion_reconst(img, tx).detach().cpu().numpy()
    cm_loss = criterion_cluster(cm, split=True).detach().cpu().numpy()
    
    c_lr =  optimizer.param_groups[0]["lr"]
        
    # print('Epoch: [{:d}]\tlr: {:.6f}\taccuracy: {:.1f}\thomog: {:.1f}'.format( 
    #         epoch+1, 
    #         c_lr,
    #         accuracy( lbl, pred )*100,
    #         homog( lbl, pred )*100,
    #     ), flush=True, end='\t')
        
    # print('Rec:{:.6f}'.format( rc_loss * BETA ), end='\t', flush=True)
    # print('Loss:',a2s( cm_loss ), flush=True)


    umap_embedder = umap.UMAP(n_neighbors=15, random_state=42, metric='euclidean')
    umap_embeddings = umap_embedder.fit_transform(all_gamma_np)
    # all_lbl = np.array(all_lbl)
    # all_pred = np.array(all_pred)
    # all_gamma_np = np.array(all_gamma_np)
    # 检查是否保存并绘制 UMAP 嵌入图
    if is_save:
        gamma_np = gamma.detach().cpu().numpy()
        ACC = round(accuracy( lbl, pred )*100, 1)

        if gamma_np.shape[0] > 1:
            SS = round(silhouette_score(all_gamma_np, pred), 3)
            DBI = round(davies_bouldin_score(all_gamma_np, pred), 3)

        ARI = round(adjusted_rand_score(lbl, pred), 3)
        NMI = round(normalized_mutual_info_score(lbl, pred), 3)
        HS = round(homogeneity_score(lbl, pred), 3)
        CS = round(completeness_score(lbl, pred), 3)
        VM = round(v_measure_score(lbl, pred), 3)

        print(alpha, new_alpha, temp, ACC,SS,DBI,ARI,NMI,HS,CS,VM)
        plt.figure()
        plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], c=all_lbl_np, cmap="Spectral", s=0.1)
        plt.title(f'UMAP Embedding: Alpha {alpha}; New_alpha {new_alpha}; Temp {temp}')
        plt.colorbar(label="True Labels")
        savefig = f"/home/matteo/github_2/clustering_module/Example/experiments_d1105_mnist/umap_final_alpha_{alpha}_new_alpha_{new_alpha}_temp_{temp}.png"
        plt.savefig(savefig, dpi=300)
        plt.close()

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
            new_alpha=None, 
            temp=None,
            cm_loss_list=None
            ):
    model.eval()

    pred,lbl,img = [],[],None
    all_gamma = []
    all_lbl = []
    all_pred = []
    for i, (images, labels) in enumerate(dataloader):
        #import pdb;pdb.set_trace()
        
        img = images.to(device)
        tx, cm = model(img)
        _,gamma,_,_ = cm
        pred += gamma.argmax(-1).detach().cpu().tolist()
        lbl += labels.cpu().tolist()
        all_gamma.append(gamma.detach().cpu().numpy())
        all_pred += gamma.argmax(-1).detach().cpu().tolist()
        all_lbl.extend(labels.cpu().tolist())

    all_gamma_np = np.concatenate(all_gamma, axis=0)
    all_lbl_np = np.array(all_lbl)
    all_pred_np = np.array(all_pred)
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


    umap_embedder = umap.UMAP(n_neighbors=15, random_state=42, metric='euclidean')
    umap_embeddings = umap_embedder.fit_transform(all_gamma_np)
    # all_lbl = np.array(all_lbl)
    # all_pred = np.array(all_pred)
    # all_gamma_np = np.array(all_gamma_np)
    # 检查是否保存并绘制 UMAP 嵌入图
    if is_save:
        gamma_np = gamma.detach().cpu().numpy()
        ACC = round(accuracy( lbl, pred )*100, 1)

        if gamma_np.shape[0] > 1:
            SS = round(silhouette_score(all_gamma_np, pred), 3)
            DBI = round(davies_bouldin_score(all_gamma_np, pred), 3)

        ARI = round(adjusted_rand_score(lbl, pred), 3)
        NMI = round(normalized_mutual_info_score(lbl, pred), 3)
        HS = round(homogeneity_score(lbl, pred), 3)
        CS = round(completeness_score(lbl, pred), 3)
        VM = round(v_measure_score(lbl, pred), 3)

        print(alpha, new_alpha, temp, ACC,SS,DBI,ARI,NMI,HS,CS,VM)
        plt.figure()
        plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], c=all_lbl_np, cmap="Spectral", s=0.1)
        plt.title(f'UMAP Embedding: Alpha {alpha}; New_alpha {new_alpha}; Temp {temp}')
        plt.colorbar(label="True Labels")
        savefig = f"/home/matteo/github_2/clustering_module/Example/experiments_d1105_mnist/umap_final_alpha_{alpha}_new_alpha_{new_alpha}_temp_{temp}_all.png"
        plt.savefig(savefig, dpi=300)
        plt.close()
        # import pdb;pdb.set_trace()
        plot_loss_components(cm_loss_list,alpha=alpha,new_alpha=new_alpha,temp=temp)
        plot_loss_components_old(cm_loss_list,alpha=alpha,new_alpha=new_alpha,temp=temp)
    return pred, lbl, cm_loss, rc_loss
def plot_loss_components(cm_loss_list, alpha, new_alpha, temp):
    # 解包 total_loss, reconstruction_loss 和 clustering_loss
    total_losses = [item['total_loss'] for item in cm_loss_list]
    reconstruction_losses = [item['reconstruction_loss'] for item in cm_loss_list]
    clustering_losses = [item['clustering_loss'] for item in cm_loss_list]

    # 创建一个1行3列的子图
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    
    # 绘制 Reconstruction Loss
    axs[0].plot(reconstruction_losses, label='Reconstruction Loss', color='orange')
    axs[0].set_xlabel('Iteration')
    axs[0].set_ylabel('Reconstruction Loss')
    axs[0].set_title('Reconstruction Loss over Batches')
    axs[0].legend()

    # 绘制 Clustering Loss
    axs[1].plot(clustering_losses, label='Clustering Loss', color='green')
    axs[1].set_xlabel('Iteration')
    axs[1].set_ylabel('Clustering Loss')
    axs[1].set_title('Clustering Loss over Batches')
    axs[1].legend()

    # 绘制 Total Loss
    axs[2].plot(total_losses, label='Total Loss', color='blue')
    axs[2].set_xlabel('Iteration')
    axs[2].set_ylabel('Total Loss')
    axs[2].set_title('Total Loss over Batches')
    axs[2].legend()

    # 调整布局
    plt.tight_layout()
    savepath = f"/home/matteo/github_2/clustering_module/Example/experiments_d1105_mnist/alpha_{alpha}_new_alpha_{new_alpha}_temp_{temp}.png"
    plt.savefig(savepath)
def plot_loss_components_old(cm_loss_list,alpha,new_alpha,temp):
    # 解包 total_loss, reconstruction_loss, weighted_reconstruction_loss 和 clustering_loss
    total_losses = [item['total_loss'] for item in cm_loss_list]
    reconstruction_losses = [item['reconstruction_loss'] for item in cm_loss_list]
    weighted_reconstruction_losses = [item['weighted_reconstruction_loss'] for item in cm_loss_list]
    clustering_losses = [item['clustering_loss'] for item in cm_loss_list]

    # 创建一个2行2列的子图
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # 绘制 Reconstruction Loss
    axs[0, 0].plot(reconstruction_losses, label='Reconstruction Loss', color='orange')
    axs[0, 0].set_xlabel('Iteration')
    axs[0, 0].set_ylabel('Reconstruction Loss')
    axs[0, 0].set_title('Reconstruction Loss over Batches')
    axs[0, 0].legend()

    # 绘制 Weighted Reconstruction Loss
    axs[0, 1].plot(weighted_reconstruction_losses, label='Weighted Reconstruction Loss', color='purple')
    axs[0, 1].set_xlabel('Iteration')
    axs[0, 1].set_ylabel('Weighted Reconstruction Loss')
    axs[0, 1].set_title('Weighted Reconstruction Loss over Batches')
    axs[0, 1].legend()

    # 绘制 Clustering Loss
    axs[1, 0].plot(clustering_losses, label='Clustering Loss', color='green')
    axs[1, 0].set_xlabel('Iteration')
    axs[1, 0].set_ylabel('Clustering Loss')
    axs[1, 0].set_title('Clustering Loss over Batches')
    axs[1, 0].legend()

    # 绘制 Total Loss
    axs[1, 1].plot(total_losses, label='Total Loss', color='blue')
    axs[1, 1].set_xlabel('Iteration')
    axs[1, 1].set_ylabel('Total Loss')
    axs[1, 1].set_title('Total Loss over Batches')
    axs[1, 1].legend()

    # 调整布局
    plt.tight_layout()
    savepath = f"/home/matteo/github_2/clustering_module/Example/experiments_d1105_mnist/" + f"alpha_{alpha}_new_alpha_{new_alpha}_temp_{temp}_4parts.png" 
    plt.savefig(savepath)
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


def main(args):
    EPOCH = 100
    # Parameters for normalized Loss
    BATCH = 1024
    BETA = 200.
    LBD = .1

    print( BATCH, args.alpha, BETA, LBD )
    ######################################################################################
    #torch.cuda.set_device(0)


    # Define which digits to keep
    #allowed_labels = set(range(5))

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

    # train_dataset = filter_by_labels(train_dataset, allowed_labels)
    # test_dataset = filter_by_labels(test_dataset, allowed_labels)


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

    model = CV_CM(10,args.centroids).to(device)

    criterion_reconst = nn.MSELoss(reduction=('mean')).to(device)
    criterion_cluster = Clustering_Module_Loss(
                            num_clusters=args.centroids, 
                            alpha=args.alpha, 
                            lbd=1e-4,  # lbd == 0 
                            orth=True, # True ==> False 
                            normalize=True).to(device)

    optim_params = model.parameters()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3, 
        # betas=(.9,.999), 
        # eps=1e-3 
    )

    
    ######################################################################################
    cm_loss_list = []
    for epoch in range(EPOCH):
        cm_loss_list += train(model=model,
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
            if epoch == 59:
                evaluate_umap(model=model,
                                dataloader=test_loader,
                                criterion_cluster=criterion_cluster,
                                optimizer=optimizer,
                                device=device,
                                epoch=epoch,
                                criterion_reconst = criterion_reconst,
                                full=False,
                                is_save=True, 
                                alpha=args.alpha, 
                                new_alpha=args.new_alpha, 
                                temp=args.temperature,
                                cm_loss_list=cm_loss_list)

            if epoch > 50 and (epoch + 1)%10 == 0:
                with torch.no_grad(): 
                    # print(pred)
                    print( 'UPDATE ALPHA', criterion_cluster.alpha, end=' -> ')
                    freq = np.bincount( pred.astype(int), minlength=args.centroids ).astype(float) / args.temperature
                    freq = F.softmax(torch.tensor(freq), dim=-1)
                    freq = freq.numpy()
                    criterion_cluster.alpha = (criterion_cluster.alpha-1)*(1-args.new_alpha) 
                    criterion_cluster.alpha += (torch.tensor(freq).float()*args.new_alpha).to(device)
                    criterion_cluster.alpha += 1
                    #criterion_cluster.alpha = torch.clamp(criterion_cluster.alpha, min=1.01)
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
            new_alpha=args.new_alpha, 
            temp=args.temperature,
            cm_loss_list=cm_loss_list
            )



if __name__=='__main__':
    args = parse_arguments()
    main(args)
