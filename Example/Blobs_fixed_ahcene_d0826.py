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
#from custom_dataset import CustomDataset
import argparse
import matplotlib.pyplot as plt
import os
import time
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score,davies_bouldin_score, adjusted_rand_score, normalized_mutual_info_score, homogeneity_completeness_v_measure

from itertools import cycle, islice
import torch.nn.functional as F
def parse_arguments():
    """
    build and analyse the parameters of command
    return the parser(parameter object)
    """
    parser = argparse.ArgumentParser(description="Blobs_example")
    parser.add_argument("--alpha",'-a',type=float,default=1.1,required=True,help='set the Alpha which must be more than 1.0')
    parser.add_argument("--centroids",'-c',type=int, default=5, help='set the amount of centroids')
    parser.add_argument("--save_path_image",'-o', type=str, default='./',help='the savepath of plots')
    parser.add_argument("--c_alpha",'-ca',type=float, default=0.1, required=True, help='set the added alpha [0.1,1]')
    parser.add_argument("--freq",'-f', type=int, default=1,help='set the frequent')
    parser.add_argument("--seed",'-s', type=int, default=30,help='the random_state of blobs')
    parser.add_argument("--temperature", '-t', type=float, default=5, help='set the temperature of softmax')
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
    cm_loss_list = []
    for X, y in zip(*dataloader):
        X = X.float().to(device)
        cm = model(X)
        with torch.no_grad():
            cm_loss = criterion_cluster(cm, split=True).detach().cpu().numpy()
        #import pdb;pdb.set_trace()
            cm_loss_list.append(cm_loss)
        # loss_rc = criterion_reconst(X, tx)
        loss = criterion_cluster(cm)
        # compute gradient and do GD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return cm_loss_list
####################################################################################
######################################################################################
def evaluate(model,dataloader,criterion_cluster,optimizer,device, epoch,
             savepath=None,full=False,
             is_save=False, alpha=None, 
             centroids=None, c_alpha=None, 
             temp=None, seed_=None,
             cm_loss_list=None):
    model.eval()
    pred,lbl,x = [],[],None
    X_all = []
    y_all = []
    
    for i, (X, y) in enumerate(zip(*dataloader)):
        X_all += X.tolist()
        y_all.append(y)

        x = X.float().to(device)
        cm = model(x)
        #import pdb;pdb.set_trace()
        _,gamma,_,_ = cm
        pred += gamma.argmax(-1).detach().cpu().tolist()
        lbl += y.cpu().flatten().tolist()
    if cm_loss_list:
        cm_loss_1 = np.array([i[0] for i in cm_loss_list])
        cm_loss_2 = np.array([i[1] for i in cm_loss_list])
        cm_loss_3 = np.array([i[2] for i in cm_loss_list])
        cm_loss_4 = np.array([i[3] for i in cm_loss_list])
        print(np.std(cm_loss_2))
        print(np.std(cm_loss_3))
        #import pdb;pdb.set_trace()

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

    return pred, lbl, cm_loss

def avg_epoch(model,dataloader,criterion_cluster,optimizer,device):
    weights = {}
    for k in model.state_dict():
        weights[k] = model.state_dict()[k].detach()
    # switch to train mode
    model.train()
    i = 0
    for X, y in zip(*dataloader):
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

def plot_predictions(y_pred, X, C, save_path,alpha,c_alpha,centroids,temp,acc, 
                     silhouette_score_, davies_index, adjusted_score, 
                     normalized_score, homogeneity_score, completeness_score, 
                     v_measure_score_):
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
    plt.scatter(C[:,0],C[:,1], s=16, marker='s', color='k')
    plt.axis('auto')
    plt.xticks(())
    plt.yticks(())
    plt.title(f'A:{alpha},  C:{centroids},  C_A:{c_alpha},   T:{temp},')
    plt.text(0.5, -0.1, f'Acc:{acc}  Silhouette:{silhouette_score_}  Davies:{davies_index} Adjusted:{adjusted_score} \n Normalized:{normalized_score} Homogeneity:{homogeneity_score} Completeness: {completeness_score}V_measure:{v_measure_score_} ', ha='center', va='center', transform=plt.gca().transAxes)
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def main(args):
    if not os.path.exists(args.save_path_image):
        os.makedirs(args.save_path_image)
    EPOCH = 100
    # Parameters for normalized Loss
    BATCH = 100
    ALPHA = args.alpha  # modify
    print("===============")
    print(ALPHA)
    ######################################################################################
    n_samples = BATCH*100
    X_train, y_train = datasets.make_blobs(n_samples=n_samples, centers=5, random_state=args.seed) # random_state = 170, 180

    X_train = StandardScaler().fit_transform(X_train)
    X_train,y_train = torch.tensor(X_train).reshape((len(X_train)//BATCH,BATCH,-1)),torch.tensor(y_train).reshape((len(X_train)//BATCH,BATCH,-1))

    device = torch.device('cpu')
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
    cm_loss_list = []
    for epoch in range(EPOCH):
        cm_loss_list += train(model=model,
              dataloader=(X_train,y_train),
              criterion_cluster=criterion_cluster,
              optimizer=optimizer,
              device=device)

        if (epoch)%1 == 0:
            print('.',end='\r')
            pred,_,_, = evaluate(model=model,
              dataloader=(X_train,y_train),
              criterion_cluster=criterion_cluster,
              optimizer=optimizer,
              device=device,
              epoch=epoch,
              savepath=args.save_path_image,
              full=False,
              cm_loss_list=cm_loss_list)
            if (epoch+1)%10 == 0:
                with torch.no_grad(): 
                    print(pred)
                    freq = np.bincount( pred.astype(int), minlength=args.centroids ).astype(float) / args.temperature
                    freq = F.softmax(torch.tensor(freq), dim=-1)
                    freq = freq.numpy()
                    criterion_cluster.alpha = criterion_cluster.alpha*args.c_alpha + (torch.tensor(freq+1).float()*(1 - args.c_alpha)).to(device)
                    print(criterion_cluster.alpha)

    evaluate(model=model,
              dataloader=(X_train,y_train),
              criterion_cluster=criterion_cluster,
              optimizer=optimizer,
              epoch=EPOCH,
              device=device,
              full=False)
    print('>>> End Training')
    evaluate(model=model,
              dataloader=(X_train,y_train),
              criterion_cluster=criterion_cluster,
              optimizer=optimizer,
              device=device,
              full=True,
              epoch=EPOCH,)
    print('>>> Average Epoch')
    avg_epoch(model=model,
              dataloader=(X_train,y_train),
              criterion_cluster=criterion_cluster,
              optimizer=optimizer,
              device=device)

    evaluate(model=model,
              dataloader=(X_train,y_train),
              criterion_cluster=criterion_cluster,
              optimizer=optimizer,
              device=device,
              epoch=EPOCH,
              savepath=args.save_path_image,
              full=True,
              is_save=True,
              alpha=args.alpha,
              centroids=args.centroids,
              c_alpha=args.c_alpha,
              temp=args.temperature,
              seed_=args.seed
              )
    
    x_data = [i for i in range(len(cm_loss_list))]
    cm_loss_1 = np.array([i[0] for i in cm_loss_list])
    cm_loss_2 = np.array([i[1] for i in cm_loss_list])
    cm_loss_3 = np.array([i[2] for i in cm_loss_list])
    cm_loss_4 = np.array([i[3] for i in cm_loss_list])
    cm_loss_5 = cm_loss_1 + cm_loss_2
    cm_loss_6 = cm_loss_1 + cm_loss_3
    cm_loss_7 = cm_loss_2 + cm_loss_3
    cm_loss_8 = cm_loss_1 + cm_loss_2 + cm_loss_3
    # 创建包含8个子图的图形对象，并设置figsize
    fig, axs = plt.subplots(8, 1, figsize=(12, 28))  # 调整figsize来适配8个子图

    # 遍历并绘制每个子图
    for s, (i, j) in enumerate(zip(
        ["loss_E1", "loss_E2", "loss_E3", "loss_E4", "loss_E1 + loss_E2", "loss_E1 + loss_E3", "loss_E2 + loss_E3", "loss_E1 + loss_E2 + loss_E3"],
        [cm_loss_1, cm_loss_2, cm_loss_3, cm_loss_4, cm_loss_5, cm_loss_6, cm_loss_7, cm_loss_8]
    )):
        axs[s].plot(x_data, j, color='r', label='Scatter Plot')
        axs[s].set_title(i)
        axs[s].legend()
    fig.suptitle(f'A:{args.alpha} C:{args.centroids},  C_A:{args.c_alpha},   T:{args.temperature}', fontsize=16)

    # 调整布局
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # rect 参数留出总标题的位置

    # 保存图形
    plt.savefig(f'./loss_d0815/{args.seed}_a_{args.alpha}_ca_{args.c_alpha}_t_{args.temperature}.png', format='png', dpi=300)
    plt.close()


if __name__=='__main__':
    args = parse_arguments()
    main(args)
