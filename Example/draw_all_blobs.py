import argparse
import matplotlib.pyplot as plt
from sklearn import datasets
import os

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
import pandas as pd
import os
import time
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score,davies_bouldin_score, adjusted_rand_score, normalized_mutual_info_score, homogeneity_completeness_v_measure

from itertools import cycle, islice
import torch.nn.functional as F

def plot_blobs(X_train, y_train, seed, output_dir):
    # Define a color map for the different clusters
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
                len(np.unique(y_train)),
            )
        )
    )
    
    # Create scatter plot of the blobs
    plt.figure(figsize=(8, 6))
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.get_cmap('tab10', len(np.unique(y_train))), s=10, edgecolor='k')
    
    plt.title(f'Blobs Plot (seed={seed})')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(label='Cluster label')
    
    # Save the plot to the specified directory
    plt.savefig(os.path.join(output_dir, f'blobs_seed_{seed}.png'))
    plt.close()
def main():
    parser = argparse.ArgumentParser(description='Generate and plot blobs data with different seeds.')
    parser.add_argument('--output_dir', '-o', type=str, required=True, help='Directory to save the plots')
    args = parser.parse_args()
    
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    BATCH = 100
    n_samples = BATCH * 100
    
    for seed in range(1, 200):  # Seed values from 1 to 199
        X_train, y_train = datasets.make_blobs(n_samples=n_samples, centers=5, random_state=seed)
        
        # Plot and save the blobs
        plot_blobs(X_train, y_train, seed, output_dir)

if __name__ == "__main__":
    main()
