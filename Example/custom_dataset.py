import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class CustomDataset(Dataset):
    def __init__(self, X, y=None, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self,index):
        sample = self.X[index]
        if self.transform:
            sample = self.transform(sample)
        if self.y is not None:
            return sample, self.y[index]

