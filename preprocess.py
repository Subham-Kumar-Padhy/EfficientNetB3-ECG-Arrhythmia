import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class ECGDataset(Dataset):
    def __init__(self, split='train'):
        data = np.load(f'data/{split}_images.npy')
        labels = np.load(f'data/{split}_labels.npy')
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def load_dataset(split='train', return_loader=False, batch_size=32):
    dataset = ECGDataset(split=split)
    if return_loader:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=(split == 'train'))
        return loader
    else:
        return dataset
