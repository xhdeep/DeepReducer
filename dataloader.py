import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class SoundDataset(Dataset):
    def __init__(self):
        self.df = pd.read_csv('train_path.csv')
        self.nums = len(self.df)

    def __getitem__(self, item):
        p1 = self.df.iloc[item, 0]
        p2 = self.df.iloc[item, 1]
        input = np.load(p1)
        output = np.load(p2)
        input = torch.tensor(input)
        output = torch.tensor(output)
        return input, output

    def __len__(self):
        return self.nums