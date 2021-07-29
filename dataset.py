import glob
import random
import os
from torch._C import Value
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torch

class ImageDataset(Dataset):
    def __init__(self, root, transforms_ = None, unaligned = False, mode = 'train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        #img A,B
        self.files_A = sorted(glob.glob(os.path.join(root, mode+'A')+ '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, mode+'B')+ '/*.*'))

        self.train_dataA = []
        self.train_dataB = []


    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))

        if self.unaligned:
            item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B)-1)]))
        else:
            item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]))
        return {'A': item_A, 'B':item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))