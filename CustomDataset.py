import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms, utils
import torchvision.datasets as datasets

class CustomDataset(Dataset):

    def __init__(self, root_dir):

        self.root_dir = root_dir

        # dataset
        self.image_dataset = []
        self.classes = []
        self.caption = []

        for set in ["Hard", "Mid", "Easy"]:
            data_file = os.path.join(self.root_dir, set, "data.txt")
            with open(data_file, 'r') as f:
                for line in f:
                    info = line.split(',')
                    self.image_dataset.append(info[0])
                    self.classes.append(info[1])
                    caption = info[2]
                    if info[2][-1] == '\n': # get rid of \n
                        caption = info[2][:-1]
                    self.caption.append(caption)
        print("dataset built with", len(self.classes), "instances")
        print("get item will return: (img_path, img_class, img_caption)")
        

    def __len__(self):
        return len(self.image_dataset)

    def __getitem__(self, idx):
        return self.image_dataset[idx], self.classes[idx], self.caption[idx]