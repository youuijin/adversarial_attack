#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import numpy as np
import collections
from PIL import Image
import csv
import random
import pandas as pd


class Imagenet(Dataset):
    """
    put mini-imagenet files as :
    root :
        |- images/*.jpg includes all imgeas
        |- train.csv
        |- test.csv
        |- val.csv
    NOTICE: meta-learning is different from general supervised learning, especially the concept of batch and set.
    batch: contains several sets
    sets: conains n_way * k_shot for meta-train set, n_way * n_query for meta-test set.
    """
    

    def __init__(self, root, mode, n_way, resize, startidx=0):
        """
        :param root: root path of mini-imagenet
        :param mode: train, val or test
        :param batchsz: batch size of sets, not batch of imgs
        :param n_way:
        :param k_shot:
        :param k_query: num of qeruy imgs per class
        :param resize: resize to
        :param startidx: start to index label from startidx
        """
        np.random.seed(222)
        random.seed(222)
        self.n_way = n_way  # n-way
        self.resize = resize  # resize to
        self.startidx = startidx  # index label not from 0, but from startidx
        #print('shuffle DB :%s, %d-way, resize:%d' % (mode, n_way, resize))

        if mode == 'train':
            self.transform = transforms.Compose([lambda x: Image.open(x).convert('RGB'),
                                                 transforms.Resize((self.resize, self.resize)),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0, 0, 0), (1, 1, 1))
                                                 ])
        else:
            self.transform = transforms.Compose([lambda x: Image.open(x).convert('RGB'),
                                                 transforms.Resize((self.resize, self.resize)),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0, 0, 0), (1, 1, 1))
                                                 ])

        self.path = os.path.join(root, 'imagesTest')  # image path

        self.get_datas(os.path.join(root, mode + '_res.csv'))

    def get_datas(self, csvf):
        class_mapping = {
            'n01532829': 0,
            'n01704323': 1,
            'n01843383': 2,
            'n01910747': 3,
            'n02089867': 4
        }

        self.data = pd.read_csv(csvf)
        x_data = []
        y_data = []

        for i in range(len(self.data)):
            image_file = self.data.iloc[i][0]
            class_name = self.data.iloc[i][1]
            class_label = class_mapping[class_name]

            image_path = os.path.join(self.path, image_file)

            #image = Image.open(image_path)
            image = self.transform(image_path)
            #print(image.shape)
            
            x_data.append(image)
            y_data.append(class_label)

        self.x = torch.stack(x_data)
        self.y = torch.tensor(y_data)

        return self.x, self.y


    def __getitem__(self, index):
        """
        index means index of sets, 0<= index <= batchsz-1
        :param index:
        :return:
        """
        return self.x[index], self.y[index]

    def __len__(self):
        # as we have built up to batchsz of sets, you can sample some small batch size of sets.
        return len(self.data)


if __name__ == '__main__':
    pass