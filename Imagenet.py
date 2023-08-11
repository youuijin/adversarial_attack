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
    

    def __init__(self, root, mode, n_way, resize, color=3, startidx=0):
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
        self.n_way = n_way  # n-way
        self.resize = resize  # resize to
        self.color = color
        self.startidx = startidx  # index label not from 0, but from startidx
        #print('shuffle DB :%s, %d-way, resize:%d' % (mode, n_way, resize))

        self.transform = transforms.Compose([lambda x: Image.open(x).convert('RGB'),
                                                 transforms.Resize((self.resize, self.resize)),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0, 0, 0), (1, 1, 1))
                                                 ])
        # number of data per class
        if mode == 'train':
            self.data_len = 480
        elif mode == 'test' or mode =='val':
            self.data_len = 60
        else:
            self.data_len = 1

        self.path = os.path.join(root, 'imagesTest')  # image path

        self.get_datas(os.path.join(root, mode + '_res.csv'))

    def get_datas(self, csvf):
        class_mapping = {
            'n01532829': 0,
            'n01704323': 1,
            'n01843383': 2,
            'n01910747': 3,
            'n02089867': 4,
            'n02120079': 5,
            'n04418357': 6, 
            'n03146219': 7,
            'n02108551': 8,
            'n04612504': 9
        }
        skip = []
        for i in range((10-self.n_way)*self.data_len):
            skip.append(self.data_len*self.n_way+1+i)
        self.data = pd.read_csv(csvf, skiprows=skip)
        x_data = []
        y_data = []

        #for i in range(len(self.data)):
        for i in range(self.data_len*self.n_way):
            image_file = self.data.iloc[i][0]
            class_name = self.data.iloc[i][1]
            class_label = class_mapping[class_name]

            image_path = os.path.join(self.path, image_file)
            image = self.transform(image_path)
            
            x_data.append(image)
            y_data.append(class_label)

        self.x = torch.stack(x_data)
        self.y = torch.tensor(y_data)

        if self.color==1:
            self.x = torch.mean(self.x, dim=1, keepdim=True)

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
# %%
