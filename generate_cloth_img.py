# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 08:51:18 2018

@author: user
"""

import argparse

import os
import os.path as osp

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import chainer
from chainer import cuda
from chainer.datasets import get_cifar10
from chainer import dataset
from chainer import Variable
from chainer import serializers
import chainer.functions as F
import numpy as np
from sklearn.model_selection import train_test_split
from skimage.io import imread

import network


# Load data
class LoadDataset(dataset.DatasetMixin):
    def __init__(self, split, return_image=False):
        assert split in ('train', 'val')
        ids = self._get_ids()
        iter_train, iter_val = train_test_split(
            ids, test_size=0.2, random_state=np.random.RandomState(1234))
        self.ids = iter_train if split == 'train' else iter_val
        self._return_image = return_image

    def __len__(self):
        return len(self.ids)

    def _get_ids(self):
        ids = []
        dataset_dir = chainer.dataset.get_dataset_directory(
            '2019_11_28_pr2')
        for data_id in os.listdir(dataset_dir):
            ids.append(osp.join(dataset_dir , data_id))
        return ids

    def img_to_datum(self, img):
        img = img.copy()
        datum = img.astype(np.float32)
        datum = datum[:, :, ::-1] #RGB -> BGR
        datum = datum.transpose((2, 0, 1))
        return datum

    def get_example(self, i):
        id = self.ids[i]
        image_file = osp.join(id , "image.png")
        img = imread(image_file)
        datum = self.img_to_datum(img)
        if self._return_image:
            return img
        else:
            return datum, datum

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--model', '-m', type=str, default="./results/cloth/model")
    parser.add_argument('--begin', '-b', type=int, default=0)
    args = parser.parse_args()

    # Set up a neural network to train.
    test = LoadDataset(split='val')

    model = network.CAE(3,3, return_out=True)
    
    if args.model != None:
        print( "loading model from " + args.model )
        serializers.load_npz(args.model, model)
    
    # Show 64 images
    fig = plt.figure(figsize=(6,6))
    plt.title("Original images: first rows,\n Predicted images: second rows")
    plt.axis('off')
    plt.tight_layout()
    
    pbar = tqdm(total=8)
    #import ipdb; ipdb.set_trace()
    for i in range(2):
        for j in range(2):
            ax = fig.add_subplot(4, 2, i*4+j+1, xticks=[], yticks=[])
            x, t = test[i*2+j]
            xT = x.transpose(1, 2, 0)
            xT = xT.astype(np.uint8)
            ax.imshow(xT, cmap=plt.cm.bone, interpolation='nearest')
            
            x = np.expand_dims(x, 0)
            t = np.expand_dims(t, 0)
    
            if args.gpu >= 0:
                cuda.get_device_from_id(0).use()
                model.to_gpu()
                x = cuda.cupy.array(x)
                t = cuda.cupy.array(t)
            
            predicted, loss = model(Variable(x), Variable(t))
            #print(predicted.shape)
            #print(loss)   
            
            predicted = F.transpose(predicted[0], (1, 2, 0))
            predicted = cuda.to_cpu(predicted.data) #Variable to numpy
            predicted = predicted * 255
            predicted = predicted.astype(np.uint8) 
            ax = fig.add_subplot(4, 2, i*4+j+3, xticks=[], yticks=[])
            ax.imshow(predicted, cmap=plt.cm.bone, interpolation='nearest')

            pbar.update(1)
            
    pbar.close()
   
    plt.savefig("result.png")
    plt.show()
    plt.close()

if __name__ == '__main__':
    main()
