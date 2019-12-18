# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 08:51:18 2018

@author: user
"""
import os
import os.path as osp

import argparse

import chainer
from chainer import training
from chainer.training import extensions
from chainer import iterators, optimizers, serializers
from chainer import cuda
from chainer.datasets import get_cifar10
from chainer import dataset
import numpy as np
from sklearn.model_selection import train_test_split
from skimage.io import imread
from skimage.transform import resize
import network_deep

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
            '2019_12_03')
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
        img = resize(img, (50 , 50))
        datum = self.img_to_datum(img)
        if self._return_image:
            return img
        else:
            return datum, datum

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--model', '-m', type=str, default=None)
    parser.add_argument('--opt', type=str, default=None)
    parser.add_argument('--epoch', '-e', type=int, default=50)
    parser.add_argument('--lr', '-l', type=float, default=0.001)
    parser.add_argument('--batch', '-b', type=int, default=1)
    parser.add_argument('--noplot', dest='plot', action='store_false')
    args = parser.parse_args()

    # Set up a neural network to train.
    train = LoadDataset(split='train')
    test = LoadDataset(split='val')

    train_iter = iterators.SerialIterator(train, batch_size=args.batch, shuffle=True)
    test_iter = iterators.SerialIterator(test, batch_size=args.batch, repeat=False, shuffle=False)
    
    # Define model
    model = network_deep.TmpCAE(3,3)
    
    # Load weight
    if args.model != None:
        print( "loading model from " + args.model )
        serializers.load_npz(args.model, model)

    if args.gpu >= 0:
        cuda.get_device_from_id(0).use()
        model.to_gpu()

    # Define optimizer
    opt = optimizers.Adam(alpha=args.lr)
    opt.setup(model)

    if args.opt != None:
        print( "loading opt from " + args.opt )
        serializers.load_npz(args.opt, opt)

    # Set up a trainer
    updater = training.StandardUpdater(train_iter, opt, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out='results')

    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))
    trainer.extend(extensions.LogReport(trigger=(10, 'iteration')))
    
    if args.plot and extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(['main/loss', 'validation/main/loss'],
                                  'epoch', file_name='loss.png'))
    
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss']))
    trainer.extend(extensions.ProgressBar(update_interval=1))
    
    # Train
    trainer.run()

    # Save results
    modelname = "./results/cloth/model"
    print( "saving model to " + modelname )
    serializers.save_npz(modelname, model)

    optname = "./results/cloth/opt"
    print( "saving opt to " + optname )
    serializers.save_npz(optname, opt) 

if __name__ == '__main__':
    main()
