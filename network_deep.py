# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 09:13:59 2018

@author: user
"""

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable
import numpy as np
from chainer import reporter
from PIL import Image

# Network definition
class CAE(chainer.Chain):

    def __init__(self, n_input, n_out, return_out=False):
        super(CAE, self).__init__(
                conv1 = L.Convolution2D(3, 32, 1, 1, 1),
                conv2 = L.Convolution2D(32, 64, 1 , 1, 1),
                conv3 = L.Convolution2D(64, 128, 1, 1, 1),
                conv4 = L.Convolution2D(128, 256, 1, 1, 1),
                l1 = L.Linear(None, 1000),
                l2 = L.Linear(1000, 50),
                dconv1 = L.Linear(50, 1000),
                dconv_mid = L.Linear(1000, 95744),
                dconv2 = L.Convolution2D(1000, 256, pad=1),
                dconv3 = L.Convolution2D(256, 128, pad=1),
                dconv4 = L.Convolution2D(128, 64, pad=1),
                dconv5 = L.Convolution2D(64, 32, pad=1),
                dconv6 = L.Convolution2D(32, 3, pad=1)
                )
        self.return_out = return_out

    def __call__(self, x, t):
        # Encoder
        import ipdb; ipdb.set_trace()
        e = F.relu(self.conv1(x))
        e = F.max_pooling_2d(e,ksize=2, stride=2,)
        e = F.relu(self.conv2(e))
        e = F.max_pooling_2d(e,ksize=2, stride=2,)
        e = F.relu(self.conv3(e))
        e = F.max_pooling_2d(e,ksize=2, stride=2,)
        e = F.relu(self.conv4(e))
        e = F.max_pooling_2d(e,ksize=2, stride=2,)
        e = F.relu(self.l1(e))
        e_out = F.relu(self.l2(e))

        # Decoder
        de = F.relu(self.dconv1(e_out))
        de = F.relu(self.dconv_mid(de))
        #de = F.relu(self.dconv2(e_out))
        # de = F.relu(self.dconv2(de.respahe(1, 256, 22, 17))
        # de = F.unpooling_2d(de, ksize=2, stride=2, cover_all=False)
        de = F.relu(self.dconv3(de.reshape(1, 256, 22, 17)))
        de = F.unpooling_2d(de, ksize=2, stride=2, cover_all=False)
        de = F.relu(self.dconv4(de))
        de = F.unpooling_2d(de, ksize=2, stride=2, cover_all=False)
        de = F.relu(self.dconv5(de))
        de = F.unpooling_2d(de, ksize=2, stride=2, cover_all=False)
        out = F.relu(self.dconv6(de))

        loss = F.mean_squared_error(out,t)

        reporter.report({'loss': loss}, self)

        if self.return_out == True:
            return out, loss
        else:
            return loss

class TmpCAE(chainer.Chain):

    def __init__(self, n_input, n_out, return_out=False):
        super(TmpCAE, self).__init__(
                conv1 = L.Convolution2D(3, 32, 1, 1, 1),
                conv2 = L.Convolution2D(32, 64, 1 , 1, 1),
                conv3 = L.Convolution2D(64, 128, 1 , 1, 1),
                dconv1 = L.Deconvolution2D(128, 64, pad=1),
                dconv2 = L.Deconvolution2D(64, 32, pad=1),
                dconv3 = L.Deconvolution2D(32, 3, pad=1)
                )
        self.return_out = return_out

    def __call__(self, x, t):
        # Encoder
        import ipdb; ipdb.set_trace()
        e = F.relu(self.conv1(x))
        e = F.max_pooling_2d(e,ksize=2, stride=2,)
        e = F.relu(self.conv2(e))
        e = F.max_pooling_2d(e,ksize=2, stride=2,)
        e_out = F.relu(self.conv3(e))

        # Decoder
        de = F.relu(self.dconv1(e_out))
        de = F.relu(self.dconv2(de))
        de = F.relu(self.dconv1(e_out))
        de = F.unpooling_2d(de, ksize=2, stride=2, cover_all=False)
        de = F.relu(self.dconv2(de))
        de = F.unpooling_2d(de, ksize=2, stride=2, cover_all=False)
        out = F.relu(self.dconv3(de))
        import ipdb; ipdb.set_trace()
        out = out.reshape(x.shape)

        loss = F.mean_squared_error(out,t)

        reporter.report({'loss': loss}, self)

        if self.return_out == True:
            return out, loss
        else:
            return loss
