from __future__ import print_function
from __future__ import division

import os
import sys
import time
import math
import random
import glob
import numpy as np
import tensorflow as tf
import cv2

from cleverhans.model import Model

sys.path.append('/home/chiba/research/tensorflow/dl_utils/')
from utils import *
from data import *
sys.path.append('/home/chiba/research/tensorflow/classifiers/')

class ClfWrapper(Model):
    def __init__(self, clf):
        self.clf = clf

    def __call__(self, *args, **kwargs):
        return self.get_probs(*args, **kwargs)
    
    def get_layer_names(self):
        if not hasattr(self, 'layer_names'):
            if 'vgg' in str(self.clf.__class__):
                self.layer_names = ['h0', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'logits']
            else: # resnet
                self.layer_names = ['h0', 'h1', 'h2', 'h3', 'logits']
        return self.layer_names
    
    def fprop(self, x):
        if 'vgg' in str(self.clf.__class__):
            codes = self.clf.model(x, 1.0, training=False, reuse=True)
        else:
            codes = self.clf.model(x, training=False, reuse=True)
        states = dict(zip(self.get_layer_names(), codes[::-1]))
        return states
    
    def get_layer(self, x, layer):
        states = self.fprop(x)
        return states[layer]
    
    def get_logits(self, x):
        states = self.fprop(x)
        logits = states['logits']
        return logits
    
    def get_probs(self, x):
        logits = self.get_logits(x)
        probs = tf.nn.softmax(logits)
        return probs
    
