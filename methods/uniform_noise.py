from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os
import copy

import numpy as np
import tensorflow as tf

sys.path.append('/home/chiba/research/tensorflow/dl_utils/')
from utils import *
from data import *

def generate_adversarial_examples(model, images, labels, eps=0.0001):
    batch_size = 1
    n_batches = len(images) // batch_size
    adversarial_images = []
    preds_org = []
    preds_adv = []
    answers = []
    grads = []
    for i in xrange(n_batches):
        images_ = images[i*batch_size:(i+1)*batch_size]
        labels_ = labels[i*batch_size:(i+1)*batch_size]

        noise = np.random.uniform(-eps, eps+np.finfo(np.float32).eps, size=images_.shape)
        adversarial_images_ = np.clip(images_ + noise, 0., 1.)

        adversarial_images.extend(adversarial_images_)
        sys.stdout.write('\r%5d/%5d'%((i+1)*batch_size, len(images)))
        sys.stdout.flush()
    sys.stdout.write('\n')

    return np.asarray(adversarial_images), np.array(images)

