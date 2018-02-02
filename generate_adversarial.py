from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os
import random
import pprint
import numpy as np
import tensorflow as tf

sys.path.append('/home/chiba/research/tensorflow/dl_utils/')
from data import *
sys.path.append('/home/chiba/research/master/classifiers/')

flags = tf.app.flags
flags.DEFINE_integer('eps', 2, 'integer value in [0, 255] which indicates the strength of a pertubation')
flags.DEFINE_string('dataset', 'fmnist', 'dataset name [fmnist, cifar10]')
flags.DEFINE_string('save_dir', 'save', 'directory to save adversarial examples')
flags.DEFINE_string('gpu_list', '0', 'gpu numbers to use')
flags.DEFINE_string('checkpoint_path', '', 'path to saved classifier model')
flags.DEFINE_string('method', 'fgs', 'method to use to generate adversarial expamples [fgs, bim, deepfool, cw]')
flags.DEFINE_string('nn_type', 'resnet', 'neural networks type [resnet, vgg]')

FLAGS = flags.FLAGS

def main(argv):
    if FLAGS.dataset == 'fmnist':
        FLAGS.input_shape = (28, 28, 1)
        FLAGS.cnn_dim = 4
        FLAGS.augment = False
        dataset = FashionMnistDataset(code_dim=0, code_init=None)
    elif FLAGS.dataset == 'cifar10':
        FLAGS.input_shape = (24, 24, 3)
        FLAGS.cnn_dim = 16
        FLAGS.augment = True
        dataset = Cifar10Dataset('/home/chiba/data/cifar10/cifar-10-batches-py', code_dim=0, code_init=None)
    else:
        raise ValueError('Dataset %s is unsupported.'%FLAGS.dataset)

    if FLAGS.nn_type == 'resnet':
        from classifiers.resnet import Classifier
    elif FLAGS.nn_type == 'vgg':
        from classifiers.vgg import Classifier
    else:
        raise ValueError('Neural Network %s is unsupported.'%FLAGS.nn_type)

    if FLAGS.method == 'fgs':
        from methods.fast_gradient_sign import generate_adversarial_examples
    elif FLAGS.method == 'bim':
        from methods.basic_iterative_method import generate_adversarial_examples
    elif FLAGS.method == 'deepfool':
        from methods.deepfool import generate_adversarial_examples
    elif FLAGS.method == 'cw':
        from methods.l2_attack import CarliniL2
    elif FLAGS.method == 'random':
        from methods.uniform_noise import generate_adversarial_examples
    # TODO: implement L-BFGS attack
    #elif FLAGS.method == 'lbfgs':
    #    from methods.l_bfgs import generate_adversarial_examples
    else:
        raise ValueError('Method %s is unsupported.'%FLAGS.method)

    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(FLAGS.__flags)

    if not os.path.exists(FLAGS.save_dir):
        os.makedirs(FLAGS.save_dir)
    #FLAGS.image_dir = os.path.join(FLAGS.save_dir, 'images')
    #if not os.path.exists(FLAGS.image_dir):
    #    os.makedirs(FLAGS.image_dir)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = FLAGS.gpu_list

    with tf.Session(config=config) as sess:
        clf = Classifier(sess, 1, FLAGS.input_shape, FLAGS.cnn_dim, '', FLAGS.augment)
        clf.build_model()
        clf.saver.restore(sess, FLAGS.checkpoint_path)

        eps = float(FLAGS.eps) / 255.
        images = dataset.test_images
        labels = dataset.test_labels
        if FLAGS.augment:
            images = crop_images(images, FLAGS.input_shape)

        #if FLAGS.method == 'lbfgs':
        #    adversarials, originals = generate_adversarial_examples(clf, images, labels, FLAGS.image_dir)

        if FLAGS.method == 'cw':
            # create random target labels
            targets = []
            for image, label in zip(images, labels):
                while True:
                    t = random.randint(0, 9)
                    if t != np.argmax(label): break
                targets.append(t)
            targets = np.asarray(targets)
            one_hot = np.zeros((len(images), 10))
            one_hot[np.arange(len(images)), targets] = 1.
            targets = one_hot

            attack = CarliniL2(sess, clf, dataset, eps, batch_size=1000, max_iterations=1000, confidence=0, boxmin=0.0, boxmax=1.0)
            adversarials = attack.attack(images, targets)
        else:
            #adversarials, originals = generate_adversarial_examples(clf, images, labels, FLAGS.image_dir, eps=eps)
            adversarials, originals = generate_adversarial_examples(clf, images, labels, eps=eps)

        adversarials.dump(os.path.join(FLAGS.save_dir, 'adversarials.pkl'))
        #originals.dump(os.path.join(FLAGS.save_dir, 'originals.pkl'))
        print('done.')

if __name__ == '__main__':
    tf.app.run()

