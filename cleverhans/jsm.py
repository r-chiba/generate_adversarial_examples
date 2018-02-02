from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import os

import numpy as np
from six.moves import xrange
import tensorflow as tf
from tensorflow.python.platform import flags

import logging
import os
from cleverhans.attacks import SaliencyMapMethod
from cleverhans.utils import pair_visual, grid_visual, AccuracyReport
from cleverhans.utils import set_log_level
from cleverhans.utils_tf import model_train, model_eval, tf_model_load

from classifier_extend import classifier_extend
sys.path.append('/home/chiba/research/master/classifiers/')
sys.path.append('/home/chiba/research/tensorflow/dl_utils/')
from utils import *
from data import *

FLAGS = flags.FLAGS


def cw_attack(dataset_name, nn_type, save_dir, eps, gpu_list,
                batch_size=128, nb_classes=10, source_samples=10000,
                model_path=os.path.join("models", "mnist")):

    # Object used to keep track of (and return) key accuracies
    report = AccuracyReport()

    if dataset_name == 'fmnist':
        input_shape = (28, 28, 1)
        cnn_dim = 4
        augment = False
        dataset = FashionMnistDataset(code_dim=0, code_init=None)
    elif dataset_name == 'cifar10':
        input_shape = (24, 24, 3)
        cnn_dim = 16
        augment = True
        dataset = Cifar10Dataset('/home/chiba/data/cifar10/cifar-10-batches-py', code_dim=0, code_init=None)

    if nn_type == 'resnet':
        from classifiers.resnet import Classifier
    elif nn_type == 'vgg':
        from classifiers.vgg import Classifier
    else:
        raise ValueError('Neural Network %s is unsupported.'%FLAGS.nn_type)

    # MNIST-specific dimensions
    img_rows, img_cols, channels = input_shape

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)

    # Create TF session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = gpu_list
    sess = tf.Session(config=config)
    print("Created TensorFlow session.")

    set_log_level(logging.DEBUG)

    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols, channels))
    y = tf.placeholder(tf.float32, shape=(None, nb_classes))

    X_test, Y_test = dataset.test_images, dataset.test_labels
    if augment:
        X_test = crop_images(X_test, input_shape)

    # Define TF model graph
    #model = make_basic_cnn()
    model = Classifier(sess, batch_size, input_shape, cnn_dim, save_dir, augment)
    model.build_model()
    model = classifier_extend(model)
    preds = model.get_probs(x)
    print("Defined TensorFlow model graph.")

    rng = np.random.RandomState([2017, 8, 30])
    # check if we've trained before, and if we have, use that pre-trained model
    if os.path.exists(model_path + ".meta"):
        tf_model_load(sess, model_path)

    # Evaluate the accuracy of the MNIST model on legitimate test examples
    eval_params = {'batch_size': batch_size}
    accuracy = model_eval(sess, x, y, preds, X_test, Y_test, args=eval_params)
    print('Test accuracy on legitimate test examples: {0}'.format(accuracy))
    report.clean_train_clean_eval = accuracy

    ###########################################################################
    # Craft adversarial examples using Carlini and Wagner's approach
    ###########################################################################
    nb_adv_per_sample = '1'
    print('Crafting ' + str(source_samples) + ' * ' + nb_adv_per_sample +
          ' adversarial examples')
    print("This could take some time ...")

    # Instantiate a CW attack object
    jsm = SaliencyMapMethod(model, back='tf', sess=sess)

    adv_inputs = X_test[:source_samples]

    adv_ys = None
    yname = "y"

    jsm_params = {'clip_min': 0., 'clip_max': 1.}

    #adv = fgm.generate_np(adv_inputs,
    #                     **fgm_params)
    #adv = np.clip(adv, adv-eps, adv+eps)

    n_batches = source_samples // batch_size
    adv = []
    for i in xrange(n_batches):
        batch = adv_inputs[i*batch_size:(i+1)*batch_size]
        adv_ = jsm.generate_np(batch,
                             **jsm_params)
        adv_ = np.clip(adv_, batch-eps, batch+eps)
        adv.extend(list(adv_))
    adv = np.asarray(adv)

    adv.dump(os.path.join(save_dir, 'adversarials.pkl'))
    adv_inputs.dump(os.path.join(save_dir, 'originals.pkl'))

    eval_params = {'batch_size': np.minimum(nb_classes, source_samples)}
    adv_accuracy = 1 - \
        model_eval(sess, x, y, preds, adv, Y_test[
                   :source_samples], args=eval_params)

    print('--------------------------------------')

    # Compute the number of adversarial examples that were successfully found
    print('Avg. rate of successful adv. examples {0:.4f}'.format(adv_accuracy))
    report.clean_train_adv_eval = 1. - adv_accuracy

    # Compute the average distortion introduced by the algorithm
    percent_perturbed = np.mean(np.sum((adv - adv_inputs)**2,
                                       axis=(1, 2, 3))**.5)
    print('Avg. L_2 norm of perturbations {0:.4f}'.format(percent_perturbed))

    # Close TF session
    sess.close()

    return report


def main(argv=None):
    if not os.path.exists(FLAGS.save_dir):
        os.makedirs(FLAGS.save_dir)

    eps = float(FLAGS.eps) / 255.
    cw_attack(FLAGS.dataset, FLAGS.nn_type, FLAGS.save_dir, eps,
                gpu_list=FLAGS.gpu_list,
                batch_size=FLAGS.batch_size,
                nb_classes=FLAGS.nb_classes,
                source_samples=FLAGS.source_samples,
                model_path=FLAGS.model_path)

if __name__ == '__main__':
    flags.DEFINE_integer('batch_size', 100, 'Size of training batches')
    flags.DEFINE_integer('nb_classes', 10, 'Number of output classes')
    flags.DEFINE_integer('source_samples', 10, 'Nb of test inputs to attack')
    flags.DEFINE_integer('eps', 2, 'integer value in [0, 255] which indicates the strength of a pertubation')
    flags.DEFINE_string('dataset', 'fmnist', 'dataset name [fmnist, cifar10]')
    flags.DEFINE_string('save_dir', 'save', 'directory to save adversarial examples')
    flags.DEFINE_string('model_path', os.path.join("models", "mnist"),
                        'Path to save or load the model file')
    flags.DEFINE_string('gpu_list', '0', 'gpu numbers to use')
    flags.DEFINE_string('nn_type', 'resnet', 'neural networks type [resnet, vgg]')

    tf.app.run()
