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

#def generate_adversarial_examples(model, images, labels, image_dir, eps=0.0001):
def generate_adversarial_examples(model, images, labels, eps=0.0001):
    #loss_per_image = tf.nn.softmax_cross_entropy_with_logits(logits=model.logits_test, labels=model.label)
    loss_per_image = model.loss
    grad_op = tf.gradients(loss_per_image, model.x)

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

        preds_org_, answers_, grads_ = run_ops_test(model,
                                        [model.prediction, model.answer, grad_op],
                                        {model.x: images_, model.label: labels_})
        adversarial_images_ = np.clip(images_ + eps * np.sign(grads_[0]).astype(np.float32), 0., 1.)
        preds_adv_ = run_ops_test(model, model.prediction, {model.x: adversarial_images_})

        adversarial_images.extend(adversarial_images_)
        preds_org.extend(preds_org_)
        preds_adv.extend(preds_adv_)
        answers.extend(answers_)
        sys.stdout.write('\r%5d/%5d'%((i+1)*batch_size, len(images)))
        sys.stdout.flush()
    sys.stdout.write('\n')

    adversarial_images = np.asarray(adversarial_images)
    preds_org = np.asarray(preds_org)
    preds_adv = np.asarray(preds_adv)
    answers = np.asarray(answers)
    
    originals = []
    adversarials = []
    n_advs = 0
    for i, (po, pa, ans, img_org, img_adv) in enumerate(zip(preds_org, preds_adv, answers, images, adversarial_images)):
        originals.append(copy.deepcopy(img_org))
        adversarials.append(copy.deepcopy(img_adv))
        if po != pa and po == ans:
            n_advs += 1
            #originals.append(copy.deepcopy(img_org))
            #adversarials.append(copy.deepcopy(img_adv))
            #img_org *= 255.
            #img_org = cv2.cvtColor(img_org, cv2.COLOR_GRAY2BGR)
            #cv2.imwrite(os.path.join(image_dir, "img_original_%d.png" % i), img_org)
            #img_adv *= 255.
            #img_adv = cv2.cvtColor(img_adv, cv2.COLOR_GRAY2BGR)
            #cv2.imwrite(os.path.join(image_dir, "img_adversarial_%d.png" % i), img_adv)
    print('n_advs: ', n_advs)
    return np.array(adversarials), np.array(originals)

