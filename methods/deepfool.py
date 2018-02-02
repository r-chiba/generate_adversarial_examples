from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os
import math
import copy

import numpy as np
import tensorflow as tf

sys.path.append('/home/chiba/research/tensorflow/dl_utils/')
from utils import *
from data import *

#def generate_adversarial_examples(model, images, labels, image_dir, overshoot=0.02, eps=0.0001, max_iter=50):
def generate_adversarial_examples(model, images, labels, overshoot=0.02, eps=0.0001, max_iter=50):
    n_classes = len(labels[0])
    adversarial_images = []
    preds_org = []
    preds_adv = []
    answers = []
    grads = []
    grad_ops = [tf.gradients(model.logits_test[:, k], model.x) for k in xrange(n_classes)]

    n_adv = 0
    for n, (image, label) in enumerate(zip(images, labels)):
        pred_org, answer = run_ops_test(model,
                                        [model.prediction, model.answer],
                                        {model.x: [image], model.label: [label]})
        pred_org, answer = pred_org[0], answer[0]

        k_0 = pred_org
        image_i = copy.copy(image)
        rs = []
        for i in xrange(max_iter):
            k_i = run_ops_test(model, model.prediction, {model.x: [image_i]})[0]
            if k_i != k_0: break
            grads = run_ops_test(model, grad_ops, {model.x: [image_i]})
            grads = [grad[0][0] for grad in grads]
            logits = run_ops_test(model, model.logits_test, {model.x: [image_i]})[0]
            w_i = np.asarray([grads[k] - grads[k_0] for k in xrange(n_classes)])
            w_i[k_0] += 1e-10
            w_i_flatten = np.reshape(w_i, [n_classes, -1])
            f_i = np.asarray([logits[k] - logits[k_0] for k in xrange(n_classes)])
            f_w = abs(f_i) / np.linalg.norm(w_i_flatten, axis=1)
            l = None
            for j in xrange(n_classes):
                if j == k_0: continue
                if l is None or f_w[l] > f_w[j]: l = j
            r_i = f_w[l] * w_i[l] / np.linalg.norm(w_i_flatten[l])
            rs.append(r_i)
            image_i = np.clip(image_i + r_i, 0.0, 1.0)
        pertubation = (1 + overshoot) * np.sum(np.array(rs), axis=0)
        pertubation = np.clip(pertubation, -eps, eps)
        adversarial_image = np.clip(image + pertubation, 0.0, 1.0)
        pred_adv = run_ops_test(model, model.prediction, {model.x: [adversarial_image]})[0]
        if pred_org != pred_adv: n_adv += 1

        adversarial_images.append(adversarial_image)
        preds_org.append(pred_org)
        preds_adv.append(pred_adv)
        answers.append(answer)
        sys.stdout.write('\r%5d/%5d : %5d/%5d'%(n+1, len(images), n_adv, len(images)))
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

