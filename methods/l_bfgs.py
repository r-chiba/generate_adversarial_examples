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

def generate_adversarial_example(model, image, target_label, eps=0.001):
    def objective_lbfgs(x_adv, model, x, c, label, x_shape):
        # objective function for L-BFGS
        r = x_adv - x
        r_norm = np.linalg.norm(r)
        x_adv = np.reshape(x_adv, x_shape)
        model_loss = run_ops_test(model, model.loss, {model.x: [x_adv], model.label: [label]})
        return c * r_norm + model_loss

    def gradient_lbfgs(x_adv, model, x, c, label, x_shape):
        # gradients w.r.t. r = x_adv - x of objective function for L-BFGS
        r = x_adv - x
        r_norm = np.linalg.norm(r)
        r_norm_grad =  r / r_norm
        x_adv = np.reshape(x_adv, x_shape)
        model_loss_grad = run_ops_test(model, model.x_grad, {model.x: [x_adv], model.label: [label]})[0]
        model_loss_grad = np.reshape(model_loss_grad, x.shape)
        r_grad = c * r_norm_grad + model_loss_grad
        return r_grad

    if isinstance(target_label, int):
        one_hot = np.zeros([10,])
        one_hot[target_label] = 1.
        target_label = one_hot

    pred_org = run_ops_test(model, model.prediction, {model.x: [image]})[0]
    if pred_org == np.argmax(target_label):
        return

    h, w, c = image.shape
    image_dim = h * w * c
    bounds = np.array([(0.0, 1.0) for i in xrange(image_dim)])
    image_flatten = np.reshape(image, [image_dim,])

    # determine initial c
    c_init = 0.01
    while True:
        res = sp.optimize.minimize(objective_lbfgs,
                                    image_flatten+0.0001,
                                    args=(model, image_flatten, c_init, target_label, image.shape),
                                    method='L-BFGS-B',
                                    jac=gradient_lbfgs,
                                    bounds=bounds)
        x_adv = np.reshape(res.x, image.shape)
        pred_adv = run_ops_test(model, model.prediction, {model.x: [x_adv]})[0]
        if pred_adv == pred_org:
            break
        c_init *= 10.

    trial = 0
    n_trials = 10
    while trial < n_trials:
        # bisection search for c
        c_min, c_max = 0.0, c_init
        i = 0
        tol = c_init / 10.0
        while c_max - c_min > tol:
            c_mid = (c_max + c_min) / 2.0
            res = sp.optimize.minimize(objective_lbfgs,
                                        image_flatten+0.0001,
                                        args=(model, image_flatten, c_mid, target_label, image.shape),
                                        method='L-BFGS-B',
                                        jac=gradient_lbfgs,
                                        bounds=bounds)
            x_adv = np.reshape(res.x, image.shape)
            pred_adv = run_ops_test(model, model.prediction, {model.x: [x_adv]})[0]
            c = c_min
            adversarial = pred_adv == np.argmax(target_label)
            if adversarial:
                c_min = c_mid
            else:
                c_max = c_mid
        
        res = sp.optimize.minimize(objective_lbfgs,
                                    image_flatten+0.0001,
                                    args=(model, image_flatten, c, target_label, image.shape),
                                    method='L-BFGS-B',
                                    jac=gradient_lbfgs,
                                    bounds=bounds)
        x_adv = np.reshape(res.x, image.shape)
        x_adv = np.clip(x_adv, image-eps, image+eps)
        pred_adv = run_ops_test(model, model.prediction, {model.x: [x_adv]})[0]
        adversarial = pred_adv == np.argmax(target_label)
        if adversarial: break
        trial += 1
    #if adversarial:
    #    print(pred_adv, np.linalg.norm(x_adv-image))
    return x_adv if adversarial else None

def generate_adversarial_examples(model, images, labels, image_dir):
    if not hasattr(model, 'logits'):
        model.logits = model.logits_test
    if not hasattr(model, 'x_grad'):
        model.x_grad = tf.gradients(model.loss, model.x)

    preds, answers = run_ops_test(model, [model.logits, model.answer],
                                    {model.x: images, model.label: labels})

    originals, adversarials = [], []
    norms = []
    for i, (img_org, pred, answer) in enumerate(zip(images, preds, answers)):
        print(i)
        pred1, pred2 = np.argsort(-pred)[:2]
        #if pred1 != answer: continue
        img_adv = generate_adversarial_example(model, img_org, pred2)
        if img_adv is not None:
            pred_adv, logits = run_ops_test(model, [model.prediction, model.logits], {model.x: [img_adv]})
            pred_adv = pred_adv[0]
            print(pred1, pred_adv, np.linalg.norm(img_adv-img_org))

            norm = np.linalg.norm(img_adv-img_org)
            norms.append(norm)

            originals.append(img_org)
            adversarials.append(img_adv)

            #img_adv_ = copy.deepcopy(img_adv)
            #img_adv_ *= 255.
            #img_adv_ = np.uint8(img_adv_)
            #if img_adv_.shape[-1] == 3:
            #    img_adv_ = cv2.cvtColor(img_adv_, cv2.COLOR_RGB2BGR)
            #else:
            #    img_adv_ = cv2.cvtColor(img_adv_, cv2.COLOR_GRAY2BGR)
            #cv2.imwrite(os.path.join(image_dir, "img_adversarial_%d.png" % i), img_adv_)
            #img_org_ = copy.deepcopy(img_org)
            #img_org_ *= 255.
            #img_org_ = np.uint8(img_org_)
            #if img_org_.shape[-1] == 3:
            #    img_org_ = cv2.cvtColor(img_org_, cv2.COLOR_RGB2BGR)
            #else:
            #    img_org_ = cv2.cvtColor(img_org_, cv2.COLOR_GRAY2BGR)
            #cv2.imwrite(os.path.join(image_dir, "img_original_%d.png" % i), img_org_)
        sys.stdout.flush()
        sys.stderr.flush()

    #norms = np.asarray(norms)
    #print(np.mean(norms), np.std(norms))

    return np.array(adversarials), np.array(originals)

