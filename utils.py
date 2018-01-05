import theano
import theano.tensor as T 

import numpy as np

import globals

floatX = globals.globals['floatX']

def conv_weight_he(o, i, w, h):
    w = (2 * np.random.randn(o, i, w, h) / (i * w * h)).astype(floatX)

    return theano.shared(w)

def fc_init_he(i, o):
    w = (2 * np.random.randn(i, o) / (i)).astype(floatX)

    return theano.shared(w)

def shared(x):
    return theano.shared(x.astype(floatX))

def RMSprop(cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - lr * g))
    return updates

def huber_loss(target, output, delta = 0.1):
    d = target - output

    l1 = (d ** 2)/2.
    l2 = delta * (abs(d) - delta / 2.)

    lf = T.switch(abs(d) <= delta, l1, l2)

    return lf.sum()