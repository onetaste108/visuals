import numpy as np
from scipy.optimize import fmin_l_bfgs_b


def get_shape(x):
    if type(x) in [tuple, list]:
        return [get_shape(y) for y in x]
    else:
        return x.shape

def flatten(x):
    if type(x) in [tuple, list]:
        return np.concatenate([flatten(y) for y in x])
    else:
        return x.flatten()
    
def unflatten(x, shape):
    if type(shape) is tuple:
        size = np.product(shape)
        return np.reshape(x[:size], shape), x[size:]
    else:
        vals = []
        for sh in shape:
            val, x = unflatten(x, sh)
            vals.append(val)
        return vals, x
    
def lbfgs(fn, x0, iters, **kwargs):
    shape = get_shape(x0)
    def fn_wrap(xf):
        x, _ = unflatten(xf, shape)
        loss, grad = fn(x)
        return loss, grad.flatten()
    xf = flatten(x0)
    kwargs["maxfun"] = iters
    kwargs["func"] = fn_wrap
    kwargs["x0"] = xf
    result = fmin_l_bfgs_b(**kwargs)
    xf = result[0]
    x, _ = unflatten(xf, shape)
    return x