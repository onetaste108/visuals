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
        return np.array(x).flatten()
    
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

def get_bounds(shape, bounds):
    if type(shape) == tuple:
        min_ = np.zeros(shape)
        max_ = np.zeros(shape)
        if bounds is None:
            min_ += np.finfo(np.float32).min
            max_ += np.finfo(np.float32).max
        else:
            bounds = np.float32(bounds)
            min_ += bounds[...,0]
            max_ += bounds[...,1]            
        return np.stack([min_,max_], -1)
    else:
        return [get_bounds(s, b) for s, b in zip(shape, bounds)]
    
def lbfgs(fn, x0, iters, bounds=None, info=False, **kwargs):
    shape = get_shape(x0)
    if bounds is not None:
        bounds = flatten(get_bounds(shape, bounds)).reshape(-1, 2)
    def fn_wrap(xf):
        x, _ = unflatten(xf, shape)
        loss, grad = fn(x)
        return np.float64(loss), np.float64(flatten(grad))
    xf = flatten(x0)
    print("XF", xf.shape)
    print("B", bounds.shape)
    kwargs["maxfun"] = iters
    kwargs["func"] = fn_wrap
    kwargs["x0"] = xf
    kwargs["bounds"] = bounds
    kwargs["factr"] = 0
    kwargs["pgtol"] = 0
    result = fmin_l_bfgs_b(**kwargs)
    xf = result[0]
    x, _ = unflatten(xf, shape)
    if info:
        return x, result[1:]
    return x

