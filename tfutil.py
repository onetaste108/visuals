import tensorflow as tf
import numpy as np

@tf.function
def histmatch(x, y, nbins=256):
    x_shape = tf.shape(x)
    y_shape = tf.shape(y)
    ch = x_shape[-1]
    size = x_shape[0]*x_shape[1]
    x_flat = tf.transpose(tf.reshape(x, [-1, ch]), [1, 0])
    y_flat = tf.transpose(tf.reshape(y, [-1, ch]), [1, 0])
    x_min = tf.reduce_min(x_flat, -1)[...,None]
    x_max = tf.reduce_max(x_flat, -1)[...,None]
    y_min = tf.reduce_min(y_flat, -1)[...,None]
    y_max = tf.reduce_max(y_flat, -1)[...,None]
    x_flat = (x_flat - x_min) / (x_max - x_min)
    y_flat = (y_flat - y_min) / (y_max - y_min)
    
    x_bins = tf.minimum(tf.cast(x_flat * tf.cast(nbins, tf.float32), tf.int32), nbins - 1)
    x_hist = tf.math.bincount(x_bins, minlength=nbins, axis=-1)
    x_cdf = tf.cumsum(x_hist, -1)
    x_cdf = tf.cast(x_cdf, tf.float32)
    x_cdf = x_cdf / x_cdf[:, -1, None]    
    
    y_bins = tf.minimum(tf.cast(y_flat * tf.cast(nbins, tf.float32), tf.int32), nbins - 1)
    y_hist = tf.math.bincount(y_bins, minlength=nbins, axis=-1)
    y_cdf = tf.cumsum(y_hist, -1)
    y_cdf = tf.cast(y_cdf, tf.float32)
    y_cdf = y_cdf / y_cdf[:, -1, None]
    
    x_match = x_cdf[:, :, None]
    y_match = y_cdf[:, None, :]
    
    where = (y_match >= x_match)
    amax = tf.argmax(where, -1)
    
    indices = tf.stack([tf.repeat(tf.range(ch)[..., None], size, -1), x_bins], -1)
    corr = tf.gather_nd(amax, indices)
    corr = tf.cast(corr, tf.float32) / tf.cast(nbins, tf.float32)
    corr = corr * (y_max-y_min) + y_min
    corr = tf.reshape(tf.transpose(corr, [1,0]), x_shape)
    return corr

@tf.function
def gram(x):
    shape = tf.cast(tf.shape(x), tf.float32)
    x = tf.reshape(x, [shape[0], -1, shape[3]])
    x = tf.matmul(tf.transpose(x, [0, 2, 1]), x)
    x = x / (shape[1]*shape[2])
    return tf.reshape(x, (shape[0], -1))