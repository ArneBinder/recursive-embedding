import scipy.sparse as sparse
import time
from functools import wraps
import errno
import os
import sys
import logging
from itertools import product
from threading import Thread

import numpy as np


# unused
def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def flatten(l):
    return [item for sublist in l for item in sublist]


def fn_timer(function):
    @wraps(function)
    def function_timer(*args, **kwargs):
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        print ("Total time running %s: %s seconds" %
               (function.func_name, str(t1-t0))
               )
        return result
    return function_timer


def list_powerset(lst):
    # the power set of the empty set has one element, the empty set
    result = [[]]
    for x in lst:
        # for every additional element in our set
        # the power set consists of the subsets that don't
        # contain this element (just take the previous power set)
        # plus the subsets that do contain the element (use list
        # comprehension to add [x] onto everything in the
        # previous power set)
        result.extend([subset + [x] for subset in result])
    return result


def avg_dif(a):
    if len(a) == 1:
        return a[0]
    l = []
    for i in range(len(a)-1):
        l.append(a[i+1] - a[i])

    return sum(l) / len(l)


def getOrAdd(strings, s, idx_alt=None):
    if idx_alt is None:
        return strings.add(s)
    else:
        return strings[s] if s in strings else idx_alt


def getOrAdd2(mapping, types, type, type_alt=None):
    try:
        res = mapping[type]
    # word doesnt occur in dictionary
    except KeyError:
        # return alternative index, if alternative type is given
        if type_alt is not None:
            return mapping[type_alt]
        res = len(mapping)
        mapping[type] = res
        types.append(type)
    return res


def incOrAdd(d, e):
    try:
        d[e] += 1
    except KeyError:
        d[e] = 1


def insert_before(position, list1, list2):
    return list1[:position] + list2 + list1[position:]


def get_default(l, idx, default):
    try:
        if idx < 0:
            return default
        return l[idx]
    except IndexError:
        return default


def logging_init():
    import tensorflow as tf
    logging_format = '%(asctime)s %(levelname)s %(message)s'
    tf.logging._logger.propagate = False
    tf.logging._handler.setFormatter(logging.Formatter(logging_format))
    tf.logging._logger.format = logging_format
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format=logging_format)


#unused
def make_parent_dir(fn):
    out_dir = os.path.abspath(os.path.join(fn, os.pardir))
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)


def dict_product(dicts):
    """
    # >>> list(dict_product(dict(number=[1,2], character='ab')))
    [{'character': 'a', 'number': 1},
     {'character': 'a', 'number': 2},
     {'character': 'b', 'number': 1},
     {'character': 'b', 'number': 2}]
    """
    return (dict(zip(dicts, x)) for x in product(*dicts.values()))


def numpy_load(filename, assert_exists=True):
    if os.path.exists(filename):
        return np.load(filename)
    elif os.path.exists('%s.npy' % filename):
        return np.load('%s.npy' % filename)
    elif os.path.exists('%s.npz' % filename):
        return sparse.load_npz('%s.npz' % filename)
    else:
        if assert_exists:
            raise IOError('file %s or %s.npy does not exist' % (filename, filename))
        return None


def numpy_dump(filename, ndarray):
    if isinstance(ndarray, sparse.csr_matrix) or isinstance(ndarray, sparse.csc_matrix):
        sparse.save_npz('%s.npz' % filename, ndarray)
    else:
        np.save('%s.npy' % filename, ndarray)


def numpy_exists(filename):
    return os.path.exists(filename) or os.path.exists('%s.npy' % filename) or os.path.exists('%s.npz' % filename)


def chunks(g, n, cut=False):
    """Yield successive n-sized chunks from generator g."""
    l = []
    for i, x in enumerate(g):
        l.append(x)
        if (i + 1) % n == 0:
            yield l
            l = []
    #if len(l) > 0 and not cut:
    #    yield l
    assert len(l) == 0 or cut, '%i elements remain after chunking with n=%i' % (len(l), n)


# similar to numpy.split, but uses sizes instead of end positions
def partition(l, sizes):
    assert sum(sizes) == len(l), 'sum(sizes): %i != len(l): %i' % (sum(sizes), len(l))
    res = []
    start = 0
    for s in sizes:
        res.append(l[start:start+s])
        start += s
    return res


# taken from https://nolanbconaway.github.io/blog/2017/softmax-numpy
def softmax(X, theta=1.0, axis=None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis=axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p