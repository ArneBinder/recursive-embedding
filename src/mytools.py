import time
from functools import wraps
import errno
import os
import sys
import logging
from tqdm import tqdm
#from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Pool
from itertools import product


def parallel_process_simple(input, func):
    p = Pool()

    return p.map(func, input)

# unused
def parallel_process(array, function, n_jobs=4, use_kwargs=False, front_num=3):
    """
        A parallel version of the map function with a progress bar.

        Args:
            array (array-like): An array to iterate over.
            function (function): A python function to apply to the elements of array
            n_jobs (int, default=16): The number of cores to use
            use_kwargs (boolean, default=False): Whether to consider the elements of array as dictionaries of
                keyword arguments to function
            front_num (int, default=3): The number of iterations to run serially before kicking off the parallel job.
                Useful for catching bugs
        Returns:
            [function(array[0]), function(array[1]), ...]
    """
    #We run the first few iterations serially to catch bugs
    if front_num > 0:
        front = [function(**a) if use_kwargs else function(a) for a in array[:front_num]]
    #If we set n_jobs to 1, just run a list comprehension. This is useful for benchmarking and debugging.
    if n_jobs==1:
        return front + [function(**a) if use_kwargs else function(a) for a in tqdm(array[front_num:])]
    #Assemble the workers
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        #Pass the elements of array into function
        if use_kwargs:
            futures = [pool.submit(function, **a) for a in array[front_num:]]
        else:
            futures = [pool.submit(function, a) for a in array[front_num:]]
        kwargs = {
            'total': len(futures),
            'unit': 'it',
            'unit_scale': True,
            'leave': True
        }
        #Print out the progress as tasks complete
        for f in tqdm(as_completed(futures), **kwargs):
            pass
    out = []
    #Get the results from the futures.
    for i, future in tqdm(enumerate(futures)):
        try:
            out.append(future.result())
        except Exception as e:
            out.append(e)
    return front + out


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


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