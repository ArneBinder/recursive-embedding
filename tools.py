import time
from functools import wraps
import errno
import os


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


def revert_mapping(mapping):
    temp = {}
    for key in mapping:
        temp[mapping[key]] = key
    return temp


def getOrAdd(d, e):
    try:
        res = d[e]
    # word doesnt occur in dictionary
    except KeyError:
        res = len(d)
        d[e] = res
        print('add to dict: ' + str(e) + ' -> ' + str(res))
    return res


def incOrAdd(d, e):
    try:
        d[e] += 1
    except KeyError:
        d[e] = 1


def insert_before(position, list1, list2):
    return list1[:position] + list2 + list1[position:]


