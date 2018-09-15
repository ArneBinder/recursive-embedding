
from threading import Thread, Lock
from datetime import datetime
import logging
import numpy as np
from scipy.sparse import vstack

from constants import LOGGING_FORMAT, M_TRAIN
from mytools import chunks
from model_fold import convert_sparse_matrix_to_sparse_tensor

logger = logging.getLogger('concurrency')
logger.setLevel(logging.DEBUG)
logger_streamhandler = logging.StreamHandler()
logger_streamhandler.setLevel(logging.DEBUG)
logger_streamhandler.setFormatter(logging.Formatter(LOGGING_FORMAT))
logger.addHandler(logger_streamhandler)
logger.propagate = False


# not used
def prepare_batches_multi(_q_in, _q_out, _forest, dataset_trees, forest_indices):
    while True:
        _i, _tree_indices_batched, _probs_batched = _q_in.get()

        if len(_tree_indices_batched) > 0:
            forest_indices_batched_np = forest_indices[np.array(_tree_indices_batched)]
            forest_indices_batched_np_flat = forest_indices_batched_np.flatten()
            n = len(_tree_indices_batched[0])
            trees_generator = list(dataset_trees(forest_indices_batched_np_flat, forest=_forest))
        else:
            trees_generator = None
            n = -1

        _q_out.put((_i, trees_generator, n, _probs_batched))
        _q_in.task_done()


def create_trees_simple(_q_in, _q_out, _iter, _forest):
    while True:
        _i, _indices = _q_in.get()
        _trees = list(_iter(indices=_indices, forest=_forest))
        _q_out.put((_i, _trees))
        _q_in.task_done()


# not used
def compile_batches_single(_q_in, _q_out, _compiler, use_pool=True):
    def _do():
        while True:
            _i, trees_generator, _n, _probs_batched = _q_in.get()

            if _n > 0:
                _compiled = _compiler.build_loom_inputs(([x] for x in trees_generator), ordered=True)
                _trees_batched = list(chunks(_compiled, _n))
            else:
                _trees_batched = []
            _q_out.put((_i, _trees_batched, None, _probs_batched))
            _q_in.task_done()

    if use_pool:
        with _compiler.multiprocessing_pool():
            _do()
    else:
        _do()


def compile_batches_simple(_q_in, res_dict, _compiler, use_pool=True):
    def _do():
        while True:
            _i, _trees = _q_in.get()
            res_dict[_i] = list(_compiler.build_loom_inputs(([t] for t in _trees), ordered=True))
            _q_in.task_done()

    if use_pool:
        with _compiler.multiprocessing_pool():
            _do()
    else:
        _do()


def prepare_batch(_forest_indices_batched_np, _probs_batched, _forest_indices_to_trees_indices, _dataset_trees,
                  _dataset_embeddings, _tree_iter, _compiler):
    embeddings = None
    trees_compiled = None

    assert len(_forest_indices_batched_np) > 0, 'empty batch (forest_indices_batched has length 0)'
    # use pre-compiled trees, if available
    if _dataset_trees is not None:
        trees_compiled = [[_dataset_trees[_forest_indices_to_trees_indices[tree_idx]] for tree_idx in tree_indices] for
                          tree_indices in _forest_indices_batched_np]
    # otherwise compile (and create forests)
    elif _compiler is not None and _tree_iter is not None:
        # forest_indices_batched_np = np.array(_forest_indices_batched)
        gen_trees_compiled = _compiler.build_loom_inputs(
            ([t] for t in _tree_iter(indices=_forest_indices_batched_np.flatten())), ordered=True)
        trees_compiled = list(chunks(gen_trees_compiled, n=_forest_indices_batched_np.shape[1]))
    # add sparse embeddings if available
    # TODO: check, if chunking is necessary
    if _dataset_embeddings is not None:
        # forest_indices_batched_np = np.array(_forest_indices_batched)
        # convert forest_indices to tree_indices
        embeddings = _dataset_embeddings[
            [_forest_indices_to_trees_indices[idx] for idx in _forest_indices_batched_np.flatten()]]
    return trees_compiled, embeddings, _probs_batched


#unused
def prepare_batches_single(_q_in, _q_out, _forest_indices_to_trees_indices, _dataset_trees, _dataset_embeddings,
                           _tree_iter, _compiler, use_pool=True, debug=False):

    def _do():
        while True:
            _i, _forest_indices_batched_np, _probs_batched = _q_in.get()

            t_start = datetime.now()
            trees_compiled, embeddings, _probs_batched = prepare_batch(
                _forest_indices_batched_np, _probs_batched, _forest_indices_to_trees_indices, _dataset_trees,
                _dataset_embeddings, _tree_iter, _compiler)
            t_delta = datetime.now() - t_start
            if debug:
                logger.debug('prepared batch %i. time_prepare: %s' % (_i, str(t_delta)))
            _q_out.put((_i, trees_compiled, embeddings, _probs_batched, t_delta))
            _q_in.task_done()

    #if _dataset_trees is None and _compiler is not None and use_pool:
    if _compiler is not None and use_pool:
        with _compiler.multiprocessing_pool():
            _do()
    else:
        _do()


#unused
class PrepareThread(Thread):
    def __init__(self, _q_in, _q_out, _forest_indices_to_trees_indices, _dataset_trees, _dataset_embeddings,
                           _tree_iter, _compiler, use_pool=True, debug=False):
        super(PrepareThread, self).__init__()
        self._q_in = _q_in
        self._q_out = _q_out
        self._forest_indices_to_trees_indices = _forest_indices_to_trees_indices
        self._dataset_trees = _dataset_trees
        self._dataset_embeddings = _dataset_embeddings
        self._tree_iter = _tree_iter
        self._compiler = _compiler
        self._use_pool = use_pool
        self._debug = debug
        self.daemon = True

    def _do(self):
        while True:
            item = self._q_in.get()
            if item is None:
                self._q_in.task_done()
                break
            _i, _forest_indices_batched_np, _probs_batched = item

            t_start = datetime.now()
            trees_compiled, embeddings, _probs_batched = prepare_batch(
                _forest_indices_batched_np, _probs_batched, self._forest_indices_to_trees_indices, self._dataset_trees,
                self._dataset_embeddings, self._tree_iter, self._compiler)
            t_delta = datetime.now() - t_start
            if self._debug:
                logger.debug('prepared batch %i. time_prepare: %s' % (_i, str(t_delta)))
            self._q_out.put((_i, trees_compiled, embeddings, _probs_batched, t_delta))
            self._q_in.task_done()

    def run(self):
        if self._use_pool and self._compiler is not None:
            with self._compiler.multiprocessing_pool():
                self._do()
        else:
            self._do()

    def join(self, timeout=None):
        self._q_in.join()
        self._q_in.put(None)
        Thread.join(self, timeout)


def process_batch(_trees_batched, _embeddings, _probs_batched, _vars, _feed_dict, _sess, _model,
                  _use_sparse_probs, _use_sparse_embeddings):
    if _trees_batched is not None:
        _feed_dict[_model.tree_model.compiler.loom_input_tensor] = _trees_batched
    if _embeddings is not None and _use_sparse_embeddings:
        _embeddings = convert_sparse_matrix_to_sparse_tensor(_embeddings)
    if _embeddings is not None:
        _feed_dict[_model.tree_model.prepared_embeddings_placeholder] = _embeddings
    # if values_gold expects a sparse tensor, convert probs_batched
    if _use_sparse_probs:
        _probs_batched = convert_sparse_matrix_to_sparse_tensor(vstack(_probs_batched))

    _feed_dict[_model.values_gold] = _probs_batched
    _res = _sess.run(_vars, _feed_dict)
    return _res


#unused
def process_batches_single(_q, _vars, _feed_dict, _res_dict, _sess, _model, _use_sparse_probs, _use_sparse_embeddings, debug=False):
    while True:
        _i, _trees_batched, _embeddings, _probs_batched, _time_prepare = _q.get()
        t_start = datetime.now()
        _res = process_batch(_trees_batched, _embeddings, _probs_batched, _vars, _feed_dict, _sess, _model,
                             _use_sparse_probs, _use_sparse_embeddings)
        t_delta = datetime.now() - t_start
        _res_dict[_i] = (_res, _time_prepare, t_delta)
        if debug:
            logger.debug('finished batch %i. time_prepare: %s\ttime_train: %s' % (_i, str(_time_prepare), str(t_delta)))
        _q.task_done()


#unused
class TrainThread(Thread):
    def __init__(self, _q, _vars, _feed_dict, _res_dict, _sess, _model, _use_sparse_probs, _use_sparse_embeddings, debug=False):
        super(TrainThread, self).__init__()
        self._q = _q
        self._vars = _vars
        self._feed_dict = _feed_dict
        self._res_dict = _res_dict
        self._sess = _sess
        self._model = _model
        self._use_sparse_probs = _use_sparse_probs
        self._use_sparse_embeddings = _use_sparse_embeddings
        self._debug = debug
        self.daemon = True

    def _do(self):
        while True:
            item = self._q.get()
            if item is None:
                self._q.task_done()
                break
            _i, _trees_batched, _embeddings, _probs_batched, _time_prepare = item
            t_start = datetime.now()
            _res = process_batch(_trees_batched, _embeddings, _probs_batched, self._vars, self._feed_dict, self._sess, self._model,
                                 self._use_sparse_probs, self._use_sparse_embeddings)
            t_delta = datetime.now() - t_start
            self._res_dict[_i] = (_res, _time_prepare, t_delta)
            if self._debug:
                logger.debug(
                    'finished batch %i. time_prepare: %s\ttime_train: %s' % (_i, str(_time_prepare), str(t_delta)))
            self._q.task_done()

    def run(self):
        self._do()

    def join(self, timeout=None):
        self._q.join()
        self._q.put(None)
        Thread.join(self, timeout)


class RunnerThread(Thread):
    def __init__(self, q_in, q_out, func, args=None, kwargs=None, compiler=None, debug=False):
        super(RunnerThread, self).__init__()
        self._func = func
        self._args = args or []
        self._kwargs = kwargs or {}
        self._q_in = q_in
        self._q_out = q_out
        self._compiler = compiler
        self._debug = debug
        self.daemon = True

    def _do(self):
        while True:
            item = self._q_in.get()
            if item is None:
                self._q_in.task_done()
                break
            _i, _args, _kwargs, _times = item
            t_start = datetime.now()
            # use '+' instead of append to allow tuples
            _current_args = _args + self._args
            _kwargs.update(self._kwargs)
            _res = self._func(*_current_args, **_kwargs)
            del _args
            del _kwargs
            t_delta = datetime.now() - t_start
            _times.append(t_delta)
            if self._debug:
                logger.debug('finished %s %i. time: %s' % (self._func.__name__, _i, str(t_delta)))
            self._q_out.put((_i, _res, {}, _times))
            self._q_in.task_done()

    def run(self):
        if self._compiler is not None:
            with self._compiler.multiprocessing_pool():
                self._do()
        else:
            self._do()

    def join(self, timeout=None):
        self._q_in.join()
        self._q_in.put(None)
        Thread.join(self, timeout)


class RecompileThread(Thread):
    def __init__(self, q, train_tree_iter, compiler, train_indices_sampler, compile_func):
        super(RecompileThread, self).__init__()
        self._q = q
        self._lock = Lock()
        self._train_tree_iter = train_tree_iter
        self._train_indices_sampler = train_indices_sampler
        self._compiler = compiler
        self._compile_func = compile_func
        self.daemon = True

    def run(self):
        with self._compiler.multiprocessing_pool():
            while True:
                #logger.debug('wait for lock (run) ...')
                self._lock.acquire()
                try:
                    _indices = self._train_indices_sampler()
                finally:
                    self._lock.release()
                if _indices is None:
                    break
                #logger.debug('compile ...')
                _trees = self._compile_func(tree_iterators={M_TRAIN: self._train_tree_iter},
                                            indices={M_TRAIN: _indices},
                                            compiler=self._compiler, use_pool=False)[M_TRAIN]
                #logger.debug('q.put() ...')
                self._q.put((_indices, _trees))

    def join(self, timeout=None):
        # stopping
        logger.debug('wait for lock (join) ...')
        self._lock.acquire()
        try:
            self._train_indices_sampler = lambda: None
        finally:
            self._lock.release()
        # remove element
        try:
            logger.debug('try task_done ...')
            self._q.task_done()
            logger.debug('q.get() ...')
            self._q.get()
        # throws value error if already empty
        except ValueError:
            logger.debug('queue was already empty')
            pass
        Thread.join(self, timeout)