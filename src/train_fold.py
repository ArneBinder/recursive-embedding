# from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import csv
from datetime import datetime, timedelta
import fnmatch
import json
import logging
import ntpath
import os
import re
# import google3
# import shutil
from multiprocessing.pool import ThreadPool

import scipy
from functools import reduce, partial
import cPickle as pickle
import Queue
from threading import Thread, Lock

import numpy as np
import six
import tensorflow as tf
import tensorflow_fold as td
from scipy.stats.mstats import spearmanr
from scipy.stats.stats import pearsonr
from sklearn import metrics
from sklearn import preprocessing as pp
#from spacy.strings import StringStore
from tensorflow.python.client import device_lib
from scipy.sparse import csr_matrix, vstack

#import lexicon as lex
from lexicon import Lexicon
import model_fold
from model_fold import MODEL_TYPE_DISCRETE, MODEL_TYPE_REGRESSION, convert_sparse_matrix_to_sparse_tensor, \
    convert_sparse_tensor_to_sparse_matrix

# model flags (saved in flags.json)
#import mytools
from mytools import numpy_load, chunks, flatten, numpy_dump, numpy_exists, logging_init
from sequence_trees import Forest
from constants import vocab_manual, IDENTITY_EMBEDDING, LOGGING_FORMAT, CM_AGGREGATE, CM_TREE, M_INDICES, M_TEST, \
    M_TRAIN, M_MODEL, M_FNAMES, M_TREES, M_TREE_ITER, M_INDICES_TARGETS, M_BATCH_ITER, M_NEG_SAMPLES, OFFSET_ID, \
    M_MODEL_NEAREST, M_INDEX_FILE_SIZES, FN_TREE_INDICES, PADDING_EMBEDDING, MT_SINGLE_DISCRETE, MT_TUPLE_DISCRETE, \
    MT_SINGLE_DISCRETE_INDEPENDENT, MT_TUPLE_DISCRETE_DEPENDENT, \
    DTYPE_IDX, UNKNOWN_EMBEDDING, M_EMBEDDINGS, M_INDICES_SAMPLER, M_TREE_ITER_TFIDF, OFFSET_MESH_ROOT, \
    MT_TUPLE_CONTINOUES, TYPE_ENTAILMENT, TYPE_POLARITY, TASK_MESH_PREDICTION, TASK_ENTAILMENT_PREDICTION, TYPE_MESH, \
    OFFSET_POLARITY_ROOT, TASK_SENTIMENT_PREDICTION, OFFSET_OTHER_ENTRY_ROOT, OFFSET_ENTAILMENT_ROOT
from config import Config, FLAGS_FN, TREE_MODEL_PARAMETERS, MODEL_PARAMETERS
#from data_iterators import data_tuple_iterator_reroot, data_tuple_iterator_dbpedianif, data_tuple_iterator, \
#    indices_dbpedianif
import data_iterators as diters
from data_iterators import batch_iter_naive, batch_iter_all, batch_iter_reroot, batch_iter_multiclass, batch_iter_simtuple
from corpus import FE_CLASS_IDS, load_class_ids
from concurrency import RunnerThread, compile_batches_simple, create_trees_simple, prepare_batch, RecompileThread, process_batch

# non-saveable flags
tf.flags.DEFINE_string('logdir',
                       # '/home/arne/ML_local/tf/supervised/log/dataPs2aggregate_embeddingsUntrainable_simLayer_modelTreelstm_normalizeTrue_batchsize250',
                       #  '/home/arne/ML_local/tf/supervised/log/dataPs2aggregate_embeddingsTrainable_simLayer_modelAvgchildren_normalizeTrue_batchsize250',
                       #  '/home/arne/ML_local/tf/supervised/log/SA/EMBEDDING_FC_dim300',
                       '/home/arne/ML_local/tf/supervised/log/SA/PRETRAINED',
                       'Directory in which to write event logs.')
tf.flags.DEFINE_string('test_files',
                       None,
                       'Set this to execute only.')
tf.flags.DEFINE_string('train_files',
                       None,
                       'If set, do not look for idx.<id>.npy files (in train data dir), '
                       'but use these index files instead (separated by comma)')
tf.flags.DEFINE_boolean('test_only',
                        False,
                        'Enable to execute evaluation only.')
tf.flags.DEFINE_boolean('dont_test',
                        False,
                        'If enabled, do not evaluate on test data.')
tf.flags.DEFINE_string('logdir_continue',
                       None,
                       'continue training with config from flags.json')
tf.flags.DEFINE_string('logdir_pretrained',
                       None,
                       # '/home/arne/ML_local/tf/supervised/log/batchsize100_embeddingstrainableTRUE_learningrate0.001_optimizerADADELTAOPTIMIZER_simmeasureSIMCOSINE_statesize50_testfileindex1_traindatapathPROCESSSENTENCE3SICKTTCMSEQUENCEICMTREE_treeembedderTREEEMBEDDINGHTUGRUSIMPLIFIED',
                       # '/home/arne/ML_local/tf/supervised/log/SA/EMBEDDING_FC/batchsize100_embeddingstrainableTRUE_learningrate0.001_optimizerADADELTAOPTIMIZER_simmeasureSIMCOSINE_statesize50_testfileindex1_traindatapathPROCESSSENTENCE3SICKTTCMAGGREGATE_treeembedderTREEEMBEDDINGFLATAVG2LEVELS',
                       'Set this to fine tune a pre-trained model. The logdir_pretrained has to contain a types file '
                       'with the filename "model.types"'
                       )
tf.flags.DEFINE_boolean('init_only',
                        False,
                        'If True, save the model without training and exit')
tf.flags.DEFINE_string('grid_config_file',
                       None,
                       'read config parameter dict from this file and execute multiple runs')
tf.flags.DEFINE_string('early_stopping_metric',
                       '',
                       'If set and early_stopping_window != 0, use this metric to estimate when to cancel training')
tf.flags.DEFINE_integer('run_count',
                        1,
                        'repeat each run this often')
tf.flags.DEFINE_boolean('debug',
                        False,
                        'enable debug mode (additional output, but slow)')
tf.flags.DEFINE_boolean('precompile',
                        True,
                        'If enabled, compile all trees once. Otherwise trees are compiled batch wise, which results in '
                        'decreased memory consumption.')
tf.flags.DEFINE_boolean('clean_train_trees',
                        False,
                        'If enabled, delete train trees after every train executionion to decrease memory consumption.'
                        'That requires re-compilation of all trees during training.')
tf.flags.DEFINE_boolean('discard_tree_embeddings',
                        False,
                        'If enabled, discard embeddings produced by the tree model and pass instead vecs of zeros.')
tf.flags.DEFINE_boolean('discard_prepared_embeddings',
                        False,
                        'If enabled, discard prepared embeddings like tf-idf and pass instead vecs of zeros.')


# flags which are not logged in logdir/flags.json
#tf.flags.DEFINE_string('master', '',
#                       'Tensorflow master to use.')
#tf.flags.DEFINE_integer('task', 0,
#                        'Task ID of the replica running the training.')
#tf.flags.DEFINE_integer('ps_tasks', 0,
#                        'Number of PS tasks in the job.')
FLAGS = tf.flags.FLAGS

# NOTE: the first entry (of both lists) defines the value used for early stopping and other statistics
#METRIC_KEYS_DISCRETE = ['roc_micro', 'ranking_loss_inv', 'f1_t10', 'f1_t33', 'f1_t50', 'f1_t66', 'f1_t90', 'acc_t10', 'acc_t33', 'acc_t50', 'acc_t66', 'acc_t90', 'precision_t10', 'precision_t33', 'precision_t50', 'precision_t66', 'precision_t90', 'recall_t10', 'recall_t33', 'recall_t50', 'recall_t66', 'recall_t90']
METRIC_KEYS_DISCRETE = ['f1_t10', 'f1_t33', 'f1_t50', 'f1_t66', 'f1_t90', 'precision_t10', 'precision_t33', 'precision_t50', 'precision_t66', 'precision_t90', 'recall_t10', 'recall_t33', 'recall_t50', 'recall_t66', 'recall_t90', 'recall@1', 'recall@2', 'recall@3', 'recall@5', 'accuracy_t50', 'accuracy_t33', 'accuracy_t66']
METRIC_DISCRETE = 'f1_t33'
#STAT_KEY_MAIN_DISCRETE = 'roc_micro'
METRIC_KEYS_REGRESSION = ['pearson_r', 'mse']
METRIC_REGRESSION = 'pearson_r'
#STAT_KEY_MAIN_REGRESSION = 'pearson_r'

TREE_EMBEDDER_PREFIX = 'TreeEmbedding_'

logger = logging.getLogger('')
logger.setLevel(logging.DEBUG)
logger_streamhandler = logging.StreamHandler()
logger_streamhandler.setLevel(logging.INFO)
logger_streamhandler.setFormatter(logging.Formatter(LOGGING_FORMAT))
#logger.addHandler(logger_streamhandler)


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def get_available_cpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'CPU']


#def get_devices():
#    return get_available_cpus() + get_available_gpus()


def get_ith_best_device(i):
    gpus = get_available_gpus()
    cpus = get_available_cpus()
    if len(gpus) > 0:
        devices = gpus
    else:
        devices = cpus
    assert len(devices) > 0, 'no devices for calculation available'
    idx = min(len(devices)-1, i)
    return devices[idx]


def emit_values(supervisor, session, step, values, writer=None, csv_writer=None):
    summary = tf.Summary()
    for name, value in six.iteritems(values):
        summary_value = summary.value.add()
        summary_value.tag = name
        summary_value.simple_value = float(value)
    if writer is not None:
        writer.add_summary(summary, step)
    else:
        supervisor.summary_computed(session, summary, global_step=step)
    if csv_writer is not None:
        values['step'] = step
        csv_writer.writerow({k: values[k] for k in values if k in csv_writer.fieldnames})


def collect_metrics(supervisor, sess, epoch, step, loss, values, values_gold, model, print_out=True, emit=True,
                    test_writer=None, test_result_writer=None):
    logger.debug('collect metrics ...')
    if test_writer is None:
        suffix = 'train'
        writer = None
        csv_writer = None
    else:
        suffix = 'test '
        writer = test_writer
        csv_writer = test_result_writer

    emit_dict = {'loss': loss}
    ms = sess.run(model.metrics)
    for k in ms.keys():
        spl = k.split(':')
        if len(spl) > 1:
            spl_t = spl[1].split(',')
            for i, v in enumerate(ms[k]):
                emit_dict[spl[0] + '_t' + spl_t[i]] = v
        else:
            emit_dict[k] = ms[k]

    if model.model_type == MODEL_TYPE_REGRESSION:
        stats_string = '\t'.join(['%s=%f' % (k, emit_dict[k]) for k in METRIC_KEYS_REGRESSION if k in emit_dict])
        info_string = 'epoch=%d step=%d %s: loss=%f\t%s' % (epoch, step, suffix, loss, stats_string)
    elif model.model_type == MODEL_TYPE_DISCRETE:

        # for all precision values, add F1 scores (assuming that recall values exist)
        for k in emit_dict.keys():
            if k.startswith('precision'):
                suf = k[len('precision'):]
                if 'recall'+suf in emit_dict.keys():
                    r = emit_dict['recall'+suf]
                    p = emit_dict[k]
                    f1 = 2 * p * r / (p + r)
                    emit_dict['f1' + suf] = f1
        stats_string = '\t'.join(['%s=%f' % (k, emit_dict[k]) for k in METRIC_KEYS_DISCRETE if k in emit_dict])
        info_string = 'epoch=%d step=%d %s: loss=%f\t%s' % (epoch, step, suffix, loss, stats_string)
    else:
        raise ValueError('unknown model type: %s. Use %s or %s.' % (model.model_type, MODEL_TYPE_DISCRETE,
                                                                    MODEL_TYPE_REGRESSION))
    if emit:
        emit_values(supervisor, sess, step, emit_dict, writer=writer, csv_writer=csv_writer)
    if print_out:
        logger.info(info_string)
    return emit_dict


class PutList(list):
    def put(self, item):
        self.append(item)


def do_epoch(supervisor, sess, model, epoch, forest_indices, batch_iter, indices_targets=None,
             tree_iter=None, dataset_trees=None, dataset_embeddings=None,
             train=True, emit=True, test_step=0, test_writer=None, test_result_writer=None,
             highest_sims_model=None, number_of_samples=None, return_values=True, debug=False):
    """
    Execute one training or testing epoch. Use precompield trees or prepared embeddings, if available. Otherwise,
    create and compile trees batch wise on the run while training (or predicting).
    :param supervisor: the training supervisor
    :param sess: the training tensorflow session
    :param model: the model
    :param epoch: number of current epoch
    :param forest_indices: indices with respect to the forest whose trees are used for training
    :param indices_targets: indices for target data. In the tree tuple case, lists with indices that point to the
                            forest. For (multi) class prediction, (sparse) class probabilities.
    :param tree_iter: iterator that requires a parameter indices pointing to forest and produces trees rooted by these
                      indices
    :param dataset_trees: precompiled trees for every index in forest_indices
    :param dataset_embeddings: prepared (sparse) embeddings for every index in forest_indices
    :param train: If True, train the model. Otherwise, predict only.
    :param emit: If True, emit statistics to the superviser.writer/test_writer
    :param test_step: If train==False, use this for statistics
    :param test_writer: If train==False, use this tensorflow statistics writer
    :param test_result_writer: If train==False, use this prediction result csv writer
    :param highest_sims_model: DEPRECATED
    :param number_of_samples: numb er of neattive samples for tree tuple settings
    :param batch_iter: one of the implemented batch iterators (batch_iter_naive, )
    :param return_values: Iff True, calculate predictions
    :param debug: enable debugging mode
    :return:
    """
    logger.debug('use %i forest_indices for this epoch' % len(forest_indices))
    #dataset_indices = np.arange(len(forest_indices))
    #np.random.shuffle(dataset_indices)
    logger.debug('reset metrics...')
    sess.run(model.reset_metrics)
    step = test_step
    feed_dict = {}
    execute_vars = {'loss': model.loss, 'update_metrics': model.update_metrics}
    if return_values:
        execute_vars['values'] = model.values_predicted
        execute_vars['values_gold'] = model.values_gold

    if train:
        assert test_writer is None, 'test_writer should be None for training'
        assert test_result_writer is None, 'test_result_writer should be None for training'
        execute_vars['train_op'] = model.train_op
        execute_vars['step'] = model.global_step
    else:
        assert test_writer is not None, 'test_writer should not be None for testing'
        assert test_result_writer is not None, 'test_result_writer not should be None for testing'
        feed_dict[model.tree_model.keep_prob] = 1.0
        assert test_writer is not None, 'training is disabled, but test_writer is not set'
        assert test_result_writer is not None, 'training is disabled, but test_result_writer is not set'

    #tree_model_batch_size = 10
    #_map = {idx: i for i, idx in enumerate(forest_indices)}
    #indices_forest_to_tree = np.vectorize(_map.get)
    indices_forest_to_tree = {idx: i for i, idx in enumerate(forest_indices)}
    nbr_embeddings_in = model.nbr_embeddings_in
    iter_args = {batch_iter_naive: [number_of_samples, forest_indices, indices_targets, indices_forest_to_tree],
                 # batch_iter_nearest is DEPRECATED
                 #batch_iter_nearest: [number_of_samples, forest_indices, indices_targets, sess,
                 #                     model.tree_model, highest_sims_model, dataset_trees, tree_model_batch_size,
                 #                     indices_forest_to_tree],
                 batch_iter_all: [forest_indices, indices_targets, number_of_samples + 1],
                 batch_iter_reroot: [forest_indices, number_of_samples],
                 batch_iter_multiclass: [forest_indices, indices_targets, nbr_embeddings_in, not debug],
                 batch_iter_simtuple: [forest_indices, indices_targets, nbr_embeddings_in, not debug]}

    assert batch_iter.strip() != '', 'empty batch_iter'
    _iter = globals()[batch_iter]
    logger.debug('use %s' % _iter.__name__)
    _batch_iter = _iter(*iter_args[_iter])
    #_result_all_dict = {}

    compilation_required = hasattr(model.tree_model, 'compiler') and hasattr(model.tree_model.compiler, 'loom_input_tensor')
    sparse_embeddings_required = hasattr(model.tree_model, 'prepared_embeddings_placeholder') \
                                 and isinstance(model.tree_model.prepared_embeddings_placeholder, tf.SparseTensor)
    sparse_probs_required = isinstance(model.values_gold, tf.SparseTensor)

    batch_queue = Queue.Queue()
    result_list = PutList()
    train_thread = RunnerThread(q_in=batch_queue, q_out=result_list, func=process_batch,
                                args=(execute_vars, feed_dict, sess, model, sparse_probs_required,
                                      sparse_embeddings_required),
                                debug=debug)
    logger.debug('start train thread (single)...')
    train_thread.start()

    prebatch_queue = Queue.Queue()
    prepare_thread = RunnerThread(q_in=prebatch_queue, q_out=batch_queue, func=prepare_batch,
                                  args=(indices_forest_to_tree, dataset_trees, dataset_embeddings, tree_iter,
                                        model.tree_model.compiler if compilation_required and dataset_trees is None else None),
                                  debug=debug)
    logger.debug('start prepare thread (single)...')
    prepare_thread.start()

    t_start = datetime.now()
    # for batch in td.group_by_batches(data_set, config.batch_size if train else len(test_set)):
    for i, batch in enumerate(td.group_by_batches(_batch_iter, config.batch_size)):
        forest_indices_batched, probs_batched = zip(*batch)
        prebatch_queue.put((i, (np.array(forest_indices_batched), probs_batched), {}, []))

    prepare_thread.join()
    train_thread.join()

    _result_all_dict = {_i: (_res, _times[0], _times[1]) for _i, _res, _, _times in result_list}
    # order results
    _result_all = [_result_all_dict[i][0] for i in range(len(_result_all_dict))]
    # time
    time_prepare = sum([_result_all_dict[i][1] for i in range(len(_result_all_dict))], timedelta())
    time_train = sum([_result_all_dict[i][2] for i in range(len(_result_all_dict))], timedelta())
    logger.debug('time_prepare: %s\ttime_train: %s\ttime_total: %s'
                 % (str(time_prepare), str(time_train), str(datetime.now() - t_start)))

    # list of dicts to dict of lists
    if len(_result_all) > 0:
        result_all = dict(zip(_result_all[0], zip(*[d.values() for d in _result_all])))
    else:
        logger.warning('EMPTY RESULT')
        result_all = {k: [] for k in execute_vars.keys()}

    # if train, set step to last executed step
    if train and len(_result_all) > 0:
        step = result_all['step'][-1]

    if return_values:
        sizes = [len(result_all['values'][i]) for i in range(len(_result_all))]
        values_all_ = np.concatenate(result_all['values'])
        if isinstance(model.values_gold, tf.SparseTensor):
            values_all_gold_ = vstack((convert_sparse_tensor_to_sparse_matrix(sm) for sm in result_all['values_gold'])).toarray()
        else:
            values_all_gold_ = np.concatenate(result_all['values_gold'])

        # sum batch losses weighted by individual batch size (can vary at last batch)
        loss_all = sum([result_all['loss'][i] * sizes[i] for i in range(len(_result_all))])
        loss_all /= sum(sizes)
    else:
        values_all_ = None
        values_all_gold_ = None
        loss_all = np.sum(result_all['loss'])

    metrics_dict = collect_metrics(supervisor, sess, epoch, step, loss_all, values_all_, values_all_gold_,
                                   model=model, emit=emit,
                                   test_writer=test_writer, test_result_writer=test_result_writer)
    return step, loss_all, values_all_, values_all_gold_, metrics_dict


def checkpoint_path(logdir, step):
    return os.path.join(logdir, 'model.ckpt-' + str(step))


def csv_test_writer(logdir, mode='w'):
    if not os.path.isdir(logdir):
        os.makedirs(logdir)
    test_result_csv = open(os.path.join(logdir, 'results.csv'), mode, buffering=1)
    fieldnames = ['step', 'loss', 'pearson_r', 'sim_avg']
    test_result_writer = csv.DictWriter(test_result_csv, fieldnames=fieldnames, delimiter='\t')
    return test_result_writer


def get_parameter_count_from_shapes(shapes, shapes_neg=(), selector_prefix='', selector_suffix='',
                                    selector_suffixes_not=()):
    def valid_name(name):
        return name.endswith(selector_suffix) and name.startswith(selector_prefix) \
               and not any([name.endswith(suf_not) for suf_not in selector_suffixes_not])

    filtered_shapes = {k: shapes[k] for k in shapes if valid_name(k) and k not in shapes_neg}
    count = 0
    for tensor_name in filtered_shapes:
        if len(shapes[tensor_name]) > 0:
            count += reduce((lambda x, y: x * y), shapes[tensor_name])
    return count, filtered_shapes


def log_shapes_info(reader, tree_embedder_prefix='TreeEmbedding/', optimizer_suffixes=('/Adam', '/Adam_1')):
    saved_shapes = reader.get_variable_to_shape_map()

    shapes_rev = {k: saved_shapes[k] for k in saved_shapes if '_reverse_' in k}
    p_count, shapes_train_rev = get_parameter_count_from_shapes(shapes_rev,
                                                                selector_suffix=optimizer_suffixes[0])
    logger.debug('(trainable) reverse parameter count: %i' % p_count)
    logger.debug(shapes_train_rev)
    saved_shapes_wo_rev = {k: saved_shapes[k] for k in saved_shapes if k not in shapes_rev}
    # logger.debug(saved_shapes)
    p_count, shapes_te_trainable = get_parameter_count_from_shapes(saved_shapes_wo_rev,
                                                                   selector_prefix=tree_embedder_prefix,
                                                                   selector_suffix=optimizer_suffixes[0])
    logger.debug('(trainable) tree embedder parameter count: %i' % p_count)
    logger.debug(shapes_te_trainable)
    p_count, shapes_te_total = get_parameter_count_from_shapes(saved_shapes_wo_rev,
                                                               shapes_neg=['/'.join(k.split('/')[:-1]) for k in
                                                                           shapes_te_trainable],
                                                               selector_prefix=tree_embedder_prefix,
                                                               selector_suffixes_not=optimizer_suffixes)
    logger.debug('(not trainable) tree embedder parameter count: %i' % p_count)
    logger.debug(shapes_te_total)
    p_count, shapes_nte_trainable = get_parameter_count_from_shapes(saved_shapes_wo_rev,
                                                                    shapes_neg=shapes_te_trainable.keys(),
                                                                    selector_suffix=optimizer_suffixes[0])
    logger.debug('(trainable) remaining parameter count: %i' % p_count)
    logger.debug(shapes_nte_trainable)
    p_count, shapes_nte_total = get_parameter_count_from_shapes(saved_shapes_wo_rev,
                                                                shapes_neg=['/'.join(k.split('/')[:-1]) for k in
                                                                            shapes_te_trainable] + shapes_te_total.keys() + [
                                                                               '/'.join(k.split('/')[:-1]) for k in
                                                                               shapes_nte_trainable],
                                                                selector_suffixes_not=optimizer_suffixes)
    logger.debug('(not trainable) remaining parameter count: %i' % p_count)
    logger.debug(shapes_nte_total)


def get_lexicon(logdir, train_data_path=None, logdir_pretrained=None, logdir_continue=None, dont_dump=False,
                no_fixed_vecs=False, all_vecs_fixed=False, var_vecs_zero=False, var_vecs_random=False,
                additional_vecs_path=None):
    checkpoint_fn = tf.train.latest_checkpoint(logdir)
    if logdir_continue:
        raise NotImplementedError('usage of logdir_continue not implemented')
        assert checkpoint_fn is not None, 'could not read checkpoint from logdir: %s' % logdir
    #old_checkpoint_fn = None
    fine_tune = False
    prev_config = None
    if checkpoint_fn is not None:
        if not checkpoint_fn.startswith(logdir):
            raise ValueError('entry in checkpoint file ("%s") is not located in logdir=%s' % (checkpoint_fn, logdir))
        prev_config = Config(logdir=logdir)
        logger.info('read lex_size from model ...')
        reader = tf.train.NewCheckpointReader(checkpoint_fn)
        log_shapes_info(reader)

        lexicon = Lexicon(filename=os.path.join(logdir, 'model'), checkpoint_reader=reader, add_vocab_manual=True,
                          load_ids_fixed=True)
    else:
        assert train_data_path is not None, 'no checkpoint found and no train_data_path given'
        lexicon = Lexicon(filename=train_data_path, load_ids_fixed=(not no_fixed_vecs), add_vocab_manual=True)

        if logdir_pretrained:
            prev_config = Config(logdir=logdir_pretrained)
            no_fixed_vecs = prev_config.no_fixed_vecs
            additional_vecs_path = prev_config.additional_vecs
            fine_tune = True

        if lexicon.has_vecs:
            #if not no_fixed_vecs and not all_vecs_fixed:
            #    lexicon.set_to_zero(indices=lexicon.ids_fixed, indices_as_blacklist=True)

            if additional_vecs_path:
                logger.info('add embedding vecs from: %s' % additional_vecs_path)
                # ATTENTION: add_lex should contain only lower case entries, because self_to_lowercase=True
                add_lex = Lexicon(filename=additional_vecs_path)
                ids_added = lexicon.add_vecs_from_other(add_lex, self_to_lowercase=True)
                #ids_added_not = [i for i in range(len(lexicon)) if i not in ids_added]
                # remove ids_added_not from lexicon.ids_fixed
                mask_added = np.zeros(len(lexicon), dtype=bool)
                mask_added[ids_added] = True
                mask_fixed = np.zeros(len(lexicon), dtype=bool)
                mask_fixed[lexicon.ids_fixed] = True
                #lexicon._ids_fixed = np.array([_id for _id in lexicon._ids_fixed if _id not in ids_added_not], dtype=lexicon.ids_fixed.dtype)
                lexicon._ids_fixed = (mask_added & mask_fixed).nonzero()[0]

            #ROOT_idx = lexicon.get_d(vocab_manual[ROOT_EMBEDDING], data_as_hashes=False)
            #IDENTITY_idx = lexicon.get_d(vocab_manual[IDENTITY_EMBEDDING], data_as_hashes=False)
            if logdir_pretrained:
                logger.info('load lexicon from pre-trained model: %s' % logdir_pretrained)
                # Check, if flags file is available (because of docker-compose file, logdir_pretrained could be just
                # train path prefix and is therefore not None, but does not point to a valid train dir).
                if os.path.exists(os.path.join(logdir_pretrained, FLAGS_FN)):
                    #old_config = Config(logdir=logdir_pretrained)
                    checkpoint_fn = tf.train.latest_checkpoint(logdir_pretrained)
                    assert checkpoint_fn is not None, 'No checkpoint file found in logdir_pretrained: ' + logdir_pretrained
                    reader_old = tf.train.NewCheckpointReader(checkpoint_fn)
                    log_shapes_info(reader_old)
                    lexicon_old = Lexicon(filename=os.path.join(logdir_pretrained, 'model'))
                    lexicon_old.init_vecs(checkpoint_reader=reader_old)
                    logger.debug('merge old lexicon into new one...')
                    lexicon.merge(lexicon_old, add_entries=True, replace_vecs=True)
                else:
                    logger.warning('logdir_pretrained is not None (%s), but no flags file found. Do not try to load '
                                   'from logdir_pretrained.' % logdir_pretrained)

            if all_vecs_fixed:
                # zero (UNKNOWN) has to remain trainable because of double assignment bug (see TreeEmbedding.embed and
                # Lexicon.transform_idx)
                lexicon.init_ids_fixed(ids_fixed=np.arange(len(lexicon) - 1, dtype=DTYPE_IDX) + 1)

            assert not (var_vecs_zero and var_vecs_random), 'use either var_vecs_zero OR (exclusive) var_vecs_random'
            if var_vecs_zero:
                lexicon.set_to_zero(indices=lexicon.ids_fixed, indices_as_blacklist=True)
            elif var_vecs_random:
                lexicon.set_to_random(indices=lexicon.ids_fixed, indices_as_blacklist=True)

            if not dont_dump:
                logger.debug('dump lexicon to: %s ...' % os.path.join(logdir, 'model'))
                lexicon.dump(filename=os.path.join(logdir, 'model'), strings_only=True)
                assert lexicon.is_filled, 'lexicon: not all vecs for all types are set (len(types): %i, len(vecs): %i)' % \
                                          (len(lexicon), len(lexicon.vecs))
        else:
            logger.warning('NO VECS AVAILABLE FOR LEXICON')



    logger.info('lexicon size: %i' % len(lexicon))
    #logger.debug('IDENTITY_idx: %i' % IDENTITY_idx)
    #logger.debug('ROOT_idx: %i' % ROOT_idx)
    return lexicon, checkpoint_fn, prev_config, fine_tune


def init_model_type(config):
    ## set index and tree getter

    num_classes = None

    # relatedness prediction (SICK)
    if config.model_type == MT_TUPLE_CONTINOUES:
        tree_iterator_args = {'max_depth': config.max_depth, 'context': config.context, 'transform': True,
                              'concat_mode': config.concat_mode}

        tree_iterator = diters.tree_iterator
        indices_getter = diters.indices_sick
        load_parents = (tree_iterator_args['context'] > 0)
        config.batch_iter = batch_iter_simtuple.__name__
    # seeAlso prediction (dbpedianif)
    elif config.model_type == MT_TUPLE_DISCRETE:
        tree_iterator_args = {'max_depth': config.max_depth, 'context': config.context, 'transform': True,
                              'concat_mode': config.concat_mode, 'link_cost_ref': config.link_cost_ref,
                              'bag_of_seealsos': False}

        tree_iterator = diters.tree_iterator
        indices_getter = diters.indices_dbpedianif
        load_parents = (tree_iterator_args['context'] > 0)
        config.batch_iter = batch_iter_naive.__name__
    # language model (reroot)
    elif config.model_type == MT_SINGLE_DISCRETE:
        if config.tree_embedder.strip() not in ['HTU_reduceSUM_mapGRU', 'HTUBatchedHead_reduceSUM_mapGRU']:
            raise NotImplementedError('reroot model only implemented for tree_embedder == '
                                      'HTU_reduceSUM_mapGRU, but it is: %s'
                                      % config.tree_embedder.strip())
        # set tree_embedder to batched head version
        config.tree_embedder = 'HTUBatchedHead_reduceSUM_mapGRU'

        config.batch_iter = batch_iter_reroot.__name__
        logger.debug('set batch_iter to %s' % config.batch_iter)
        tree_iterator_args = {
                              'max_depth': config.max_depth, 'concat_mode': CM_TREE,
                              'transform': True, 'link_cost_ref': config.link_cost_ref, 'link_cost_ref_seealso': -1}
        tree_iterator = diters.tree_iterator
        indices_getter = diters.indices_reroot
        load_parents = True
    # discrete classification
    elif config.model_type in [MT_SINGLE_DISCRETE_INDEPENDENT, MT_TUPLE_DISCRETE_DEPENDENT]:
        other_offset = None
        if config.model_type == MT_SINGLE_DISCRETE_INDEPENDENT:
            # MESH prediction
            if config.task == TASK_MESH_PREDICTION:
                classes_ids = load_class_ids(config.train_data_path, prefix_type=TYPE_MESH)
                num_classes = len(classes_ids)
                classes_root_offset = OFFSET_MESH_ROOT
            # IMDB SENTIMENT prediction
            elif config.task == TASK_SENTIMENT_PREDICTION:
                classes_ids = load_class_ids(config.train_data_path, prefix_type=TYPE_POLARITY)
                num_classes = len(classes_ids)
                classes_root_offset = OFFSET_POLARITY_ROOT
            else:
                raise NotImplementedError(
                    'Task=%s is not implemented for model_type=%s' % (config.task, config.model_type))
        # SICK ENTAILMENT prediction
        elif config.model_type == MT_TUPLE_DISCRETE_DEPENDENT:
            classes_ids = load_class_ids(config.train_data_path, prefix_type=TYPE_ENTAILMENT)
            num_classes = len(classes_ids)
            classes_root_offset = OFFSET_ENTAILMENT_ROOT
            other_offset = OFFSET_OTHER_ENTRY_ROOT + 1
        else:
            #raise NotImplementedError('Task=%s is not implemented for model_type=%s' % (config.task, config.model_type))
            raise NotImplementedError('That should not happen.')
        tree_iterator_args = {'max_depth': config.max_depth, 'context': config.context, 'transform': True,
                              'concat_mode': config.concat_mode, 'link_cost_ref': -1}
        tree_iterator = diters.tree_iterator
        indices_getter = partial(diters.indices_multiclass, classes_all_ids=classes_ids,
                                 classes_root_offset=classes_root_offset, other_offset=other_offset)

        load_parents = (tree_iterator_args['context'] > 0)

        config.batch_iter = batch_iter_multiclass.__name__
    else:
        raise NotImplementedError('model_type=%s not implemented' % config.model_type)

    return tree_iterator, tree_iterator_args, indices_getter, load_parents, num_classes


def get_index_file_names(config, parent_dir, test_files=None, test_only=False, dont_test=False):

    #if config.model_type == MT_REROOT:
    #    return [], []

    fnames_train = None
    fnames_test = None
    if FLAGS.train_files is not None and FLAGS.train_files.strip() != '' and not test_only:
        #logger.info('use train data index files: %s' % FLAGS.train_files)
        fnames_train = [os.path.join(parent_dir, fn) for fn in FLAGS.train_files.split(',')]
    if test_files is not None and test_files.strip() != '' and not dont_test:
        fnames_test = [os.path.join(parent_dir, fn) for fn in FLAGS.test_files.split(',')]
    if not test_only:
        if fnames_train is None:
            logger.info('collect train data from: ' + config.train_data_path + ' ...')
            regex = re.compile(r'%s\.idx\.\d+\.npy$' % ntpath.basename(config.train_data_path))
            _train_fnames = filter(regex.search, os.listdir(parent_dir))
            fnames_train = [os.path.join(parent_dir, fn) for fn in sorted(_train_fnames)]
        assert len(fnames_train) > 0, 'no matching train data files found for ' + config.train_data_path
        logger.info('found ' + str(len(fnames_train)) + ' train data files')
        if fnames_test is None and not dont_test:
            df_indices = [int(idx.strip()) for idx in config.dev_file_indices.split(',')]
            fnames_test = [fnames_train[idx] for idx in df_indices]
            #logger.info('use %s for testing' % str(fnames_test))
            # sort reverse to avoid index shift when deleting
            for idx in sorted(df_indices, reverse=True):
                del fnames_train[idx]
    logger.info('train with:\t%s' % ', '.join([str(fn) for fn in fnames_train or []]))
    logger.info('test with:\t%s' % ', '.join([str(fn) for fn in fnames_test or []]))
    return fnames_train, fnames_test


def check_train_test_overlap(forest_indices_train, forest_indices_train_target, forest_indices_test, forest_indices_test_target):
    logger.info('check for occurrences of test_target_id->test_id in train data...')
    del_count_all = 0
    count_all = 0
    indices_train_rev = {id_train: i for i, id_train in enumerate(forest_indices_train)}
    for i, forest_idx_test in enumerate(forest_indices_test):
        del_count = 0
        for j in range(len(forest_indices_test_target[i])):
            forest_idx_target_test = forest_indices_test_target[i][j - del_count]
            idx_train = indices_train_rev.get(forest_idx_target_test, None)
            if idx_train is not None and forest_idx_target_test in forest_indices_train_target[idx_train]:
                del forest_indices_test_target[i][j - del_count]
                del_count += 1
            else:
                count_all += 1
        del_count_all += del_count
    logger.info('deleted %i test targets. %i test targets remain.' % (del_count_all, count_all))
    return forest_indices_test_target


def compile_trees(tree_iterators, compiler, cache_dir=None, index_file_names=None, index_file_sizes=None,
                  work_forests=None, indices=None, use_pool=True):
    # save compiled trees to file, if cache_dir is given
    if cache_dir is not None:
        assert index_file_names is not None, 'caching of compiled trees to file indicated (because compile cache_dir ' \
                                             'is defined), but no index file names given.'
        assert index_file_sizes is not None, 'caching of compiled trees to file indicated (because compile cache_dir ' \
                                             'is defined), but no index file sizes given.'
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

    compiled_trees = {}
    for m in tree_iterators:
        logger.debug('compile %s trees ...' % m)
        try:
            current_tree_iter = tree_iterators[m](indices=indices[m])
        except TypeError as e:
            assert work_forests is not None, '%s tree_iterator is not satisfied, but work_forests are None' % m

            assert indices is not None, '%s tree_iterator is not satisfied, but indices are None' % m
            assert m in indices, '%s tree_iterator is not satisfied, but no indices for %s given' % (m, m)

            if len(work_forests) == 1:
                # to avoid multi-threading overhead if only one forest is available
                current_tree_iter = tree_iterators[m](forest=work_forests[0], indices=indices[m])
            else:
                #current_tree_iter = tree_iterators[m](forest=work_forests[0], indices=indices[m])
                indices_queue = Queue.Queue()
                trees_queue = Queue.Queue()
                for wf in work_forests:
                    tree_worker = Thread(target=create_trees_simple, args=(indices_queue, trees_queue,
                                                                           tree_iterators[m], wf))
                    tree_worker.setDaemon(True)
                    logger.debug('start prepare thread (multi)...')
                    tree_worker.start()

                res_dict = {}
                compile_worker = Thread(target=compile_batches_simple, args=(trees_queue, res_dict, compiler, use_pool))
                compile_worker.setDaemon(True)
                logger.debug('start compile thread (single)...')
                compile_worker.start()

                bs = 100
                for i, pos in enumerate(range(0, len(indices[m]), bs)):
                    current_indices = indices[m][pos:pos+bs]
                    indices_queue.put((i, current_indices))

                indices_queue.join()
                trees_queue.join()
                compiled_trees[m] = flatten([res_dict[i] for i in range(len(res_dict))])
                logger.info('%s dataset: compiled %i different trees' % (m, len(compiled_trees[m])))

                # TODO: handle caching
                continue

        if cache_dir is None:
            #try:
            if use_pool:
                with compiler.multiprocessing_pool():
                    compiled_trees[m] = list(compiler.build_loom_inputs(([x] for x in current_tree_iter), ordered=True))
            else:
                compiled_trees[m] = list(compiler.build_loom_inputs(([x] for x in current_tree_iter), ordered=True))
            #except TypeError:
            #    # if the tree_iterator is not satisfied, just return it again for later calling with missing arguments
            #    compiled_trees[m] = tree_iterators[m]

        else:
            compiled_trees[m] = []
            cache_fn_names = [os.path.join(cache_dir, '%s.compiled' % os.path.splitext(os.path.basename(ind_fn))[0])
                              for ind_fn in index_file_names[m]]
            # if all trees for all index files are already compiled, the tree_iter do not have to be called
            if all([os.path.exists(fn) for fn in cache_fn_names]):
                for fn in cache_fn_names:
                    logger.debug('load compiled trees from: %s' % fn)
                    #prepared_embeddings[m].extend(np.load(fn).tolist())
                    with open(fn, 'rb') as pf:
                        current_trees = pickle.load(pf)
                    compiled_trees[m].extend(current_trees)
            # otherwise, already compiled trees have to be skipped (i.e. the iterator has to be called)
            else:
                #tree_iter = tree_iterators[m]()
                for i, ind_fn in enumerate(index_file_names[m]):
                    nbr_indices = index_file_sizes[m][i]
                    #base_fn = os.path.splitext(os.path.basename(ind_fn))[0]
                    fn = cache_fn_names[i]
                    if os.path.exists(fn):
                        logger.debug('load compiled trees from: %s' % fn)
                        with open(fn, 'rb') as pf:
                            current_trees = pickle.load(pf)
                        #current_trees = np.load(fn).tolist()
                        for _ in range(nbr_indices):
                            current_tree_iter.next()
                    else:
                        if use_pool:
                            with compiler.multiprocessing_pool():
                                current_trees = list(compiler.build_loom_inputs(([current_tree_iter.next()] for _ in range(nbr_indices)), ordered=True))
                        else:
                            current_trees = list(
                                compiler.build_loom_inputs(([current_tree_iter.next()] for _ in range(nbr_indices)),
                                                           ordered=True))
                        logger.debug('dump compiled trees to: %s' % fn)
                        with open(fn, 'wb') as pf:
                            pickle.dump(current_trees, pf)

                        #np.array(current_trees).dump(fn)
                    compiled_trees[m].extend(current_trees)

        logger.info('%s dataset: compiled %i different trees' % (m, len(compiled_trees[m]) if not callable(compiled_trees[m]) else -1))

    return compiled_trees


def prepare_embeddings_tfidf(tree_iterators, d_unknown, indices, cache_dir=None, index_file_names=None, index_file_sizes=None):
    # save compiled trees to file, if cache_dir is given
    if cache_dir is not None:
        assert index_file_names is not None, 'caching of compiled trees to file indicated (because compile cache_dir ' \
                                             'is defined), but no index file names given.'
        assert index_file_sizes is not None, 'caching of compiled trees to file indicated (because compile cache_dir ' \
                                             'is defined), but no index file sizes given.'
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

    prepared_embeddings = {}
    embedding_dim = -1

    # no caching to file
    if cache_dir is None:
        assert M_TRAIN in tree_iterators, 'if no train data (iterator) is given, a directory containing a vocabulary ' \
                                          'file has to be provided.'
        logger.info('create TF-IDF embeddings for train data set ...')
        (prepared_embeddings[M_TRAIN],), vocab = diters.embeddings_tfidf(
            [tree_iterators[M_TRAIN](indices=indices[M_TRAIN])], d_unknown)
    # with caching to/from file
    else:
        cache_fn_vocab = os.path.join(cache_dir, 'vocab')
        if M_TRAIN in tree_iterators:
            cache_fn_vocab = os.path.join(cache_dir, 'vocab')
            cache_fn_names_train = [os.path.join(cache_dir, '%s.tfidf.npz' % os.path.splitext(os.path.basename(ind_fn))[0])
                                    for ind_fn in index_file_names[M_TRAIN]] if M_TRAIN in index_file_names else []
            # load train tf-idf embeddings, if all exist
            if all([os.path.exists(fn) for fn in cache_fn_names_train]) and numpy_exists(cache_fn_vocab):
                vocab_np = numpy_load(cache_fn_vocab)
                vocab = {v: i for i, v in enumerate(vocab_np)}
                embedding_dim = -1
                if M_TRAIN in tree_iterators:
                    embeddings_list = []
                    for fn in cache_fn_names_train:
                        logger.debug('load tfidf embeddings from files: %s' % fn)
                        embeddings_list.extend(scipy.sparse.load_npz(fn))
                        # check dimensions
                        current_embedding_dim = embeddings_list[-1].shape[1]
                        assert embedding_dim == -1 or embedding_dim == current_embedding_dim, \
                            'current embedding_dim: %i does not match previous one: %i' \
                            % (current_embedding_dim, embedding_dim)
                        embedding_dim = current_embedding_dim
                    prepared_embeddings[M_TRAIN] = vstack(embeddings_list)

            # otherwise recreate train tf-idf embeddings
            else:
                logger.info('create TF-IDF embeddings for TRAIN data set ...')

                def _iter(_m):
                    sizes = index_file_sizes[_m]
                    current_iter = tree_iterators[_m](indices=indices[_m])
                    for s in sizes:
                        trees = [current_iter.next() for _ in range(s)]
                        yield trees

                tree_embeddings_tfidf_train, vocab = diters.embeddings_tfidf(_iter(M_TRAIN), d_unknown)
                vocab_np = np.ones(len(vocab), dtype=int) * np.inf
                for k in vocab:
                    vocab_np[vocab[k]] = k
                assert np.max(vocab_np) < np.inf, 'some value(s) in vocab_np is not set correctly: vocab_np[%i]=%i' \
                                              % (int(np.argmax(vocab_np)), np.max(vocab_np))
                numpy_dump(cache_fn_vocab, vocab_np)

                for i, s in enumerate(index_file_sizes[M_TRAIN]):
                    fn = cache_fn_names_train[i]
                    logger.info('%s dataset (%s): use %i different trees. Dump to file: %s'
                                % (M_TRAIN, cache_fn_names_train[i], tree_embeddings_tfidf_train[i].shape[0], fn))
                    scipy.sparse.save_npz(file=fn, matrix=tree_embeddings_tfidf_train[i])
                    current_embedding_dim = tree_embeddings_tfidf_train[i].shape[1]
                    assert embedding_dim == -1 or embedding_dim == current_embedding_dim, \
                        'current embedding_dim: %i does not match previous one: %i' \
                        % (current_embedding_dim, embedding_dim)
                    embedding_dim = current_embedding_dim
                prepared_embeddings[M_TRAIN] = vstack(tree_embeddings_tfidf_train)

        # if no current train data set, we have to load a vocab from file
        else:
            vocab_np = numpy_load(cache_fn_vocab, assert_exists=True)
            vocab = {v: i for i, v in enumerate(vocab_np)}

    # create test data tf-idf embeddings with train data vocabulary
    if M_TEST in tree_iterators:
        logger.info('create TF-IDF embeddings for TEST data set ...')
        (prepared_embeddings[M_TEST],), _ = diters.embeddings_tfidf([tree_iterators[M_TEST](indices=indices[M_TEST])],
                                                                    d_unknown, vocabulary=vocab)
        current_embedding_dim = prepared_embeddings[M_TEST].shape[1]
        assert embedding_dim == -1 or embedding_dim == current_embedding_dim, \
            'current embedding_dim: %i does not match previous one: %i' % (current_embedding_dim, embedding_dim)
        embedding_dim = current_embedding_dim

    assert embedding_dim != -1, 'no data sets created'
    return prepared_embeddings, embedding_dim


def create_models(config, lexicon, tree_iterators, tree_iterators_tfidf, indices=None, data_dir=None,
                  use_inception_tree_model=False, index_file_names=None, index_file_sizes=None,
                  precompile=True, create_tfidf_embeddings=False, discard_tree_embeddings=False,
                  discard_prepared_embeddings=False, num_classes=None):

    if discard_tree_embeddings:
        logger.warning('discard tree embeddings')
    if discard_prepared_embeddings:
        logger.warning('discard prepared embeddings')

    optimizer = config.optimizer
    if optimizer is not None:
        optimizer = getattr(tf.train, optimizer)

    compiled_trees = None
    prepared_embeddings = None
    prepared_embeddings_dim = -1
    create_tfidf_embeddings = create_tfidf_embeddings or config.tree_embedder == 'tfidf'
    if create_tfidf_embeddings:
        cache_dir = None
        if data_dir is not None:
            # get tfidf config serialization and append number of train files
            # TODO: ATTENTION: if config.train_files does not consist of consecutive index files with increasing split
            # index i, idx.<i>.npy, that path is not sufficient!
            cache_dir = os.path.join(data_dir, 'cache', config.get_serialization_for_calculate_tfidf())
        d_unknown = lexicon.get_d(vocab_manual[UNKNOWN_EMBEDDING], data_as_hashes=False)
        prepared_embeddings, prepared_embeddings_dim = prepare_embeddings_tfidf(tree_iterators=tree_iterators_tfidf,
                                                                                indices=indices,
                                                                                d_unknown=d_unknown,
                                                                                cache_dir=cache_dir,
                                                                                index_file_names=index_file_names,
                                                                                index_file_sizes=index_file_sizes)

    if config.tree_embedder == 'tfidf':
        model_tree = model_fold.DummyTreeModel(embeddings_dim=prepared_embeddings_dim, #tree_count=tree_count,
                                               keep_prob=config.keep_prob, sparse=True,
                                               root_fc_sizes=[int(s) for s in ('0' + config.root_fc_sizes).split(',')],
                                               discard_tree_embeddings=discard_tree_embeddings,
                                               discard_prepared_embeddings=discard_prepared_embeddings)

    else:
        tree_embedder = getattr(model_fold, TREE_EMBEDDER_PREFIX + config.tree_embedder)
        kwargs = {}
        if issubclass(tree_embedder, model_fold.TreeEmbedding_FLATconcat):
            #kwargs['sequence_length'] = model_fold.FLAT_MAX_SIZE
            kwargs['sequence_length'] = config.sequence_length #or 500
            for k in tree_iterators.keys():
                tree_iterators[k] = partial(tree_iterators[k], max_size_plain=kwargs['sequence_length'])
            _padding_idx = lexicon.get_d(vocab_manual[PADDING_EMBEDDING], data_as_hashes=False)
            kwargs['padding_id'] = lexicon.transform_idx(_padding_idx)

        # nbr_trees_out has to be defined for the reroot model because TreeEmbedding_HTUBatchedHead generates a
        # sequence of trees with unspecified length
        if config.model_type == MT_SINGLE_DISCRETE:
            nbr_trees_out = config.neg_samples + 1
        else:
            nbr_trees_out = None

        model_tree = model_fold.SequenceTreeModel(lex_size_fix=lexicon.len_fixed,
                                                  lex_size_var=lexicon.len_var,
                                                  tree_embedder=tree_embedder,
                                                  dimension_embeddings=lexicon.vec_size,
                                                  state_size=config.state_size,
                                                  leaf_fc_size=config.leaf_fc_size,
                                                  # add a leading '0' to allow an empty string
                                                  root_fc_sizes=[int(s) for s in
                                                                 ('0' + config.root_fc_sizes).split(',')],
                                                  keep_prob_default=config.keep_prob,
                                                  nbr_trees_out=nbr_trees_out,  # required for reroot model
                                                  prepared_embeddings_dim=prepared_embeddings_dim,
                                                  prepared_embeddings_sparse=True,
                                                  discard_tree_embeddings=discard_tree_embeddings,
                                                  discard_prepared_embeddings=discard_prepared_embeddings,
                                                  # data_transfomed=data_transformed
                                                  # keep_prob_fixed=config.keep_prob # to enable full head dropout
                                                  **kwargs
                                                  )
        cache_dir = None
        if config.model_type != MT_SINGLE_DISCRETE and data_dir is not None:
            cache_dir = os.path.join(data_dir, 'cache', config.get_serialization_for_compile_trees())

        if precompile:
            compiled_trees = compile_trees(tree_iterators=tree_iterators, compiler=model_tree.compiler,
                                           cache_dir=None if not config.dump_trees else cache_dir,
                                           index_file_names=index_file_names, index_file_sizes=index_file_sizes,
                                           indices=indices)
        elif M_TEST in tree_iterators:
            # TODO: check, if correct
            compiled_trees = compile_trees(tree_iterators={M_TEST: tree_iterators[M_TEST]},
                                           compiler=model_tree.compiler,
                                           cache_dir=None if not config.dump_trees else cache_dir,
                                           index_file_names={M_TEST: index_file_names[M_TEST]},
                                           index_file_sizes={M_TEST: index_file_sizes[M_TEST]},
                                           indices={M_TEST: indices[M_TEST]})

    if use_inception_tree_model:
        inception_tree_model = model_fold.DummyTreeModel(embeddings_dim=model_tree.tree_output_size, sparse=False,
                                                         # TODO: check, if disabling this is correct
                                                         #nbr_trees_out=nbr_trees_out,
                                                         keep_prob=config.keep_prob, root_fc_sizes=0)
    else:
        inception_tree_model = model_tree

    # dbpedianif
    if config.model_type == MT_TUPLE_DISCRETE:
        model = model_fold.TreeTupleModel_with_candidates(tree_model=inception_tree_model,
                                                          nbr_embeddings_in=config.neg_samples + 2,
                                                          fc_sizes=[int(s) for s in ('0' + config.fc_sizes).split(',')],
                                                          optimizer=optimizer,
                                                          learning_rate=config.learning_rate,
                                                          clipping_threshold=config.clipping,
                                                          use_circular_correlation=config.use_circular_correlation
                                                          )

    # reroot
    elif config.model_type == MT_SINGLE_DISCRETE:
        model = model_fold.TreeSingleModel_with_candidates(tree_model=inception_tree_model,
                                                           nbr_embeddings_in=config.neg_samples + 1,
                                                           fc_sizes=[int(s) for s in ('0' + config.fc_sizes).split(',')],
                                                           optimizer=optimizer,
                                                           learning_rate=config.learning_rate,
                                                           clipping_threshold=config.clipping,
                                                           use_circular_correlation=config.use_circular_correlation
                                                           )
    # BIOASQ MESH
    elif config.model_type == MT_SINGLE_DISCRETE_INDEPENDENT:
        #classes_ids = numpy_load(filename='%s.%s' % (config.train_data_path, FE_CLASS_IDS))
        model = model_fold.TreeMultiClassModel(tree_model=inception_tree_model,
                                               fc_sizes=[int(s) for s in ('0' + config.fc_sizes).split(',')],
                                               optimizer=optimizer,
                                               learning_rate=config.learning_rate,
                                               clipping_threshold=config.clipping,
                                               num_classes=num_classes,
                                               nbr_embeddings_in=1,
                                               independent_classes=True
                                               )
    # SICK ENTAILMENT
    elif config.model_type == MT_TUPLE_DISCRETE_DEPENDENT:
        #classes_ids = numpy_load(filename='%s.%s.%s' % (config.train_data_path, TYPE_ENTAILMENT, FE_CLASS_IDS))
        model = model_fold.TreeMultiClassModel(tree_model=inception_tree_model,
                                               fc_sizes=[int(s) for s in ('0' + config.fc_sizes).split(',')],
                                               optimizer=optimizer,
                                               learning_rate=config.learning_rate,
                                               clipping_threshold=config.clipping,
                                               num_classes=num_classes,
                                               nbr_embeddings_in=2,
                                               independent_classes=False
                                               )
    # sim tuple
    elif config.model_type == MT_TUPLE_CONTINOUES:
        model = model_fold.TreeTupleModel(tree_model=inception_tree_model,
                                          #fc_sizes=[int(s) for s in ('0' + config.fc_sizes).split(',')],
                                          optimizer=optimizer,
                                          learning_rate=config.learning_rate,
                                          clipping_threshold=config.clipping,
                                          )

    else:
        raise NotImplementedError('model_type=%s not implemented' % config.model_type)

    return model_tree, model, prepared_embeddings, compiled_trees


def create_models_nearest(prepared_embeddings, model_tree):
    models_nearest = {}
    # set up highest_sims_models
    current_device = get_ith_best_device(1)
    with tf.device(current_device):
        for m in prepared_embeddings.keys():
            logger.debug('create nearest %s model on device: %s' % (m, str(current_device)))
            if isinstance(model_tree, model_fold.DummyTreeModel):
                s = prepared_embeddings[m].shape[0]
            else:
                s = len(prepared_embeddings[m])
            logger.debug('create %s model_highest_sims (number_of_embeddings=%i, embedding_size=%i)' % (
                m, s, model_tree.tree_output_size))
            models_nearest[m] = model_fold.HighestSimsModel(
                number_of_embeddings=s,
                embedding_size=model_tree.tree_output_size,
            )
    return models_nearest


# unused
def blank_kwargs(kwargs, discard_kwargs):
    discard_kwargs_split = {dk.split('.')[0]: dk.split('.')[1:] for dk in discard_kwargs}
    new_kwargs = {}
    for k, v in kwargs.items():
        if k in discard_kwargs_split:
            dk_remaining = discard_kwargs_split[k]
            if len(dk_remaining) > 0:
                assert isinstance(v, dict) or isinstance(v, Config), \
                    'value has to be dict like if subentry is selected via dotted notation (e.g. "config.logdir")'
                # ATTENTION: deepcopy should work on v
                new_kwargs[k] = copy.deepcopy(v)
                p = new_kwargs[k]
                for k_deep in dk_remaining[:-1]:
                    p = p[k_deep]
                del p[dk_remaining[-1]]

        else:
            new_kwargs[k] = v
    return new_kwargs


def exec_cached(cache, func, discard_kwargs=(), add_kwargs=None, *args, **kwargs):
    if cache is None:
        logger.debug('do not use cache because it is None')
        return func(*args, **kwargs)
    key_kwargs = {}
    if discard_kwargs != 'all':
        key_kwargs.update(kwargs)
    if add_kwargs is not None:
        key_kwargs.update(add_kwargs)
    key = json.dumps({'func': func.__name__, 'args': args, 'kwargs': {k: v for k, v in key_kwargs.items() if k not in discard_kwargs}}, sort_keys=True)
    if key not in cache:
        cache[key] = func(*args, **kwargs)
    else:
        logger.debug('use cached value(s) for: %s' % key)
    return cache[key]


def execute_session(supervisor, model_tree, lexicon, init_only, loaded_from_checkpoint, meta, test_writer,
                    test_result_writer, logdir, cache=None, debug=False, clean_train_trees=False):
    with supervisor.managed_session() as sess:
        if lexicon.is_filled:
            logger.info('init embeddings with external vectors...')
            feed_dict = {}
            model_vars = []
            if not isinstance(model_tree, model_fold.DummyTreeModel):
                if lexicon.len_fixed > 0:
                    feed_dict[model_tree.embedder.lexicon_fix_placeholder] = lexicon.vecs_fixed
                    model_vars.append(model_tree.embedder.lexicon_fix_init)
                if lexicon.len_var > 0:
                    feed_dict[model_tree.embedder.lexicon_var_placeholder] = lexicon.vecs_var
                    model_vars.append(model_tree.embedder.lexicon_var_init)
                sess.run(model_vars, feed_dict=feed_dict)

        if init_only:
            supervisor.saver.save(sess, checkpoint_path(logdir, 0))
            return

        # TRAINING #################################################################################################

        # do initial test epoch
        if M_TEST in meta:
            #if not loaded_from_checkpoint or M_TRAIN not in meta:
            _, _, values_all, values_all_gold, stats_dict = do_epoch(
                supervisor,
                sess=sess,
                model=meta[M_TEST][M_MODEL],
                dataset_trees=meta[M_TEST][M_TREES] if M_TREES in meta[M_TEST] else None,
                dataset_embeddings=meta[M_TEST][M_EMBEDDINGS],
                forest_indices=meta[M_TEST][M_INDICES],
                indices_targets=meta[M_TEST][M_INDICES_TARGETS],
                epoch=0,
                train=False,
                emit=not loaded_from_checkpoint,
                test_writer=test_writer,
                test_result_writer=test_result_writer,
                number_of_samples=meta[M_TEST][M_NEG_SAMPLES],
                # number_of_samples=None,
                #highest_sims_model=meta[M_TEST][M_MODEL_NEAREST] if M_MODEL_NEAREST in meta[M_TEST] else None,
                batch_iter=meta[M_TEST][M_BATCH_ITER],
                debug=debug,
                #work_forests=work_forests
            )
            if M_TRAIN not in meta:
                if values_all is None or values_all_gold is None:
                    logger.warning('Predicted and gold values are None. Passed return_values=False?')
                else:
                    values_all.dump(os.path.join(logdir, 'sims.np'))
                    values_all_gold.dump(os.path.join(logdir, 'sims_gold.np'))
                #lexicon.dump(filename=os.path.join(logdir, 'model'), strings_only=True)
                return stats_dict
        else:
            stats_dict = None
        # clear vecs in lexicon to clean up memory
        #if cache is None or cache == {}:
        lexicon.init_vecs()

        logger.info('training the model')
        model_for_metric = M_TEST if M_TEST in meta else M_TRAIN

        if FLAGS.early_stopping_metric is not None and FLAGS.early_stopping_metric.strip() != '':
            assert FLAGS.early_stopping_metric in METRIC_KEYS_DISCRETE + METRIC_KEYS_REGRESSION, \
                'early_stopping_metric=%s not in available metrics: %s' \
                % (FLAGS.early_stopping_metric, ', '.join(METRIC_KEYS_DISCRETE + METRIC_KEYS_REGRESSION))
            metric = FLAGS.early_stopping_metric.strip()
        else:
            if meta[model_for_metric][M_MODEL].model_type == MODEL_TYPE_DISCRETE:
                metric = METRIC_DISCRETE
            elif meta[model_for_metric][M_MODEL].model_type == MODEL_TYPE_REGRESSION:
                metric = METRIC_REGRESSION
            else:
                raise ValueError('no metric defined for model_type=%s' % meta[model_for_metric][M_MODEL].model_type)
        logger.info('use %s as metric for early stopping with window size %i' % (metric, config.early_stopping_window))

        # init recompile for reroot model
        if config.model_type == MT_SINGLE_DISCRETE:
            # only one queue element is necessary
            train_tree_queue = Queue.Queue(1)
            if M_TREES in meta[M_TRAIN] and meta[M_TRAIN][M_TREES] is not None \
                    and M_INDICES in meta[M_TRAIN] and meta[M_TRAIN][M_INDICES] is not None:
                train_tree_queue.put((meta[M_TRAIN][M_INDICES], meta[M_TRAIN][M_TREES]))
            recompile_thread = RecompileThread(q=train_tree_queue, train_tree_iter=meta[M_TRAIN][M_TREE_ITER],
                                               compiler=meta[M_TRAIN][M_MODEL].tree_model.compiler,
                                               train_indices_sampler=meta[M_TRAIN][M_INDICES_SAMPLER],
                                               compile_func=compile_trees)
            recompile_thread.start()
        else:
            train_tree_queue = None
            recompile_thread = None

        if stats_dict is not None:
            stat_queue = [stats_dict]
        else:
            # NOTE: this depends on metric (pearson/mse/roc/...)
            METRIC_MIN_INIT = -1
            stat_queue = [{metric: METRIC_MIN_INIT}]
        max_queue_length = 0
        for epoch, shuffled in enumerate(td.epochs(items=range(len(meta[M_TRAIN][M_INDICES])), n=config.epochs, shuffle=True), 1):

            # TRAIN

            # re-create and compile trees for reroot (language) model
            if train_tree_queue is not None:
                logger.debug('wait for compiled trees ...')
                meta[M_TRAIN][M_INDICES], meta[M_TRAIN][M_TREES] = train_tree_queue.get()
                train_tree_queue.task_done()

            step_train, loss_train, _, _, stats_train = do_epoch(
                supervisor, sess,
                model=meta[M_TRAIN][M_MODEL],
                dataset_trees=meta[M_TRAIN][M_TREES] if M_TREES in meta[M_TRAIN] else None,
                forest_indices=meta[M_TRAIN][M_INDICES],
                dataset_embeddings=meta[M_TRAIN][M_EMBEDDINGS],
                tree_iter=meta[M_TRAIN][M_TREE_ITER],
                indices_targets=meta[M_TRAIN][M_INDICES_TARGETS],
                epoch=epoch,
                number_of_samples=meta[M_TRAIN][M_NEG_SAMPLES],
                #highest_sims_model=meta[M_TRAIN][M_MODEL_NEAREST] if M_MODEL_NEAREST in meta[M_TRAIN] else None,
                batch_iter=meta[M_TRAIN][M_BATCH_ITER],
                return_values=False,
                debug=debug,
                #work_forests=work_forests
            )

            if M_TREES in meta[M_TRAIN] and (clean_train_trees or train_tree_queue is not None):
                logger.debug('delete train trees')
                del meta[M_TRAIN][M_TREES]

            # TEST

            if M_TEST in meta:
                step_test, loss_test, _, _, stats_test = do_epoch(
                    supervisor, sess,
                    model=meta[M_TEST][M_MODEL],
                    dataset_trees=meta[M_TEST][M_TREES] if M_TREES in meta[M_TEST] else None,
                    dataset_embeddings=meta[M_TEST][M_EMBEDDINGS],
                    forest_indices=meta[M_TEST][M_INDICES],
                    indices_targets=meta[M_TEST][M_INDICES_TARGETS],
                    number_of_samples=meta[M_TEST][M_NEG_SAMPLES],
                    epoch=epoch,
                    train=False,
                    test_step=step_train,
                    test_writer=test_writer,
                    test_result_writer=test_result_writer,
                    highest_sims_model=meta[M_TEST][M_MODEL_NEAREST] if M_MODEL_NEAREST in meta[M_TEST] else None,
                    batch_iter=meta[M_TEST][M_BATCH_ITER],
                    return_values=False,
                    debug=debug,
                    #work_forests=work_forests
                )
            else:
                step_test, loss_test, stats_test = step_train, loss_train, stats_train

            # EARLY STOPPING ###############################################################################

            stat = round(stats_test[metric], 6)

            prev_max = max(stat_queue, key=lambda t: t[metric])[metric]
            # stop, if current metric is not bigger than previous values. The amount of regarded
            # previous values is set by config.early_stopping_window
            if stat > prev_max:
                stat_queue = []
            else:
                if len(stat_queue) >= max_queue_length:
                    max_queue_length = len(stat_queue) + 1
            stat_queue.append(stats_test)
            stat_queue_sorted = sorted(stat_queue, reverse=True, key=lambda t: t[metric])
            rank = stat_queue_sorted.index(stats_test)

            # write out queue length
            emit_values(supervisor, sess, step_test, values={'queue_length': len(stat_queue), 'rank': rank},
                        writer=test_writer if M_TEST in meta else None)
            logger.info(
                '%s rank (of %i):\t%i\tdif: %f\tmax_queue_length: %i'
                % (metric, len(stat_queue), rank, (stat - prev_max), max_queue_length))

            if len(stat_queue) == 1 or not config.early_stopping_window or epoch == 0:
                supervisor.saver.save(sess, checkpoint_path(logdir, step_train))

            if 0 < config.early_stopping_window < len(stat_queue):
                #logger.info('last metrics (last rank: %i): %s' % (rank, str(stat_queue)))
                logger.info('last metrics (rank: %i): %s' % (rank, str(stat_queue[-1])))
                logger.info('best metrics: %s' % str(stat_queue_sorted[0]))
                if recompile_thread is not None:
                    logger.debug('wait for recompile_thread ...')
                    recompile_thread.join()
                return stat_queue_sorted[0], cache


def execute_run(config, logdir_continue=None, logdir_pretrained=None, test_files=None, init_only=None, test_only=None,
                cache=None, precompile=True, debug=False, discard_tree_embeddings=False,
                discard_prepared_embeddings=False):
    # config.set_run_description()
    #try:
    #    config.run_description
    #except AttributeError:
    #    config.set_run_description()

    logdir = logdir_continue or os.path.join(FLAGS.logdir, config.run_description)
    logger.info('logdir: %s' % logdir)
    if not os.path.isdir(logdir):
        os.makedirs(logdir)

    fh_debug = logging.FileHandler(os.path.join(logdir, 'train-debug.log'))
    fh_debug.setLevel(logging.DEBUG)
    fh_debug.setFormatter(logging.Formatter(LOGGING_FORMAT))
    logger.addHandler(fh_debug)
    fh_info = logging.FileHandler(os.path.join(logdir, 'train-info.log'))
    fh_info.setLevel(logging.INFO)
    fh_info.setFormatter(logging.Formatter(LOGGING_FORMAT))
    logger.addHandler(fh_info)

    # GET CHECKPOINT or PREPARE LEXICON ################################################################################

    ## get lexicon
    lexicon, checkpoint_fn, prev_config, fine_tune = get_lexicon(
        logdir=logdir, train_data_path=config.train_data_path, logdir_pretrained=logdir_pretrained,
        logdir_continue=logdir_continue, no_fixed_vecs=config.no_fixed_vecs, all_vecs_fixed=config.all_vecs_fixed,
        var_vecs_zero=config.var_vecs_zero, var_vecs_random=config.var_vecs_random,
        additional_vecs_path=config.additional_vecs)
    # use previous tree model config values
    #restore_only_tree_embedder = prev_config is not None and config.model_type != prev_config.model_type
    #if prev_config is not None and config.model_type != prev_config.model_type:
    #    if config.model_type != prev_config.model_type:
        #    reuse_parameters = TREE_MODEL_PARAMETERS
    #        restore_only_tree_embedder = True
        #else:
        #    reuse_parameters = MODEL_PARAMETERS
        #logger.info('use (tree) model parameters from previous model: %s'
        #            % ', '.join(['%s: %s' % (p, prev_config.__getattr__(p)) for p in reuse_parameters]))
        #for p in reuse_parameters:
        #    v = prev_config.__getattr__(p)
        #    config.__setattr__(p, v)

    loaded_from_checkpoint = checkpoint_fn is not None and not fine_tune
    if loaded_from_checkpoint:
        # create test result writer
        test_result_writer = csv_test_writer(os.path.join(logdir, 'test'), mode='a')
    else:
        # create test result writer
        test_result_writer = csv_test_writer(os.path.join(logdir, 'test'))
        test_result_writer.writeheader()
        # dump config
        config.dump(logdir=logdir)


    # TRAINING and TEST DATA ###########################################################################################

    meta = {}
    parent_dir = os.path.abspath(os.path.join(config.train_data_path, os.pardir))

    ## handle train/test index files
    fnames_train, fnames_test = get_index_file_names(config=config, parent_dir=parent_dir, test_files=test_files,
                                                     test_only=test_only, dont_test=FLAGS.dont_test)
    if not (init_only or fnames_train is None or len(fnames_train) == 0):
        meta[M_TRAIN] = {M_FNAMES: fnames_train}
    if not (fnames_test is None or len(fnames_test) == 0):
        meta[M_TEST] = {M_FNAMES: fnames_test}

    tree_iterator, tree_iterator_args, indices_getter, load_parents, num_classes = init_model_type(config)
    tree_iterator_args_tfidf = None
    if config.use_tfidf or config.tree_embedder == 'tfidf':
        tree_iterator_args_tfidf = tree_iterator_args.copy()
        tree_iterator_args_tfidf['concat_mode'] = CM_AGGREGATE
        tree_iterator_args_tfidf['context'] = 0
        tree_iterator_args_tfidf['max_size_plain'] = config.sequence_length

    if debug:
        tree_iterator_args['debug'] = True

    # load forest data
    lexicon_root_fn = '%s.root.id' % config.train_data_path
    if Lexicon.exist(lexicon_root_fn, types_only=True):
        logging.info('load lexicon_roots from %s' % lexicon_root_fn)
        lexicon_roots = Lexicon(filename=lexicon_root_fn, load_vecs=False)
    else:
        lexicon_roots = None
    forest = Forest(filename=config.train_data_path, lexicon=lexicon, load_parents=load_parents, lexicon_roots=lexicon_roots)

    #if config.model_type == MT_REROOT:
    logger.debug('set ids to IDENTITY')
    d_identity = lexicon.get_d(s=vocab_manual[IDENTITY_EMBEDDING], data_as_hashes=False)
    forest.data[forest.roots + OFFSET_ID] = d_identity

    # TODO: use this?
    #if config.model_type == MT_REROOT:
    #    logger.info('transform data ...')
    #    data_transformed = [forest.lexicon.transform_idx(forest.data[idx], root_id_pos=forest.root_id_pos) for idx in range(len(forest))]
    #else:
    #    data_transformed = None

    # calc indices (ids, indices, other_indices) from index files
    logger.info('calc indices from index files ...')
    for m in meta:
        assert M_FNAMES in meta[m], 'no %s fnames found' % m
        meta[m][M_INDICES], meta[m][M_INDICES_TARGETS], meta[m][M_INDEX_FILE_SIZES] = indices_getter(index_files=meta[m][M_FNAMES], forest=forest)
        # dump tree indices
        if not loaded_from_checkpoint:
            numpy_dump(os.path.join(logdir, '%s.%s' % (FN_TREE_INDICES, m)), meta[m][M_INDICES])

    if config.model_type == MT_TUPLE_DISCRETE:
        if M_TEST in meta and M_TRAIN in meta \
                and meta[M_TRAIN][M_INDICES_TARGETS] is not None and meta[M_TEST][M_INDICES_TARGETS] is not None:
            meta[M_TEST][M_INDICES_TARGETS] = check_train_test_overlap(forest_indices_train=meta[M_TRAIN][M_INDICES],
                                                                       forest_indices_train_target=meta[M_TRAIN][M_INDICES_TARGETS],
                                                                       forest_indices_test=meta[M_TEST][M_INDICES],
                                                                       forest_indices_test_target=meta[M_TEST][M_INDICES_TARGETS])

    # set batch iterators and numbers of negative samples
    if M_TEST in meta:
        meta[M_TEST][M_BATCH_ITER] = config.batch_iter
        meta[M_TEST][M_NEG_SAMPLES] = config.neg_samples
    if M_TRAIN in meta:
        meta[M_TRAIN][M_BATCH_ITER] = config.batch_iter
        meta[M_TRAIN][M_NEG_SAMPLES] = config.neg_samples

    # set tree iterator
    for m in meta:
        if config.model_type == MT_SINGLE_DISCRETE:
            nbr_indices = config.nbr_trees or 1000
            if m == M_TEST and config.nbr_trees_test:
                nbr_indices = config.nbr_trees_test
            logger.info('%s: use %i indices per epoch (forest size: %i)' % (m, nbr_indices, len(forest)))

            root_indices = meta[m][M_INDICES]
            # create a mapping to all data that will be used in training
            indices_mapping = np.concatenate([np.arange(forest.roots[root_idx],
                                                        forest.roots[root_idx + 1] if root_idx + 1 < len(
                                                            forest.roots) else len(forest)) for root_idx in
                                              root_indices])

            def _sample_indices():
                logger.debug('select %i new root indices (selected data size: %i)' % (nbr_indices, len(indices_mapping)))
                if debug:
                    logger.warning('use %i FIXED indices (debug: True)' % nbr_indices)
                    assert nbr_indices <= len(indices_mapping), 'nbr_indices (%i) is higher then selected data size (%i)' \
                                                                % (nbr_indices, len(indices_mapping))
                    _indices = np.arange(nbr_indices, dtype=DTYPE_IDX)
                else:
                    _indices = np.random.randint(len(indices_mapping), size=nbr_indices)
                return indices_mapping[_indices]

            # overwrite root indices with index sampler
            #meta[m][M_INDICES] = np.zeros(nbr_indices)
            meta[m][M_INDICES_SAMPLER] = _sample_indices
            meta[m][M_INDICES] = _sample_indices()
            meta[m][M_TREE_ITER] = partial(diters.reroot_wrapper,
                                           tree_iter=tree_iterator, forest=forest,
                                           neg_samples=meta[m][M_NEG_SAMPLES], #nbr_indices=nbr_indices,
                                           indices_mapping=indices_mapping, **tree_iterator_args)
        else:
            #if precompile:
            meta[m][M_TREE_ITER] = partial(tree_iterator, forest=forest, **tree_iterator_args)
            if tree_iterator_args_tfidf is not None:
                meta[m][M_TREE_ITER_TFIDF] = partial(tree_iterator, forest=forest, **tree_iterator_args_tfidf)
            #else:
            #    meta[m][M_TREE_ITER] = partial(tree_iterator, **tree_iterator_args)

    # MODEL DEFINITION #################################################################################################

    current_device = get_ith_best_device(0)
    logger.info('create tensorflow graph on device: %s ...' % str(current_device))
    with tf.device(current_device):
        with tf.Graph().as_default() as graph:
            logger.debug('trainable lexicon entries: %i' % lexicon.len_var)
            logger.debug('fixed lexicon entries:     %i' % lexicon.len_fixed)

            model_tree, model, prepared_embeddings, compiled_trees = create_models(
                config=config, lexicon=lexicon,  #tree_count=tree_count, #logdir=logdir,
                tree_iterators={m: meta[m][M_TREE_ITER] for m in meta},
                tree_iterators_tfidf={m: meta[m][M_TREE_ITER_TFIDF] for m in meta if M_TREE_ITER_TFIDF in meta[m]},
                #cache=cache,
                data_dir=parent_dir,
                index_file_names={m: meta[m][M_FNAMES] for m in meta},
                index_file_sizes={m: meta[m][M_INDEX_FILE_SIZES] for m in meta},
                #work_forests=work_forests if precompile else None,
                indices={m: meta[m][M_INDICES] for m in meta},
                precompile=precompile,
                create_tfidf_embeddings=config.use_tfidf,
                discard_tree_embeddings=discard_tree_embeddings,
                discard_prepared_embeddings=discard_prepared_embeddings,
                num_classes=num_classes
            )

            #models_nearest = create_models_nearest(model_tree=model_tree,
            #                                       prepared_embeddings={m: prepared_embeddings[m] for m in meta.keys()
            #                                                            if M_BATCH_ITER in meta[m]
            #                                                            and meta[m][M_BATCH_ITER].strip() == batch_iter_nearest.__name__})

            # set model(s) and prepared embeddings
            for m in meta:
                meta[m][M_MODEL] = model
                #if m in models_nearest:
                #    meta[m][M_MODEL_NEAREST] = models_nearest[m]
                if compiled_trees is not None and m in compiled_trees:
                    meta[m][M_TREES] = compiled_trees[m]
                else:
                    meta[m][M_TREES] = None
                if prepared_embeddings is not None and m in prepared_embeddings:
                    meta[m][M_EMBEDDINGS] = prepared_embeddings[m]
                else:
                    meta[m][M_EMBEDDINGS] = None


            # PREPARE TRAINING #########################################################################################

            if fine_tune:
                logger.info('restore from old_checkpoint (except lexicon, step and optimizer vars): %s ...'
                            % checkpoint_fn)
                optimizer_vars = meta[M_TRAIN][M_MODEL].optimizer_vars() + [meta[M_TRAIN][M_MODEL].global_step] \
                                 + ((meta[M_TEST][M_MODEL].optimizer_vars() + [
                    meta[M_TEST][M_MODEL].global_step]) if M_TEST in meta and meta[M_TEST][M_MODEL] != meta[M_TRAIN][M_MODEL] else [])

                lexicon_vars = [v for v in set(model_fold.get_lexicon_vars()) if v not in optimizer_vars]
                tree_embedder_vars = [v for v in set(model_fold.get_tree_embedder_vars()) if v not in optimizer_vars]

                if config.model_type != prev_config.model_type:
                    restore_vars = tree_embedder_vars
                else:
                    restore_vars = [item for item in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if
                                    item not in lexicon_vars + optimizer_vars]
                logger.debug('restore vars: %s' % str(restore_vars))
                pre_train_saver = tf.train.Saver(restore_vars)
            else:
                pre_train_saver = None

            #for var in tf.trainable_variables:  # or tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'your desired scope name'):
            for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
                tf.summary.histogram(var.name.replace(':', '/'), var)
                with tf.name_scope(var.name.replace(':', '/')):
                    with tf.name_scope('summaries'):
                        mean = tf.reduce_mean(var)
                        tf.summary.scalar('mean', mean)
                    with tf.name_scope('stddev'):
                        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
                    tf.summary.scalar('stddev', stddev)
                    tf.summary.scalar('max', tf.reduce_max(var))
                    tf.summary.scalar('min', tf.reduce_min(var))

            def load_pretrain(sess):
                pre_train_saver.restore(sess, checkpoint_fn)

            # Set up the supervisor.
            supervisor = tf.train.Supervisor(
                # saver=None,# my_saver,
                logdir=logdir,
                #is_chief=(FLAGS.task == 0),
                save_summaries_secs=10,
                save_model_secs=0,
                summary_writer=tf.summary.FileWriter(os.path.join(logdir, 'train'), graph),
                init_fn=load_pretrain if fine_tune else None
            )
            #if dev_iterator is not None or test_iterator is not None:
            test_writer = tf.summary.FileWriter(os.path.join(logdir, 'test'), graph) if M_TEST in meta else None
            #sess = supervisor.PrepareSession(FLAGS.master)
            #sess = supervisor.PrepareSession('')
            # TODO: try
            #sess = supervisor.PrepareSession(FLAGS.master, config=tf.ConfigProto(log_device_placement=True))

            #if precompile:
            #    work_forests = None
            res = execute_session(supervisor, model_tree, lexicon, init_only, loaded_from_checkpoint, meta, test_writer,
                                  test_result_writer, logdir, cache, debug, clean_train_trees=not precompile)
            logger.removeHandler(fh_info)
            logger.removeHandler(fh_debug)
            supervisor.stop()
            return res


def add_metrics(d, stats, metric_main=None, prefix=''):
    if metric_main is not None and metric_main.strip() != '':
        assert metric_main in stats, 'manually defined metric_main=%s not found in stats keys: %s' % (metric_main, ', '.join(stats.keys()))
        if metric_main in METRIC_KEYS_REGRESSION:
            metric_keys = METRIC_KEYS_REGRESSION
        elif metric_main in METRIC_KEYS_DISCRETE:
            metric_keys = METRIC_KEYS_DISCRETE
        else:
            raise ValueError('metric_main=%s has to be in either %s or %s'
                             % (metric_main, ', '.join(METRIC_KEYS_REGRESSION), ', '.join(METRIC_DISCRETE)))
    else:
        if METRIC_REGRESSION in stats:
            metric_keys = METRIC_KEYS_REGRESSION
            metric_main = METRIC_REGRESSION
        elif METRIC_DISCRETE in stats:
            metric_keys = METRIC_KEYS_DISCRETE
            metric_main = METRIC_DISCRETE
        else:
            raise ValueError('stats has to contain either %s or %s' % (METRIC_REGRESSION, METRIC_DISCRETE))
    for k in metric_keys:
        if k in stats:
            d[prefix + k] = stats[k]
    return metric_main


if __name__ == '__main__':
    logging_init()
    # account for prefix if started via docker-compose.yml
    if FLAGS.logdir_continue is not None and FLAGS.logdir_continue.strip() == '/root/train/':
        logdir_continue = None
    else:
        logdir_continue = FLAGS.logdir_continue
    if FLAGS.logdir_pretrained is not None and FLAGS.logdir_pretrained.strip() == '/root/train/':
        logdir_pretrained = None
    else:
        logdir_pretrained = FLAGS.logdir_pretrained

    # Handle multiple logdir_continue's
    # ATTENTION: discards any FLAGS (e.g. provided as argument) contained in default_config!
    if logdir_continue is not None and ',' in logdir_continue:
        logdirs = logdir_continue.split(',')
        logger.info('execute %i runs ...' % len(logdirs))
        stats_prefix = 'score_'
        with open(os.path.join(FLAGS.logdir, 'scores_new.tsv'), 'w') as csvfile:
            fieldnames = Config(logdir=logdirs[0]).as_dict().keys() \
                         + [stats_prefix + k for k in METRIC_KEYS_DISCRETE + METRIC_KEYS_REGRESSION]
            score_writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter='\t', extrasaction='ignore')
            score_writer.writeheader()
            for i, logdir in enumerate(logdirs, 1):
                logger.info('START RUN %i of %i' % (i, len(logdirs)))
                config = Config(logdir=logdir)
                config_dict = config.as_dict()
                stats, _ = execute_run(config, logdir_continue=logdir, logdir_pretrained=logdir_pretrained,
                                       test_files=FLAGS.test_files, init_only=FLAGS.init_only,
                                       test_only=FLAGS.test_only,
                                       precompile=FLAGS.precompile, debug=FLAGS.debug,
                                       discard_tree_embeddings=FLAGS.discard_tree_embeddings,
                                       discard_prepared_embeddings=FLAGS.discard_prepared_embeddings,
                                       )

                add_metrics(config_dict, stats, metric_main=FLAGS.early_stopping_metric, prefix=stats_prefix)
                score_writer.writerow(config_dict)
                csvfile.flush()
    else:
        config = Config(logdir=logdir_continue, logdir_pretrained=logdir_pretrained)
        USE_CACHE = False
        # get default config (or load from logdir_continue/logdir_pretrained)
        config.init_flags()
        # pylint: disable=protected-access
        FLAGS._parse_flags()
        # pylint: enable=protected-access
        # keep (TREE_)MODEL_PARAMETERS
        config.update_with_flags(FLAGS, keep_model_parameters=logdir_continue or logdir_pretrained)
        if FLAGS.grid_config_file is not None and FLAGS.grid_config_file.strip() != '':

            scores_fn = os.path.join(FLAGS.logdir, 'scores.tsv')
            fieldnames_loaded = None
            run_descriptions_done = []
            scores_done = []
            if os.path.isfile(scores_fn):
                #file_mode = 'a'
                with open(scores_fn, 'r') as csvfile:
                    scores_done_reader = csv.DictReader(csvfile, delimiter='\t')
                    fieldnames_loaded = scores_done_reader.fieldnames
                    scores_done = list(scores_done_reader)
                run_descriptions_done = [s_d['run_description'] for s_d in scores_done]
                logger.debug('already finished: %s' % ', '.join(run_descriptions_done))
            #else:
                #file_mode = 'w'

            logger.info('write scores to: %s' % scores_fn)

            parameters_fn = os.path.join(FLAGS.logdir, FLAGS.grid_config_file)
            f_ext = os.path.splitext(parameters_fn)[1]
            if f_ext == '.json':
                logger.info('load grid parameters from json: %s' % parameters_fn)
                with open(parameters_fn, 'r') as infile:
                    grid_parameters = json.load(infile)
                parameters_keys, settings = config.explode(grid_parameters, fieldnames_loaded)
            elif f_ext in ['.jl', '.jsonl']:
                logger.info('load parameter settings from json lines: %s' % parameters_fn)
                with open(parameters_fn, 'r') as infile:
                    list_parameters = [json.loads(line) for line in infile.readlines() if line.strip() != "" and line.strip()[0] != '#']
                assert len(list_parameters) > 0, 'parameters file does not contain any setting'
                parameters_keys, settings = config.create_new_configs(list_parameters, fieldnames_loaded)
            else:
                raise ValueError('Unknown parameters file extension: %s. Use ".json" for json files (indicates grid '
                                 'search) and ".jsonl" or ".jl" for json line files (indicates individual settings per '
                                 'line)' % f_ext)

            stats_prefix_dev = 'dev_best_'
            stats_prefix_test = 'test_'

            fieldnames_expected = sorted(list(parameters_keys)) + [stats_prefix_dev + k for k in METRIC_KEYS_DISCRETE + METRIC_KEYS_REGRESSION] \
                                  + [stats_prefix_test + k for k in METRIC_KEYS_DISCRETE + METRIC_KEYS_REGRESSION] + ['run_description']
            #assert fieldnames_loaded is None or set(fieldnames_loaded) == set(fieldnames_expected), 'field names in tsv file are not as expected'
            #fieldnames = fieldnames_loaded or fieldnames_expected
            with open(scores_fn, 'w') as csvfile:
                score_writer = csv.DictWriter(csvfile, fieldnames=fieldnames_expected, delimiter='\t', extrasaction='ignore')
                #if file_mode == 'w':
                score_writer.writeheader()
                score_writer.writerows(scores_done)
                csvfile.flush()

                logger.info('execute %i different settings, repeat each %i times' % (len(settings), FLAGS.run_count))
                for c, d in settings:
                    assert c.early_stopping_window > 0, 'early_stopping_window has to be set (i.e. >0) if multiple runs are executed'
                    cache_dev = {}
                    cache_test = {}
                    for i in range(FLAGS.run_count):
                        c.set_run_description()
                        run_desc_backup = c.run_description

                        logger.info(
                            'start run ==============================================================================')
                        c.run_description = os.path.join(run_desc_backup, str(i))
                        logdir = os.path.join(FLAGS.logdir, c.run_description)

                        # check, if the test file exists, before executing the run
                        train_data_dir = os.path.abspath(os.path.join(c.train_data_path, os.pardir))
                        use_test_files = False
                        if FLAGS.test_files is not None and FLAGS.test_files.strip() != '':
                            for t_fn in FLAGS.test_files.strip().split(','):
                                current_test_fname = os.path.join(train_data_dir, t_fn)
                                assert os.path.isfile(current_test_fname), 'could not find test file: %s' % current_test_fname
                                use_test_files = True

                        # skip already processed
                        if os.path.isdir(logdir) and c.run_description in run_descriptions_done:
                            logger.debug('skip config for logdir: %s' % logdir)
                            c.run_description = run_desc_backup
                            continue

                        # train
                        metrics_dev, cache_dev = execute_run(c, cache=cache_dev if USE_CACHE else None,
                                                             precompile=FLAGS.precompile,
                                                             debug=FLAGS.debug)
                        main_metric = add_metrics(d, metrics_dev, metric_main=FLAGS.early_stopping_metric, prefix=stats_prefix_dev)
                        logger.info('best dev score (%s): %f' % (main_metric, metrics_dev[main_metric]))

                        # test
                        if use_test_files:
                            metrics_test, cache_test = execute_run(c, logdir_continue=logdir, test_only=True,
                                                                   precompile=FLAGS.precompile,
                                                                   test_files=FLAGS.test_files,
                                                                   cache=cache_test if USE_CACHE else None,
                                                                   debug=FLAGS.debug)
                            main_metric = add_metrics(d, metrics_test, metric_main=FLAGS.early_stopping_metric,
                                                      prefix=stats_prefix_test)
                            logger.info('test score (%s): %f' % (main_metric, metrics_test[main_metric]))
                        d['run_description'] = c.run_description

                        c.run_description = run_desc_backup
                        score_writer.writerow(d)
                        csvfile.flush()

        # default: execute single run
        else:
            config.set_run_description()
            execute_run(config, logdir_continue=logdir_continue, logdir_pretrained=logdir_pretrained,
                        test_files=FLAGS.test_files, init_only=FLAGS.init_only, test_only=FLAGS.test_only,
                        precompile=FLAGS.precompile, debug=FLAGS.debug,
                        discard_tree_embeddings=FLAGS.discard_tree_embeddings,
                        discard_prepared_embeddings=FLAGS.discard_prepared_embeddings,)
