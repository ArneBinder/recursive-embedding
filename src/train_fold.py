# from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import fnmatch
import json
import logging
import ntpath
import os
import re
# import google3
# import shutil
import scipy
from functools import reduce, partial

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
import mytools
from mytools import numpy_load
from sequence_trees import Forest
from constants import vocab_manual, KEY_HEAD, KEY_CHILDREN, ROOT_EMBEDDING, IDENTITY_EMBEDDING, DTYPE_OFFSET, TYPE_REF, \
    TYPE_REF_SEEALSO, UNKNOWN_EMBEDDING, UNIQUE_EMBEDDING, TYPE_SECTION_SEEALSO, LOGGING_FORMAT, CM_AGGREGATE, CM_TREE, M_INDICES, M_TEST, \
    M_TRAIN, M_MODEL, M_FNAMES, M_TREES, M_DATA, M_TREE_ITER, M_INDICES_TARGETS, M_BATCH_ITER, M_NEG_SAMPLES, \
    M_MODEL_NEAREST, FN_TREE_INDICES
from config import Config
#from data_iterators import data_tuple_iterator_reroot, data_tuple_iterator_dbpedianif, data_tuple_iterator, \
#    indices_dbpedianif
import data_iterators as diters
from corpus import FE_CLASS_IDS

# non-saveable flags
tf.flags.DEFINE_string('logdir',
                       # '/home/arne/ML_local/tf/supervised/log/dataPs2aggregate_embeddingsUntrainable_simLayer_modelTreelstm_normalizeTrue_batchsize250',
                       #  '/home/arne/ML_local/tf/supervised/log/dataPs2aggregate_embeddingsTrainable_simLayer_modelAvgchildren_normalizeTrue_batchsize250',
                       #  '/home/arne/ML_local/tf/supervised/log/SA/EMBEDDING_FC_dim300',
                       '/home/arne/ML_local/tf/supervised/log/SA/PRETRAINED',
                       'Directory in which to write event logs.')
tf.flags.DEFINE_string('test_file',
                       None,
                       'Set this to execute only.')
tf.flags.DEFINE_string('train_files',
                       None,
                       'If set, do not look for idx.<id>.npy files (in train data dir), '
                       'but use these index files instead (separated by comma)')
tf.flags.DEFINE_boolean('test_only',
                        False,
                        'Enable to execute evaluation only.')
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
tf.flags.DEFINE_integer('run_count',
                        1,
                        'repeat each run this often')
tf.flags.DEFINE_boolean('debug',
                        False,
                        'enable debug mode (additional output, but slow)')

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
METRIC_KEYS_DISCRETE = ['roc', 'f1_t10', 'f1_t33', 'f1_t50', 'f1_t66', 'f1_t90', 'precision_t10', 'precision_t33', 'precision_t50', 'precision_t66', 'precision_t90', 'recall_t10', 'recall_t33', 'recall_t50', 'recall_t66', 'recall_t90']
METRIC_DISCRETE = 'f1_t50'
#STAT_KEY_MAIN_DISCRETE = 'roc_micro'
METRIC_KEYS_REGRESSION = ['pearson_r', 'mse']
METRIC_REGRESSION = 'pearson_r'
#STAT_KEY_MAIN_REGRESSION = 'pearson_r'

MT_MULTICLASS = 'multiclass'
MT_REROOT = 'reroot'
MT_TREETUPLE = 'tuple'

logger = logging.getLogger('')
logger.setLevel(logging.DEBUG)
logger_streamhandler = logging.StreamHandler()
logger_streamhandler.setLevel(logging.INFO)
logger_streamhandler.setFormatter(logging.Formatter(LOGGING_FORMAT))
#logger.addHandler(logger_streamhandler)

DT_PROBS = np.float32


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
    # if sim is not None and sim_gold is not None:
    if model.model_type == MODEL_TYPE_REGRESSION:
        if values is not None and values_gold is not None:
            p_r = pearsonr(values, values_gold)
            s_r = spearmanr(values, values_gold)
            mse = np.mean(np.square(values - values_gold))
            emit_dict.update({
                'pearson_r': p_r[0],
                'spearman_r': s_r[0],
                'mse': mse
            })
        #info_string = 'epoch=%d step=%d: loss_%s=%f\tpearson_r_%s=%f\tavg=%f\tvar=%f\tgold_avg=%f\tgold_var=%f' \
        #              % (epoch, step, suffix, loss, suffix, p_r[0], np.average(values), np.var(values),
        #                 np.average(values_gold), np.var(values_gold))

        stats_string = '\t'.join(['%s=%f' % (k, emit_dict[k]) for k in METRIC_KEYS_REGRESSION if k in emit_dict])
        info_string = 'epoch=%d step=%d %s: loss=%f\t%s' % (epoch, step, suffix, loss, stats_string)
    elif model.model_type == MODEL_TYPE_DISCRETE:
        #filtered = np.argwhere(values_gold[:, 0] == 1).flatten()
        #if len(filtered) < len(values_gold):
        #    logger.warning('discarded %i (of %i) values for evalution (roc)'
        #                   % (len(values_gold) - len(filtered), len(values_gold)))
        #roc = metrics.roc_auc_score(values_gold[filtered].flatten(), values[filtered].flatten())
        #emit_dict['roc_micro'] = metrics.roc_auc_score(values_gold, values, average='micro')
        #emit_dict['roc_micro'] = sess.run(model.auc, {model.eval_gold_placeholder: values_gold, model.eval_predictions_placeholder: values})
        #roc_samples = metrics.roc_auc_score(values_gold, values, average='samples')
        ms = sess.run(model.metrics)
        for k in ms.keys():
            spl = k.split(':')
            if len(spl) > 1:
                spl_t = spl[1].split(',')
                for i, v in enumerate(ms[k]):
                    emit_dict[spl[0] + '_t' + spl_t[i]] = v
            else:
                emit_dict[k] = ms[k]

        for k in emit_dict.keys():
            if k.startswith('precision'):
                suf = k[len('precision'):]
                if 'recall'+suf in emit_dict.keys():
                    r = emit_dict['recall'+suf]
                    p = emit_dict[k]
                    f1 = 2 * p * r / (p + r)
                    emit_dict['f1' + suf] = f1



        #emit_dict['ranking_loss_inv'] = 1.0 - metrics.label_ranking_loss(values_gold, values)

        #emit_dict.update({
        #    'roc_micro': roc_micro,
        #    #'roc_samples': roc_samples,
        #    'ranking_loss_inv': ranking_loss_inv,
        #    'f1_t50': f1_t50,
        #    'f1_t33': f1_t33,
        #    'f1_t66': f1_t66,
        #    'acc_t50': acc_t50,
        #    'acc_t33': acc_t33,
        #    'acc_t66': acc_t66,
        #})
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


def batch_iter_naive(number_of_samples, forest_indices, forest_indices_targets, idx_forest_to_idx_trees, sampler=None):

    if sampler is None:
        def sampler(idx_target):
            # sample from [1, len(dataset_indices)-1] (inclusive); size +1 to hold the correct target and +1 to hold
            # the origin/reference (idx)
            sample_indices = np.random.random_integers(len(forest_indices) - 1, size=number_of_samples + 1 + 1)
            # replace sampled correct target with 0 (0 is located outside the sampled values, so statistics remain correct)
            # (e.g. do not sample the correct target)
            sample_indices[sample_indices == idx_forest_to_idx_trees[idx_target]] = 0
            return sample_indices

    indices = np.arange(len(forest_indices))
    np.random.shuffle(indices)
    for i in indices:
        idx = forest_indices[i]
        for idx_target in forest_indices_targets[i]:
            sample_indices = sampler(idx_target)
            # set the first to the correct target
            sample_indices[1] = idx_forest_to_idx_trees[idx_target]
            # set the 0th to the origin/reference
            sample_indices[0] = idx_forest_to_idx_trees[idx]

            # convert candidates to ids
            candidate_indices = forest_indices[sample_indices[1:]]
            ix = np.isin(candidate_indices, forest_indices_targets[i])
            probs = np.zeros(shape=len(candidate_indices), dtype=DT_PROBS)
            probs[ix] = 1

            yield sample_indices, probs


def batch_iter_nearest(number_of_samples, forest_indices, forest_indices_targets, sess, tree_model,
                       highest_sims_model, dataset_trees, tree_model_batch_size, idx_forest_to_idx_trees):
    _tree_embeddings = []
    feed_dict = {}
    if isinstance(tree_model, model_fold.DummyTreeModel):
        for start in range(0, dataset_trees.shape[0], tree_model_batch_size):
            feed_dict[tree_model.embeddings_placeholder] = convert_sparse_matrix_to_sparse_tensor(dataset_trees[start:start+tree_model_batch_size])
            current_tree_embeddings = sess.run(tree_model.embeddings_all, feed_dict)
            _tree_embeddings.append(current_tree_embeddings)
    else:
        for batch in td.group_by_batches(dataset_trees, tree_model_batch_size):
            feed_dict[tree_model.compiler.loom_input_tensor] = batch
            current_tree_embeddings = sess.run(tree_model.embeddings_all, feed_dict)
            _tree_embeddings.append(current_tree_embeddings)
    dataset_trees_embedded = np.concatenate(_tree_embeddings)
    logger.debug('calculated %i embeddings ' % len(dataset_trees_embedded))

    s = dataset_trees_embedded.shape[0]
    # calculate cosine sim for all combinations by tree-index ([0..tree_count-1])
    normed = pp.normalize(dataset_trees_embedded, norm='l2')
    logger.debug('normalized %i embeddings' % s)

    current_device = get_ith_best_device(1)
    with tf.device(current_device):
        logger.debug('calc nearest on device: %s' % str(current_device))
        neg_sample_indices = np.zeros(shape=(s, number_of_samples), dtype=np.int32)

        # initialize normed embeddings
        sess.run(highest_sims_model.normed_embeddings_init,
                 feed_dict={highest_sims_model.normed_embeddings_placeholder: normed})
        for i in range(s):
            current_sims = sess.run(highest_sims_model.sims,
                                    {
                                        highest_sims_model.reference_idx: i,
                                    })
            current_sims[i] = 0
            current_indices = np.argpartition(current_sims, -number_of_samples)[-number_of_samples:]
            neg_sample_indices[i, :] = current_indices

    # TODO: clear normed_embeddings (or move to second gpu?)
    #sess.run(highest_sims_model.normed_embeddings_init,
    #         feed_dict={highest_sims_model.normed_embeddings_placeholder: normed})
    logger.debug('created nearest indices')

    def sampler(idx_target):
        sample_indices = np.zeros(shape=number_of_samples + 1 + 1, dtype=np.int32)
        sample_indices[2:] = neg_sample_indices[idx_forest_to_idx_trees[idx_target]]
        return sample_indices

    for sample_indices, probs in batch_iter_naive(number_of_samples, forest_indices, forest_indices_targets,
                                                  idx_forest_to_idx_trees, sampler=sampler):
        yield sample_indices, probs


#def batch_iter_reroot(forest_indices, number_of_samples, data_transformed):
#    for idx in forest_indices:
#        samples = np.random.choice(data_transformed, size=number_of_samples+1)
#        samples[0] = data_transformed[idx]
#
#        #samples = forest.lexicon.transform_indices(samples, root_id_pos=forest.root_id_pos)
#
#        probs = np.zeros(shape=number_of_samples + 1, dtype=DT_PROBS)
#        probs[samples == samples[0]] = 1
#
#        yield [idx], probs, samples


def batch_iter_reroot(forest_indices, number_of_samples):
    for idx in np.arange(len(forest_indices)):
        probs = np.zeros(shape=number_of_samples + 1, dtype=DT_PROBS)
        probs[0] = 1
        yield [idx], probs


def batch_iter_all(forest_indices, forest_indices_targets, batch_size):
    for i in range(len(forest_indices)):
        ix = np.isin(forest_indices, forest_indices_targets[i])
        probs = np.zeros(shape=len(ix), dtype=DT_PROBS)
        probs[ix] = 1
        for start in range(0, len(probs), batch_size):
            # do not yield, if it is not full (end of the dataset)
            if start+batch_size > len(probs):
                continue
            sampled_indices = np.arange(start - 1, start + batch_size)
            sampled_indices[0] = i
            current_probs = probs[start:start + batch_size]
            yield sampled_indices, current_probs, None


def batch_iter_multiclass(forest_indices, indices_targets, indices_forest_to_tree):
    indices = np.arange(len(forest_indices))
    np.random.shuffle(indices)
    for i in indices:
        yield [indices_forest_to_tree[forest_indices[i]]], indices_targets[i]


def do_epoch(supervisor, sess, model, epoch, forest_indices, indices_targets=None, dataset_trees=None,
             train=True, emit=True, test_step=0, test_writer=None, test_result_writer=None,
             highest_sims_model=None, number_of_samples=None, batch_iter='',
             dataset_iterator=None, return_values=True):

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
        execute_vars['train_op'] = model.train_op
        execute_vars['step'] = model.global_step
    else:
        feed_dict[model.tree_model.keep_prob] = 1.0
        assert test_writer is not None, 'training is disabled, but test_writer is not set'
        assert test_result_writer is not None, 'training is disabled, but test_result_writer is not set'

    tree_model_batch_size = 10
    indices_forest_to_tree = {idx: i for i, idx in enumerate(forest_indices)}
    iter_args = {batch_iter_naive: [number_of_samples, forest_indices, indices_targets, indices_forest_to_tree],
                 batch_iter_nearest: [number_of_samples, forest_indices, indices_targets, sess,
                                      model.tree_model, highest_sims_model, dataset_trees, tree_model_batch_size,
                                      indices_forest_to_tree],
                 batch_iter_all: [forest_indices, indices_targets, number_of_samples + 1],
                 batch_iter_reroot: [forest_indices, number_of_samples],
                 batch_iter_multiclass: [forest_indices, indices_targets, indices_forest_to_tree]}

    if batch_iter is not None and batch_iter.strip() != '':
        _iter = globals()[batch_iter]
    else:
        if isinstance(model, model_fold.SequenceTreeRerootModel):
            _iter = batch_iter_reroot
        else:
            if number_of_samples is None:
                _iter = batch_iter_all
            elif highest_sims_model is not None:
                _iter = batch_iter_nearest
            else:
                _iter = batch_iter_naive
    logger.debug('use %s' % _iter.__name__)
    _batch_iter = _iter(*iter_args[_iter])
    _result_all = []

    # TODO: do this in parallel with train execution
    if dataset_iterator is not None:
        logger.debug('re-generate trees with new samples...')
        with model.tree_model.compiler.multiprocessing_pool():
            dataset_trees = list(model.tree_model.compiler.build_loom_inputs(map(lambda x: [x], dataset_iterator()), ordered=True))
        logger.debug('re-generated %i trees' % len(dataset_trees))

    # for batch in td.group_by_batches(data_set, config.batch_size if train else len(test_set)):
    for batch in td.group_by_batches(_batch_iter, config.batch_size):
        tree_indices_batched, probs_batched = zip(*batch)
        if hasattr(model.tree_model, 'compiler') and hasattr(model.tree_model.compiler, 'loom_input_tensor'):
            trees_batched = [[dataset_trees[tree_idx] for tree_idx in tree_indices] for tree_indices in tree_indices_batched]
            feed_dict[model.tree_model.compiler.loom_input_tensor] = trees_batched
        else:
            tree_indices_batched_np = np.array(tree_indices_batched)
            trees_batched = dataset_trees[tree_indices_batched_np.flatten()]
            feed_dict[model.tree_model.embeddings_placeholder] = convert_sparse_matrix_to_sparse_tensor(trees_batched)

        # if values_gold expects a sparse tensor, convert probs_batched
        if isinstance(model.values_gold, tf.SparseTensor):
            probs_batched = convert_sparse_matrix_to_sparse_tensor(vstack(probs_batched))

        feed_dict[model.values_gold] = probs_batched
        _result_all.append(sess.run(execute_vars, feed_dict))

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


def get_parameter_count_from_shapes(shapes, selector_suffix='/Adadelta'):
    count = 0
    for tensor_name in shapes:
        if tensor_name.endswith(selector_suffix):
            count += reduce((lambda x, y: x * y), shapes[tensor_name])
    return count


#def get_dataset_size(index_files=None):
#    if index_files is None:
#        return 0
#    ds = sum([len(numpy_load(ind_file, assert_exists=True)) for ind_file in index_files])
#    logger.debug('dataset size: %i' % ds)
#    return ds

def get_lexicon(logdir, train_data_path=None, logdir_pretrained=None, logdir_continue=None, dont_dump=False,
                no_fixed_vecs=False, additional_vecs_path=None):
    checkpoint_fn = tf.train.latest_checkpoint(logdir)
    if logdir_continue:
        assert checkpoint_fn is not None, 'could not read checkpoint from logdir: %s' % logdir
    old_checkpoint_fn = None
    if checkpoint_fn is not None:
        if not checkpoint_fn.startswith(logdir):
            raise ValueError('entry in checkpoint file ("%s") is not located in logdir=%s' % (checkpoint_fn, logdir))
        logger.info('read lex_size from model ...')
        reader = tf.train.NewCheckpointReader(checkpoint_fn)
        saved_shapes = reader.get_variable_to_shape_map()
        logger.debug(saved_shapes)
        logger.debug('parameter count: %i' % get_parameter_count_from_shapes(saved_shapes))

        lexicon = Lexicon(filename=os.path.join(logdir, 'model'), load_ids_fixed=(not no_fixed_vecs))
        # assert len(lexicon) == saved_shapes[model_fold.VAR_NAME_LEXICON][0]
        #ROOT_idx = lexicon.get_d(vocab_manual[ROOT_EMBEDDING], data_as_hashes=False)
        #IDENTITY_idx = lexicon.get_d(vocab_manual[IDENTITY_EMBEDDING], data_as_hashes=False)
        try:
            lexicon.init_vecs(checkpoint_reader=reader)
        except AssertionError:
            logger.warning('no embedding vecs found in model')
            lexicon.init_vecs()
    else:
        assert train_data_path is not None, 'no checkpoint found and no train_data_path given'
        lexicon = Lexicon(filename=train_data_path, load_ids_fixed=(not no_fixed_vecs))

        # TODO: check this!
        if not no_fixed_vecs:
            lexicon.set_to_zero(indices=lexicon.ids_fixed, indices_as_blacklist=True)
            lexicon.add_flag(indices=lexicon.ids_fixed)

        # TODO: check this!
        if additional_vecs_path is not None and additional_vecs_path.strip() != '':
            logger.info('add embedding vecs from: %s' % additional_vecs_path)
            # ATTENTION: add_lex should contain only lower case entries, because self_to_lowercase=True
            add_lex = Lexicon(filename=additional_vecs_path)
            ids_added = set(lexicon.add_vecs_from_other(add_lex, self_to_lowercase=True))
            ids_added_not = [i for i in range(len(lexicon)) if i not in ids_added]
            # remove ids_added_not from lexicon.ids_fixed
            lexicon._ids_fixed = np.array([_id for _id in lexicon._ids_fixed if _id not in ids_added_not], dtype=lexicon.ids_fixed.dtype)

        #ROOT_idx = lexicon.get_d(vocab_manual[ROOT_EMBEDDING], data_as_hashes=False)
        #IDENTITY_idx = lexicon.get_d(vocab_manual[IDENTITY_EMBEDDING], data_as_hashes=False)
        if logdir_pretrained:
            logger.info('load lexicon from pre-trained model: %s' % logdir_pretrained)
            old_checkpoint_fn = tf.train.latest_checkpoint(logdir_pretrained)
            assert old_checkpoint_fn is not None, 'No checkpoint file found in logdir_pretrained: ' + logdir_pretrained
            reader_old = tf.train.NewCheckpointReader(old_checkpoint_fn)
            lexicon_old = Lexicon(filename=os.path.join(logdir_pretrained, 'model'))
            lexicon_old.init_vecs(checkpoint_reader=reader_old)
            lexicon.merge(lexicon_old, add=False, remove=False)

        # lexicon.replicate_types(suffix=constants.SEPARATOR + constants.vocab_manual[constants.BACK_EMBEDDING])
        # lexicon.pad()

        if not dont_dump:
            lexicon.dump(filename=os.path.join(logdir, 'model'), strings_only=True)
            assert lexicon.is_filled, 'lexicon: not all vecs for all types are set (len(types): %i, len(vecs): %i)' % \
                                      (len(lexicon), len(lexicon.vecs))



    logger.info('lexicon size: %i' % len(lexicon))
    #logger.debug('IDENTITY_idx: %i' % IDENTITY_idx)
    #logger.debug('ROOT_idx: %i' % ROOT_idx)
    return lexicon, checkpoint_fn, old_checkpoint_fn


def init_model_type(config):
    ## set index and tree getter
    if config.model_type == 'simtuple':
        raise NotImplementedError('model_type=%s is deprecated' % config.model_type)
        tree_iterator_args = {'root_idx': ROOT_idx, 'split': True, 'extensions': config.extensions.split(','),
                              'max_depth': config.max_depth, 'context': config.context, 'transform': True}
        tree_iterator = diters.data_tuple_iterator

        tree_count = 2  # [1.0, <sim_value>]   # [first_sim_entry, second_sim_entry]
        #discrete_model = False
        load_parents = False
    elif config.model_type == MT_TREETUPLE:
        tree_iterator_args = {'max_depth': config.max_depth, 'context': config.context, 'transform': True,
                              'concat_mode': config.concat_mode, 'link_cost_ref': config.link_cost_ref,
                              'bag_of_seealsos': False}
        if config.tree_embedder == 'tfidf':
            tree_iterator_args['concat_mode'] = CM_AGGREGATE
            tree_iterator_args['context'] = 0
        # if FLAGS.debug:
        #    root_strings_store = StringStore()
        #    root_strings_store.from_disk('%s.root.id.string' % config.train_data_path)
        #    tree_iterator_args['root_strings'] = [s for s in root_strings_store]

        tree_iterator = diters.tree_iterator
        indices_getter = diters.indices_dbpedianif
        # tuple_size = config.neg_samples + 1
        tree_count = 1
        #discrete_model = True
        load_parents = (tree_iterator_args['context'] > 0)
    # elif config.model_type == 'tuple_single':
    #    tree_iterator_args = {'max_depth': config.max_depth, 'context': config.context, 'transform': True,
    #                          'concat_mode': config.concat_mode, 'link_cost_ref': config.link_cost_ref,
    #                          'get_seealso_ids': True}
    #    #if FLAGS.debug:
    #    #    root_strings_store = StringStore()
    #    #    root_strings_store.from_disk('%s.root.id.string' % config.train_data_path)
    #    #    data_iterator_args['root_strings'] = [s for s in root_strings_store]
    #
    #    tree_iterator = diters.tree_iterator
    #    indices_getter = diters.indices_dbpedianif
    #    tuple_size = config.neg_samples + 1
    #
    #    discrete_model = True
    #    load_parents = (config.context is not None and config.context > 0)
    elif config.model_type == MT_REROOT:
        # if config.cut_indices is not None:
        #    indices = np.arange(config.cut_indices)
        #    # avoid looking for train index files
        #    meta[M_TRAIN][M_FNAMES] = []
        # else:
        #    indices = None
        config.batch_iter = batch_iter_reroot.__name__
        logger.debug('set batch_iter to %s' % config.batch_iter)
        config.batch_iter_test = config.batch_iter
        logger.debug('set batch_iter_test to %s' % config.batch_iter_test)
        neg_samples = config.neg_samples
        config.neg_samples_test = config.neg_samples
        logger.debug('set neg_samples_test to %i (neg_samples)' % config.neg_samples_test)
        tree_count = neg_samples + 1
        tree_iterator_args = {'neg_samples': neg_samples, 'max_depth': config.max_depth, 'concat_mode': CM_TREE,
                              'transform': True, 'link_cost_ref': config.link_cost_ref, 'link_cost_ref_seealso': -1}
        # tree_iterator = diters.data_tuple_iterator_reroot
        tree_iterator = diters.tree_iterator

        def _get_indices(index_files, forest, **unused):
            # TODO: remove count
            #indices = np.fromiter(diters.index_iterator(index_files), count=1000, dtype=np.int32)
            # TODO: remove dummy indices
            #indices = np.arange(1000, dtype=np.int32)
            number_of_indices = config.cut_indices or 1000
            logger.info('use %i fixed indices per epoch (forest size: %i)' % (number_of_indices, len(forest)))
            indices = np.random.randint(len(forest), size=number_of_indices)
            return indices, None

        #indices_getter = diters.indices_as_ids
        indices_getter = _get_indices
        # del meta[M_TEST]
        #discrete_model = True
        load_parents = True
    # elif config.model_type == 'tfidf':
    #    tree_iterator_args = {'max_depth': config.max_depth, 'context': config.context, 'transform': True,
    #                          'concat_mode': CM_AGGREGATE}
    #    tree_iterator = diters.tree_iterator
    #    indices_getter = diters.indices_dbpedianif
    #    # tuple_size = config.neg_samples + 1
    #    tuple_size = 1
    #    discrete_model = True
    #    load_parents = False
    elif config.model_type == MT_MULTICLASS:
        classes_ids = numpy_load(filename='%s.%s' % (config.train_data_path, FE_CLASS_IDS))
        logger.info('number of classes to predict: %i' % len(classes_ids))

        tree_iterator_args = {'max_depth': config.max_depth, 'context': config.context, 'transform': True,
                              'concat_mode': config.concat_mode, 'link_cost_ref': -1}
        if config.tree_embedder == 'tfidf':
            tree_iterator_args['concat_mode'] = CM_AGGREGATE
            tree_iterator_args['context'] = 0
        tree_iterator = diters.tree_iterator
        indices_getter = partial(diters.indices_bioasq, classes_ids=classes_ids)
        tree_count = 1
        load_parents = (tree_iterator_args['context'] > 0)

        config.batch_iter = batch_iter_multiclass.__name__
        config.batch_iter_test = batch_iter_multiclass.__name__
    else:
        raise NotImplementedError('model_type=%s not implemented' % config.model_type)

    return tree_iterator, tree_iterator_args, indices_getter, load_parents, tree_count


def get_index_file_names(config, parent_dir, test_files=None, test_only=None):

    fnames_train = None
    fnames_test = None
    if FLAGS.train_files is not None and FLAGS.train_files != '':
        logger.info('use train data index files: %s' % FLAGS.train_files)
        fnames_train = [os.path.join(parent_dir, fn) for fn in FLAGS.train_files.split(',')]
    if test_files is not None and test_files != '':
        fnames_test = [os.path.join(parent_dir, fn) for fn in FLAGS.test_files.split(',')]
    if not test_only:
        #if M_FNAMES not in meta[M_TRAIN]:
        if fnames_train is None:
            logger.info('collect train data from: ' + config.train_data_path + ' ...')
            regex = re.compile(r'%s\.idx\.\d+\.npy$' % ntpath.basename(config.train_data_path))
            _train_fnames = filter(regex.search, os.listdir(parent_dir))
            fnames_train = [os.path.join(parent_dir, fn) for fn in sorted(_train_fnames)]
        assert len(fnames_train) > 0, 'no matching train data files found for ' + config.train_data_path
        logger.info('found ' + str(len(fnames_train)) + ' train data files')
        #if M_TEST in meta and M_FNAMES not in meta[M_TEST]:
        if fnames_test is None:
            fnames_test = [fnames_train[config.dev_file_index]]
            logger.info('use %s for testing' % str(fnames_test))
            del fnames_train[config.dev_file_index]
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


def compile_trees(tree_iterators, compiler):
    prepared_embeddings = {}
    for m in tree_iterators:
        logger.info('create %s data set (tree-embeddings) ...' % m)
        with compiler.multiprocessing_pool():
            prepared_embeddings[m] = list(
                compiler.build_loom_inputs(map(lambda x: [x], tree_iterators[m]()), ordered=True))
            logger.info('%s dataset: compiled %i different trees' % (m, len(prepared_embeddings[m])))

    return prepared_embeddings


def prepare_embeddings_tfidf(tree_iterators, logdir):
    prepared_embeddings = {}

    # check, if tfidf data exists
    tfidf_exist = []
    for i, m in enumerate([M_TRAIN, M_TEST]):
        fn_tfidf_data = os.path.join(logdir, 'embeddings_tfidf.%s.npz' % m)
        if os.path.exists(fn_tfidf_data):
            tfidf_exist.append(m)
    # if tfidf data exists, load tfidf_data and indices
    if len(tfidf_exist) > 0:
        embedding_dim = -1
        prepared_embeddings = {}
        tree_indices = {}
        indices_nbr = 0
        for m in tfidf_exist:
            fn_tfidf_data = os.path.join(logdir, 'embeddings_tfidf.%s.npz' % m)
            fn_tree_indices = os.path.join(logdir, '%s.%s.npy' % (FN_TREE_INDICES, m))
            if os.path.exists(fn_tfidf_data):
                assert os.path.exists(fn_tree_indices), \
                    'found tfidf data (%s), but no related indices file (%s)' % (fn_tfidf_data, fn_tree_indices)
                prepared_embeddings[m] = scipy.sparse.load_npz(fn_tfidf_data)
                tree_indices[m] = numpy_load(fn_tree_indices)
                logging.info('%s dataset: use %i different tree embeddings' % (m, prepared_embeddings[m].shape[0]))
                current_embedding_dim = prepared_embeddings[m].shape[1]
                assert embedding_dim == -1 or embedding_dim == current_embedding_dim, 'current embedding_dim: %i does not match previous one: %i' % (
                    current_embedding_dim, embedding_dim)
                embedding_dim = current_embedding_dim
                indices_nbr += len(tree_indices[m])
        assert embedding_dim != -1, 'no data sets created'
        logging.debug('number of tfidf_indices: %i with dimension: %i' % (indices_nbr, embedding_dim))
    else:
        logger.info('create %s data sets (tf-idf) ...' % ', '.join(tree_iterators.keys()))
        assert logdir is not None, 'no logdir provided to dump tfidf embeddings'
        _tree_embeddings_tfidf = diters.embeddings_tfidf([tree_iterators[m]() for m in tree_iterators.keys()])
        embedding_dim = -1
        for i, m in enumerate(tree_iterators.keys()):
            prepared_embeddings[m] = _tree_embeddings_tfidf[i]
            scipy.sparse.save_npz(file=os.path.join(logdir, 'embeddings_tfidf.%s.npz' % m),
                                  matrix=prepared_embeddings[m])
            logger.info('%s dataset: use %i different trees' % (m, prepared_embeddings[m].shape[0]))
            current_embedding_dim = prepared_embeddings[m].shape[1]
            assert embedding_dim == -1 or embedding_dim == current_embedding_dim, \
                'current embedding_dim: %i does not match previous one: %i' % (current_embedding_dim, embedding_dim)
            embedding_dim = current_embedding_dim
        assert embedding_dim != -1, 'no data sets created'

    return prepared_embeddings, embedding_dim


def create_models(config, lexicon, tree_count, tree_iterators, tree_indices, logdir=None, use_inception_tree_model=False):

    #prepared_embeddings = {}
    optimizer = config.optimizer
    if optimizer is not None:
        optimizer = getattr(tf.train, optimizer)

    if not config.tree_embedder == 'tfidf':
        tree_embedder = getattr(model_fold, config.tree_embedder)
        kwargs = {}
        if issubclass(tree_embedder, model_fold.TreeEmbedding_FLATconcat):
            # TODO: this value should depend on max_size_plain, see data_iterators.tree_iterator
            kwargs['sequence_length'] = 100
            _padding_idx = lexicon.get_d(vocab_manual[UNIQUE_EMBEDDING], data_as_hashes=False)
            kwargs['padding_id'] = lexicon.transform_idx(_padding_idx)

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
                                                  tree_count=tree_count,
                                                  # data_transfomed=data_transformed
                                                  # tree_count=1,
                                                  # keep_prob_fixed=config.keep_prob # to enable full head dropout
                                                  **kwargs
                                                  )
        #if config.model_type != MT_REROOT:
        prepared_embeddings = compile_trees(tree_iterators=tree_iterators, compiler=model_tree.compiler)
        #else:
        #    prepared_embeddings = None
    else:
        prepared_embeddings, embedding_dim = prepare_embeddings_tfidf(tree_iterators=tree_iterators, logdir=logdir)

        model_tree = model_fold.DummyTreeModel(embeddings_dim=embedding_dim, tree_count=tree_count,
                                               keep_prob=config.keep_prob, sparse=True,
                                               root_fc_sizes=[int(s) for s in
                                                              ('0' + config.root_fc_sizes).split(',')], )

    if use_inception_tree_model:
        inception_tree_model = model_fold.DummyTreeModel(embeddings_dim=model_tree.tree_output_size, sparse=False,
                                                         tree_count=tree_count, keep_prob=config.keep_prob, root_fc_sizes=0)
    else:
        inception_tree_model = model_tree

    # if config.model_type == 'simtuple':
    #    model = model_fold.SimilaritySequenceTreeTupleModel(tree_model=model_tree,
    #                                                        optimizer=optimizer,
    #                                                        learning_rate=config.learning_rate,
    #                                                        sim_measure=sim_measure,
    #                                                        clipping_threshold=config.clipping)
    if config.model_type == MT_TREETUPLE:
        model = model_fold.TreeTupleModel_with_candidates(tree_model=inception_tree_model,
                                                          fc_sizes=[int(s) for s in ('0' + config.fc_sizes).split(',')],
                                                          optimizer=optimizer,
                                                          learning_rate=config.learning_rate,
                                                          clipping_threshold=config.clipping,
                                                          )

    elif config.model_type == MT_REROOT:
        model = model_fold.TreeSingleModel_with_candidates(tree_model=inception_tree_model,
                                                           fc_sizes=[int(s) for s in ('0' + config.fc_sizes).split(',')],
                                                           optimizer=optimizer,
                                                           learning_rate=config.learning_rate,
                                                           clipping_threshold=config.clipping,
                                                           )
    elif config.model_type == MT_MULTICLASS:
        classes_ids = numpy_load(filename='%s.%s' % (config.train_data_path, FE_CLASS_IDS))
        model = model_fold.TreeMultiClassModel(tree_model=inception_tree_model,
                                               fc_sizes=[int(s) for s in ('0' + config.fc_sizes).split(',')],
                                               optimizer=optimizer,
                                               learning_rate=config.learning_rate,
                                               clipping_threshold=config.clipping,
                                               num_classes=len(classes_ids)
                                               )
    else:
        raise NotImplementedError('model_type=%s not implemented' % config.model_type)

    return model_tree, model, prepared_embeddings, tree_indices


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


def execute_run(config, logdir_continue=None, logdir_pretrained=None, test_file=None, init_only=None, test_only=None, cache=None):
    config.set_run_description()

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
    lexicon, checkpoint_fn, old_checkpoint_fn = get_lexicon(logdir=logdir,
                                                            train_data_path=config.train_data_path,
                                                            logdir_pretrained=logdir_pretrained,
                                                            logdir_continue=logdir_continue,
                                                            no_fixed_vecs=config.no_fixed_vecs,
                                                            additional_vecs_path=config.additional_vecs)
    loaded_from_checkpoint = checkpoint_fn is not None
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
    fnames_train, fnames_test = get_index_file_names(config=config, parent_dir=parent_dir, test_files=test_file,
                                                     test_only=test_only)
    if not (test_only or init_only):
        meta[M_TRAIN] = {M_FNAMES: fnames_train}
    meta[M_TEST] = {M_FNAMES: fnames_test}

    tree_iterator, tree_iterator_args, indices_getter, load_parents, tree_count = init_model_type(config)

    # load forest data
    lexicon_root_fn = '%s.root.id' % config.train_data_path
    if Lexicon.exist(lexicon_root_fn, types_only=True):
        logging.info('load lexicon_roots from %s' % lexicon_root_fn)
        lexicon_roots = Lexicon(filename=lexicon_root_fn, load_vecs=False)
    else:
        lexicon_roots = None
    forest = Forest(filename=config.train_data_path, lexicon=lexicon, load_parents=load_parents, lexicon_roots=lexicon_roots)
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
        meta[m][M_INDICES], meta[m][M_INDICES_TARGETS] = indices_getter(index_files=meta[m][M_FNAMES], forest=forest)
        # dump tree indices
        if not loaded_from_checkpoint:
            mytools.numpy_dump(os.path.join(logdir, '%s.%s' % (FN_TREE_INDICES, m)), meta[m][M_INDICES])

    if config.model_type == MT_TREETUPLE:
        if M_TEST in meta and M_TRAIN in meta \
                and meta[M_TRAIN][M_INDICES_TARGETS] is not None and meta[M_TEST][M_INDICES_TARGETS] is not None:
            meta[M_TEST][M_INDICES_TARGETS] = check_train_test_overlap(forest_indices_train=meta[M_TRAIN][M_INDICES],
                                                                       forest_indices_train_target=meta[M_TRAIN][M_INDICES_TARGETS],
                                                                       forest_indices_test=meta[M_TEST][M_INDICES],
                                                                       forest_indices_test_target=meta[M_TEST][M_INDICES_TARGETS])

    # set batch iterators and numbers of negative samples
    if M_TEST in meta:
        if config.batch_iter_test is '':
            meta[M_TEST][M_BATCH_ITER] = config.batch_iter
        else:
            meta[M_TEST][M_BATCH_ITER] = config.batch_iter_test
        if config.neg_samples_test is '':
            meta[M_TEST][M_NEG_SAMPLES] = config.neg_samples
        else:
            meta[M_TEST][M_NEG_SAMPLES] = int(config.neg_samples_test)
    if M_TRAIN in meta:
        meta[M_TRAIN][M_BATCH_ITER] = config.batch_iter
        meta[M_TRAIN][M_NEG_SAMPLES] = config.neg_samples

    # set tree iterator
    for m in meta:
        meta[m][M_TREE_ITER] = partial(tree_iterator, indices=meta[m][M_INDICES], forest=forest, **tree_iterator_args)
        if config.model_type == MT_REROOT:
            meta[m][M_TREE_ITER] = partial(diters.reroot_wrapper, trees=list(meta[m][M_TREE_ITER]()),
                                           neg_samples=meta[m][M_NEG_SAMPLES], forest=forest, transform=True)

    # MODEL DEFINITION #################################################################################################

    current_device = get_ith_best_device(0)
    logger.info('create tensorflow graph on device: %s ...' % str(current_device))
    with tf.device(current_device):
        with tf.Graph().as_default() as graph:
            #with tf.device(tf.train.replica_device_setter(FLAGS.ps_tasks)):
            logger.debug('trainable lexicon entries: %i' % lexicon.len_var)
            logger.debug('fixed lexicon entries:     %i' % lexicon.len_fixed)

            model_tree, model, prepared_embeddings, tree_indices = create_models(
                config=config, lexicon=lexicon,  tree_count=tree_count, logdir=logdir,
                tree_iterators={m: meta[m][M_TREE_ITER] for m in meta},
                tree_indices={m: meta[m][M_INDICES] for m in meta})

            models_nearest = create_models_nearest(model_tree=model_tree,
                                                   prepared_embeddings={m: prepared_embeddings[m] for m in meta.keys()
                                                                        if M_BATCH_ITER in meta[m]
                                                                        and meta[m][M_BATCH_ITER].strip() == batch_iter_nearest.__name__})

            # set model(s) and prepared embeddings
            for m in meta:
                meta[m][M_MODEL] = model
                if m in models_nearest:
                    meta[m][M_MODEL_NEAREST] = models_nearest[m]
                if prepared_embeddings is not None:
                    meta[m][M_TREES] = prepared_embeddings[m]
                else:
                    meta[m][M_TREES] = None
                meta[m][M_INDICES] = tree_indices[m]


            # PREPARE TRAINING #########################################################################################

            if old_checkpoint_fn is not None:
                logger.info(
                    'restore from old_checkpoint (except lexicon, step and optimizer vars): %s ...' % old_checkpoint_fn)
                lexicon_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=model_fold.VAR_NAME_LEXICON_VAR) \
                               + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=model_fold.VAR_NAME_LEXICON_FIX)
                #optimizer_vars = model_train.optimizer_vars() + [model_train.global_step] \
                #                 + ((model_test.optimizer_vars() + [
                #    model_test.global_step]) if model_test is not None and model_test != model_train else [])
                optimizer_vars = meta[M_TRAIN][M_MODEL].optimizer_vars() + [meta[M_TRAIN][M_MODEL].global_step] \
                                 + ((meta[M_TEST][M_MODEL].optimizer_vars() + [
                    meta[M_TEST][M_MODEL].global_step]) if M_TEST in meta and meta[M_TEST][M_MODEL] != meta[M_TRAIN][M_MODEL] else [])

                restore_vars = [item for item in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if
                                item not in lexicon_vars + optimizer_vars]
                pre_train_saver = tf.train.Saver(restore_vars)
            else:
                pre_train_saver = None

            def load_pretrain(sess):
                pre_train_saver.restore(sess, old_checkpoint_fn)

            # Set up the supervisor.
            supervisor = tf.train.Supervisor(
                # saver=None,# my_saver,
                logdir=logdir,
                #is_chief=(FLAGS.task == 0),
                save_summaries_secs=10,
                save_model_secs=0,
                summary_writer=tf.summary.FileWriter(os.path.join(logdir, 'train'), graph),
                init_fn=load_pretrain if pre_train_saver is not None else None
            )
            #if dev_iterator is not None or test_iterator is not None:
            if M_TEST in meta:
                test_writer = tf.summary.FileWriter(os.path.join(logdir, 'test'), graph)
            #sess = supervisor.PrepareSession(FLAGS.master)
            sess = supervisor.PrepareSession('')
            # TODO: try
            #sess = supervisor.PrepareSession(FLAGS.master, config=tf.ConfigProto(log_device_placement=True))

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
                logger.removeHandler(fh_info)
                logger.removeHandler(fh_debug)
                return

            # TRAINING #################################################################################################

            # do initial test epoch
            if M_TEST in meta:
                if not loaded_from_checkpoint or M_TRAIN not in meta:
                    _, _, values_all, values_all_gold, stats_dict = do_epoch(
                        supervisor,
                        sess=sess,
                        model=meta[M_TEST][M_MODEL],
                        dataset_trees=meta[M_TEST][M_TREES],# if M_TREES in meta[M_TEST] else None,
                        forest_indices=meta[M_TEST][M_INDICES],
                        indices_targets=meta[M_TEST][M_INDICES_TARGETS],
                        #dataset_trees_embedded=meta[M_TEST][M_TREE_EMBEDDINGS] if M_TREE_EMBEDDINGS in meta[M_TEST] else None,
                        epoch=0,
                        train=False,
                        emit=True,
                        test_writer=test_writer,
                        test_result_writer=test_result_writer,
                        number_of_samples=meta[M_TEST][M_NEG_SAMPLES],
                        #number_of_samples=None,
                        highest_sims_model=meta[M_TEST][M_MODEL_NEAREST] if M_MODEL_NEAREST in meta[M_TEST] else None,
                        batch_iter=meta[M_TEST][M_BATCH_ITER])
                if M_TRAIN not in meta:
                    if values_all is None or values_all_gold is None:
                        logger.warning('Predicted and gold values are None. Passed return_values=False?')
                    else:
                        values_all.dump(os.path.join(logdir, 'sims.np'))
                        values_all_gold.dump(os.path.join(logdir, 'sims_gold.np'))
                    logger.removeHandler(fh_info)
                    logger.removeHandler(fh_debug)
                    lexicon.dump(filename=os.path.join(logdir, 'model'))
                    return stats_dict


            # clear vecs in lexicon to clean up memory
            lexicon.init_vecs()

            logger.info('training the model')
            loss_test_best = 9999
            stat_queue = []
            if M_TEST in meta:
                if meta[M_TEST][M_MODEL].model_type == MODEL_TYPE_DISCRETE:
                    stat_key = METRIC_DISCRETE
                elif meta[M_TEST][M_MODEL].model_type == MODEL_TYPE_REGRESSION:
                    stat_key = METRIC_REGRESSION
                else:
                    raise ValueError('stat_key not defined for model_type=%s' % meta[M_TEST][M_MODEL].model_type)
                # NOTE: this depends on stat_key (pearson/mse/roc/...)
                TEST_MIN_INIT = -1
                stat_queue = [{stat_key: TEST_MIN_INIT}]
            step_train = sess.run(meta[M_TRAIN][M_MODEL].global_step)
            max_queue_length = 0
            for epoch, shuffled in enumerate(td.epochs(items=range(len(meta[M_TRAIN][M_INDICES])), n=config.epochs, shuffle=True), 1):

                # train
                if not config.early_stopping_window or len(stat_queue) > 0:
                    step_train, loss_train, _, _, stats_train = do_epoch(
                        supervisor, sess,
                        model=meta[M_TRAIN][M_MODEL],
                        dataset_trees=meta[M_TRAIN][M_TREES],# if M_TREES in meta[M_TRAIN] else None,
                        #dataset_trees_embedded=meta[M_TRAIN][M_TREE_EMBEDDINGS] if M_TREE_EMBEDDINGS in meta[M_TRAIN] else None,
                        forest_indices=meta[M_TRAIN][M_INDICES],
                        indices_targets=meta[M_TRAIN][M_INDICES_TARGETS],
                        epoch=epoch,
                        number_of_samples=meta[M_TRAIN][M_NEG_SAMPLES],
                        highest_sims_model=meta[M_TRAIN][M_MODEL_NEAREST] if M_MODEL_NEAREST in meta[M_TRAIN] else None,
                        batch_iter=meta[M_TRAIN][M_BATCH_ITER],
                        dataset_iterator=meta[M_TRAIN][M_TREE_ITER] if config.model_type == MT_REROOT else None,
                        return_values=False
                    )

                if M_TEST in meta:

                    # test
                    step_test, loss_test, _, _, stats_test = do_epoch(
                        supervisor, sess,
                        model=meta[M_TEST][M_MODEL],
                        dataset_trees=meta[M_TEST][M_TREES],# if M_TREES in meta[M_TEST] else None,
                        # #dataset_trees_embedded=meta[M_TEST][M_TREE_EMBEDDINGS] if M_TREE_EMBEDDINGS in meta[M_TEST] else None,
                        forest_indices=meta[M_TEST][M_INDICES],
                        indices_targets=meta[M_TEST][M_INDICES_TARGETS],
                        number_of_samples=meta[M_TEST][M_NEG_SAMPLES],
                        #number_of_samples=None,
                        epoch=epoch,
                        train=False,
                        test_step=step_train,
                        test_writer=test_writer,
                        test_result_writer=test_result_writer,
                        highest_sims_model=meta[M_TEST][M_MODEL_NEAREST] if M_MODEL_NEAREST in meta[M_TEST] else None,
                        batch_iter=meta[M_TEST][M_BATCH_ITER],
                        return_values=False
                    )

                    if loss_test < loss_test_best:
                        loss_test_best = loss_test

                    # EARLY STOPPING ###############################################################################

                    stat = round(stats_test[stat_key], 6)

                    prev_max = max(stat_queue, key=lambda t: t[stat_key])[stat_key]
                    # stop, if current test pearson r is not bigger than previous values. The amount of regarded
                    # previous values is set by config.early_stopping_window
                    if stat > prev_max:
                        stat_queue = []
                    else:
                        if len(stat_queue) >= max_queue_length:
                            max_queue_length = len(stat_queue) + 1
                    stat_queue.append(stats_test)
                    stat_queue_sorted = sorted(stat_queue, reverse=True, key=lambda t: t[stat_key])
                    rank = stat_queue_sorted.index(stats_test)

                    # write out queue length
                    emit_values(supervisor, sess, step_test, values={'queue_length': len(stat_queue), 'rank': rank},
                                writer=test_writer)

                    logger.info(
                        '%s rank (of %i):\t%i\tdif: %f\tmax_queue_length: %i'
                        % (stat_key, len(stat_queue), rank, (stat - prev_max), max_queue_length))
                    if 0 < config.early_stopping_window < len(stat_queue):
                        logger.info('last metrics (last rank: %i): %s' % (rank, str(stat_queue)))
                        logger.removeHandler(fh_info)
                        logger.removeHandler(fh_debug)
                        return stat_queue_sorted[0]

                    # do not save, if score was not the best
                    # if rank > len(stat_queue) * 0.05:
                    if len(stat_queue) > 1 and config.early_stopping_window:
                        # auto restore if enabled
                        #if config.auto_restore:
                        #    supervisor.saver.restore(sess, tf.train.latest_checkpoint(logdir))
                        pass
                    else:
                        # don't save after first epoch if config.early_stopping_window > 0
                        if prev_max > TEST_MIN_INIT or not config.early_stopping_window:
                            supervisor.saver.save(sess, checkpoint_path(logdir, step_train))
                else:
                    # save model after each step if not dev model is set (training a language model)
                    supervisor.saver.save(sess, checkpoint_path(logdir, step_train))

            logger.removeHandler(fh_info)
            logger.removeHandler(fh_debug)


def add_metrics(d, stats, prefix=''):
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
    mytools.logging_init()
    logger.debug('test')

    # Handle multiple logdir_continue's
    # ATTENTION: discards any FLAGS (e.g. provided as argument) contained in default_config!
    if FLAGS.logdir_continue is not None and ',' in FLAGS.logdir_continue:
        logdirs = FLAGS.logdir_continue.split(',')
        logger.info('execute %i runs ...' % len(logdirs))
        stats_prefix = 'score_'
        with open(os.path.join(FLAGS.logdir, 'scores_new.tsv'), 'w') as csvfile:
            fieldnames = Config(logdir_continue=logdirs[0]).as_dict().keys() \
                         + [stats_prefix + k for k in METRIC_KEYS_DISCRETE + METRIC_KEYS_REGRESSION]
            score_writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter='\t', extrasaction='ignore')
            score_writer.writeheader()
            for i, logdir in enumerate(logdirs, 1):
                logger.info('START RUN %i of %i' % (i, len(logdirs)))
                config = Config(logdir_continue=logdir)
                config_dict = config.as_dict()
                stats = execute_run(config, logdir_continue=logdir, logdir_pretrained=FLAGS.logdir_pretrained,
                                    test_file=FLAGS.test_file, init_only=FLAGS.init_only, test_only=FLAGS.test_only)

                add_metrics(config_dict, stats, prefix=stats_prefix)
                score_writer.writerow(config_dict)
                csvfile.flush()
    else:
        config = Config(logdir_continue=FLAGS.logdir_continue, logdir_pretrained=FLAGS.logdir_pretrained)

        # get default config (or load from logdir_continue/logdir_pretrained)
        config.init_flags()
        # pylint: disable=protected-access
        FLAGS._parse_flags()
        # pylint: enable=protected-access
        config.update_with_flags(FLAGS)
        if FLAGS.grid_config_file is not None:

            parameters_fn = os.path.join(FLAGS.logdir, FLAGS.grid_config_file)
            logger.info('load grid parameters from: %s' % parameters_fn)
            with open(parameters_fn, 'r') as infile:
                grid_parameters = json.load(infile)

            scores_fn = os.path.join(FLAGS.logdir, 'scores.tsv')
            fieldnames_loaded = None
            if os.path.isfile(scores_fn):
                file_mode = 'a'
                with open(scores_fn, 'r') as csvfile:
                    scores_done_reader = csv.DictReader(csvfile, delimiter='\t')
                    fieldnames_loaded = scores_done_reader.fieldnames
                    scores_done = list(scores_done_reader)
                run_descriptions_done = [s_d['run_description'] for s_d in scores_done]
                logger.debug('already finished: %s' % ', '.join(run_descriptions_done))
            else:
                file_mode = 'w'
                run_descriptions_done = []
            logger.info('write scores to: %s' % scores_fn)

            stats_prefix_dev = 'dev_best_'
            stats_prefix_test = 'test_'
            fieldnames_expected = grid_parameters.keys() + [stats_prefix_dev + k for k in METRIC_KEYS_DISCRETE + METRIC_KEYS_REGRESSION] \
                                  + [stats_prefix_test + k for k in METRIC_KEYS_DISCRETE + METRIC_KEYS_REGRESSION] + ['run_description']
            assert fieldnames_loaded is None or set(fieldnames_loaded) == set(fieldnames_expected), 'field names in tsv file are not as expected'
            fieldnames = fieldnames_loaded or fieldnames_expected
            with open(scores_fn, file_mode) as csvfile:
                score_writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter='\t', extrasaction='ignore')
                if file_mode == 'w':
                    score_writer.writeheader()
                    csvfile.flush()

                for i in range(FLAGS.run_count):
                    for c, d in config.explode(grid_parameters):
                        c.set_run_description()
                        run_desc_backup = c.run_description

                        logger.info(
                            'start run ==============================================================================')
                        c.run_description = os.path.join(run_desc_backup, str(i))
                        logdir = os.path.join(FLAGS.logdir, c.run_description)

                        # check, if the test file exists, before executing the run
                        train_data_dir = os.path.abspath(os.path.join(c.train_data_path, os.pardir))
                        if FLAGS.test_file is not None and FLAGS.test_file.trim() != '':
                            test_fname = os.path.join(train_data_dir, FLAGS.test_file)
                            assert os.path.isfile(test_fname), 'could not find test file: %s' % test_fname
                        else:
                            test_fname = None

                        # skip already processed
                        if os.path.isdir(logdir) and c.run_description in run_descriptions_done:
                            logger.debug('skip config for logdir: %s' % logdir)
                            c.run_description = run_desc_backup
                            continue

                        # train
                        metrics_dev = execute_run(c)
                        main_metric = add_metrics(d, metrics_dev, prefix=stats_prefix_dev)
                        logger.info('best dev score (%s): %f' % (main_metric, metrics_dev[main_metric]))

                        # test
                        if test_fname is not None:
                            metrics_test = execute_run(c, logdir_continue=logdir, test_only=True, test_file=FLAGS.test_file)
                            main_metric = add_metrics(d, metrics_test, prefix=stats_prefix_test)
                            logger.info('test score (%s): %f' % (main_metric, metrics_test[main_metric]))
                        d['run_description'] = c.run_description

                        c.run_description = run_desc_backup
                        score_writer.writerow(d)
                        csvfile.flush()

        # default: execute single run
        else:
            execute_run(config, logdir_continue=FLAGS.logdir_continue, logdir_pretrained=FLAGS.logdir_pretrained,
                        test_file=FLAGS.test_file, init_only=FLAGS.init_only, test_only=FLAGS.test_only)
