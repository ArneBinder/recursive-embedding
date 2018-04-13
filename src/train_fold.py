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
from functools import reduce, partial

import numpy as np
import six
import tensorflow as tf
import tensorflow_fold as td
from scipy.stats.mstats import spearmanr
from scipy.stats.stats import pearsonr
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing as pp
from spacy.strings import StringStore

import lexicon as lex
import model_fold
from model_fold import MODEL_TYPE_DISCRETE, MODEL_TYPE_REGRESSION

# model flags (saved in flags.json)
import mytools
from mytools import numpy_load
from sequence_trees import Forest
from constants import vocab_manual, KEY_HEAD, KEY_CHILDREN, ROOT_EMBEDDING, IDENTITY_EMBEDDING, DTYPE_OFFSET, TYPE_REF, \
    TYPE_REF_SEEALSO, UNKNOWN_EMBEDDING, TYPE_SECTION_SEEALSO, LOGGING_FORMAT
from config import Config
#from data_iterators import data_tuple_iterator_reroot, data_tuple_iterator_dbpedianif, data_tuple_iterator, \
#    indices_dbpedianif
import data_iterators as diters

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
tf.flags.DEFINE_string('master', '',
                       'Tensorflow master to use.')
tf.flags.DEFINE_integer('task', 0,
                        'Task ID of the replica running the training.')
tf.flags.DEFINE_integer('ps_tasks', 0,
                        'Number of PS tasks in the job.')
FLAGS = tf.flags.FLAGS

# NOTE: the first entry (of both lists) defines the value used for early stopping and other statistics
STAT_KEYS_DISCRETE = ['roc']
STAT_KEYS_REGRESSION = ['pearson_r', 'mse']

logger = logging.getLogger('')
logger.setLevel(logging.DEBUG)
logger_streamhandler = logging.StreamHandler()
logger_streamhandler.setLevel(logging.INFO)
logger_streamhandler.setFormatter(logging.Formatter(LOGGING_FORMAT))
#logger.addHandler(logger_streamhandler)

M_INDICES = 'indices'
M_TEST = 'test'
M_TRAIN = 'train'
M_MODEL = 'model'
M_FNAMES = 'fnames'
M_TREES = 'trees'
M_TREE_EMBEDDINGS = 'tree_embeddings'
#M_TARGETS = 'targets'
M_DATA = 'data'
M_IDS = 'ids'
M_TREE_ITER = 'tree_iterator'
M_IDS_TARGET = 'ids_target'


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


def collect_stats(supervisor, sess, epoch, step, loss, values, values_gold, model_type, print_out=True, emit=True,
                  test_writer=None, test_result_writer=None):

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
    if model_type == MODEL_TYPE_REGRESSION:
        p_r = pearsonr(values, values_gold)
        s_r = spearmanr(values, values_gold)
        mse = np.mean(np.square(values - values_gold))
        emit_dict.update({
            'pearson_r': p_r[0],
            'spearman_r': s_r[0],
            'mse': mse
        })
        info_string = 'epoch=%d step=%d: loss_%s=%f\tpearson_r_%s=%f\tavg=%f\tvar=%f\tgold_avg=%f\tgold_var=%f' \
                      % (epoch, step, suffix, loss, suffix, p_r[0], np.average(values), np.var(values),
                         np.average(values_gold), np.var(values_gold))
    elif model_type == MODEL_TYPE_DISCRETE:
        roc = roc_auc_score(values_gold, values)
        emit_dict.update({
            'roc': roc
        })
        info_string = 'epoch=%d step=%d: loss_%s=%f\troc=%f' % (epoch, step, suffix, loss, roc)
    else:
        raise ValueError('unknown model type: %s. Use %s or %s.' % (model_type, MODEL_TYPE_DISCRETE,
                                                                    MODEL_TYPE_REGRESSION))
    if emit:
        emit_values(supervisor, sess, step, emit_dict, writer=writer, csv_writer=csv_writer)
    if print_out:
        logger.info(info_string)
    return emit_dict


def do_epoch(supervisor, sess, model, epoch, dataset_ids,
             dataset_target_ids=None, dataset_trees=None,
             dataset_trees_embedded=None, train=True, emit=True, test_step=0, test_writer=None, test_result_writer=None,
             highest_sims_model=None, number_of_samples=0):  # , discrete_model=False):

    id_to_idx = {_id: i for i, _id in enumerate(dataset_ids)}

    #if dataset_indices is None:
    dataset_indices = np.arange(len(dataset_ids))

    np.random.shuffle(dataset_indices)

    def id_iter():
        if dataset_target_ids is None:
            for idx in dataset_indices:
                _id = dataset_ids[idx]
                yield (idx, _id, _id)
        else:
            for idx in dataset_indices:
                for target_id in dataset_target_ids[idx]:
                    yield (idx, dataset_ids[idx], target_id)

    def batch_iter_naive():
        for idx, _id, _target_id in id_iter():
            # sample from [1, len(dataset_indices)-1] (inclusive); size +1 to hold the correct target and +1 to hold
            # the origin/reference (idx)
            samples = np.random.random_integers(len(dataset_indices)-1, size=number_of_samples+1+1)
            # replace "self" with 0
            samples[samples == idx] = 0
            # set the first to the correct target
            samples[1] = id_to_idx[_target_id]
            # set the 0th to the origin/reference
            samples[0] = idx

            # convert candidates to ids
            candidate_ids = dataset_ids[samples[1:]]
            all_targets = dataset_target_ids[idx]
            ix = np.isin(candidate_ids, all_targets)
            probs = np.zeros(shape=len(candidate_ids), dtype=np.int32)
            probs[ix] = 1

            yield samples, probs

    step = test_step
    feed_dict = {}
    execute_vars = {'loss': model.loss, 'values_gold': model.values_gold, 'values': model.values_predicted}

    if train:
        execute_vars['train_op'] = model.train_op
        execute_vars['step'] = model.global_step
    else:
        if not isinstance(model.tree_model, model_fold.DummyTreeModel):
            feed_dict[model.tree_model.keep_prob] = 1.0


    if False:
    #if highest_sims_model is not None:
        tree_count = model.tree_model.tree_count
        tree_output_size = model.tree_model.tree_output_size
        indices_number = config.batch_size // tree_count
        _tree_embeddings = []
        orginal_data = []
        for batch in td.group_by_batches(dataset_trees, config.batch_size):
            orginal_data.extend(batch)
            feed_dict[model.tree_model.compiler.loom_input_tensor] = batch
            current_tree_embeddings = sess.run(model.tree_model.embeddings_all, feed_dict)
            _tree_embeddings.append(current_tree_embeddings.reshape((-1, tree_count, tree_output_size)))
        _tree_embeddings_all = np.concatenate(_tree_embeddings)
        logger.debug('%i * %i embeddings calculated' % (len(_tree_embeddings_all), tree_count))
        # calculate cosine sim for all combinations by tree-index ([0..tree_count-1])
        s = _tree_embeddings_all.shape[0]
        sim_batch_size = 1000
        normed = pp.normalize(_tree_embeddings_all.reshape((-1, tree_output_size)), norm='l2')\
            .reshape((s, tree_count, tree_output_size))
        #sims = []
        #_indices = []
        neg_sample_indices = np.zeros(shape=(s, tree_count, indices_number), dtype=np.int32)
        for t in range(tree_count):
            # exclude identity: -eye
            #current_sims = -np.eye(s, dtype=np.float32)

            sess.run(highest_sims_model.normed_embeddings_init,
                     feed_dict={highest_sims_model.normed_embeddings_placeholder: normed[:, t, :]})

            #sess.run(highest_sims_model.normed_embeddings,
            #                        {
            #                            highest_sims_model.normed_embeddings: normed[:, t, :]
            #                        })
            for i in range(s):
                #current_sims = np.zeros(s, dtype=np.float32)
                #current_sims[i] = -1.0
                #for j in range(0, s, sim_batch_size):
                #current_sims[i, :] += np.sum(normed[i, t, :] * normed[:, t, :], axis=-1)
                    #j_end = min(j+sim_batch_size, s)
                    #sims_batch = sess.run(highest_sims_model.sims,
                    #                               {
                    #                                   highest_sims_model.normed_reference_embedding: normed[i, t, :],
                    #                                   highest_sims_model.normed_embeddings: normed[j:j_end, t, :]
                    #                               })
                    #current_sims[j:j_end] += sims_batch
                current_sims = sess.run(highest_sims_model.sims,
                                                   {
                                                       highest_sims_model.reference_idx: i,
                                                       #highest_sims_model.normed_embeddings: normed[:, t, :]
                                                   })
                current_sims[i] = 0
                #tiled = np.tile(normed[:, t, :], (s, 1)).reshape((s, s, tree_output_size))
                #tiled_trans = np.transpose(tiled, axes=[1, 0, 2])
                #current_sims = np.sum(tiled_trans * tiled, axis=-1)
                current_indices = np.argpartition(current_sims, -indices_number)[-indices_number:]
                neg_sample_indices[i, t, :] = current_indices
            #_indices.append(current_indices)
            #sims.append(current_sims)
        neg_sample_indices = neg_sample_indices.reshape((s, -1))
        #neg_sample_indices = np.concatenate(_indices, axis=-1)
        logger.debug('created neg_sample_indices')

        #batch_iter = [[orginal_data[idx] for idx in [i] + n_indices] for i, n_indices in enumerate(neg_sample_indices)]
        batch_iter = []
        for i, n_indices in enumerate(neg_sample_indices):
            _current_indices = np.concatenate(([i], n_indices))
            _current_batch_data = [orginal_data[idx] for idx in _current_indices]
            batch_iter.append(_current_batch_data)
        logger.debug('created highest sims batches')
    else:
        batch_iter = td.group_by_batches(batch_iter_naive(), config.batch_size)

    _result_all = []

    # for batch in td.group_by_batches(data_set, config.batch_size if train else len(test_set)):
    for batch in td.group_by_batches(batch_iter_naive(), config.batch_size):
        tree_indices_batched, probs_batched = zip(*batch)
        #tree_indices_batches_np = np.array(tree_indices_batched)
        if not isinstance(model.tree_model, model_fold.DummyTreeModel):
            trees_batched = [[dataset_trees[tree_idx] for tree_idx in tree_indices] for tree_indices in tree_indices_batched]
            #trees_batched = [[dataset_trees.next() for tree_idx in tree_indices] for tree_indices in tree_indices_batched]
            feed_dict[model.tree_model.compiler.loom_input_tensor] = trees_batched
        else:
            tree_embeddings_batched = [[dataset_trees_embedded[tree_idx] for tree_idx in tree_indices] for tree_indices in tree_indices_batched]
            feed_dict[model.tree_model.embeddings_placeholder] = tree_embeddings_batched
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

    sizes = [len(result_all['values'][i]) for i in range(len(_result_all))]
    values_all_ = np.concatenate(result_all['values'])
    values_all_gold_ = np.concatenate(result_all['values_gold'])

    # sum batch losses weighted by individual batch size (can vary at last batch)
    loss_all = sum([result_all['loss'][i] * sizes[i] for i in range(len(_result_all))])
    loss_all /= sum(sizes)

    stats_dict = collect_stats(supervisor, sess, epoch, step, loss_all, values_all_, values_all_gold_,
                               model_type=model.model_type, emit=emit,
                               test_writer=test_writer, test_result_writer=test_result_writer)
    return step, loss_all, values_all_, values_all_gold_, stats_dict


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


def execute_run(config, logdir_continue=None, logdir_pretrained=None, test_file=None, init_only=None, test_only=None):
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

    checkpoint_fn = tf.train.latest_checkpoint(logdir)
    if logdir_continue:
        assert checkpoint_fn is not None, 'could not read checkpoint from logdir: %s' % logdir
    old_checkpoint_fn = None
    if checkpoint_fn:
        if not checkpoint_fn.startswith(logdir):
            raise ValueError('entry in checkpoint file ("%s") is not located in logdir=%s' % (checkpoint_fn, logdir))
        logger.info('read lex_size from model ...')
        reader = tf.train.NewCheckpointReader(checkpoint_fn)
        saved_shapes = reader.get_variable_to_shape_map()
        logger.debug(saved_shapes)
        logger.debug('parameter count: %i' % get_parameter_count_from_shapes(saved_shapes))
        # create test result writer
        test_result_writer = csv_test_writer(os.path.join(logdir, 'test'), mode='a')
        lexicon = lex.Lexicon(filename=os.path.join(logdir, 'model'), load_ids_fixed=(not config.no_fixed_vecs))
        #assert len(lexicon) == saved_shapes[model_fold.VAR_NAME_LEXICON][0]
        ROOT_idx = lexicon.get_d(vocab_manual[ROOT_EMBEDDING], data_as_hashes=False)
        IDENTITY_idx = lexicon.get_d(vocab_manual[IDENTITY_EMBEDDING], data_as_hashes=False)
        lexicon.init_vecs(checkpoint_reader=reader)
    else:
        lexicon = lex.Lexicon(filename=config.train_data_path, load_ids_fixed=(not config.no_fixed_vecs))
        ROOT_idx = lexicon.get_d(vocab_manual[ROOT_EMBEDDING], data_as_hashes=False)
        IDENTITY_idx = lexicon.get_d(vocab_manual[IDENTITY_EMBEDDING], data_as_hashes=False)
        if logdir_pretrained:
            logger.info('load lexicon from pre-trained model: %s' % logdir_pretrained)
            old_checkpoint_fn = tf.train.latest_checkpoint(logdir_pretrained)
            assert old_checkpoint_fn is not None, 'No checkpoint file found in logdir_pretrained: ' + logdir_pretrained
            reader_old = tf.train.NewCheckpointReader(old_checkpoint_fn)
            lexicon_old = lex.Lexicon(filename=os.path.join(logdir_pretrained, 'model'))
            lexicon_old.init_vecs(checkpoint_reader=reader_old)
            lexicon.merge(lexicon_old, add=False, remove=False)

        #lexicon.replicate_types(suffix=constants.SEPARATOR + constants.vocab_manual[constants.BACK_EMBEDDING])
        #lexicon.pad()

        lexicon.dump(filename=os.path.join(logdir, 'model'), strings_only=True)
        assert lexicon.is_filled, 'lexicon: not all vecs for all types are set (len(types): %i, len(vecs): %i)' % \
                                  (len(lexicon), len(lexicon.vecs))

        # dump config
        config.dump(logdir=logdir)

        # create test result writer
        test_result_writer = csv_test_writer(os.path.join(logdir, 'test'))
        test_result_writer.writeheader()

    logger.info('lexicon size: %i' % len(lexicon))
    logger.debug('IDENTITY_idx: %i' % IDENTITY_idx)
    logger.debug('ROOT_idx: %i' % ROOT_idx)

    # TRAINING and TEST DATA ###########################################################################################

    meta = {}

    parent_dir = os.path.abspath(os.path.join(config.train_data_path, os.pardir))
    if not (test_only or init_only):
        meta[M_TRAIN] = {}
    meta[M_TEST] = {}

    if config.model_type == 'simtuple':
        tree_iterator_args = {'root_idx': ROOT_idx, 'split': True, 'extensions': config.extensions.split(','),
                              'max_depth': config.max_depth, 'context': config.context, 'transform': True}
        tree_iterator = diters.data_tuple_iterator

        tuple_size = 2  # [1.0, <sim_value>]   # [first_sim_entry, second_sim_entry]
        discrete_model = False
        load_parents = False
    elif config.model_type == 'tuple':
        tree_iterator_args = {'max_depth': config.max_depth, 'context': config.context, 'transform': True,
                              'concat_mode': config.concat_mode, 'link_cost_ref': config.link_cost_ref,
                              'bag_of_seealsos': False}
        #if FLAGS.debug:
        #    root_strings_store = StringStore()
        #    root_strings_store.from_disk('%s.root.id.string' % config.train_data_path)
        #    tree_iterator_args['root_strings'] = [s for s in root_strings_store]

        tree_iterator = diters.tree_iterator
        indices_getter = diters.indices_dbpedianif
        tuple_size = config.neg_samples + 1
        discrete_model = True
        load_parents = (config.context is not None and config.context > 0)
    #elif config.model_type == 'tuple_single':
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
    elif config.model_type == 'reroot':
        if config.cut_indices is not None:
            indices = np.arange(config.cut_indices)
            # avoid looking for train index files
            meta[M_TRAIN][M_FNAMES] = []
        else:
            indices = None
        neg_samples = config.neg_samples
        tuple_size = neg_samples + 1
        tree_iterator_args = {'indices': indices, 'neg_samples': neg_samples, 'max_depth': config.max_depth,
                              'transform': True, 'link_cost_ref': config.link_cost_ref, 'link_cost_ref_seealso': -1}
        tree_iterator = diters.data_tuple_iterator_reroot
        del meta[M_TEST]
        discrete_model = True
        load_parents = True
    elif config.model_type == 'tfidf':
        tree_iterator_args = {'max_depth': config.max_depth, 'context': config.context, 'transform': True,
                              'concat_mode': config.concat_mode, 'link_cost_ref': config.link_cost_ref}
        tree_iterator = diters.data_single_iterator_dbpedianif_context_tfidf

        # TODO: set correct values!
        tuple_size = None  # [1.0, <sim_value>]   # [first_sim_entry, second_sim_entry]
        discrete_model = True

        load_parents = False
    else:
        raise NotImplementedError('model_type=%s not implemented' % config.model_type)

    #if M_TRAIN in meta:
    #    meta[M_TRAIN][M_DATA_ITER] = partial(data_iterator, **data_iterator_args)
    #if M_TEST in meta:
    #    meta[M_TEST][M_DATA_ITER] = partial(data_iterator, **data_iterator_args)

    if FLAGS.train_files is not None and FLAGS.train_files != '':
        logger.info('use train data index files: %s' % FLAGS.train_files)
        meta[M_TRAIN][M_FNAMES] = [os.path.join(parent_dir, fn) for fn in FLAGS.train_files.split(',')]
    if test_file is not None and test_file != '':
        meta[M_TEST][M_FNAMES] = [os.path.join(parent_dir, test_file)]

    if not (test_only or init_only):
        if M_FNAMES not in meta[M_TRAIN]:
            logger.info('collect train data from: ' + config.train_data_path + ' ...')
            regex = re.compile(r'%s\.idx\.\d+\.npy$' % ntpath.basename(config.train_data_path))
            train_fnames = filter(regex.search, os.listdir(parent_dir))
            meta[M_TRAIN][M_FNAMES] = [os.path.join(parent_dir, fn) for fn in sorted(train_fnames)]
        assert len(meta[M_TRAIN][M_FNAMES]) > 0, 'no matching train data files found for ' + config.train_data_path
        logger.info('found ' + str(len(meta[M_TRAIN][M_FNAMES])) + ' train data files')
        if M_TEST in meta and M_FNAMES not in meta[M_TEST]:
            meta[M_TEST][M_FNAMES] = [meta[M_TRAIN][M_FNAMES][config.dev_file_index]]
            logger.info('use %s for testing' % str(meta[M_TEST][M_FNAMES]))
            del meta[M_TRAIN][M_FNAMES][config.dev_file_index]

    forest = Forest(filename=config.train_data_path, lexicon=lexicon, load_parents=load_parents)

    # calc indices (ids, indices, other_indices) from index files
    logger.info('calc indices from index files ...')
    for m in meta:
        assert M_FNAMES in meta[m], 'no %s fnames found' % m
        meta[m][M_IDS], meta[m][M_INDICES], meta[m][M_IDS_TARGET] = indices_getter(index_files=meta[m][M_FNAMES], forest=forest)
        # set tree iterator
        meta[m][M_TREE_ITER] = tree_iterator(indices=meta[m][M_INDICES], forest=forest, **tree_iterator_args)

    # MODEL DEFINITION #################################################################################################

    optimizer = config.optimizer
    if optimizer is not None:
        optimizer = getattr(tf.train, optimizer)

    sim_measure = getattr(model_fold, config.sim_measure)
    tree_embedder = getattr(model_fold, config.tree_embedder)

    logger.info('create tensorflow graph ...')
    with tf.Graph().as_default() as graph:
        with tf.device(tf.train.replica_device_setter(FLAGS.ps_tasks)):
            logger.debug('trainable lexicon entries: %i' % lexicon.len_var)
            logger.debug('fixed lexicon entries:     %i' % lexicon.len_fixed)

            if not config.model_type == 'tfidf':
                model_tree = model_fold.SequenceTreeModel(lex_size_fix=lexicon.len_fixed,
                                                          lex_size_var=lexicon.len_var,
                                                          tree_embedder=tree_embedder,
                                                          dimension_embeddings=lexicon.vec_size,
                                                          state_size=config.state_size,
                                                          leaf_fc_size=config.leaf_fc_size,
                                                          # add a leading '0' to allow an empty string
                                                          root_fc_sizes=[int(s) for s in ('0' + config.root_fc_sizes).split(',')],
                                                          keep_prob=config.keep_prob,
                                                          tree_count=tuple_size,
                                                          #discrete_values_gold=discrete_model
                                                          # keep_prob_fixed=config.keep_prob # to enable full head dropout
                                                          )
                for m in meta:
                    logger.info('create %s data set ...' % m)
                    with model_tree.compiler.multiprocessing_pool():
                        # meta[m][M_TREES], meta[m][M_IDS], meta[m][M_TARGETS] = zip(*meta[m][M_INDEX_ITER](sequence_trees=forest))
                        meta[m][M_TREES] = list(model_tree.compiler.build_loom_inputs(meta[m][M_TREE_ITER], ordered=True))
                        #meta[m][M_TREES] = list(meta[m][M_TREE_ITER])
                        #logger.info('%s data size: %s' % (m, len(meta[m][M_TREES])))
            else:
                max_embeddings_dim = -1
                for m in meta:
                    logger.info('create %s data set ...' % m)
                    # if model is None, it is tfidf
                    meta[m][M_TREE_EMBEDDINGS] = diters.embeddings_tfidf(meta[m][M_TREE_ITER])
                    current_embeddings_dim = meta[m][M_TREE_EMBEDDINGS].shape[1]
                    if current_embeddings_dim > max_embeddings_dim:
                        max_embeddings_dim = current_embeddings_dim
                    #assert embeddings_dim is None or meta[m][M_TREE_EMBEDDINGS].shape[1] == embeddings_dim, 'embedding_dims are different: %i != %i' % (meta[m][M_TREE_EMBEDDINGS].shape[1], embeddings_dim)
                    logger.info('%s data size: %s' % (m, len(meta[m][M_TREE_EMBEDDINGS])))

                # pad embedding dimensions
                for m in meta:
                    current_embeddings_dim = meta[m][M_TREE_EMBEDDINGS].shape[1]
                    if current_embeddings_dim < max_embeddings_dim:
                        shape = meta[m][M_TREE_EMBEDDINGS].shape
                        padded = np.zeros(shape=[shape[0], max_embeddings_dim])
                        padded[:shape[0], :shape[1]] = meta[m][M_TREE_EMBEDDINGS]
                        meta[m][M_TREE_EMBEDDINGS] = padded

                model_tree = model_fold.DummyTreeModel(embeddings_dim=max_embeddings_dim, tree_count=tuple_size)

            if config.model_type == 'simtuple':
                model = model_fold.SimilaritySequenceTreeTupleModel(tree_model=model_tree,
                                                                    optimizer=optimizer,
                                                                    learning_rate=config.learning_rate,
                                                                    sim_measure=sim_measure,
                                                                    clipping_threshold=config.clipping)
            elif config.model_type == 'tuple':
                model = model_fold.TreeTupleModel_with_candidates(tree_model=model_tree,
                                                                  optimizer=optimizer,
                                                                  learning_rate=config.learning_rate,
                                                                  clipping_threshold=config.clipping,
                                                                  candidate_count=tuple_size-1)
                #meta[M_TRAIN]['model_highest_sims'] = model_fold.HighestSimsModel(embedding_size=lexicon.vec_size,
                #                                                                  number_of_embeddings=len(meta[M_TRAIN][M_TREES]))
                #meta[M_TEST]['model_highest_sims'] = model_fold.HighestSimsModel(embedding_size=lexicon.vec_size,
                #                                                                 number_of_embeddings=len(meta[M_TEST][M_TREES]))
            elif config.model_type == 'reroot':
                model = model_fold.SequenceTreeRerootModel(tree_model=model_tree,
                                                           optimizer=optimizer,
                                                           learning_rate=config.learning_rate,
                                                           clipping_threshold=config.clipping)
            else:
                raise NotImplementedError('model_type=%s not implemented' % config.model_type)

            # set model
            for m in meta:
                #if M_INDEX_ITER in meta[m]:
                meta[m][M_MODEL] = model


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
                is_chief=(FLAGS.task == 0),
                save_summaries_secs=10,
                save_model_secs=0,
                summary_writer=tf.summary.FileWriter(os.path.join(logdir, 'train'), graph),
                init_fn=load_pretrain if pre_train_saver is not None else None
            )
            #if dev_iterator is not None or test_iterator is not None:
            if M_TEST in meta:
                test_writer = tf.summary.FileWriter(os.path.join(logdir, 'test'), graph)
            sess = supervisor.PrepareSession(FLAGS.master)
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

            #forest = Forest(filename=config.train_data_path, lexicon=lexicon, load_parents=load_parents)

            # TODO:
            # add code for TF-IDF model here:
            #     1) train_iterator/dev_iterator/test_iterator output to lists (of (n-)tuples)
            #     2) lists to -> tf-idf doc representations
            #     3) regression on train
            #     4) eval on dev/test

            #with model_tree.compiler.multiprocessing_pool():

            if M_TEST in meta:
                logger.info('create test data set ...')
                if M_TRAIN not in meta:
                    step, loss_all, values_all, values_all_gold, stats_dict = do_epoch(supervisor,
                                                                                       sess=sess,
                                                                                       model=meta[M_TEST][M_MODEL],
                                                                                       dataset_trees=meta[M_TEST][M_TREES] if M_TREES in meta[M_TEST] else None,
                                                                                       dataset_ids=meta[M_TEST][M_IDS],
                                                                                       dataset_target_ids=meta[M_TEST][M_IDS_TARGET],
                                                                                       dataset_trees_embedded=meta[M_TEST][M_TREE_EMBEDDINGS] if M_TREE_EMBEDDINGS in meta[M_TEST] else None,
                                                                                       epoch=0,
                                                                                       train=False,
                                                                                       emit=False,
                                                                                       number_of_samples=config.neg_samples,
                                                                                       highest_sims_model=meta[M_TEST]['model_highest_sims'] if 'model_highest_sims' in meta[M_TEST] else None)
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
                    stat_key = STAT_KEYS_DISCRETE[0]
                elif meta[M_TEST][M_MODEL].model_type == MODEL_TYPE_REGRESSION:
                    stat_key = STAT_KEYS_REGRESSION[0]
                else:
                    raise ValueError('stat_key not defined for model_type=%s' % meta[M_TEST][M_MODEL].model_type)
                # NOTE: this depends on stat_key (pearson/mse/roc/...)
                TEST_MIN_INIT = -1
                stat_queue = [{stat_key: TEST_MIN_INIT}]
            step_train = sess.run(meta[M_TRAIN][M_MODEL].global_step)
            max_queue_length = 0
            for epoch, shuffled in enumerate(td.epochs(items=range(len(meta[M_TRAIN][M_IDS])), n=config.epochs, shuffle=True), 1):

                # train
                if not config.early_stop_queue or len(stat_queue) > 0:
                    step_train, loss_train, _, _, stats_train = do_epoch(supervisor, sess,
                                                                         model=meta[M_TRAIN][M_MODEL],
                                                                         dataset_trees=meta[M_TRAIN][M_TREES] if M_TREES in meta[M_TRAIN] else None,
                                                                         dataset_trees_embedded=meta[M_TRAIN][M_TREE_EMBEDDINGS] if M_TREE_EMBEDDINGS in meta[M_TRAIN] else None,
                                                                         dataset_ids=meta[M_TRAIN][M_IDS],
                                                                         dataset_target_ids=meta[M_TRAIN][M_IDS_TARGET],
                                                                         epoch=epoch,
                                                                         number_of_samples=config.neg_samples,
                                                                         highest_sims_model=meta[M_TRAIN]['model_highest_sims'] if 'model_highest_sims' in meta[M_TRAIN] else None)

                if M_TEST in meta:

                    # test
                    step_test, loss_test, sim_all, sim_all_gold, stats_test = do_epoch(supervisor, sess,
                                                                                       model=meta[M_TEST][M_MODEL],
                                                                                       dataset_trees=meta[M_TEST][M_TREES] if M_TREES in meta[M_TEST] else None,
                                                                                       dataset_trees_embedded=meta[M_TEST][M_TREE_EMBEDDINGS] if M_TREE_EMBEDDINGS in meta[M_TEST] else None,
                                                                                       dataset_ids=meta[M_TEST][M_IDS],
                                                                                       dataset_target_ids=meta[M_TEST][M_IDS_TARGET],
                                                                                       number_of_samples=config.neg_samples,
                                                                                       epoch=epoch,
                                                                                       train=False,
                                                                                       test_step=step_train,
                                                                                       test_writer=test_writer,
                                                                                       test_result_writer=test_result_writer,
                                                                                       highest_sims_model=meta[M_TEST]['model_highest_sims'] if 'model_highest_sims' in meta[M_TEST] else None)

                    if loss_test < loss_test_best:
                        loss_test_best = loss_test

                    # EARLY STOPPING ###############################################################################

                    stat = round(stats_test[stat_key], 6)

                    prev_max = max(stat_queue, key=lambda t: t[stat_key])[stat_key]
                    # stop, if current test pearson r is not bigger than previous values. The amount of regarded
                    # previous values is set by config.early_stop_queue
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
                    if 0 < config.early_stop_queue < len(stat_queue):
                        logger.info('last test %s: %s, last rank: %i' % (stat_key, str(stat_queue), rank))
                        logger.removeHandler(fh_info)
                        logger.removeHandler(fh_debug)
                        return stat_queue_sorted[0]

                    # do not save, if score was not the best
                    # if rank > len(stat_queue) * 0.05:
                    if len(stat_queue) > 1 and config.early_stop_queue:
                        # auto restore if enabled
                        #if config.auto_restore:
                        #    supervisor.saver.restore(sess, tf.train.latest_checkpoint(logdir))
                        pass
                    else:
                        # don't save after first epoch if config.early_stop_queue > 0
                        if prev_max > TEST_MIN_INIT or not config.early_stop_queue:
                            supervisor.saver.save(sess, checkpoint_path(logdir, step_train))
                else:
                    # save model after each step if not dev model is set (training a language model)
                    supervisor.saver.save(sess, checkpoint_path(logdir, step_train))

            logger.removeHandler(fh_info)
            logger.removeHandler(fh_debug)


def set_stat_values(d, stats, prefix=''):
    if STAT_KEYS_REGRESSION[0] in stats:
        _stat_keys = STAT_KEYS_REGRESSION
    elif STAT_KEYS_DISCRETE[0] in stats:
        _stat_keys = STAT_KEYS_DISCRETE
    else:
        raise ValueError('stats has to contain either %s or %s' % (STAT_KEYS_REGRESSION[0], STAT_KEYS_DISCRETE[0]))
    for k in _stat_keys:
        d[prefix + k] = stats[k]
    return _stat_keys


if __name__ == '__main__':
    mytools.logging_init()
    logger.debug('test')
    # tf.app.run()
    # ATTENTION: discards any FLAGS (e.g. provided as argument) contained in default_config!
    if FLAGS.logdir_continue is not None and ',' in FLAGS.logdir_continue:
        logdirs = FLAGS.logdir_continue.split(',')
        logger.info('execute %i runs ...' % len(logdirs))
        stats_prefix = 'score_'
        with open(os.path.join(FLAGS.logdir, 'scores_new.tsv'), 'w') as csvfile:
            #fieldnames = Config(logdir_continue=logdirs[0]).as_dict().keys() + ['score_pearson', 'score_mse']
            # TODO: adapt for model_type==MODEL_TYPE_DISCRETE
            fieldnames = Config(logdir_continue=logdirs[0]).as_dict().keys() + [stats_prefix + k for k in STAT_KEYS_REGRESSION]
            score_writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter='\t')
            score_writer.writeheader()
            for i, logdir in enumerate(logdirs, 1):
                logger.info('START RUN %i of %i' % (i, len(logdirs)))
                config = Config(logdir_continue=logdir)
                config_dict = config.as_dict()
                stats = execute_run(config, logdir_continue=logdir, logdir_pretrained=FLAGS.logdir_pretrained,
                                    test_file=FLAGS.test_file, init_only=FLAGS.init_only, test_only=FLAGS.test_only)

                set_stat_values(config_dict, stats, prefix=stats_prefix)
                score_writer.writerow(config_dict)
                csvfile.flush()
    else:
        config = Config(logdir_continue=FLAGS.logdir_continue, logdir_pretrained=FLAGS.logdir_pretrained)
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

            # mytools.make_parent_dir(scores_fn) #logdir has to contain grid_config_file
            #fieldnames_expected = grid_parameters.keys() + ['pearson_dev_best', 'pearson_test', 'mse_dev_best',
            #                                                'mse_test', 'run_description']
            stats_prefix_dev = 'dev_best_'
            stats_prefix_test = 'test_'
            # TODO: adapt for model_type==MODEL_TYPE_DISCRETE
            fieldnames_expected = grid_parameters.keys() + [stats_prefix_dev + k for k in STAT_KEYS_REGRESSION] \
                                  + [stats_prefix_test + k for k in STAT_KEYS_REGRESSION] + ['run_description']
            assert fieldnames_loaded is None or set(fieldnames_loaded) == set(fieldnames_expected), 'field names in tsv file are not as expected'
            fieldnames = fieldnames_loaded or fieldnames_expected
            with open(scores_fn, file_mode) as csvfile:
                score_writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter='\t')
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

                        train_data_dir = os.path.abspath(os.path.join(c.train_data_path, os.pardir))
                        if FLAGS.test_file is not None:
                            test_fname = os.path.join(train_data_dir, FLAGS.test_file)
                            assert os.path.isfile(test_fname), 'could not find test file: %s' % test_fname
                        else:
                            test_fname = None

                        # skip already processed
                        if os.path.isdir(logdir) and c.run_description in run_descriptions_done:
                            logger.debug('skip config for logdir: %s' % logdir)
                            c.run_description = run_desc_backup
                            continue

                        #d['pearson_dev_best'], d['mse_dev_best'] = execute_run(c)
                        stats_dev = execute_run(c)
                        stat_keys = set_stat_values(d, stats_dev, prefix=stats_prefix_dev)

                        logger.info('best dev score (%s): %f' % (stat_keys[0], stats_dev[stat_keys[0]]))
                        if test_fname is not None:
                            #d['pearson_test'], d['mse_test'] = execute_run(c, logdir_continue=logdir, test_only=True, test_file=FLAGS.test_file)
                            stats_test = execute_run(c, logdir_continue=logdir, test_only=True, test_file=FLAGS.test_file)
                            #logger.info('test score (%s): %f' % (stat_key, stats_dev[stat_key]))
                            set_stat_values(d, stats_dev, prefix=stats_prefix_test)
                            logger.info('test score (%s): %f' % (stat_keys[0], stats_dev[stat_keys[0]]))
                        else:
                            for k in stat_keys:
                                d[k] = -1.0
                        d['run_description'] = c.run_description

                        c.run_description = run_desc_backup
                        score_writer.writerow(d)
                        csvfile.flush()

        # default: execute single run
        else:
            execute_run(config, logdir_continue=FLAGS.logdir_continue, logdir_pretrained=FLAGS.logdir_pretrained,
                        test_file=FLAGS.test_file, init_only=FLAGS.init_only, test_only=FLAGS.test_only)
