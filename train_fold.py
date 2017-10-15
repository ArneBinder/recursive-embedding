# from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import fnmatch
import json
import ntpath
import os
# import google3
import shutil

from scipy.stats.stats import pearsonr
from scipy.stats.mstats import spearmanr
import six
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import logging
import sys

import corpus
import model_fold
import similarity_tree_tuple_pb2
import tensorflow_fold as td
import numpy as np

flags = {'train_data_path': [tf.flags.DEFINE_string,
                             # '/media/arne/WIN/Users/Arne/ML/data/corpora/ppdb/process_sentence3_ns1/PPDB_CMaggregate',
                             # '/media/arne/WIN/Users/Arne/ML/data/corpora/sick/process_sentence2/SICK_CMaggregate',
                             '/media/arne/WIN/Users/Arne/ML/data/corpora/sick/process_sentence3/SICK_tt_CMaggregate',
                             # '/media/arne/WIN/Users/Arne/ML/data/corpora/sick/process_sentence2/SICK_tt_CMsequence_ICMtree',
                             # '/media/arne/WIN/Users/Arne/ML/data/corpora/sick/process_sentence3/SICK_tt_CMsequence_ICMtree',
                             # '/media/arne/WIN/Users/Arne/ML/data/corpora/sick/process_sentence4/SICK_tt_CMsequence_ICMtree',
                             # '/media/arne/WIN/Users/Arne/ML/data/corpora/debate_cluster/process_sentence3/HASAN_CMaggregate',
                             # '/media/arne/WIN/Users/Arne/ML/data/corpora/debate_cluster/process_sentence3/HASAN_CMaggregate_NEGSAMPLES0',
                             # '/media/arne/WIN/Users/Arne/ML/data/corpora/debate_cluster/process_sentence3/HASAN_CMsequence_ICMtree_NEGSAMPLES0',
                             #   '/media/arne/WIN/Users/Arne/ML/data/corpora/debate_cluster/process_sentence3/HASAN_CMsequence_ICMtree_NEGSAMPLES1',
                             'TF Record file containing the training dataset of sequence tuples.'],
         'old_logdir': [tf.flags.DEFINE_string,
                        None,
                        #'/home/arne/ML_local/tf/supervised/log/batchsize100_embeddingstrainableTRUE_learningrate0.001_optimizerADADELTAOPTIMIZER_simmeasureSIMCOSINE_statesize50_testfileindex1_traindatapathPROCESSSENTENCE3SICKTTCMSEQUENCEICMTREE_treeembedderTREEEMBEDDINGHTUGRUSIMPLIFIED',
                        # '/home/arne/ML_local/tf/supervised/log/SA/EMBEDDING_FC/batchsize100_embeddingstrainableTRUE_learningrate0.001_optimizerADADELTAOPTIMIZER_simmeasureSIMCOSINE_statesize50_testfileindex1_traindatapathPROCESSSENTENCE3SICKTTCMAGGREGATE_treeembedderTREEEMBEDDINGFLATAVG2LEVELS',
                        'Set this to fine tune a pre-trained model. The old_logdir has to contain a types file with the filename "model.types"',
                        None],
         'batch_size': [tf.flags.DEFINE_integer,
                        100,
                        'How many samples to read per batch.'],
         'epochs': [tf.flags.DEFINE_integer,
                    1000000,
                    'The number of epochs.',
                    None],
         'test_file_index': [tf.flags.DEFINE_integer,
                             1,
                             'Which file of the train data files should be used as test data.'],
         # TODO: rename to 'lexicon_trainable' (not yet done because of compatibility)
         'embeddings_trainable': [tf.flags.DEFINE_boolean,
                                  # False,
                                  True,
                                  'Iff enabled, fine tune the embeddings.'],
         'sim_measure': [tf.flags.DEFINE_string,
                         'sim_cosine',
                         'similarity measure implementation (tensorflow) from model_fold for similarity score '
                         'calculation. Currently implemented:'
                         '"sim_cosine" -> cosine'
                         '"sim_layer" -> similarity measure similar to the one defined in [Tai, Socher 2015]'
                         '"sim_manhattan" -> l1-norm based similarity measure (taken from MaLSTM) [Mueller et al., 2016]'],
         'tree_embedder': [tf.flags.DEFINE_string,
                           'TreeEmbedding_FLAT_AVG',
                           'TreeEmbedder implementation from model_fold that produces a tensorflow fold block on '
                           'calling which accepts a sequence tree and produces an embedding. '
                           'Currently implemented (see model_fold.py):'
                           '"TreeEmbedding_TREE_LSTM"           -> TreeLSTM'
                           '"TreeEmbedding_HTU_GRU"             -> Headed Tree Unit, using a GRU for order aware '
                           '                                       and summation for order unaware composition'
                           '"TreeEmbedding_FLAT_AVG"            -> Averaging applied to first level children '
                           '                                       (discarding the root)'
                           '"TreeEmbedding_FLAT_AVG_2levels"    -> Like TreeEmbedding_FLAT_AVG, but concatenating first'
                           '                                       second level children (e.g. dep-edge embedding) to '
                           '                                       the first level children (e.g. token embeddings)'
                           '"TreeEmbedding_FLAT_LSTM"           -> LSTM applied to first level children (discarding the'
                           '                                       root)'
                           '"TreeEmbedding_FLAT_LSTM_2levels"   -> Like TreeEmbedding_FLAT_LSTM, but concatenating '
                           '                                       first second level children (e.g. dependency-edge '
                           '                                       type embedding) to the first level children '
                           '                                       (e.g. token embeddings)'
                           ],
         'leaf_fc_activation': [tf.flags.DEFINE_string,
                                     # None,
                                     'tanh',
                                     'If not None, apply a fully connected layer with this activation function before composition',
                                     None],
         'root_fc_activation': [tf.flags.DEFINE_string,
                                  None,
                                  # 'tanh',
                                  'If not None, apply a fully connected layer with this activation function after composition',
                                  None],
         'leaf_fc_size': [tf.flags.DEFINE_integer,
                                     # 0,
                                     50,
                                     'If not 0, apply a fully connected layer with this size before composition'
                          ],
         'root_fc_size': [tf.flags.DEFINE_integer,
                                  # 0,
                                  50,
                                  'If not 0, apply a fully connected layer with this size after composition'
                          ],
         'state_size': [tf.flags.DEFINE_integer,
                        50,
                        'size of the composition layer'],
         'learning_rate': [tf.flags.DEFINE_float,
                           0.02,
                           # 'tanh',
                           'learning rate'],
         'optimizer': [tf.flags.DEFINE_string,
                       'AdadeltaOptimizer',
                       'optimizer'],
         'early_stop': [tf.flags.DEFINE_boolean,
                       True,
                       'If True, stop training when test loss decreases (queued)',
                       None],
         'logdir': [tf.flags.DEFINE_string,
                    # '/home/arne/ML_local/tf/supervised/log/dataPs2aggregate_embeddingsUntrainable_simLayer_modelTreelstm_normalizeTrue_batchsize250',
                    # '/home/arne/ML_local/tf/supervised/log/dataPs2aggregate_embeddingsTrainable_simLayer_modelAvgchildren_normalizeTrue_batchsize250',
                    #'/home/arne/ML_local/tf/supervised/log/SA/EMBEDDING_FC_dim300',
                    '/home/arne/ML_local/tf/supervised/log/SA/NEW',
                    'Directory in which to write event logs.',
                    None]
         }

for flag in flags:
    v = flags[flag]
    v[0](flag, v[1], v[2])

# flags which are not logged in logdir/flags.json
tf.flags.DEFINE_string('master', '',
                       'Tensorflow master to use.')
tf.flags.DEFINE_integer('task', 0,
                        'Task ID of the replica running the training.')
tf.flags.DEFINE_integer('ps_tasks', 0,
                        'Number of PS tasks in the job.')

FLAGS = tf.flags.FLAGS
PROTO_PACKAGE_NAME = 'recursive_dependency_embedding'
s_root = os.path.dirname(__file__)
# Make sure serialized_message_to_tree can find the similarity_tree_tuple proto:
td.proto_tools.map_proto_source_tree_path('', os.path.dirname(__file__))
td.proto_tools.import_proto_file('similarity_tree_tuple.proto')


def iterate_over_tf_record_protos(table_paths, message_type, multiple_epochs=True):
    while True:
        count = 0
        for table_path in table_paths:
            for v in tf.python_io.tf_record_iterator(table_path):
                res = td.proto_tools.serialized_message_to_tree(PROTO_PACKAGE_NAME + '.' + message_type.__name__, v)
                res['id'] = count
                yield res
                count += 1
        if not multiple_epochs:
            logging.info('records read: ' + str(count))
            break


def emit_values(supervisor, session, step, values, writer=None, csv_writer=None):
    summary = tf.Summary()
    for name, value in six.iteritems(values):
        summary_value = summary.value.add()
        summary_value.tag = name
        summary_value.simple_value = float(value)
    if writer:
        writer.add_summary(summary, step)
    else:
        supervisor.summary_computed(session, summary, global_step=step)
    if csv_writer:
        values['step'] = step
        csv_writer.writerow({k: values[k] for k in values if k in csv_writer.fieldnames})


def checkpoint_path(logdir, step):
    return os.path.join(logdir, 'model.ckpt-' + str(step))


def csv_test_writer(logdir, mode='w'):
    if not os.path.isdir(logdir):
        os.makedirs(logdir)
    test_result_csv = open(os.path.join(logdir, 'results.csv'), mode, buffering=1)
    fieldnames = ['step', 'loss', 'pearson_r', 'sim_avg']
    test_result_writer = csv.DictWriter(test_result_csv, fieldnames=fieldnames, delimiter='\t')
    return test_result_writer


def main(unused_argv):
    logging_format = '%(asctime)s %(message)s'
    tf.logging._logger.propagate = False
    tf.logging._logger.format = logging_format
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format=logging_format)
    logging.info('collect train data from: ' + FLAGS.train_data_path + ' ...')
    parent_dir = os.path.abspath(os.path.join(FLAGS.train_data_path, os.pardir))
    train_fnames = fnmatch.filter(os.listdir(parent_dir), ntpath.basename(FLAGS.train_data_path) + '.train.*')
    train_fnames = [os.path.join(parent_dir, fn) for fn in train_fnames]
    logging.info('found ' + str(len(train_fnames)) + ' train data files')
    test_fname = train_fnames[FLAGS.test_file_index]
    logging.info('use ' + test_fname + ' for testing')
    del train_fnames[FLAGS.test_file_index]

    train_iterator = iterate_over_tf_record_protos(
        train_fnames, similarity_tree_tuple_pb2.SimilarityTreeTuple, multiple_epochs=False)

    test_iterator = iterate_over_tf_record_protos(
        [test_fname], similarity_tree_tuple_pb2.SimilarityTreeTuple, multiple_epochs=False)

    # DEBUG
    # vecs, types = corpus.create_or_read_dict(FLAGS.train_data_path)
    # lex_size = vecs.shape[0]
    # embedding_dim = vecs.shape[1]

    run_desc = []
    for flag in sorted(flags.keys()):
        # throw the type away
        flags[flag] = flags[flag][1:]
        # get real flag value
        new_value = getattr(FLAGS, flag)
        flags[flag][0] = new_value

        # collect run description
        if len(flags[flag]) < 3:
            flag_name = flag.replace('_', '')
            flag_value = str(new_value).replace('_', '')
            # if flag_value is a path, take only the last two subfolders
            flag_value = ''.join(flag_value.split(os.sep)[-2:])
            run_desc.append(flag_name.lower() + flag_value.upper())
        # if a short version is set, use it. if it is set to None, add this flag not to the run_descriptions
        elif flags[flag][2]:
            run_desc.append(flag.replace('_', '').lower() + str(flags[flag][2]).replace('_', '').upper())

    flags['run_description'] = ['_'.join(run_desc), 'short string description of the current run']
    logging.info('serialized run description: ' + flags['run_description'][0])

    logdir = os.path.join(FLAGS.logdir, flags['run_description'][0])
    if not os.path.isdir(logdir):
        os.makedirs(logdir)
    checkpoint_fn = tf.train.latest_checkpoint(logdir)
    old_checkpoint_fn = None
    vecs = None
    if checkpoint_fn:
        logging.info('read lex_size from model ...')
        reader = tf.train.NewCheckpointReader(checkpoint_fn)
        saved_shapes = reader.get_variable_to_shape_map()
        embed_shape = saved_shapes[model_fold.VAR_NAME_LEXICON]
        lex_size = embed_shape[0]
        # create test result writer
        test_result_writer = csv_test_writer(os.path.join(logdir, 'test'), mode='a')
    else:
        vecs, types = corpus.create_or_read_dict(FLAGS.train_data_path)
        if FLAGS.old_logdir:
            old_checkpoint_fn = tf.train.latest_checkpoint(FLAGS.old_logdir)
            assert old_checkpoint_fn is not None, 'No checkpoint file found in old_logdir: ' + FLAGS.old_logdir
            reader_old = tf.train.NewCheckpointReader(old_checkpoint_fn)
            vecs_old = reader_old.get_tensor(model_fold.VAR_NAME_LEXICON)
            types_old = corpus.read_types(os.path.join(FLAGS.old_logdir, 'model'))
            vecs, types = corpus.merge_dicts(vecs1=vecs, types1=types, vecs2=vecs_old, types2=types_old, add=False,
                                             remove=False)
            # save types file in log dir
            corpus.write_dict(os.path.join(logdir, 'model'), types=types)
        else:
            # save types file in log dir
            shutil.copyfile(FLAGS.train_data_path + '.type', os.path.join(logdir, 'model.type'))
        lex_size = vecs.shape[0]
        # write flags for current run
        with open(os.path.join(logdir, 'flags.json'), 'w') as outfile:
            json.dump(flags, outfile, indent=2, sort_keys=True)
        # create test result writer
        test_result_writer = csv_test_writer(os.path.join(logdir, 'test'))
        test_result_writer.writeheader()

    logging.info('lex_size: ' + str(lex_size))

    #leaf_fc_activation = FLAGS.leaf_fc_activation
    #if leaf_fc_activation:
    #    leaf_fc_activation = getattr(tf.nn, leaf_fc_activation)
    #root_fc_activation = FLAGS.root_fc_activation
    #if root_fc_activation:
    #    root_fc_activation = getattr(tf.nn, root_fc_activation)
    optimizer = FLAGS.optimizer
    if FLAGS.optimizer:
        optimizer = getattr(tf.train, optimizer)

    sim_measure = getattr(model_fold, FLAGS.sim_measure)
    tree_embedder = getattr(model_fold, FLAGS.tree_embedder)

    logging.info('create tensorflow graph ...')
    with tf.Graph().as_default() as graph:
        with tf.device(tf.train.replica_device_setter(FLAGS.ps_tasks)):
            # Build the graph.
            model = model_fold.SimilaritySequenceTreeTupleModel(lex_size=lex_size,
                                                                tree_embedder=tree_embedder,
                                                                state_size=FLAGS.state_size,
                                                                learning_rate=FLAGS.learning_rate,
                                                                optimizer=optimizer,
                                                                sim_measure=sim_measure,
                                                                lexicon_trainable=FLAGS.embeddings_trainable,
                                                                #leaf_fc_activation=leaf_fc_activation,
                                                                #root_fc_activation=root_fc_activation)
                                                                leaf_fc_size=FLAGS.leaf_fc_size,
                                                                root_fc_size=FLAGS.root_fc_size)

            if old_checkpoint_fn is not None:
                logging.info(
                    'restore from old_checkpoint (except lexicon, step and optimizer vars): ' + old_checkpoint_fn + ' ...')
                lexicon_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=model_fold.VAR_NAME_LEXICON)
                optimizer_vars = model.optimizer_vars()
                restore_vars = [item for item in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if
                                item not in [model.global_step] + lexicon_vars + optimizer_vars]
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
            test_writer = tf.summary.FileWriter(os.path.join(logdir, 'test'), graph)
            sess = supervisor.PrepareSession(FLAGS.master)

            if vecs is not None:
                logging.info('init embeddings with external vectors...')
                sess.run(model.tree_embedder.lexicon_init, feed_dict={model.tree_embedder.lexicon_placeholder: vecs})

            def collect_values(step, loss, sim, sim_gold, train, print_out=False, emit=True):
                if train:
                    suffix = 'train'
                    writer = None
                    csv_writer = None
                else:
                    suffix = 'test '
                    writer = test_writer
                    csv_writer = test_result_writer

                p_r_train = pearsonr(sim, sim_gold)
                s_r_train = spearmanr(sim, sim_gold)

                if emit:
                    emit_values(supervisor, sess, step,
                                {'loss': loss,
                                 'pearson_r': p_r_train[0],
                                 'pearson_r_p': p_r_train[1],
                                 'spearman_r': s_r_train[0],
                                 'spearman_r_p': s_r_train[1],
                                 'sim_avg': np.average(sim)
                                 },
                                writer=writer,
                                csv_writer=csv_writer)
                if print_out:
                    logging.info(
                        (
                            'epoch=%d step=%d: loss_' + suffix + '=%f\tpearson_r_' + suffix + '=%f\tsim_avg=%f\tsim_gold_avg=%f\tsim_gold_var=%f') % (
                            epoch, step, loss, p_r_train[0], np.average(sim),
                            np.average(sim_gold), np.var(sim_gold)))

            with model.compiler.multiprocessing_pool():
                logging.info('create test data set ...')
                test_set = list(model.compiler.build_loom_inputs(test_iterator))
                logging.info('test data size: ' + str(len(test_set)))

                logging.info('create train data set ...')
                # data_train = list(train_iterator)
                train_set = model.compiler.build_loom_inputs(train_iterator)
                # logging.info('train data size: ' + str(len(data_train)))
                # dev_feed_dict = compiler.build_feed_dict(dev_trees)
                logging.info('training the model')
                test_losses = []
                for epoch, shuffled in enumerate(td.epochs(train_set, FLAGS.epochs, shuffle=False), 1):

                    def do_epoch(data_set, train=True):

                        sim_all = []
                        sim_all_gold = []
                        sim_all_jaccard = []
                        ids_all = []
                        loss_all = 0.0
                        step = None
                        #for batch in td.group_by_batches(data_set, FLAGS.batch_size if train else len(test_set)):
                        for batch in td.group_by_batches(data_set, FLAGS.batch_size):
                            train_feed_dict = {model.compiler.loom_input_tensor: batch}
                            if train:
                                _, step, batch_loss, sim, sim_gold, sim_jaccard, ids = sess.run(
                                    [model.train_op, model.global_step, model.loss, model.sim, model.gold_similarities, model.sim_jaccard, model.id],
                                    train_feed_dict)
                                #collect_values(step, batch_loss, sim, sim_gold, train=train)
                                # vars for print out: take only last result
                                #sim_all = [sim]
                                #sim_all_gold = [sim_gold]
                                # multiply with current batch size (can abbreviate from FLAGS.batch_size at last batch)
                                #loss_all = batch_loss * len(batch)
                            else:
                                step, batch_loss, sim, sim_gold, sim_jaccard, ids = sess.run(
                                    [model.global_step, model.loss, model.sim, model.gold_similarities, model.sim_jaccard, model.id],
                                    train_feed_dict)
                                # take average in test case
                            sim_all.append(sim)
                            sim_all_gold.append(sim_gold)
                            sim_all_jaccard.append(sim_jaccard)
                            ids_all.append(ids)
                            # multiply with current batch size (can abbreviate from FLAGS.batch_size at last batch)
                            loss_all += batch_loss * len(batch)
                        #print(np.concatenate(sim_all).tolist())
                        #print(np.concatenate(sim_all_gold).tolist())
                        sim_all_ = np.concatenate(sim_all)
                        sim_all_gold_ = np.concatenate(sim_all_gold)
                        sim_all_jaccard_ = np.concatenate(sim_all_jaccard)
                        ids_all_ = np.concatenate(ids_all)
                        #print(sim_all_.tolist())
                        #print(sim_all_gold_.tolist())
                        #print((sim_all_gold_ * 4.0 + 1.0).tolist())
                        #print(sim_all_jaccard_.tolist())
                        #print(ids_all_.tolist())
                        collect_values(step, loss_all / len(sim_all_), sim_all_, sim_all_gold_,
                                       train=train, print_out=True)#, emit=(not train))
                        return step, loss_all / len(sim_all_)

                    # test
                    _, loss_test = do_epoch(test_set, train=False)
                    loss_test = round(loss_test, 6) #100000000

                    # stop, if 5 different previous test losses are smaller than current loss
                    if loss_test not in test_losses:
                        test_losses.append(loss_test)
                    if max(test_losses) == loss_test and len(test_losses) > 5 and FLAGS.early_stop:
                        logging.info('last test losses: ' + str(test_losses))
                        break
                    else:
                        if len(test_losses) > 5:
                            del test_losses[0]

                        # train
                        step_train, _ = do_epoch(shuffled)

                        supervisor.saver.save(sess, checkpoint_path(logdir, step_train))

                        current_lexicon = sess.run(model.tree_embedder.lexicon_var)
                        current_lexicon.dump(os.path.join(logdir, 'model.vec'))
                        # test_result_csv.close()


if __name__ == '__main__':
    tf.app.run()
