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
import math

flags = {'train_data_path': [tf.flags.DEFINE_string,
                             # '/media/arne/WIN/Users/Arne/ML/data/corpora/ppdb/process_sentence3_ns1/PPDB_CMaggregate',
                             # '/media/arne/WIN/Users/Arne/ML/data/corpora/sick/process_sentence2/SICK_CMaggregate',
                             #'/media/arne/WIN/Users/Arne/ML/data/corpora/sick/process_sentence3/SICK_tt_CMaggregate',
                             # '/media/arne/WIN/Users/Arne/ML/data/corpora/sick/process_sentence2/SICK_CMsequence_ICMtree',
                             # '/media/arne/WIN/Users/Arne/ML/data/corpora/sick/process_sentence3/SICK_CMsequence_ICMtree',
                             '/media/arne/WIN/Users/Arne/ML/data/corpora/debate_cluster/process_sentence3/HASAN_CMaggregate',
                             'TF Record file containing the training dataset of sequence tuples.'],
         'old_logdir': [tf.flags.DEFINE_string,
                        None,
                        # '/home/arne/ML_local/tf/supervised/log/batchsize100_embeddingfcactivationNONE_embeddingstrainableTRUE_learningrate0.001_optimizerADADELTAOPTIMIZER_outputfcactivationNONE_simmeasureSIMCOSINE_testfileindex-1_traindatapathPROCESSSENTENCE3NS1PPDBCMAGGREGATE_treeembedderTREEEMBEDDINGFLATLSTM50',
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
         'embeddings_trainable': [tf.flags.DEFINE_boolean,
                                  True,
                                  # True,
                                  'Iff enabled, fine tune the embeddings.'],
         'sim_measure': [tf.flags.DEFINE_string,
                         'sim_cosine',
                         'similarity measure implementation (tensorflow) from model_fold for similarity score calculation. Currently implemented:'
                         '"sim_cosine" -> cosine'
                         '"sim_layer" -> similarity measure similar to the one defined in [Tai, Socher 2015]'
                         '"sim_manhattan" -> l1-norm based similarity measure (taken from MaLSTM) [Mueller et al., 2016]'],
         'tree_embedder': [tf.flags.DEFINE_string,
                           'TreeEmbedding_FLAT_AVG',
                           'Tree embedder implementation from model_fold that produces a tensorflow fold block on calling which accepts a sequence tree and produces an embedding. '
                           'Currently implemented:'
                           '"TreeEmbedding_TREE_LSTM" -> '
                           '"TreeEmbedding_HTU_GRU" -> '
                           '"TreeEmbedding_HTU_GRU_simplified" -> '
                           '"TreeEmbedding_FLAT_AVG" -> '
                           '"TreeEmbedding_FLAT_AVG_2levels" -> '
                           '"TreeEmbedding_FLAT_LSTM" -> '
                           '"TreeEmbedding_FLAT_LSTM50" -> '
                           '"TreeEmbedding_FLAT_LSTM_2levels" -> '],
         'embedding_fc_activation': [tf.flags.DEFINE_string,
                                     None,
                                     # 'tanh',
                                     'If not None, apply a fully connected layer with this activation function before composition',
                                     None],
         'output_fc_activation': [tf.flags.DEFINE_string,
                                  None,
                                  # 'tanh',
                                  'If not None, apply a fully connected layer with this activation function after composition',
                                  None],
         'learning_rate': [tf.flags.DEFINE_float,
                           0.001,
                           # 'tanh',
                           'learning rate'],
         'optimizer': [tf.flags.DEFINE_string,
                       'AdadeltaOptimizer',
                       'optimizer'],
         'logdir': [tf.flags.DEFINE_string,
                    # '/home/arne/ML_local/tf/supervised/log/dataPs2aggregate_embeddingsUntrainable_simLayer_modelTreelstm_normalizeTrue_batchsize250',
                    # '/home/arne/ML_local/tf/supervised/log/dataPs2aggregate_embeddingsTrainable_simLayer_modelAvgchildren_normalizeTrue_batchsize250',
                    '/home/arne/ML_local/tf/supervised/log',
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
    count = 0
    while True:
        if multiple_epochs:
            logging.debug('start epoche: ' + str(count))
        for table_path in table_paths:
            for v in tf.python_io.tf_record_iterator(table_path):
                yield td.proto_tools.serialized_message_to_tree(PROTO_PACKAGE_NAME + '.' + message_type.__name__, v)
        count += 1
        if not multiple_epochs:
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
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
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
            flag_value = ''.join(flag_value.split(os.sep)[-2:])
            run_desc.append(flag_name.lower() + flag_value.upper())
        # if a short version is set, use it. if it is set to None, add this flag not to the run_descriptions
        elif flags[flag][2]:
            run_desc.append(flag.replace('_', '').lower() + str(flags[flag][2]).replace('_', '').upper())

    flags['run_description'] = ['_'.join(run_desc), 'short string description of the current run']
    logging.info('short run description: ' + flags['run_description'][0])

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
        embed_shape = saved_shapes[model_fold.VAR_NAME_EMBEDDING]
        lex_size = embed_shape[0]
        # create test result writer
        test_result_writer = csv_test_writer(os.path.join(logdir, 'test'), mode='a')
    else:
        vecs, types = corpus.create_or_read_dict(FLAGS.train_data_path)
        if FLAGS.old_logdir:
            old_checkpoint_fn = tf.train.latest_checkpoint(FLAGS.old_logdir)
            reader_old = tf.train.NewCheckpointReader(old_checkpoint_fn)
            vecs_old = reader_old.get_tensor(model_fold.VAR_NAME_EMBEDDING)
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

    embedding_fc_activation = FLAGS.embedding_fc_activation
    if embedding_fc_activation:
        embedding_fc_activation = getattr(tf.nn, embedding_fc_activation)
    output_fc_activation = FLAGS.output_fc_activation
    if output_fc_activation:
        output_fc_activation = getattr(tf.nn, output_fc_activation)
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
                                                                learning_rate=FLAGS.learning_rate,
                                                                optimizer=optimizer,
                                                                sim_measure=sim_measure,
                                                                embeddings_trainable=FLAGS.embeddings_trainable,
                                                                embedding_fc_activation=embedding_fc_activation,
                                                                output_fc_activation=output_fc_activation)

            # Set up the supervisor.
            supervisor = tf.train.Supervisor(
                # saver=None,# my_saver,
                logdir=logdir,
                is_chief=(FLAGS.task == 0),
                save_summaries_secs=10,
                save_model_secs=300,
                summary_writer=tf.summary.FileWriter(os.path.join(logdir, 'train'), graph))
            test_writer = tf.summary.FileWriter(os.path.join(logdir, 'test'), graph)
            sess = supervisor.PrepareSession(FLAGS.master)

            if old_checkpoint_fn is not None:
                logging.info('restore from old_checkpoint (except embeddings and step): ' + old_checkpoint_fn + ' ...')
                embedding_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=model_fold.VAR_NAME_EMBEDDING)
                restore_vars = [item for item in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if
                                item not in [model.global_step] + embedding_vars]
                restore_saver = tf.train.Saver(restore_vars)
                restore_saver.restore(sess, old_checkpoint_fn)

            if vecs is not None:
                logging.info('init embeddings with external vectors...')
                sess.run(model.embedding_init, feed_dict={model.embedding_placeholder: vecs})

            # prepare test set
            logging.info('create test data set ...')
            batch_test = list(test_iterator)  # [next(test_iterator) for _ in xrange(test_size)]
            fdict_test = model.build_feed_dict(batch_test)
            logging.info('test data size: ' + str(len(batch_test)))

            with model.compiler.multiprocessing_pool():
                logging.info('create train data set ...')
                data_train = list(train_iterator)
                train_set = model.compiler.build_loom_inputs(data_train)
                logging.info('train data size: ' + str(len(data_train)))
                # dev_feed_dict = compiler.build_feed_dict(dev_trees)
                # dev_hits_best = 0.0
                logging.info('training the model')
                for epoch, shuffled in enumerate(td.epochs(train_set, FLAGS.epochs), 1):

                    # test
                    step, loss_test, sim_test, sim_gold_test, mse_comp_test = sess.run(
                        [model.global_step, model.loss, model.sim, model.gold_similarities, model.mse_compare],
                        feed_dict=fdict_test)
                    p_r_test = pearsonr(sim_gold_test, sim_test)
                    emit_values(supervisor, sess, step,
                                {  # 'mse': loss_test,  # to stay comparable with previous runs
                                    'loss': loss_test,
                                    'pearson_r': p_r_test[0],
                                    'pearson_r_p': p_r_test[1],
                                    'sim_avg': np.average(sim_test),
                                    'mse_compare': np.average(mse_comp_test)},
                                writer=test_writer,
                                csv_writer=test_result_writer)
                    print('epoch=%d step=%d: loss_test =%f\tpearson_r_test =%f\tsim_avg=%f\tsim_gold_avg=%f\tsim_gold_var=%f\tTEST' % (
                        epoch, step, loss_test, p_r_test[0], np.average(sim_test), np.average(sim_gold_test), np.var(sim_gold_test)))

                    # print(es1[0].tolist())
                    # print(es2[0].tolist())
                    # cos_ = pairwise_distances([es1[0], es2[0]], metric='cosine')
                    # print(cos_)
                    # print(sim_test[0])

                    # train
                    batch_step = 0
                    show = True
                    for batch in td.group_by_batches(shuffled, FLAGS.batch_size):
                        train_feed_dict = {model.compiler.loom_input_tensor: batch}
                        _, step, batch_loss, sim_train, sim_gold_train = sess.run(
                            [model.train_op, model.global_step, model.loss, model.sim, model.gold_similarities],
                            train_feed_dict)
                        # train_loss += batch_loss
                        p_r_train = pearsonr(sim_gold_train, sim_train)

                        emit_values(supervisor, sess, step,
                                    {  # 'mse': batch_loss,  # to stay comparable with previous runs
                                        'loss': batch_loss,
                                        'pearson_r': p_r_train[0],
                                        'pearson_r_p': p_r_train[1],
                                        'sim_avg': np.average(sim_train)
                                    })
                        batch_step += 1
                        if show:  # or True:
                            print('epoch=%d step=%d: loss_train=%f\tpearson_r_train=%f\tsim_avg=%f\tsim_gold_avg=%f\tsim_gold_var=%f' % (
                                epoch, step, batch_loss, p_r_train[0], np.average(sim_train),
                                np.average(sim_gold_train), np.var(sim_gold_train)))
                            show = False
                    supervisor.saver.save(sess, checkpoint_path(logdir, step))
                    # test_result_csv.close()


if __name__ == '__main__':
    tf.app.run()
