#from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import fnmatch
import ntpath
import os
# import google3
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
                             '/media/arne/WIN/Users/Arne/ML/data/corpora/sick/process_sentence2/SICK_CMaggregate',
                             'TF Record file containing the training dataset of sequence tuples.']}

tf.flags.DEFINE_string(
    'train_data_path',
    #'data/corpora/sick/process_sentence6/SICK.train',
    #'/media/arne/WIN/Users/Arne/ML/data/corpora/sick/process_sentence2/SICK_tree',
    '/media/arne/WIN/Users/Arne/ML/data/corpora/sick/process_sentence2/SICK_CMaggregate',
    'TF Record file containing the training dataset of sequence tuples.')
#tf.flags.DEFINE_string(
#    'test_data_path',
#    #'data/corpora/sick/process_sentence6/SICK.test',
#    '/media/arne/WIN/Users/Arne/ML/data/corpora/sick/process_sentence3/SICK_tree',
#    'TF Record file containing the test dataset of sequence tuples.')
#tf.flags.DEFINE_string(
#    'train_dict_path', 'data/nlp/spacy/dict.vecs',
#    'Numpy array which is used to initialize the embedding vectors.')
tf.flags.DEFINE_integer(
    'batch_size', 250, 'How many samples to read per batch.')
    #'batch_size', 2, 'How many samples to read per batch.')
tf.flags.DEFINE_integer('epochs',
                        1000,
                        'The number of epochs.')
#tf.flags.DEFINE_integer( # use size of embeddings loaded from numpy array
#    'embedding_length', 300,
#    'How long to make the embedding vectors.')
tf.flags.DEFINE_integer(
    #'max_steps', 1000000,
    'max_steps', 1000,
    'The maximum number of batches to run the trainer for.')
tf.flags.DEFINE_integer(
    'test_data_size', 1000,
    'The size of the test set.')
tf.flags.DEFINE_string('run_description',
                       None,
                       #'dataPs2_embeddingsUntrainable_simCosine_modelDefault',
                       'label extension for the name of the run when visualizing with tensorboard')
tf.flags.DEFINE_integer(
    'test_file_index', -1,
    'Which file of the train data files should be used as test data.')
tf.flags.DEFINE_boolean(
    'embeddings_trainable',
    True,
    'Iff enabled, fine tune the embeddings.'
)
tf.flags.DEFINE_boolean(
    'normalize',
    True,
    'Iff enabled, normalize sequence embeddings before application of sim_measure.'
)
tf.flags.DEFINE_string('sim_measure',
                       #'sim_layer',
                       'sim_cosine',
                       'similarity measure implementation (tensorflow) from model_fold for similarity score calculation. Currently implemented:'
                       '"sim_cosine" -> cosine'
                       '"sim_layer" -> similarity measure defined in [Tai, Socher 2015]')
tf.flags.DEFINE_string('tree_embedder',
                       #'TreeEmbedding_TreeLSTM',
                       'TreeEmbedding_AVG_children',
                       'Tree embedder implementation from model_fold that produces a tensorflow fold block on calling which accepts a sequence tree and produces an embedding. '
                       'Currently implemented:'
                       '"TreeEmbedding_TreeLSTM" -> '
                       '"TreeEmbedding_HTU" -> '
                       '"TreeEmbedding_HTU_simplified" -> '
                       '"TreeEmbedding_AVG_children" -> '
                       '"TreeEmbedding_AVG_children_2levels" -> '
                       )

# Replication flags:
tf.flags.DEFINE_string('logdir',
                       #'/home/arne/ML_local/tf/supervised/log/dataPs2aggregate_embeddingsUntrainable_simLayer_modelTreelstm_normalizeTrue_batchsize250',
                       '/home/arne/ML_local/tf/supervised/log/dataPs2aggregate_embeddingsTrainable_simCosine_modelAvgchildren_normalizeTrue_batchsize250',
                       'Directory in which to write event logs.')
tf.flags.DEFINE_string('master', '',
                       'Tensorflow master to use.')
tf.flags.DEFINE_integer('task', 0,
                        'Task ID of the replica running the training.')
tf.flags.DEFINE_integer('ps_tasks', 0,
                        'Number of PS tasks in the job.')
FLAGS = tf.flags.FLAGS


# Find the root of the bazel repository.
def source_root():
  root = __file__
  #print('root1: ' + str(root))
  #for _ in xrange(5):
  for _ in xrange(1):
    root = os.path.dirname(root)
  #print('root: '+str(root))
  return root


#CALCULATOR_SOURCE_ROOT = source_root()
#CALCULATOR_PROTO_FILE = ('tensorflow_fold/loom/'
#                         'calculator_example/calculator.proto')
#CALCULATOR_PROTO_FILE = ('calculator_new.proto')
#CALCULATOR_EXPRESSION_PROTO = ('tensorflow_fold.loom.'
#                               'calculator_example.CalculatorExpression')
#CALCULATOR_EXPRESSION_PROTO = ('my_example.CalculatorExpression')

PROTO_PACKAGE_NAME = 'recursive_dependency_embedding'


# Make sure serialized_message_to_tree can find the calculator example proto:
td.proto_tools.map_proto_source_tree_path('', source_root())
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


def emit_values(supervisor, session, step, values, writer=None):
    summary = tf.Summary()
    for name, value in six.iteritems(values):
        summary_value = summary.value.add()
        summary_value.tag = name
        summary_value.simple_value = float(value)
    if writer:
        writer.add_summary(summary, step)
    else:
        supervisor.summary_computed(session, summary, global_step=step)


def normed_loss(batch_loss, batch_size):
    return math.sqrt(batch_loss / batch_size)


def checkpoint_path(step):
    return os.path.join(FLAGS.logdir, 'model.ckpt-'+str(step))


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
    vecs, types = corpus.create_or_read_dict(FLAGS.train_data_path)
    lex_size = vecs.shape[0]
    #embedding_dim = vecs.shape[1]

    checkpoint_fn = tf.train.latest_checkpoint(FLAGS.logdir)
    if checkpoint_fn:
        logging.info('read lex_size from model ...')
        reader = tf.train.NewCheckpointReader(checkpoint_fn)
        saved_shapes = reader.get_variable_to_shape_map()
        embed_shape = saved_shapes[model_fold.VAR_NAME_EMBEDDING]
        lex_size = embed_shape[0]

    logging.info('lex_size = '+str(lex_size))

    #print('load embeddings from: '+FLAGS.train_dict_path + ' ...')
    #embeddings_np = np.load(FLAGS.train_dict_path)

    #embedding_dim = embeddings_np.shape[1]
    #lex_size = 1300000
    #print('load mappings from: ' + data_fn + '.mapping ...')
    #mapping = pickle.load(open(data_fn + '.mapping', "rb"))
    #assert lex_size >= embeddings_np.shape[0], 'len(embeddings) > lex_size. Can not cut the lexicon!'

    #embeddings_padded = np.lib.pad(embeddings_np, ((0, lex_size - embeddings_np.shape[0]), (0, 0)), 'mean')
    #embeddings_padded = np.ones(shape=(1300000, 300)) #np.lib.pad(embeddings_np, ((0, lex_size - embeddings_np.shape[0]), (0, 0)), 'mean')

    #print('embeddings_np.shape: '+str(embeddings_np.shape))
    #print('embeddings_padded.shape: ' + str(embeddings_padded.shape))

    print('create tensorflow graph ...')
    with tf.Graph().as_default() as graph:
        with tf.device(tf.train.replica_device_setter(FLAGS.ps_tasks)):
            embed_w = tf.Variable(tf.constant(0.0, shape=[lex_size, model_fold.DIMENSION_EMBEDDINGS]),
                                  trainable=FLAGS.embeddings_trainable, name=model_fold.VAR_NAME_EMBEDDING)

            embedding_placeholder = tf.placeholder(tf.float32, [lex_size, model_fold.DIMENSION_EMBEDDINGS])
            embedding_init = embed_w.assign(embedding_placeholder)

            # Build the graph.
            #aggregator_ordered_scope_name = 'aggregator_ordered'
            sim_measure = getattr(model_fold, FLAGS.sim_measure)
            tree_embedder = getattr(model_fold, FLAGS.tree_embedder)
            model = model_fold.SimilaritySequenceTreeTupleModel(embed_w, tree_embedder=tree_embedder, normalize=FLAGS.normalize, sim_measure=sim_measure) #, aggregator_ordered_scope_name)
            loss = model.loss
            #sim_cosine = model.cosine_similarities
            sim_gold = model.gold_similarities
            sim = model.sim
            mse = model.mse
            compiler = model.compiler

            #accuracy = model.accuracy
            train_op = model.train_op
            global_step = model.global_step

            summary_path = os.path.join(FLAGS.logdir, '')
            if FLAGS.run_description:
                summary_path += FLAGS.run_description + '_'

            test_writer = tf.summary.FileWriter(summary_path + 'test', graph)

            # collect important variables
            #tree_embedder_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=model_fold.DEFAULT_SCOPE_TREE_EMBEDDER)
            #save_vars = tree_embedder_vars + [embed_w, global_step]
            #my_saver = tf.train.Saver(save_vars)

            #missing_vars = [item for item in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if item not in save_vars]
            #init_missing = tf.variables_initializer(missing_vars)

            #embeddings_1 = model.tree_embeddings_1
            #embeddings_2 = model.tree_embeddings_2

            #cosine_similarities = model.cosine_similarities

            # Set up the supervisor.
            supervisor = tf.train.Supervisor(
                #saver=None,# my_saver,
                logdir=FLAGS.logdir,
                is_chief=(FLAGS.task == 0),
                save_summaries_secs=10,
                save_model_secs=300,
                summary_writer=tf.summary.FileWriter(summary_path + 'train', graph))
            sess = supervisor.PrepareSession(FLAGS.master)

            if checkpoint_fn is None:
                print('init embeddings with external vectors...')
                sess.run(embedding_init, feed_dict={embedding_placeholder: vecs})
                #sess.run(init_missing)
                step = 0
            else:
                step = reader.get_tensor(model_fold.VAR_NAME_GLOBAL_STEP)
                #my_saver.restore(sess, checkpoint_fn)

            # prepare test set
            #test_size = FLAGS.test_data_size
            batch_test = list(test_iterator) #[next(test_iterator) for _ in xrange(test_size)]
            fdict_test = model.build_feed_dict(batch_test)
            #step = 0

            with compiler.multiprocessing_pool():
                print('training the model')
                train_set = compiler.build_loom_inputs(list(train_iterator))
                # dev_feed_dict = compiler.build_feed_dict(dev_trees)
                # dev_hits_best = 0.0
                for epoch, shuffled in enumerate(td.epochs(train_set, FLAGS.epochs), 1):

                    # test
                    loss_test, sim_test, sim_gold_test = sess.run([loss, sim, sim_gold], feed_dict=fdict_test)
                    p_r_test = pearsonr(sim_gold_test, sim_test)
                    #loss_test_normed = loss_test
                    emit_values(supervisor, sess, step,
                                {'mse': loss_test,  # to stay comparable with previous runs
                                 'loss': loss_test,
                                 'pearson_r': p_r_test[0],
                                 'pearson_r_p': p_r_test[1],
                                 'sim_avg': np.average(sim_test)},
                                writer=test_writer)
                    print('epoch=%d step=%d: loss_test=%f pearson_r_test=%f' % (
                        epoch, step, loss_test, p_r_test[0]))

                    # train
                    #train_loss = 0.0
                    batch_step = 0
                    for batch in td.group_by_batches(shuffled, FLAGS.batch_size):
                        train_feed_dict = {compiler.loom_input_tensor: batch}
                        _, step, batch_loss, sim_train, sim_gold_train = sess.run([train_op, global_step, loss, sim, sim_gold], train_feed_dict)
                        #train_loss += batch_loss
                        #loss_batch_normed = batch_loss
                        p_r_train = pearsonr(sim_gold_train, sim_train)

                        emit_values(supervisor, sess, step,
                                    {'mse': batch_loss,  # to stay comparable with previous runs
                                     'loss': batch_loss,
                                     'pearson_r': p_r_train[0],
                                     'pearson_r_p': p_r_train[1],
                                     'sim_avg': np.average(sim_train)
                                     })
                        batch_step += 1
                        #print(sim_train.tolist())

                        print('epoch=%d step=%d: loss_train=%f pearson_r_train=%f sim_avg=%f sim_gold_avg=%f' % (
                            epoch, step, batch_loss, p_r_train[0], np.average(sim_train),
                            np.average(sim_gold_train)))
                    supervisor.saver.save(sess, checkpoint_path(step))

if __name__ == '__main__':
    tf.app.run()
