# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Runs the trainer for the calculator example.

This file is a minor modification to loom/calculator_example/train.py.
To run, first make the data set:

  ./tensorflow_fold/loom/calculator_example/make_dataset \
    --output_path=DIR/calc_data.dat

Then run the trainer:

  ./tensorflow_fold/blocks/examples/calculator/train \
    --train_data_path=DIR/calc_data.dat
"""
#from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
# import google3
import six
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import model_fold
import similarity_tree_tuple_pb2
import tensorflow_fold as td
import numpy as np

tf.flags.DEFINE_string(
    'train_data_path', 'data/corpora/sick/process_sentence3/SICK_train',
    'TF Record file containing the training dataset of sequence tuples.')
tf.flags.DEFINE_string(
    'test_data_path', 'data/corpora/sick/process_sentence3/SICK_test',
    'TF Record file containing the test dataset of sequence tuples.')
tf.flags.DEFINE_string(
    'train_dict_path', 'data/nlp/spacy/dict.vecs',
    'Numpy array which is used to initialize the embedding vectors.')
tf.flags.DEFINE_integer(
    'batch_size', 250, 'How many samples to read per batch.')
    #'batch_size', 2, 'How many samples to read per batch.')
#tf.flags.DEFINE_integer( # use size of embeddings loaded from numpy array
#    'embedding_length', 300,
#    'How long to make the embedding vectors.')
tf.flags.DEFINE_integer(
    #'max_steps', 1000000,
    'max_steps', 100,
    'The maximum number of batches to run the trainer for.')
tf.flags.DEFINE_integer(
    'test_data_size', 10000,
    'The size of the test set.')

# Replication flags:
tf.flags.DEFINE_string('logdir', 'data/log',
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


# Make sure serialized_message_to_tree can find the calculator example proto:
td.proto_tools.map_proto_source_tree_path('', source_root())
td.proto_tools.import_proto_file('similarity_tree_tuple.proto')


def iterate_over_tf_record_protos(table_path, unused_message_type):
  while True:
    for v in tf.python_io.tf_record_iterator(table_path):
      yield td.proto_tools.serialized_message_to_tree(
          'recursive_dependency_embedding.SimilarityTreeTuple', v)


def emit_values(supervisor, session, step, values):
  summary = tf.Summary()
  for name, value in six.iteritems(values):
    summary_value = summary.value.add()
    summary_value.tag = name
    summary_value.simple_value = float(value)
  supervisor.summary_computed(session, summary, global_step=step)


def main(unused_argv):
    train_data_fn = FLAGS.train_data_path
    print('use training data: '+train_data_fn)
    train_iterator = iterate_over_tf_record_protos(
        train_data_fn, similarity_tree_tuple_pb2.SimilarityTreeTuple)

    test_data_fn = FLAGS.train_data_path
    print('use test data: '+test_data_fn)
    test_iterator = iterate_over_tf_record_protos(
        test_data_fn, similarity_tree_tuple_pb2.SimilarityTreeTuple)

    # DEBUG
    print('load embeddings from: '+FLAGS.train_dict_path + ' ...')
    embeddings_np = np.load(FLAGS.train_dict_path)

    embedding_dim = embeddings_np.shape[1]
    lex_size = 1300000
    #print('load mappings from: ' + data_fn + '.mapping ...')
    #mapping = pickle.load(open(data_fn + '.mapping', "rb"))
    assert lex_size >= embeddings_np.shape[0], 'len(embeddings) > lex_size. Can not cut the lexicon!'

    embeddings_padded = np.lib.pad(embeddings_np, ((0, lex_size - embeddings_np.shape[0]), (0, 0)), 'mean')
    #embeddings_padded = np.ones(shape=(1300000, 300)) #np.lib.pad(embeddings_np, ((0, lex_size - embeddings_np.shape[0]), (0, 0)), 'mean')

    print('embeddings_np.shape: '+str(embeddings_np.shape))
    print('embeddings_padded.shape: ' + str(embeddings_padded.shape))

    print('create tensorflow graph ...')
    with tf.Graph().as_default():
        with tf.device(tf.train.replica_device_setter(FLAGS.ps_tasks)):
            embed_w = tf.Variable(tf.constant(0.0, shape=[lex_size, embedding_dim]),
                            trainable=True, name="embed_w")

            embedding_placeholder = tf.placeholder(tf.float32, [lex_size, embedding_dim])
            embedding_init = embed_w.assign(embedding_placeholder)

            # Build the graph.
            classifier = model_fold.SequenceTupleModel(lex_size, embedding_dim, embed_w)
            loss = classifier.loss
            #accuracy = classifier.accuracy
            train_op = classifier.train_op
            global_step = classifier.global_step

            embeddings_1 = classifier.embeddings_1
            embeddings_2 = classifier.embeddings_2

            cosine_similarities = classifier.cosine_similarities

            # Set up the supervisor.
            supervisor = tf.train.Supervisor(
                logdir=FLAGS.logdir,
                is_chief=(FLAGS.task == 0),
                save_summaries_secs=10,
                save_model_secs=300)
            sess = supervisor.PrepareSession(FLAGS.master)
            sess.run(embedding_init, feed_dict={embedding_placeholder: embeddings_padded})

            # prepare test set
            test_size = FLAGS.test_data_size
            batch_test = [next(test_iterator) for _ in xrange(test_size)]
            fdict_test = classifier.build_feed_dict(batch_test)

            # Run the trainer.
            for _ in xrange(FLAGS.max_steps):
                batch = [next(train_iterator) for _ in xrange(FLAGS.batch_size)]
                fdict = classifier.build_feed_dict(batch)

                _, step, loss_v, embeds_1, embeds_2, sims = sess.run(
                    [train_op, global_step, loss, embeddings_1, embeddings_2, cosine_similarities],
                    feed_dict=fdict)

                emit_values(supervisor, sess, step,
                            {'loss_train': loss_v / FLAGS.batch_size})

                if step % 10 == 0:
                    (loss_test,) = sess.run([loss], feed_dict=fdict_test)
                    emit_values(supervisor, sess, step,
                            {'loss_test': loss_test / test_size})
                    print('step=%d: loss=%f loss_train=%f' % (step, loss_v / FLAGS.batch_size, loss_test / test_size))
                else:
                    print('step=%d: loss=%f' % (step, loss_v / FLAGS.batch_size))

if __name__ == '__main__':
  tf.app.run()
