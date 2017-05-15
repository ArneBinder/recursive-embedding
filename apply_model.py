from __future__ import print_function
import tensorflow as tf
import tensorflow_fold as td
import model_fold
import preprocessing
import spacy
import pickle
import pprint
import os

# Replication flags:
tf.flags.DEFINE_string('logdir', '/home/arne/tmp/tf/log',
                       'Directory in which to write event logs.')
tf.flags.DEFINE_string('model_path', '/home/arne/tmp/tf/log/model.ckpt-748',
                       'model file')
tf.flags.DEFINE_string('data_mapping_path', 'data/nlp/spacy/dict.mapping',
                       'model file')
tf.flags.DEFINE_string('master', '',
                       'Tensorflow master to use.')
tf.flags.DEFINE_integer('task', 0,
                        'Task ID of the replica running the training.')
tf.flags.DEFINE_integer('ps_tasks', 0,
                        'Number of PS tasks in the job.')
FLAGS = tf.flags.FLAGS

PROTO_PACKAGE_NAME = 'recursive_dependency_embedding'
PROTO_CLASS = 'SequenceNode'

def parse_iterator(sequences, parser, sentence_processor, data_maps):
    pp = pprint.PrettyPrinter(indent=2)
    for s in sequences:
        seq_tree = preprocessing.build_sequence_tree_from_str(s, sentence_processor, parser, data_maps)
        pp.pprint(seq_tree)
        yield seq_tree.SerializeToString()

lex_size = 1300000
embedding_dim = 300

def main(unused_argv):
    print('load spacy ...')
    nlp = spacy.load('en')
    nlp.pipeline = [nlp.tagger, nlp.parser]
    print('load data_mapping from: '+FLAGS.data_mapping_path + ' ...')
    data_maps = pickle.load(open(FLAGS.data_mapping_path, "rb"))

    with tf.Graph().as_default():
        with tf.device(tf.train.replica_device_setter(FLAGS.ps_tasks)):
            embed_w = tf.Variable(tf.constant(0.0, shape=[lex_size, embedding_dim]),
                                  trainable=True, name='embeddings')
            embedder = model_fold.SequenceTreeEmbedding(embed_w)
            tree_embeddings = embedder.tree_embeddings

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()

            # Later, launch the model, use the saver to restore variables from disk, and
            # do some work with the model.
            with tf.Session() as sess:
                # Restore variables from disk.
                print('restore model ...')
                saver.restore(sess, FLAGS.model_path)
                # Do some work with the model
                print('parse input ...')
                batch = list(parse_iterator(['Hallo.', 'Hallo!', 'Hallo?', 'Hallo'],
                                            nlp, preprocessing.process_sentence3, data_maps))
                fdict = embedder.build_feed_dict(batch)
                print('calculate tree embeddings ...')
                batch_embeddings, = sess.run(tree_embeddings, feed_dict=fdict)
                print(batch_embeddings)


if __name__ == '__main__':
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    td.proto_tools.map_proto_source_tree_path('', ROOT_DIR)
    td.proto_tools.import_proto_file('sequence_node.proto')
    tf.app.run()
