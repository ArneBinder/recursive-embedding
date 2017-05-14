from __future__ import print_function
import tensorflow as tf
import tensorflow_fold as td
import model_fold
import preprocessing
import spacy
import pickle
import pprint
import os
import sequence_node_sequence_pb2
import numpy as np

# Replication flags:
tf.flags.DEFINE_string('logdir', '/home/arne/tmp/tf/log',
                       'Directory in which to write event logs.')
tf.flags.DEFINE_string('model_path', '/home/arne/tmp/tf/log/model.ckpt-976',
                       'model file')
tf.flags.DEFINE_string('data_mapping_path', 'data/nlp/spacy/dict.mapping',
                       'model file')
tf.flags.DEFINE_string('train_dict_path', 'data/nlp/spacy/dict.vecs',
                       'Numpy array which is used to initialize the embedding vectors.')
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
    #pp = pprint.PrettyPrinter(indent=2)
    for (s, idx_correct) in sequences:
        seq_tree_seq = sequence_node_sequence_pb2.SequenceNodeSequence()
        seq_tree_seq.idx_correct = idx_correct
        for s2 in s:
            new_tree = seq_tree_seq.trees.add()
            preprocessing.build_sequence_tree_from_str(s2, sentence_processor, parser, data_maps, seq_tree=new_tree)
        #pp.pprint(seq_tree_seq)
        yield td.proto_tools.serialized_message_to_tree('recursive_dependency_embedding.SequenceNodeSequence', seq_tree_seq.SerializeToString())

lex_size = 1300000
embedding_dim = 300

def main(unused_argv):
    print('load spacy ...')
    nlp = spacy.load('en')
    nlp.pipeline = [nlp.tagger, nlp.parser]
    print('load data_mapping from: '+FLAGS.data_mapping_path + ' ...')
    data_maps = pickle.load(open(FLAGS.data_mapping_path, "rb"))

    print('load embeddings from: ' + FLAGS.train_dict_path + ' ...')
    embeddings_np = np.load(FLAGS.train_dict_path)

    embedding_dim = embeddings_np.shape[1]
    lex_size = 1300000
    # print('load mappings from: ' + data_fn + '.mapping ...')
    # mapping = pickle.load(open(data_fn + '.mapping', "rb"))
    assert lex_size >= embeddings_np.shape[0], 'len(embeddings) > lex_size. Can not cut the lexicon!'
    embeddings_padded = np.lib.pad(embeddings_np, ((0, lex_size - embeddings_np.shape[0]), (0, 0)), 'mean')

    with tf.Graph().as_default():
        with tf.device(tf.train.replica_device_setter(FLAGS.ps_tasks)):
            embed_w = tf.Variable(tf.constant(0.0, shape=[lex_size, embedding_dim]), trainable=True, name='embeddings')
            embedding_placeholder = tf.placeholder(tf.float32, [lex_size, embedding_dim])
            embedding_init = embed_w.assign(embedding_placeholder)

            trainer = model_fold.SequenceTreeEmbeddingSequence(embed_w)

            softmax_correct = trainer.softmax_correct
            loss = trainer.loss
            train_op = trainer.train_op
            global_step = trainer.global_step

            # collect important variables
            scoring_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=model_fold.DEFAULT_SCORING_SCOPE)

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()

            # Later, launch the model, use the saver to restore variables from disk, and
            # do some work with the model.
            with tf.Session() as sess:

                # exclude embedding, will be initialized afterwards
                init_vars = [v for v in tf.global_variables() if v != embed_w]
                tf.variables_initializer(init_vars).run()
                print('init embeddings with external vectors...')
                sess.run(embedding_init, feed_dict={embedding_placeholder: embeddings_padded})

                #tf.variables_initializer(tf.global_variables()).run()

                # Restore variables from disk.
                #print('restore model ...')
                #saver.restore(sess, FLAGS.model_path)
                # Do some work with the model
                print('parse input ...')
                #batch = list(parse_iterator([(['Hallo.', 'Hallo!', 'Hallo?', 'Hallo'], 0), (['Hallo.', 'Hallo!', 'Hallo?', 'Hallo'], 0)],
                #                            nlp, preprocessing.process_sentence3, data_maps))
                batch = list(parse_iterator(
                    [(['Hallo.'], 0)],
                    nlp, preprocessing.process_sentence3, data_maps))

                fdict = trainer.build_feed_dict(batch)
                print('calculate tree embeddings ...')
                #_, step, loss_v = sess.run([train_op, global_step, loss], feed_dict=fdict)
                _, step, loss_v, smax_correct = sess.run([train_op, global_step, loss, softmax_correct], feed_dict=fdict)
                #print(loss_v)
                print('step=%d: loss=%f' % (step, loss_v))
                print(smax_correct)


if __name__ == '__main__':
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    td.proto_tools.map_proto_source_tree_path('', ROOT_DIR)
    td.proto_tools.import_proto_file('sequence_node.proto')
    td.proto_tools.import_proto_file('sequence_node_sequence.proto')
    tf.app.run()
