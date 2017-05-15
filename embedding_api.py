from __future__ import print_function
import tensorflow as tf
import tensorflow_fold as td
import model_fold
import preprocessing
import spacy
import pickle
import pprint
import os
import json, time
from flask import Flask, request
from flask_cors import CORS
import numpy as np

# Replication flags:
#tf.flags.DEFINE_string('logdir', '/home/arne/tmp/tf/log',
#                       'Directory in which to write event logs.')
tf.flags.DEFINE_string('model_path', '/home/arne/tmp/tf/log/model.ckpt-748',
                       'model file')
tf.flags.DEFINE_string('data_mapping_path', 'data/nlp/spacy/dict.mapping',
                       'model file')
#tf.flags.DEFINE_string('master', '',
#                       'Tensorflow master to use.')
#tf.flags.DEFINE_integer('task', 0,
#                        'Task ID of the replica running the training.')
tf.flags.DEFINE_integer('ps_tasks', 0,
                        'Number of PS tasks in the job.')
FLAGS = tf.flags.FLAGS

PROTO_PACKAGE_NAME = 'recursive_dependency_embedding'
PROTO_CLASS = 'SequenceNode'

##################################################
# API part
##################################################
app = Flask(__name__)
cors = CORS(app)


@app.route("/api/embed", methods=['POST'])
def embed():
    start = time.time()

    data = request.data.decode("utf-8")
    if data == "":
        params = request.form
        sequences = json.loads(params['sequences'])
    else:
        params = json.loads(data)
        sequences = params['sequences']

    ##################################################
    # Tensorflow part
    ##################################################
    batch = list(parse_iterator(sequences, nlp, preprocessing.process_sentence3, data_maps))
    fdict = embedder.build_feed_dict(batch)
    _tree_embeddings, = sess.run(tree_embeddings, feed_dict=fdict)
    ##################################################
    # END Tensorflow part
    ##################################################

    json_data = json.dumps({'embeddings':  np.array(_tree_embeddings).tolist()})
    print('Embeddings requested for: '+str(sequences))
    print("Time spent handling the request: %f" % (time.time() - start))

    return json_data


def parse_iterator(sequences, parser, sentence_processor, data_maps):
    #pp = pprint.PrettyPrinter(indent=2)
    for s in sequences:
        seq_tree = preprocessing.build_sequence_tree_from_str(s, sentence_processor, parser, data_maps, expand_dict=False)
        #pp.pprint(seq_tree)
        yield seq_tree.SerializeToString()

lex_size = 1300000
embedding_dim = 300


if __name__ == '__main__':
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    td.proto_tools.map_proto_source_tree_path('', ROOT_DIR)
    td.proto_tools.import_proto_file('sequence_node.proto')

    print('load spacy ...')
    nlp = spacy.load('en')
    nlp.pipeline = [nlp.tagger, nlp.parser]
    print('load data_mapping from: ' + FLAGS.data_mapping_path + ' ...')
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
            sess = tf.Session()
            # Restore variables from disk.
            print('restore model ...')
            saver.restore(sess, FLAGS.model_path)

    print('Starting the API')
    app.run()

