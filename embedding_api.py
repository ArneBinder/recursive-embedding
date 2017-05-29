from __future__ import print_function
import tensorflow as tf
import tensorflow_fold as td

import constants
import model_fold
import preprocessing
import spacy
import pickle
import os
import json, time
from flask import Flask, request
from flask_cors import CORS
import numpy as np
# from scipy import spatial # crashes!
from sklearn.metrics import pairwise_distances

tf.flags.DEFINE_string('model_dir', '/home/arne/ML_local/tf/log',  # '/home/arne/tmp/tf/log',
                       'directory containing the model')
tf.flags.DEFINE_string('data_mapping_path',
                       '/media/arne/WIN/Users/Arne/ML/data/corpora/wikipedia/process_sentence7/WIKIPEDIA_articles10000_maxdepth10.mapping',
                       # 'data/corpora/sick/process_sentence3/SICK.mapping', #'data/nlp/spacy/dict.mapping',
                       'model file')
tf.flags.DEFINE_string('sentence_processor', 'process_sentence2',  # 'process_sentence8',#'process_sentence3',
                       'Defines which NLP features are taken into the embedding trees.')
tf.flags.DEFINE_string('tree_mode',
                       None,
                       # 'aggregate',
                       #  'sequence',
                       'How to structure the tree. '
                       + '"sequence" -> parents point to next token, '
                       + '"aggregate" -> parents point to an added, artificial token (TERMINATOR) '
                         'in the end of the token sequence,'
                       + 'None -> use parsed dependency tree')

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


def parse_iterator(sequences, parser, sentence_processor, data_maps, tree_mode):
    # pp = pprint.PrettyPrinter(indent=2)
    for s in sequences:
        seq_tree = preprocessing.build_sequence_tree_from_str(s, sentence_processor, parser, data_maps,
                                                              tree_mode=tree_mode, expand_dict=False)
        # pp.pprint(seq_tree)
        yield seq_tree.SerializeToString()


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
    batch = list(parse_iterator(sequences, nlp, sentence_processor, data_maps, FLAGS.tree_mode))
    fdict = embedder.build_feed_dict(batch)
    _tree_embeddings, = sess.run(tree_embeddings, feed_dict=fdict)
    ##################################################
    # END Tensorflow part
    ##################################################

    json_data = json.dumps({'embeddings': np.array(_tree_embeddings).tolist()})
    print('Embeddings requested for: ' + str(sequences))
    print("Time spent handling the request: %f" % (time.time() - start))

    return json_data


@app.route("/api/distance", methods=['POST'])
def sim():
    start = time.time()

    data = request.data.decode("utf-8")
    if data == "":
        params = request.form
        sequences = json.loads(params['embeddings'])
    else:
        params = json.loads(data)
        sequences = params['embeddings']

    embeddings = np.array(sequences)
    result = pairwise_distances(embeddings, metric="cosine")  # spatial.distance.cosine(embeddings[0], embeddings[1])

    json_data = json.dumps({'distance': result.tolist()})
    print('Similarity requested')  # : '+str(sequences))
    print("Time spent handling the request: %f" % (time.time() - start))

    return json_data


if __name__ == '__main__':
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    td.proto_tools.map_proto_source_tree_path('', ROOT_DIR)
    td.proto_tools.import_proto_file('sequence_node.proto')

    # We retrieve our checkpoint fullpath
    checkpoint = tf.train.get_checkpoint_state(FLAGS.model_dir)
    input_checkpoint = checkpoint.model_checkpoint_path

    reader = tf.train.NewCheckpointReader(input_checkpoint)
    print('extract lexicon size from model: ' + input_checkpoint + ' ...')
    saved_shapes = reader.get_variable_to_shape_map()
    embed_shape = saved_shapes[model_fold.VAR_NAME_EMBEDDING]
    lex_size = embed_shape[0]

    print('load spacy ...')
    nlp = spacy.load('en')
    nlp.pipeline = [nlp.tagger, nlp.parser]
    print('load data_mapping from: ' + FLAGS.data_mapping_path + ' ...')
    data_maps = pickle.load(open(FLAGS.data_mapping_path, "rb"))

    sentence_processor = getattr(preprocessing, FLAGS.sentence_processor)

    with tf.Graph().as_default():
        with tf.device(tf.train.replica_device_setter(FLAGS.ps_tasks)):
            embed_w = tf.Variable(tf.constant(0.0, shape=[lex_size, model_fold.DIMENSION_EMBEDDINGS]),
                                  trainable=True, name='embeddings')
            embedder = model_fold.SequenceTreeEmbedding(embed_w)
            tree_embeddings = embedder.tree_embeddings

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()

            # Later, launch the model, use the saver to restore variables from disk, and
            # do some work with the model.
            sess = tf.Session()
            # Restore variables from disk.
            print('restore model from: ' + input_checkpoint + '...')
            saver.restore(sess, input_checkpoint)

    print('Starting the API')
    app.run()
