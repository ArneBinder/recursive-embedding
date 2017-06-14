from __future__ import print_function

import json
import logging
import os
import sys
import time

import numpy as np
import spacy
import tensorflow as tf
import tensorflow_fold as td
from flask import Flask, request, send_file
from flask_cors import CORS
from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering
# from scipy import spatial # crashes!
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import kneighbors_graph

import constants
import corpus
import model_fold
import preprocessing
import visualize as vis

tf.flags.DEFINE_string('model_dir', '/home/arne/ML_local/tf/log',  # '/home/arne/tmp/tf/log',
                       'directory containing the model')
tf.flags.DEFINE_string('data_mapping',
                       '/media/arne/WIN/Users/Arne/ML/data/corpora/wikipedia/process_sentence8/WIKIPEDIA_articles10000_offset0',
                       # 'data/corpora/sick/process_sentence3/SICK.mapping', #'data/nlp/spacy/dict.mapping',
                       'Path to the "<data_mapping>.type" file (for visualization) and optional the ".vec" file, '
                       'see flag: load_embeddings')
tf.flags.DEFINE_boolean('load_embeddings', False,
                        'Load embeddings from numpy array located at "<data_mapping>.vec"')
tf.flags.DEFINE_string('sentence_processor', 'process_sentence7',  # 'process_sentence8',#'process_sentence3',
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
app = Flask(__name__, static_url_path='')
cors = CORS(app)


def form_data_to_dict(form_data):
    result = {}
    for k in form_data:
        try:
            result[k] = json.loads(form_data[k])
        except ValueError:
            result[k] = form_data[k]
    return result


def get_or_calc_embeddings(data):
    if data == "":
        params = form_data_to_dict(request.form)
    else:
        params = json.loads(data)

    if 'embeddings' in params:
        embeddings_str = params['embeddings']
        embeddings = np.array(embeddings_str)
    elif 'sequences' in params:
        sequences = params['sequences']
        tree_mode = FLAGS.tree_mode
        if 'tree_mode' in params:
            tree_mode = params['tree_mode']
            assert tree_mode in [None, 'sequence', 'aggregate', 'tree'], 'unknown tree_mode=' + tree_mode
            logging.info('use tree_mode=' + tree_mode)

        sentence_processor = getattr(preprocessing, FLAGS.sentence_processor)
        if 'sentence_processor' in params:
            sentence_processor = getattr(preprocessing, params['sentence_processor'])
            logging.info('use sentence_processor=' + sentence_processor.__name__)

        embeddings = get_embeddings(sequences=sequences, sentence_processor=sentence_processor, tree_mode=tree_mode)
    else:
        raise ValueError('no embeddings or sequences found in request')

    return embeddings, params


@app.route("/api/embed", methods=['POST'])
def embed():
    start = time.time()
    logging.info('Embeddings requested')
    embeddings, _ = get_or_calc_embeddings(request.data.decode("utf-8"))

    json_data = json.dumps({'embeddings': embeddings.tolist()})
    logging.info("Time spent handling the request: %f" % (time.time() - start))

    return json_data


@app.route("/api/distance", methods=['POST'])
def sim():
    start = time.time()
    logging.info('Distance requested')
    embeddings, _ = get_or_calc_embeddings(request.data.decode("utf-8"))

    result = pairwise_distances(embeddings, metric='cosine')  # spatial.distance.cosine(embeddings[0], embeddings[1])

    json_data = json.dumps({'distance': result.tolist()})
    logging.info("Time spent handling the request: %f" % (time.time() - start))

    return json_data


@app.route("/api/cluster", methods=['POST'])
def cluster():
    start = time.time()
    logging.info('Clusters requested')
    embeddings, _ = get_or_calc_embeddings(request.data.decode("utf-8"))

    labels, meta, best_idx = get_cluster_ids(embeddings=np.array(embeddings))
    json_data = json.dumps({'cluster_labels': labels, 'meta_data': meta, 'best_idx': best_idx})
    logging.info("Time spent handling the request: %f" % (time.time() - start))

    return json_data


@app.route("/api/visualize", methods=['POST'])
def visualize():
    start = time.time()

    data = request.data.decode("utf-8")
    if data == "":
        params = request.form
        sequences = json.loads(params['sequences'])
    else:
        params = json.loads(data)
        sequences = params['sequences']

    logging.info('Visualization requested for: ' + str(sequences))

    tree_mode = FLAGS.tree_mode
    if 'tree_mode' in params:
        tree_mode = params['tree_mode']
        assert tree_mode in [None, 'sequence', 'aggregate', 'tree'], 'unknown tree_mode=' + tree_mode
        logging.info('use tree_mode=' + tree_mode)

    sentence_processor = getattr(preprocessing, FLAGS.sentence_processor)
    if 'sentence_processor' in params:
        sentence_processor = getattr(preprocessing, params['sentence_processor'])
        logging.info('use sentence_processor=' + sentence_processor.__name__)

    parsed_datas = list(parse_iterator(sequences, nlp, sentence_processor, data_maps, tree_mode))
    vis.visualize_list(parsed_datas, types, file_name=vis.TEMP_FN)
    logging.info("Time spent handling the request: %f" % (time.time() - start))
    return send_file(vis.TEMP_FN)


def get_cluster_ids(embeddings):
    logging.info('get clusters ...')
    k_min = 3
    k_max = (embeddings.shape[0] / 3) + 2  # minimum viable clustering
    knn_min = 3
    knn_max = 12
    labels = []
    meta = []
    best_idx = -1
    best_score = -1
    idx = 0
    for k in range(k_min, k_max):
        for knn in range(knn_min, knn_max):
            connectivity = kneighbors_graph(embeddings, n_neighbors=knn, include_self=False)
            clusters = AgglomerativeClustering(n_clusters=k, linkage="ward", affinity='euclidean',
                                               connectivity=connectivity).fit(embeddings)
            sscore = metrics.silhouette_score(embeddings, clusters.labels_, metric='euclidean')
            # print "{:<3}\t{:<3}\t{:<6}".format(k, knn,  "%.4f" % sscore)
            labels.append(clusters.labels_.tolist())
            meta.append([sscore.astype(float), knn, k])
            if sscore > best_score:
                # record best silh
                best_score = sscore
                best_idx = idx
            idx += 1
    return labels, meta, best_idx


def get_embeddings(sequences, sentence_processor, tree_mode):
    logging.info('get embeddings ...')
    ##################################################
    # Tensorflow part
    ##################################################
    batch = [preprocessing.build_sequence_tree_from_parse(parsed_data).SerializeToString() for parsed_data in
             parse_iterator(sequences, nlp, sentence_processor, data_maps, tree_mode)] #list(seq_tree_iterator(sequences, nlp, sentence_processor, data_maps, tree_mode))
    fdict = embedder.build_feed_dict(batch)
    _tree_embeddings, = sess.run(tree_embeddings, feed_dict=fdict)
    ##################################################
    # END Tensorflow part
    ##################################################
    return np.array(_tree_embeddings)


def seq_tree_iterator(sequences, parser, sentence_processor, data_maps, tree_mode):
    # pp = pprint.PrettyPrinter(indent=2)
    for s in sequences:
        seq_tree = preprocessing.build_sequence_tree_from_str(s, sentence_processor, parser, data_maps,
                                                              tree_mode=tree_mode, expand_dict=False)
        # pp.pprint(seq_tree)
        yield seq_tree.SerializeToString()


def parse_iterator(sequences, parser, sentence_processor, data_maps, tree_mode):
    for s in sequences:
        seq_data, seq_parents, _ = preprocessing.read_data(preprocessing.identity_reader, sentence_processor, parser,
                                                           data_maps, args={'content': s}, tree_mode=tree_mode,
                                                           expand_dict=False)
        yield seq_data, seq_parents


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    td.proto_tools.map_proto_source_tree_path('', ROOT_DIR)
    td.proto_tools.import_proto_file('sequence_node.proto')

    # We retrieve our checkpoint fullpath
    checkpoint = tf.train.get_checkpoint_state(FLAGS.model_dir)
    input_checkpoint = checkpoint.model_checkpoint_path

    if FLAGS.load_embeddings:
        logging.info('load new embeddings from: '+FLAGS.data_mapping+'.vec ...')
        embeddings_np = np.load(FLAGS.data_mapping+'.vec')
        lex_size = embeddings_np.shape[0]
    else:
        reader = tf.train.NewCheckpointReader(input_checkpoint)
        logging.info('extract lexicon size from model: ' + input_checkpoint + ' ...')
        saved_shapes = reader.get_variable_to_shape_map()
        embed_shape = saved_shapes[model_fold.VAR_NAME_EMBEDDING]
        lex_size = embed_shape[0]

    logging.info('load spacy ...')
    nlp = spacy.load('en')
    nlp.pipeline = [nlp.tagger, nlp.entity, nlp.parser]

    logging.info('read types ...')
    types = corpus.read_types(FLAGS.data_mapping)
    logging.info('dict size: ' + str(len(types)))
    data_maps = corpus.mapping_from_list(types)

    with tf.Graph().as_default():
        with tf.device(tf.train.replica_device_setter(FLAGS.ps_tasks)):
            embed_w = tf.Variable(tf.constant(0.0, shape=[lex_size, model_fold.DIMENSION_EMBEDDINGS]),
                                  trainable=True, name='embeddings')
            embedding_placeholder = tf.placeholder(tf.float32, [lex_size, model_fold.DIMENSION_EMBEDDINGS])
            embedding_init = embed_w.assign(embedding_placeholder)
            embedder = model_fold.SequenceTreeEmbedding(embed_w)
            tree_embeddings = embedder.tree_embeddings

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()

            # Later, launch the model, use the saver to restore variables from disk, and
            # do some work with the model.
            sess = tf.Session()
            # Restore variables from disk.
            logging.info('restore model from: ' + input_checkpoint + '...')
            saver.restore(sess, input_checkpoint)

            if FLAGS.load_embeddings:
                print('init embeddings with external vectors ...')
                sess.run(embedding_init, feed_dict={embedding_placeholder: embeddings_np})

    logging.info('Starting the API')
    app.run()
