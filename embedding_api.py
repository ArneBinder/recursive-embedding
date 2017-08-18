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
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering
# from scipy import spatial # crashes!
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import normalize
#from google.protobuf.json_format import MessageToJson

import constants
import corpus
import model_fold
import preprocessing
import visualize as vis

tf.flags.DEFINE_string('model_dir',
                       '/home/arne/ML_local/tf/supervised/log/PRETRAINED/batchsize100_embeddingstrainableTRUE_learningrate0.001_optimizerADADELTAOPTIMIZER_simmeasureSIMCOSINE_statesize50_testfileindex1_traindatapathPROCESSSENTENCE3HASANCMSEQUENCEICMTREENEGSAMPLES1_treeembedderTREEEMBEDDINGHTUGRU',
                       #/model.ckpt-122800',
                       #'/home/arne/ML_local/tf/supervised/log/applyembeddingfcTRUE_batchsize100_embeddingstrainableTRUE_normalizeTRUE_simmeasureSIMCOSINE_testfileindex-1_traindatapathPROCESSSENTENCE3SICKCMAGGREGATE_treeembedderTREEEMBEDDINGFLATLSTM',
                       #'/home/arne/ML_local/tf/log/final_model',
                       'Directory containing the model and a checkpoint file or the direct path to a '
                       'model (without extension) and the model.type file containing the string dict.')
tf.flags.DEFINE_string('external_embeddings',
                       #'/media/arne/WIN/Users/Arne/ML/data/corpora/wikipedia/process_sentence8_/WIKIPEDIA_articles10000_offset0',
                       None,
                       'If not None, load embeddings from numpy array located at "<external_embeddings>.vec" and type '
                       'string mappings from "<external_embeddings>.type" file and merge them into the embeddings '
                       'from the loaded model ("<model_dir>/[model].type").')
#tf.flags.DEFINE_boolean('load_embeddings', False,
#                        'Load embeddings from numpy array located at "<dict_file>.vec"')
tf.flags.DEFINE_string('sentence_processor', 'process_sentence7',  # 'process_sentence8',#'process_sentence3',
                       'Defines which NLP features are taken into the embedding trees.')
tf.flags.DEFINE_string('default_concat_mode',
                       'sequence',
                       'How to concat the trees returned by one parser call (e.g. trees in one document). '
                       + '"sequence" -> roots point to next root, '
                       + '"aggregate" -> roots point to an added, artificial token (AGGREGATOR) '
                         'in the end of the token sequence'
                         'None -> do not concatenate at all')

tf.flags.DEFINE_string('default_inner_concat_mode',
                       'tree',
                       'How to concatenate the token (sub-)trees in process_sentence. '
                       '\n"tree" -> use dependency tree structure, '
                       '\n"sequence" -> roots point to next root, '
                       '\n"aggregate" -> roots point to an added, artificial token (AGGREGATOR) '
                       'in the end of the token sequence '
                       '\nNone -> do not concatenate at all')
tf.flags.DEFINE_boolean('merge_nlp_embeddings',
                        #True,
                        False,
                        'If True, merge embeddings from nlp framework (spacy) into loaded embeddings.')
tf.flags.DEFINE_string('save_final_model_path',
                        None,
                        #'/home/arne/ML_local/tf/temp/log/final_model',
                        'If not None, save the final model (after integration of external and/or nlp '
                        'embeddings) to <save_final_model_path> and the types to <save_final_model_path>.type for '
                        'further usages.')
tf.flags.DEFINE_string('sim_measure',
                       'sim_cosine',
                       'similarity measure implementation (tensorflow) from model_fold for similarity score calculation. Currently implemented:'
                       '"sim_cosine" -> cosine'
                       '"sim_layer" -> similarity measure similar to the one defined in [Tai, Socher 2015]'
                       '"sim_manhattan" -> l1-norm based similarity measure (taken from MaLSTM) [Mueller et al., 2016]')
#tf.flags.DEFINE_string('tree_embedder',
#                           'TreeEmbedding_FLAT_LSTM_2levels',
#                           'Tree embedder implementation from model_fold that produces a tensorflow fold block on calling which accepts a sequence tree and produces an embedding. '
#                           'Currently implemented:'
#                           '"TreeEmbedding_TREE_LSTM" -> '
#                           '"TreeEmbedding_HTU_GRU" -> '
#                           '"TreeEmbedding_HTU_GRU_simplified" -> '
#                           '"TreeEmbedding_FLAT_AVG" -> '
#                           '"TreeEmbedding_FLAT_AVG_2levels" -> '
#                           '"TreeEmbedding_FLAT_LSTM" -> '
#                           '"TreeEmbedding_FLAT_LSTM_2levels" -> ')

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


class InvalidUsage(Exception):
    status_code = 400

    def __init__(self, message, status_code=None, payload=None):
        Exception.__init__(self)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        rv = dict(self.payload or ())
        rv['message'] = self.message
        return rv


@app.errorhandler(InvalidUsage)
def handle_invalid_usage(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response


def form_data_to_dict(form_data):
    result = {}
    for k in form_data:
        try:
            result[k] = json.loads(form_data[k])
        except ValueError:
            result[k] = form_data[k]
    return result


def make_serializable(d):
    if type(d) == dict:
        for k in d:
            d[k] = make_serializable(d[k])
    elif type(d) == list:
        for i in range(len(d)):
            d[i] = make_serializable(d[i])
    elif type(d) == np.ndarray:
        d = d.tolist()

    return d


def filter_result(r):
    if 'whitelist' in r:
        wl = r['whitelist']
        for k in r.keys():
            if k not in wl:
                del r[k]
    elif 'blacklist' in r:
        bl = r['blacklist']
        for k in r.keys():
            if k in bl:
                del r[k]
    return r


def get_params(data):
    if data == "":
        return form_data_to_dict(request.form)
    else:
        return json.loads(data)


def get_or_calc_sequence_data(params):
    if 'data_sequences' in params:
        params['data_sequences'] = np.array(params['data_sequences'])
    elif 'sequences' in params:
        sequences = params['sequences']
        concat_mode = FLAGS.default_concat_mode
        inner_concat_mode = FLAGS.default_inner_concat_mode
        if 'concat_mode' in params:
            concat_mode = params['concat_mode']
            assert concat_mode in constants.concat_modes, 'unknown concat_mode=' + concat_mode
            logging.info('use concat_mode=' + concat_mode)
        if 'inner_concat_mode' in params:
            inner_concat_mode = params['inner_concat_mode']
            assert inner_concat_mode in constants.concat_modes, 'unknown inner_concat_mode=' + inner_concat_mode
            logging.info('use inner_concat_mode=' + concat_mode)

        sentence_processor = getattr(preprocessing, FLAGS.sentence_processor)
        if 'sentence_processor' in params:
            sentence_processor = getattr(preprocessing, params['sentence_processor'])
            logging.info('use sentence_processor=' + sentence_processor.__name__)

        params['data_sequences'] = list(corpus.parse_iterator(sequences, nlp, sentence_processor, data_maps, concat_mode, inner_concat_mode))

    else:
        raise ValueError('no sequences or data_sequences found in request')


def get_or_calc_embeddings(params):
    if 'embeddings' in params:
        params['embeddings'] = np.array(params['embeddings'])
    elif 'data_sequences' or 'sequences' in params:
        get_or_calc_sequence_data(params)

        data_sequences = params['data_sequences']
        max_depth = 999
        if 'max_depth' in params:
            max_depth = int(params['max_depth'])
        #batch = [json.loads(MessageToJson(preprocessing.build_sequence_tree_from_parse(parsed_data))) for parsed_data in
        #         data_sequences]
        batch = [preprocessing.build_sequence_tree_dict_from_parse(parsed_data, max_depth) for parsed_data in data_sequences]

        if len(batch) > 0:
            fdict = embedder.build_feed_dict(batch)
            embeddings = sess.run(embedder.tree_embeddings, feed_dict=fdict)
            if embedder.scoring_enabled:
                fdict_scoring = embedder.build_scoring_feed_dict(embeddings)
                params['scores'] = sess.run(embedder.scores, feed_dict=fdict_scoring)
        else:
            embeddings = np.zeros(shape=(0, model_fold.DIMENSION_EMBEDDINGS), dtype=np.float32)
            scores = np.zeros(shape=(0,), dtype=np.float32)
            params['scores'] = scores
        params['embeddings'] = embeddings
    else:
        raise ValueError('no embeddings or sequences found in request')


@app.route("/api/embed", methods=['POST'])
def embed():
    try:
        start = time.time()
        logging.info('Embeddings requested')
        params = get_params(request.data.decode("utf-8"))
        get_or_calc_embeddings(params)

        json_data = json.dumps(filter_result(make_serializable(params)))
        logging.info("Time spent handling the request: %f" % (time.time() - start))
    except Exception as e:
        raise InvalidUsage(e.message)

    return json_data


@app.route("/api/distance", methods=['POST'])
def distance():
    try:
        start = time.time()
        logging.info('Distance requested')
        params = get_params(request.data.decode("utf-8"))
        get_or_calc_embeddings(params)

        result = pairwise_distances(params['embeddings'], metric='cosine')  # spatial.distance.cosine(embeddings[0], embeddings[1])
        params['distances'] = result.tolist()
        json_data = json.dumps(filter_result(make_serializable(params)))
        logging.info("Time spent handling the request: %f" % (time.time() - start))
    except Exception as e:
        raise InvalidUsage(e.message)

    return json_data


@app.route("/api/similarity", methods=['POST'])
def sim():
    try:
        start = time.time()
        logging.info('Similarity requested')
        params = get_params(request.data.decode("utf-8"))
        get_or_calc_embeddings(params)

        count = len(params['embeddings'])
        sims = sess.run(embedder.sim, feed_dict={embedder.e1_placeholder: np.repeat(params['embeddings'], count, axis=0),
                                                 embedder.e2_placeholder: np.tile(params['embeddings'], (count, 1))})

        params['similarities'] = sims.reshape((count, count)).tolist()
        json_data = json.dumps(filter_result(make_serializable(params)))
        logging.info("Time spent handling the request: %f" % (time.time() - start))
    except Exception as e:
        raise InvalidUsage(e.message)

    return json_data


@app.route("/api/cluster", methods=['POST'])
def cluster():
    try:
        start = time.time()
        logging.info('Clusters requested')
        params = get_params(request.data.decode("utf-8"))
        get_or_calc_embeddings(params)

        labels, meta, best_idx = get_cluster_ids(embeddings=np.array(params['embeddings']))
        params['cluster_labels'] = labels
        params['meta_data'] = meta
        params['best_idx'] = best_idx
        json_data = json.dumps(filter_result(make_serializable(params)))
        logging.info("Time spent handling the request: %f" % (time.time() - start))
    except Exception as e:
        raise InvalidUsage(e.message)
    return json_data


@app.route("/api/norm", methods=['POST'])
def norm():
    try:
        start = time.time()
        logging.info('Norms requested')
        params = get_params(request.data.decode("utf-8"))
        get_or_calc_embeddings(params)

        _, norms = normalize(params['embeddings'], norm='l2', axis=1, copy=False, return_norm=True)

        params['norms'] = norms.tolist()
        json_data = json.dumps(filter_result(make_serializable(params)))
        logging.info("Time spent handling the request: %f" % (time.time() - start))
    except Exception as e:
        raise InvalidUsage(e.message)

    return json_data


@app.route("/api/visualize", methods=['POST'])
def visualize():
    try:
        start = time.time()
        logging.info('Visualizations requested')
        params = get_params(request.data.decode("utf-8"))
        get_or_calc_sequence_data(params)

        vis.visualize_list(params['data_sequences'], types, file_name=vis.TEMP_FN)
        logging.info("Time spent handling the request: %f" % (time.time() - start))
    except Exception as e:
        raise InvalidUsage(e.message)
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


#unused
def seq_tree_iterator(sequences, parser, sentence_processor, data_maps, inner_concat_mode):
    # pp = pprint.PrettyPrinter(indent=2)
    for s in sequences:
        seq_tree = preprocessing.build_sequence_tree_from_str(str_=s, sentence_processor=sentence_processor,
                                                              parser=parser, data_maps=data_maps,
                                                              inner_concat_mode=inner_concat_mode, expand_dict=False)
        # pp.pprint(seq_tree)
        yield seq_tree.SerializeToString()


def all_subclasses(cls):
    return cls.__subclasses__() + [g for s in cls.__subclasses__() for g in all_subclasses(s)]


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    td.proto_tools.map_proto_source_tree_path('', ROOT_DIR)
    td.proto_tools.import_proto_file('sequence_node.proto')

    # We retrieve our checkpoint fullpath
    checkpoint = tf.train.get_checkpoint_state(FLAGS.model_dir)
    if checkpoint:
        # use latest checkpoint in model_dir
        input_checkpoint = checkpoint.model_checkpoint_path
        types_fn = os.path.join(os.path.dirname(input_checkpoint), 'model')
    else:
        input_checkpoint = FLAGS.model_dir
        types_fn = FLAGS.model_dir

    logging.info('load spacy ...')
    nlp = spacy.load('en')
    nlp.pipeline = [nlp.tagger, nlp.entity, nlp.parser]

    logging.info('read types ...')
    types = corpus.read_types(types_fn)
    reader = tf.train.NewCheckpointReader(input_checkpoint)
    saved_shapes = reader.get_variable_to_shape_map()

    #available_embedder = ['TreeEmbedding_TREE_LSTM',
    #                      'TreeEmbedding_HTU_GRU',
    #                      'TreeEmbedding_HTU_GRU_simplified',
    #                      'TreeEmbedding_FLAT_AVG',
    #                      'TreeEmbedding_FLAT_AVG_2levels',
    #                      'TreeEmbedding_FLAT_LSTM',
    #                      'TreeEmbedding_FLAT_LSTM_2levels']

    available_embedder = [cls.__name__ for cls in all_subclasses(model_fold.TreeEmbedding)]

    tree_embedder_names = [en for en in available_embedder if len([vn for vn in saved_shapes if vn.startswith(en + '/')]) > 0]
    assert len(tree_embedder_names) <= 1, 'found vars for multiple tree embedders: ' + ', '.join(tree_embedder_names)
    if len(tree_embedder_names) == 0:
        logging.info('No tree embedder vars found in model. Use "TreeEmbedding_FLAT_AVG".')
        tree_embedder_names = ['TreeEmbedding_FLAT_AVG']
    else:
        logging.info('Tree embedder vars found in model. Use "' + tree_embedder_names[0] + '".')
    tree_embedder = getattr(model_fold, tree_embedder_names[0])

    fc_embedding_var_names = [vn for vn in saved_shapes if vn.startswith(tree_embedder_names[0]
                                                                         + '/' + model_fold.VAR_PREFIX_FC_EMBEDDING)]
    if len(fc_embedding_var_names):
        logging.info('found embedding_fc vars: ' + ', '.join(fc_embedding_var_names) + '. Apply fully connected layer to embeddings before composition.')

    scoring_var_names = [vn for vn in saved_shapes if vn.startswith(model_fold.DEFAULT_SCOPE_SCORING)]
    if len(scoring_var_names) > 0:
        logging.info('found scoring vars: ' + ', '.join(scoring_var_names) + '. Enable scoring functionality.')
    else:
        logging.info('no scoring vars found. Disable scoring functionality.')

    sim_measure = getattr(model_fold, FLAGS.sim_measure)

    if not FLAGS.merge_nlp_embeddings and not FLAGS.external_embeddings:
        logging.info('extract lexicon size from model: ' + input_checkpoint + ' ...')
        #saved_shapes = reader.get_variable_to_shape_map()
        embed_shape = saved_shapes[model_fold.VAR_NAME_EMBEDDING]
        lex_size = embed_shape[0]
    else:
        logging.info('extract embeddings from model: ' + input_checkpoint + ' ...')
        embeddings_np = reader.get_tensor(model_fold.VAR_NAME_EMBEDDING)
        if FLAGS.external_embeddings:
            logging.info('read external types: '+FLAGS.external_embeddings+'.type ...')
            external_types = corpus.read_types(FLAGS.external_embeddings)
            logging.info('load external embeddings from: '+FLAGS.external_embeddings+'.vec ...')
            external_vecs = np.load(FLAGS.external_embeddings+'.vec')
            embeddings_np, types = corpus.merge_dicts(embeddings_np, types, external_vecs, external_types, add=True, remove=False)
        if FLAGS.merge_nlp_embeddings:
            logging.info('extract nlp embeddings and types ...')
            nlp_vecs, nlp_types = corpus.get_dict_from_vocab(nlp.vocab)
            logging.info('merge nlp embeddings into loaded embeddings ...')
            embeddings_np, types = corpus.merge_dicts(embeddings_np, types, nlp_vecs, nlp_types, add=True, remove=False)
        lex_size = embeddings_np.shape[0]

    logging.info('dict size: ' + str(len(types)))
    assert len(types) == lex_size, 'count of types (' +str(len(types)) + ') does not match count of embedding vectors (' + str(lex_size) + ')'
    data_maps = corpus.mapping_from_list(types)

    with tf.Graph().as_default():
        with tf.device(tf.train.replica_device_setter(FLAGS.ps_tasks)):

            embedder = model_fold.SequenceTreeEmbedding(lex_size=lex_size,
                                                        tree_embedder=tree_embedder,
                                                        state_size=50,
                                                        sim_measure=sim_measure,
                                                        scoring_enabled=len(scoring_var_names) > 0,
                                                        embedding_fc_activation=None,
                                                        output_fc_activation=None
                                                        #apply_embedding_fc=len(fc_embedding_var_names) > 0,
                                                        )

            if FLAGS.external_embeddings or FLAGS.merge_nlp_embeddings:
                vars_all = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
                vars_without_embed = [v for v in vars_all if v != embedder.embeddings_var]
                if len(vars_without_embed) > 0:
                    saver = tf.train.Saver(var_list=vars_without_embed)
                else:
                    saver = None
            else:
                saver = tf.train.Saver()

            sess = tf.Session()
            # Restore variables from disk.
            if saver:
                logging.info('restore model from: ' + input_checkpoint + '...')
                saver.restore(sess, input_checkpoint)

            if FLAGS.external_embeddings or FLAGS.merge_nlp_embeddings:
                logging.info('init embeddings with external vectors ...')
                sess.run(embedder.embeddings_init, feed_dict={embedder.embeddings_placeholder: embeddings_np})

            if FLAGS.save_final_model_path:
                logging.info('save final model to: ' + FLAGS.save_final_model_path + ' ...')
                saver_final = tf.train.Saver()
                saver_final.save(sess, FLAGS.save_final_model_path, write_meta_graph=False, write_state=False)
                corpus.write_dict(FLAGS.save_final_model_path, types=types)

    logging.info('Starting the API')
    app.run()
