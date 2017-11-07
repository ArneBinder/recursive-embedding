from __future__ import print_function

import ast
import json
import logging
import os
import sys
import time

import numpy as np
import spacy
import tensorflow as tf
import tensorflow_fold as td

import mytools
import sequence_trees
from flask import Flask, request, send_file, jsonify, Response
from flask_cors import CORS
from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering
# from scipy import spatial # crashes!
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import normalize
# from google.protobuf.json_format import MessageToJson

import constants
import corpus
import corpus_simtuple
import model_fold
import preprocessing
import visualize as vis
import lexicon as lex

tf.flags.DEFINE_string('data_source',
                       # '/media/arne/WIN/ML/data/corpora/SICK/process_sentence3_marked/SICK_CMaggregate',
                       '/home/arne/ML_local/tf/supervised/log/SA/FINETUNE/PPDB/restoreFALSE_batchs100_keepprob0.9_leaffc0_learningr0.05_lextrainTRUE_optADADELTAOPTIMIZER_rootfc0_smSIMCOSINE_state50_testfilei1_dataPROCESSSENTENCE3MARKEDSICKCMAGGREGATE_teTREEEMBEDDINGFLATLSTM',
                       # '/media/arne/WIN/ML/data/corpora/SICK/process_sentence3_marked/SICK_CMaggregate',
                       # '/home/arne/ML_local/tf/supervised/log/SA/DUMMY/restoreFALSE_batchs100_keepprob0.9_leaffc0_learningr0.05_lextrainTRUE_optADADELTAOPTIMIZER_rootfc0_smSIMCOSINE_state50_testfilei1_dataPROCESSSENTENCE3MARKEDSICKOHCMAGGREGATE_teTREEEMBEDDINGFLATAVG',
                       # '/home/arne/ML_local/tf/supervised/log/PRETRAINED/batchsize100_embeddingstrainableTRUE_learningrate0.001_optimizerADADELTAOPTIMIZER_simmeasureSIMCOSINE_statesize50_testfileindex1_traindatapathPROCESSSENTENCE3HASANCMSEQUENCEICMTREENEGSAMPLES1_treeembedderTREEEMBEDDINGHTUGRU',
                       # '/home/arne/ML_local/tf/supervised/log/batchsize100_embeddingstrainableTRUE_learningrate0.001_optimizerADADELTAOPTIMIZER_simmeasureSIMCOSINE_statesize50_testfileindex1_traindatapathPROCESSSENTENCE3SICKTTCMSEQUENCEICMTREE_treeembedderTREEEMBEDDINGHTUGRU',
                       # '/home/arne/ML_local/tf/supervised/log/BACKUP_batchsize100_embeddingstrainableTRUE_learningrate0.001_optimizerADADELTAOPTIMIZER_simmeasureSIMCOSINE_statesize50_testfileindex1_traindatapathPROCESSSENTENCE3SICKTTCMSEQUENCEICMTREE_treeembedderTREEEMBEDDINGHTUGRU',
                       # /model.ckpt-122800',
                       # '/home/arne/ML_local/tf/supervised/log/applyembeddingfcTRUE_batchsize100_embeddingstrainableTRUE_normalizeTRUE_simmeasureSIMCOSINE_testfileindex-1_traindatapathPROCESSSENTENCE3SICKCMAGGREGATE_treeembedderTREEEMBEDDINGFLATLSTM',
                       # '/home/arne/ML_local/tf/log/final_model',
                       'Directory containing the model and a checkpoint file or the direct path to a '
                       'model (without extension) and the model.type file containing the string dict.')
tf.flags.DEFINE_string('external_lexicon',
                       # '/media/arne/WIN/Users/Arne/ML/data/corpora/wikipedia/process_sentence8_/WIKIPEDIA_articles10000_offset0',
                       None,
                       'If not None, load embeddings from numpy array located at "<external_lexicon>.vec" and type '
                       'string mappings from "<external_lexicon>.type" file and merge them into the embeddings '
                       'from the loaded model ("<data_source>/[model].type").')
# tf.flags.DEFINE_boolean('load_embeddings', False,
#                        'Load embeddings from numpy array located at "<dict_file>.vec"')
tf.flags.DEFINE_string('default_sentence_processor', 'process_sentence3',  # 'process_sentence8',#'process_sentence3',
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
tf.flags.DEFINE_boolean('merge_nlp_lexicon',
                        # True,
                        False,
                        'If True, merge embeddings from nlp framework (spacy) into loaded embeddings.')
tf.flags.DEFINE_string('save_final_model_path',
                       None,
                       # '/home/arne/ML_local/tf/temp/log/final_model',
                       'If not None, save the final model (after integration of external and/or nlp '
                       'embeddings) to <save_final_model_path> and the types to <save_final_model_path>.type for '
                       'further usages.')

tf.flags.DEFINE_integer('ps_tasks', 0,
                        'Number of PS tasks in the job.')
FLAGS = tf.flags.FLAGS
mytools.logging_init()

flags_fn = os.path.join(FLAGS.data_source, 'flags.json')
if os.path.isfile(flags_fn):
    with open(flags_fn, 'r') as infile:
        model_flags = json.load(infile)
    for flag in model_flags:
        v = model_flags[flag]
        getattr(tf.flags, v[0])('model_' + flag, v[1], v[2])
else:
    tf.flags.DEFINE_string('model_train_data_path', FLAGS.data_source, '')

# PROTO_PACKAGE_NAME = 'recursive_dependency_embedding'
# PROTO_CLASS = 'SequenceNode'

sess = None
embedder = None
data_maps = None
types = None
nlp = None

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


def parse_params(params, prev={}):
    result = prev
    for param in params:
        v_ = params[param]
        try:
            v = ast.literal_eval(v_)
        except ValueError:
            v = v_
        except SyntaxError:
            v = v_
            logging.warning('Syntax error while parsing "%s". Assume it is a string.' % v_)
        result[param] = v
    return result


def get_params(request):
    data = request.data.decode("utf-8")
    params = {}
    if data != "":
        params = json.loads(data)
    params = parse_params(request.args, params)
    params = parse_params(request.form, params)
    if request.headers.environ['HTTP_ACCEPT'] != '*/*':
        params['HTTP_ACCEPT'] = request.headers.environ['HTTP_ACCEPT']

    return params


def get_or_calc_sequence_data(params):
    if 'data_sequences' in params:
        params['data_sequences'] = np.array(params['data_sequences'])
    elif 'train_file' in params:
        fn = '%s.%s' % (FLAGS.model_train_data_path, params['train_file'])
        if os.path.isfile(fn):
            # get sequence data from sim tuple file
            params['data_sequences'] = []
            params['scores_gold'] = []
            start = params.get('start', 0)
            end = params.get('end', -1)
            if 'new_format' in params:
                for i, sim_tuple in enumerate(corpus_simtuple.iterate_scored_tree_data([fn])):
                    if i < start:
                        continue
                    if 0 <= end <= i:
                        break

                    data1, parent1 = sequence_trees.sequence_node_to_sequence_trees(sim_tuple['tree'])
                    params['data_sequences'].append([data1, parent1])
                    params['scores_gold'].append(sim_tuple['score'])
            else:
                for i, sim_tuple in enumerate(corpus_simtuple.iterate_sim_tuple_data([fn])):
                    if i < start:
                        continue
                    if 0 <= end <= i:
                        break
                    data1, parent1 = sequence_trees.sequence_node_to_sequence_trees(sim_tuple['first'])
                    data2, parent2 = sequence_trees.sequence_node_to_sequence_trees(sim_tuple['second'])
                    params['data_sequences'].append([data1, parent1])
                    params['data_sequences'].append([data2, parent2])
                    params['scores_gold'].append(sim_tuple['similarity'])
            params['scores_gold'] = np.array(params['scores_gold'])
        else:
            raise IOError('could not open "%s"' % fn)
    elif 'sequences' in params:
        sequences = [s.decode("utf-8") for s in params['sequences']]
        concat_mode = FLAGS.default_concat_mode
        inner_concat_mode = FLAGS.default_inner_concat_mode
        if 'concat_mode' in params:
            concat_mode = params['concat_mode']
            assert concat_mode in constants.concat_modes, 'unknown concat_mode=%s' % concat_mode
            logging.info('use concat_mode=%s' % concat_mode)
        if 'inner_concat_mode' in params:
            inner_concat_mode = params['inner_concat_mode']
            assert inner_concat_mode in constants.concat_modes, 'unknown inner_concat_mode=%s' % inner_concat_mode
            logging.info('use inner_concat_mode=%s' % inner_concat_mode)

        sentence_processor = getattr(preprocessing, FLAGS.default_sentence_processor)
        if 'sentence_processor' in params:
            sentence_processor = getattr(preprocessing, params['sentence_processor'])
            logging.info('use sentence_processor=%s' % sentence_processor.__name__)

        init_nlp()
        params['data_sequences'] = list(
            corpus.parse_iterator(sequences, nlp, sentence_processor, data_maps, concat_mode, inner_concat_mode))

    else:
        raise ValueError('no sequences or data_sequences found in request')


def get_or_calc_embeddings(params):
    if 'embeddings' in params:
        params['embeddings'] = np.array(params['embeddings'])
    elif 'data_sequences' or 'sequences' in params:
        get_or_calc_sequence_data(params)

        data_sequences = params['data_sequences']
        max_depth = 150
        if 'max_depth' in params:
            max_depth = int(params['max_depth'])
        # batch = [json.loads(MessageToJson(preprocessing.build_sequence_tree_from_parse(parsed_data))) for parsed_data in
        #         data_sequences]
        batch = [sequence_trees.build_sequence_tree_dict_from_parse(parsed_data, max_depth) for parsed_data in
                 data_sequences]

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
        params = get_params(request)
        get_or_calc_embeddings(params)

        # debug
        #params['embeddings'].dump('api_request_embeddings')
        #if 'scores_gold' in params:
        #    params['scores_gold'].dump('api_request_scores_gold')
        # debug end

        return_type = params.get('HTTP_ACCEPT', False) or 'application/json'
        json_data = json.dumps(filter_result(make_serializable(params)))
        response = Response(json_data, mimetype=return_type)
        logging.info("Time spent handling the request: %f" % (time.time() - start))
    except Exception as e:
        raise InvalidUsage(e.message)
    return response


@app.route("/api/distance", methods=['POST'])
def distance():
    try:
        start = time.time()
        logging.info('Distance requested')
        params = get_params(request)
        get_or_calc_embeddings(params)

        result = pairwise_distances(params['embeddings'],
                                    metric='cosine')  # spatial.distance.cosine(embeddings[0], embeddings[1])
        params['distances'] = result.tolist()
        return_type = params.get('HTTP_ACCEPT', False) or 'application/json'
        json_data = json.dumps(filter_result(make_serializable(params)))
        response = Response(json_data, mimetype=return_type)
        logging.info("Time spent handling the request: %f" % (time.time() - start))
    except Exception as e:
        raise InvalidUsage(e.message)

    return response


@app.route("/api/similarity", methods=['POST'])
def sim():
    try:
        start = time.time()
        logging.info('Similarity requested')
        params = get_params(request)
        get_or_calc_embeddings(params)

        count = len(params['embeddings'])
        sims = sess.run(embedder.sim,
                        feed_dict={embedder.e1_placeholder: np.repeat(params['embeddings'], count, axis=0),
                                   embedder.e2_placeholder: np.tile(params['embeddings'], (count, 1))})

        params['similarities'] = sims.reshape((count, count)).tolist()
        return_type = params.get('HTTP_ACCEPT', False) or 'application/json'
        json_data = json.dumps(filter_result(make_serializable(params)))
        response = Response(json_data, mimetype=return_type)
        logging.info("Time spent handling the request: %f" % (time.time() - start))
    except Exception as e:
        raise InvalidUsage(e.message)

    return response


@app.route("/api/cluster", methods=['POST'])
def cluster():
    try:
        start = time.time()
        logging.info('Clusters requested')
        params = get_params(request)
        get_or_calc_embeddings(params)

        labels, meta, best_idx = get_cluster_ids(embeddings=np.array(params['embeddings']))
        params['cluster_labels'] = labels
        params['meta_data'] = meta
        params['best_idx'] = best_idx
        return_type = params.get('HTTP_ACCEPT', False) or 'application/json'
        json_data = json.dumps(filter_result(make_serializable(params)))
        response = Response(json_data, mimetype=return_type)
        logging.info("Time spent handling the request: %f" % (time.time() - start))
    except Exception as e:
        raise InvalidUsage(e.message)
    return response


@app.route("/api/norm", methods=['POST'])
def norm():
    try:
        start = time.time()
        logging.info('Norms requested')
        params = get_params(request)
        get_or_calc_embeddings(params)

        _, norms = normalize(params['embeddings'], norm='l2', axis=1, copy=False, return_norm=True)

        params['norms'] = norms.tolist()
        return_type = params.get('HTTP_ACCEPT', False) or 'application/json'
        json_data = json.dumps(filter_result(make_serializable(params)))
        response = Response(json_data, mimetype=return_type)
        logging.info("Time spent handling the request: %f" % (time.time() - start))
    except Exception as e:
        raise InvalidUsage(e.message)

    return response


@app.route("/api/visualize", methods=['POST'])
def visualize():
    try:
        start = time.time()
        logging.info('Visualizations requested')
        params = get_params(request)

        get_or_calc_sequence_data(params)

        mode = params.get('mode', 'image')
        if mode == 'image':
            vis.visualize_list(params['data_sequences'], types, file_name=vis.TEMP_FN)
            response = send_file(vis.TEMP_FN)
        elif mode == 'text':
            params['sequences'] = []
            for data, parents in params['data_sequences']:
                texts = [" ".join(t_list) for t_list in vis.get_text((data, parents), types, params.get('prefix_blacklist', None))]
                params['sequences'].append(texts)

            return_type = params.get('HTTP_ACCEPT', False) or 'application/json'
            json_data = json.dumps(filter_result(make_serializable(params)))

            ## DEBUG
            #with open('temp_response.json', 'w') as f:
            #    f.write(json_data)
            ## DEBUG END
            response = Response(json_data, mimetype=return_type)
        else:
            ValueError('Unknown mode=%s. Use "image" (default) or "text".')
        logging.info("Time spent handling the request: %f" % (time.time() - start))
    except Exception as e:
        raise InvalidUsage(e.message)
    return response


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


# unused # deprecated
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


def init_nlp():
    global nlp

    if nlp is None:
        logging.info('load spacy ...')
        nlp = spacy.load('en')
        nlp.pipeline = [nlp.tagger, nlp.entity, nlp.parser]


def main(unused_argv):
    global sess, embedder, data_maps, types

    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    td.proto_tools.map_proto_source_tree_path('', ROOT_DIR)
    td.proto_tools.import_proto_file('sequence_node.proto')
    td.proto_tools.import_proto_file('scored_tree.proto')

    # We retrieve our checkpoint fullpath
    checkpoint = tf.train.get_checkpoint_state(FLAGS.data_source)
    # if a checkpoint file exist, take data_source as model dir
    if checkpoint:
        logging.info('Model checkpoint found in "%s". Load the tensorflow model.' % FLAGS.data_source)
        # use latest checkpoint in data_source
        input_checkpoint = checkpoint.model_checkpoint_path
        types_fn = os.path.join(FLAGS.data_source, 'model')
        # types_fn = os.path.join(os.path.dirname(input_checkpoint), 'model')
        reader = tf.train.NewCheckpointReader(input_checkpoint)
        # loaded_lex_size = saved_shapes[model_fold.VAR_NAME_LEXICON][0]
        logging.info('extract embeddings from model: ' + input_checkpoint + ' ...')
        lexicon_np = reader.get_tensor(model_fold.VAR_NAME_LEXICON)
    # take data_source as corpus path
    else:
        logging.info('No model checkpoint found in "%s". Load train data corpus from: "%s"' % (
        FLAGS.data_source, FLAGS.model_train_data_path))
        types_fn = FLAGS.model_train_data_path
        logging.info('load embeddings from: ' + types_fn + '.vec ...')
        lexicon_np = np.load(types_fn + '.vec')

    logging.info('read types ...')
    types = lex.read_types(types_fn)

    if FLAGS.external_lexicon:
        logging.info('read external types: ' + FLAGS.external_lexicon + '.type ...')
        external_vecs, external_types = lex.read_dict(FLAGS.external_lexicon)
        lexicon_np, types = lex.merge_dicts(lexicon_np, types, external_vecs, external_types, add=True, remove=False)
    if FLAGS.merge_nlp_lexicon:
        logging.info('extract nlp embeddings and types ...')
        init_nlp()
        nlp_vecs, nlp_types = lex.get_dict_from_vocab(nlp.vocab)
        logging.info('merge nlp embeddings into loaded embeddings ...')
        lexicon_np, types = lex.merge_dicts(lexicon_np, types, nlp_vecs, nlp_types, add=True, remove=False)
    lex_size = lexicon_np.shape[0]

    logging.info('dict size: ' + str(len(types)))
    assert len(types) == lex_size, 'count of types (' + str(
        len(types)) + ') does not match count of embedding vectors (' + str(lex_size) + ')'
    data_maps = lex.mapping_from_list(types)

    # load model
    if checkpoint:
        tree_embedder = getattr(model_fold, FLAGS.model_tree_embedder)

        saved_shapes = reader.get_variable_to_shape_map()
        scoring_var_names = [vn for vn in saved_shapes if vn.startswith(model_fold.DEFAULT_SCOPE_SCORING)]
        if len(scoring_var_names) > 0:
            logging.info('found scoring vars: ' + ', '.join(scoring_var_names) + '. Enable scoring functionality.')
        else:
            logging.info('no scoring vars found. Disable scoring functionality.')

        sim_measure = getattr(model_fold, FLAGS.model_sim_measure)

        with tf.Graph().as_default():
            with tf.device(tf.train.replica_device_setter(FLAGS.ps_tasks)):
                tree_model = model_fold.SequenceTreeModel(lex_size=lex_size,
                                                          tree_embedder=tree_embedder,
                                                          state_size=FLAGS.model_state_size,
                                                          lexicon_trainable=False,
                                                          leaf_fc_size=FLAGS.model_leaf_fc_size,
                                                          root_fc_size=FLAGS.model_root_fc_size,
                                                          keep_prob=1.0)

                embedder = model_fold.SequenceTreeEmbedding(tree_model=tree_model,
                                                            sim_measure=sim_measure,
                                                            scoring_enabled=len(scoring_var_names) > 0
                                                            )

                if FLAGS.external_lexicon or FLAGS.merge_nlp_lexicon:
                    vars_all = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
                    vars_without_embed = [v for v in vars_all if v != tree_model.embedder.lexicon_var]
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

                if FLAGS.external_lexicon or FLAGS.merge_nlp_lexicon:
                    logging.info('init embeddings with external vectors ...')
                    sess.run(embedder.tree_model.embedder.lexicon_init,
                             feed_dict={embedder.tree_model.embedder.lexicon_placeholder: lexicon_np})

                if FLAGS.save_final_model_path:
                    logging.info('save final model to: ' + FLAGS.save_final_model_path + ' ...')
                    saver_final = tf.train.Saver()
                    saver_final.save(sess, FLAGS.save_final_model_path, write_meta_graph=False, write_state=False)
                    lex.write_dict(FLAGS.save_final_model_path, types=types)

    logging.info('Starting the API')
    app.run()


if __name__ == '__main__':
    tf.app.run()
