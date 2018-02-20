from __future__ import print_function

import ast
import json
import logging
import os
import sys
import time
import re

import numpy as np
import spacy
import tensorflow as tf

from flask import Flask, request, send_file, jsonify, Response
from flask_cors import CORS
from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering
# from scipy import spatial # crashes!
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
# from google.protobuf.json_format import MessageToJson
import svgutils.transform as sg

import constants
import corpus_simtuple
import preprocessing
from lexicon import Lexicon
from sequence_trees import Forest
from config import Config
from constants import TYPE_REF, TYPE_REF_SEEALSO, DTYPE_HASH, DTYPE_IDX
import data_iterators

TEMP_FN_SVG = 'temp_forest.svg'

tf.flags.DEFINE_string('data_source',
                       #'/media/arne/WIN/ML/data/corpora/SICK/process_sentence3_marked/SICK_CMaggregate',
                       None,
                       #'/home/arne/ML_local/tf/supervised/log/SA/FINETUNE/PPDB/restoreFALSE_batchs100_keepprob0.9_leaffc0_learningr0.05_lextrainTRUE_optADADELTAOPTIMIZER_rootfc0_smSIMCOSINE_state50_testfilei1_dataPROCESSSENTENCE3MARKEDSICKCMAGGREGATE_teTREEEMBEDDINGFLATLSTM',
                       'Directory containing the model and a checkpoint file or the direct path to a '
                       'model (without extension) and the model.type file containing the string dict.')
tf.flags.DEFINE_string('external_lexicon',
                       # '/media/arne/WIN/Users/Arne/ML/data/corpora/wikipedia/process_sentence8_/WIKIPEDIA_articles10000_offset0',
                       None,
                       'If not None, load embeddings from numpy array located at "<external_lexicon>.vec" and type '
                       'string mappings from "<external_lexicon>.type" file and merge them into the embeddings '
                       'from the loaded model ("<data_source>/[model].type").')
tf.flags.DEFINE_string('default_sentence_processor', 'process_sentence1',
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

nlp = None
sess = None
model_tree = None
lexicon = None
forest = None
data_path = None


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


def parse_iterator(sequences, sentence_processor, concat_mode, inner_concat_mode):
    init_nlp()
    for s in sequences:
        _forest = lexicon.read_data(reader=preprocessing.identity_reader,
                                    sentence_processor=sentence_processor,
                                    parser=nlp,
                                    reader_args={'content': s},
                                    concat_mode=concat_mode,
                                    inner_concat_mode=inner_concat_mode,
                                    expand_dict=False,
                                    reader_roots_args={
                                        'root_label': constants.vocab_manual[constants.IDENTITY_EMBEDDING]})
        yield _forest.forest


def get_or_calc_sequence_data(params):
    global data_path

    if 'sequences' in params:
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

        params['data_sequences'] = list(parse_iterator(sequences, sentence_processor, concat_mode, inner_concat_mode))

    if 'data_sequences' in params:
        d_list, p_list = zip(*params['data_sequences'])
        data_as_hashes = params.get('data_as_hashes', False)
        root_ids = params.get('root_ids', None)
        current_forest = Forest(data=sum(d_list, []), parents=sum(p_list, []), lexicon=lexicon,
                                data_as_hashes=data_as_hashes, root_ids=root_ids)
    else:
        init_forest(data_path)
        current_forest = forest

    if 'idx_tuple_file' in params:
        fn = '%s.%s' % (data_path, params['idx_tuple_file'])
        if not os.path.isfile(fn):
            raise IOError('could not open idx_tuple_file=%s' % fn)
        assert 'data_iterator' in params, 'parameter data_iterator is not given, can not iterate idx_tuple_file'
        data_iterator = getattr(data_iterators, params['data_iterator'])

        max_depth = params.get('max_depth', 100)
        context = params.get('context', 0)

        data_iterator_args = {'index_files': [fn], 'sequence_trees': current_forest, 'max_depth': max_depth,
                              'context': context, 'transform': False}

        if current_forest.data_as_hashes:
            current_forest.hashes_to_indices()

        if 'link_cost_ref' in params:
            data_iterator_args['link_cost_ref'] = params['link_cost_ref']
        if 'link_cost_ref_seealso' in params:
            data_iterator_args['link_cost_ref_seealso'] = params['link_cost_ref_seealso']
        if 'concat_mode' in params:
            data_iterator_args['concat_mode'] = params['concat_mode']

        params['sequences'] = []
        params['data_sequences'] = []
        params['scores_gold'] = []

        params['data_as_hashes'] = current_forest.data_as_hashes
        tuple_start = params.get('tuple_start', 0)
        tuple_end = params.get('tuple_end', -1)

        for i, (tree_dicts, probs) in enumerate(data_iterator(**data_iterator_args)):
            if i < tuple_start:
                continue
            if 0 <= tuple_end <= i:
                break
            for tree_dict in tree_dicts:
                vis_forest = Forest(tree_dict=tree_dict, lexicon=current_forest.lexicon,
                                    data_as_hashes=current_forest.data_as_hashes)

                params['data_sequences'].append([vis_forest.data, vis_forest.parents])
                token_list = vis_forest.get_text_plain(blacklist=params.get('prefix_blacklist', None))
                params['sequences'].append(token_list)
            for prob in probs:
                params['scores_gold'].append(prob)
    elif 'root_start' in params:
        params['sequences'] = []
        params['data_sequences'] = []
        params['data_as_hashes'] = current_forest.data_as_hashes
        roots = current_forest.roots
        root_start = params.get('root_start', 0)
        root_end = params.get('root_end', len(roots))
        assert root_start <= root_end, 'ERROR: root_start=%i > root_end=%i' % (root_start, root_end)
        assert root_end <= len(roots), 'ERROR: root_end=%i > len(roots)=%i' % (root_end, len(roots))
        for tree in current_forest.trees(root_indices=roots[root_start:root_end]):
            params['data_sequences'].append([tree[0].tolist(), tree[1].tolist()])
        for data_sequence in params['data_sequences']:
            token_list = Forest(forest=data_sequence, lexicon=lexicon,
                                           data_as_hashes=params['data_as_hashes']).get_text_plain(
                blacklist=params.get('prefix_blacklist', None))
            params['sequences'].append(" ".join(token_list))
    elif 'idx_start' in params:
        params['sequences'] = []
        params['data_sequences'] = []
        params['data_as_hashes'] = current_forest.data_as_hashes
        idx_start = params.get('idx_start', 0)
        idx_end = params.get('idx_end', len(current_forest))
        assert idx_start <= idx_end, 'ERROR: idx_start=%i > idx_end=%i' % (idx_start, idx_end)
        assert idx_end <= len(current_forest), 'ERROR: root_end=%i > len(roots)=%i' % (idx_end, len(current_forest))
        data_sequence = [current_forest.data[idx_start:idx_end].tolist(), current_forest.parents[idx_start:idx_end].tolist()]
        params['data_sequences'] = [data_sequence]
        token_list = current_forest.get_text_plain(blacklist=params.get('prefix_blacklist', None), start=idx_start, end=idx_end)
        params['sequences'] = [token_list]
    elif 'idx' in params:
        params['sequences'] = []
        params['data_sequences'] = []
        max_depth = params.get('max_depth', 10)
        context = params.get('context', 0)
        idx = params['idx']
        link_costs = {}
        if current_forest.data_as_hashes:
            current_forest.hashes_to_indices()
        if 'link_cost_ref' in params:
            d_ref = lexicon.get_d(TYPE_REF, data_as_hashes=current_forest.data_as_hashes)
            link_costs[d_ref] = params['link_cost_ref']
        if 'link_cost_ref_seealso' in params:
            d_ref = lexicon.get_d(TYPE_REF_SEEALSO, data_as_hashes=current_forest.data_as_hashes)
            link_costs[d_ref] = params['link_cost_ref_seealso']
        tree_dict = current_forest.get_tree_dict(idx=idx, max_depth=max_depth, context=context, transform=False,
                                         link_costs=link_costs)
        vis_forest = Forest(tree_dict=tree_dict, lexicon=current_forest.lexicon, data_as_hashes=current_forest.data_as_hashes)
        params['data_as_hashes'] = vis_forest.data_as_hashes
        params['data_sequences'] = [[vis_forest.data, vis_forest.parents]]
        token_list = vis_forest.get_text_plain(blacklist=params.get('prefix_blacklist', None))
        params['sequences'] = [token_list]


def get_or_calc_embeddings(params):
    if 'embeddings' in params:
        params['embeddings'] = np.array(params['embeddings'])
    elif 'data_sequences' or 'sequences' in params:
        assert model_tree is not None, 'No model loaded. To load a model, use endpoint: /api/load?path=path_to_model'
        get_or_calc_sequence_data(params)
        data_sequences = params['data_sequences']
        max_depth = 150
        if 'max_depth' in params:
            max_depth = int(params['max_depth'])

        batch = [[[Forest(forest=parsed_data, lexicon=lexicon).get_tree_dict(max_depth=max_depth, transform=True)], []]
                 for parsed_data in data_sequences]

        if len(batch) > 0:
            fdict = model_tree.build_feed_dict(batch)
            embeddings = sess.run(model_tree.embeddings_all, feed_dict=fdict)
            # if embedder.scoring_enabled:
            #    fdict_scoring = embedder.build_scoring_feed_dict(embeddings)
            #    params['scores'] = sess.run(embedder.scores, feed_dict=fdict_scoring)
        else:
            embeddings = np.zeros(shape=(0, model_tree.dimension_embeddings), dtype=np.float32)
            scores = np.zeros(shape=(0,), dtype=np.float32)
            params['scores'] = scores
        params['embeddings'] = embeddings
    else:
        raise ValueError('no embeddings or sequences found in request')


def concat_visualizations_svg(file_name, count):
    file_names = [file_name + '.' + str(i) for i in range(count)]
    # plots = [fig.getroot() for fig in map(sg.fromfile, file_names)]
    images = map(sg.fromfile, file_names)
    widths, heights = zip(*(i.get_size() for i in images))

    rx = re.compile(r"[-+]?\d*\.\d+|\d+", re.VERBOSE)
    widths = [float(rx.search(w).group()) for w in widths]
    heights = [float(rx.search(h).group()) for h in heights]

    total_height = 0
    plots = []
    for i, image in enumerate(images):
        plot = image.getroot()
        plot.moveto(0, total_height)
        plots.append(plot)
        total_height += heights[i]
    max_width = max(widths)

    fig = sg.SVGFigure(max_width, total_height)
    fig.append(plots)
    fig.save(file_name)

    for fn in file_names:
        os.remove(fn)


@app.route("/api/embed", methods=['POST'])
def embed():
    try:
        start = time.time()
        logging.info('Embeddings requested')
        params = get_params(request)
        get_or_calc_embeddings(params)

        # debug
        # params['embeddings'].dump('api_request_embeddings')
        # if 'scores_gold' in params:
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
        logging.info('Distance requested')
        params = get_params(request)
        get_or_calc_embeddings(params)

        result = cosine_similarity(params['embeddings'])
        # result = pairwise_distances(params['embeddings'],
        #                            metric='cosine')  # spatial.distance.cosine(embeddings[0], embeddings[1])
        params['similarities'] = result.tolist()
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

        mode = params.get('vis_mode', 'image')
        if mode == 'image':
            for i, data_sequence in enumerate(params['data_sequences']):
                if 'root_ids' in params:
                    root_ids = params['root_ids'][i]
                else:
                    root_ids = None
                forest_temp = Forest(forest=data_sequence, lexicon=lexicon, data_as_hashes=params['data_as_hashes'],
                                     root_ids=root_ids)
                forest_temp.visualize(TEMP_FN_SVG + '.' + str(i))
            concat_visualizations_svg(TEMP_FN_SVG, len(params['data_sequences']))

            response = send_file(TEMP_FN_SVG)
            os.remove(TEMP_FN_SVG)
        elif mode == 'text':
            return_type = params.get('HTTP_ACCEPT', False) or 'application/json'
            json_data = json.dumps(filter_result(make_serializable(params)))
            response = Response(json_data, mimetype=return_type)
        else:
            ValueError('Unknown mode=%s. Use "image" (default) or "text".')
        logging.info("Time spent handling the request: %f" % (time.time() - start))
    except Exception as e:
        raise InvalidUsage(e.message)
    return response


@app.route("/api/show_tree_rerooted", methods=['POST'])
def show_rooted_tree_dict():
    try:
        start = time.time()
        logging.info('Reroot requested')
        params = get_params(request)
        init_forest(data_path)
        idx = params['idx']
        max_depth = params.get('max_depth', 9999)
        rerooted_dict = forest.get_tree_dict_rooted(idx, max_depth=max_depth)
        rerooted_forest = Forest(tree_dict=rerooted_dict, lexicon=lexicon)

        mode = params.get('vis_mode', 'image')
        if mode == 'image':
            rerooted_forest.visualize(filename=TEMP_FN_SVG)
            response = send_file(TEMP_FN_SVG)
        elif mode == 'text':
            return_type = params.get('HTTP_ACCEPT', False) or 'application/json'
            json_data = json.dumps(filter_result(make_serializable(params)))
            response = Response(json_data, mimetype=return_type)
        else:
            ValueError('Unknown mode=%s. Use "image" (default) or "text".')
        logging.info("Time spent handling the request: %f" % (time.time() - start))
    except Exception as e:
        raise InvalidUsage(e.message)
    return response


@app.route("/api/show_tree", methods=['POST'])
def show_enhanced_tree_dict():
    try:
        start = time.time()
        logging.info('Show tree requested')
        params = get_params(request)
        init_forest(data_path)
        root = params['root']
        context = params.get('context', 0)
        max_depth = params.get('max_depth', 9999)
        tree_dict = forest.get_tree_dict(forest.roots[root], max_depth=max_depth, context=context)
        _forest = Forest(tree_dict=tree_dict, lexicon=lexicon)

        mode = params.get('vis_mode', 'image')
        if mode == 'image':
            _forest.visualize(filename=TEMP_FN_SVG)
            response = send_file(TEMP_FN_SVG)
        elif mode == 'text':
            return_type = params.get('HTTP_ACCEPT', False) or 'application/json'
            json_data = json.dumps(filter_result(make_serializable(params)))
            response = Response(json_data, mimetype=return_type)
        else:
            ValueError('Unknown mode=%s. Use "image" (default) or "text".')
        logging.info("Time spent handling the request: %f" % (time.time() - start))
    except Exception as e:
        raise InvalidUsage(e.message)
    return response


@app.route("/api/load", methods=['POST'])
def load_data_source():
    try:
        start = time.time()
        logging.info('Reload requested')
        params = get_params(request)
        path = params['path']
        main(path)

        logging.info("Time spent handling the request: %f" % (time.time() - start))
    except Exception as e:
        raise InvalidUsage(e.message)
    return "reload successful"


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


# unused
def all_subclasses(cls):
    return cls.__subclasses__() + [g for s in cls.__subclasses__() for g in all_subclasses(s)]


def init_nlp():
    global nlp

    if nlp is None:
        logging.info('load spacy ...')
        nlp = spacy.load('en')
        #nlp.pipeline = [nlp.tagger, nlp.entity, nlp.parser]


def init_forest(data_path):
    global forest
    if forest is None:
        assert data_path is not None, 'No data loaded. Use /api/load to load a corpus.'
        assert Forest.exist(data_path), 'Could not open corpus: %s' % data_path
        forest = Forest(filename=data_path, lexicon=lexicon)


def main(data_source):
    global sess, model_tree, lexicon, data_path, forest
    sess = None
    model_tree = None
    lexicon = None
    forest = None
    data_path = None

    if data_source is None:
        logging.info('Start api without data source. Use /api/load before any other request.')
        return

    # We retrieve our checkpoint fullpath
    checkpoint = tf.train.get_checkpoint_state(data_source)
    # if a checkpoint file exist, take data_source as model dir
    if checkpoint:
        logging.info('Model checkpoint found in "%s". Load the tensorflow model.' % data_source)
        # use latest checkpoint in data_source
        input_checkpoint = checkpoint.model_checkpoint_path
        lexicon = Lexicon(filename=os.path.join(data_source, 'model'))
        reader = tf.train.NewCheckpointReader(input_checkpoint)
        logging.info('extract embeddings from model: ' + input_checkpoint + ' ...')
        lexicon.init_vecs(checkpoint_reader=reader)
    # take data_source as corpus path
    else:
        data_path = data_source
        logging.info('No model checkpoint found in "%s". Load as train data corpus.' % data_source)
        assert Lexicon.exist(data_source, types_only=True), 'No lexicon found at: %s' % data_source
        logging.info('load lexicon from: %s' % data_source)
        lexicon = Lexicon(filename=data_source, load_vecs=False)

    if FLAGS.external_lexicon:
        logging.info('read external types: ' + FLAGS.external_lexicon + '.type ...')
        lexicon_external = Lexicon(filename=FLAGS.external_lexicon)
        lexicon.merge(lexicon_external, add=True, remove=False)
    if FLAGS.merge_nlp_lexicon:
        logging.info('extract nlp embeddings and types ...')
        init_nlp()
        lexicon_nlp = Lexicon(nlp_vocab=nlp.vocab)
        logging.info('merge nlp embeddings into loaded embeddings ...')
        lexicon.merge(lexicon_nlp, add=True, remove=False)

    # has to happen after integration of additional lexicon data (external_lexicon or merge_nlp_lexicon)
    if not checkpoint:
        lexicon.replicate_types(suffix=constants.SEPARATOR + constants.vocab_manual[constants.BACK_EMBEDDING])
    else:
        lexicon.pad()
        assert lexicon.is_filled, 'lexicon: not all vecs for all types are set (len(types): %i, len(vecs): %i)' % \
                                  (len(lexicon), len(lexicon.vecs))

    # load model
    if checkpoint:
        import model_fold
        model_config = Config(logdir_continue=data_source)
        data_path = model_config.train_data_path
        tree_embedder = getattr(model_fold, model_config.tree_embedder)

        saved_shapes = reader.get_variable_to_shape_map()
        scoring_var_names = [vn for vn in saved_shapes if vn.startswith(model_fold.DEFAULT_SCOPE_SCORING)]
        if len(scoring_var_names) > 0:
            logging.info('found scoring vars: ' + ', '.join(scoring_var_names) + '. Enable scoring functionality.')
        else:
            logging.info('no scoring vars found. Disable scoring functionality.')

        # sim_measure = getattr(model_fold, model_config.sim_measure)

        with tf.Graph().as_default():
            with tf.device(tf.train.replica_device_setter(FLAGS.ps_tasks)):
                logging.debug('trainable lexicon entries: %i' % lexicon.len_var)
                logging.debug('fixed lexicon entries:     %i' % lexicon.len_fixed)
                model_tree = model_fold.SequenceTreeModel(lex_size_fix=lexicon.len_fixed,
                                                          lex_size_var=lexicon.len_var,
                                                          dimension_embeddings=lexicon.vec_size,
                                                          tree_embedder=tree_embedder,
                                                          state_size=model_config.state_size,
                                                          #lexicon_trainable=False,
                                                          leaf_fc_size=model_config.leaf_fc_size,
                                                          root_fc_size=model_config.root_fc_size,
                                                          keep_prob=1.0,
                                                          tree_count=1,
                                                          prob_count=0)

                if FLAGS.external_lexicon or FLAGS.merge_nlp_lexicon:
                    vars_all = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
                    vars_without_embed = [v for v in vars_all if v != model_tree.embedder.lexicon_var]
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
                    sess.run([model_tree.embedder.lexicon_var_init, model_tree.embedder.lexicon_fix_init],
                             feed_dict={model_tree.embedder.lexicon_var_placeholder: lexicon.vecs_var,
                                        model_tree.embedder.lexicon_fix_placeholder: lexicon.vecs_fixed})

                if FLAGS.save_final_model_path:
                    logging.info('save final model to: ' + FLAGS.save_final_model_path + ' ...')
                    saver_final = tf.train.Saver()
                    saver_final.save(sess, FLAGS.save_final_model_path, write_meta_graph=False, write_state=False)
                    lexicon.dump(FLAGS.save_final_model_path)
                # clear vecs to clean up memory
                lexicon.init_vecs()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    #mytools.logging_init()
    FLAGS._parse_flags()
    main(FLAGS.data_source)
    logging.info('Starting the API')
    app.run(host='0.0.0.0', port=5000)
