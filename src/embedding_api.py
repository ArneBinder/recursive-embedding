from __future__ import print_function

import ast
import json
import logging
import os
import scipy
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
import preprocessing
from lexicon import Lexicon
from sequence_trees import Forest
from config import Config
from constants import TYPE_REF, TYPE_REF_SEEALSO, DTYPE_HASH, DTYPE_IDX, DTYPE_OFFSET, KEY_HEAD, KEY_CHILDREN, M_TREES, \
    M_TRAIN, M_TEST, M_INDICES, FN_TREE_INDICES
import data_iterators
from data_iterators import CONTEXT_ROOT_OFFEST
import data_iterators as diter
from train_fold import get_lexicon, create_models, convert_sparse_matrix_to_sparse_tensor

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
model_tuple = None
lexicon = None
forest = None
data_path = None
tfidf_data = None
tfidf_indices = None
tfidf_root_ids = None
embedding_indices = None
embeddings = None


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


def parse_iterator(sequences, sentence_processor, concat_mode, inner_concat_mode, expand_lexicon=False):
    init_nlp()
    for s in sequences:
        _forest = lexicon.read_data(reader=preprocessing.identity_reader,
                                    sentence_processor=sentence_processor,
                                    parser=nlp,
                                    reader_args={'content': s},
                                    concat_mode=concat_mode,
                                    inner_concat_mode=inner_concat_mode,
                                    expand_dict=expand_lexicon,
                                    reader_roots_args={
                                        'root_label': constants.vocab_manual[constants.IDENTITY_EMBEDDING]})
        yield _forest.forest


def get_or_calc_sequence_data(params):
    global data_path, lexicon

    if params.get('clear_lexicon', 'false').lower() in ['true', '1'] or lexicon is None:
        lexicon = Lexicon()
        params['clear_lexicon'] = 'true'
    else:
        params['clear_lexicon'] = 'false'

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

        params['data_sequences'] = list(parse_iterator(sequences, sentence_processor, concat_mode, inner_concat_mode,
                                                       expand_lexicon=params.get('clear_lexicon', 'false').lower() in ['true', '1']))

    if 'data_sequences' in params:
        d_list, p_list = zip(*params['data_sequences'])
        data_as_hashes = params.get('data_as_hashes', False)
        root_ids = params.get('root_ids', None)
        #current_forest = Forest(data=sum(d_list, []), parents=sum(p_list, []), lexicon=lexicon,
        #                        data_as_hashes=data_as_hashes, root_ids=root_ids)
        current_forest = Forest(data=np.concatenate(d_list), parents=np.concatenate(p_list), lexicon=lexicon,
                                data_as_hashes=data_as_hashes, root_ids=root_ids)
    else:
        init_forest(data_path)
        current_forest = forest

    if 'indices_getter' in params:

        #if not os.path.isfile(fn):
        #    raise IOError('could not open idx_file=%s' % fn)
        #assert 'data_iterator' in params, 'parameter data_iterator is not given, can not iterate idx_file'
        #assert 'indices_getter' in params, 'parameter indices_getter is not given, can not iterate idx_file'
        indices_getter = getattr(data_iterators, params['indices_getter'])

        if current_forest.data_as_hashes:
            current_forest.hashes_to_indices()

        max_depth = params.get('max_depth', 100)
        context = params.get('context', 0)
        #bag_of_seealsos = (params.get('bag_of_seealsos', 'true').lower() in ['true', '1'])

        params['transformed_idx'] = True
        tree_iterator_args = {'sequence_trees': current_forest, 'max_depth': max_depth,
                              'context': context, 'transform': params['transformed_idx'],
                              #'bag_of_seealsos': bag_of_seealsos
                              }

        if 'link_cost_ref' in params:
            tree_iterator_args['link_cost_ref'] = params['link_cost_ref']
        if 'link_cost_ref_seealso' in params:
            tree_iterator_args['link_cost_ref_seealso'] = params['link_cost_ref_seealso']
        if 'concat_mode' in params:
            tree_iterator_args['concat_mode'] = params['concat_mode']

        params['sequences'] = []
        params['data_sequences'] = []
        params['scores_gold'] = []

        params['data_as_hashes'] = current_forest.data_as_hashes
        tuple_start = params.get('tuple_start', 0)
        tuple_end = params.get('tuple_end', -1)

        fn = '%s.%s' % (data_path, params.get('idx_file', None))

        indices, indices_targets_unused = indices_getter(index_files=[fn], forest=current_forest)
        # set tree iterator
        tree_iter = data_iterators.tree_iterator(indices=indices, forest=current_forest, **tree_iterator_args)

        for i, tree_dict in enumerate(tree_iter):
            if i < tuple_start:
                continue
            if 0 <= tuple_end <= i:
                break
            #for tree_dict in tree_dicts:
            vis_forest = Forest(tree_dict=tree_dict, lexicon=current_forest.lexicon,
                                data_as_hashes=current_forest.data_as_hashes, root_ids=current_forest.root_ids,
                                lexicon_roots=current_forest.lexicon_roots)

            params['data_sequences'].append([vis_forest.data, vis_forest.parents])
            token_list = vis_forest.get_text_plain(blacklist=params.get('prefix_blacklist', None),
                                                   transformed=params['transformed_idx'])
            params['sequences'].append(token_list)
            #for prob in probs:
            #    params['scores_gold'].append(prob)
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
                                data_as_hashes=params['data_as_hashes'], root_ids=current_forest.root_ids,
                                lexicon_roots=current_forest.lexicon_roots).get_text_plain(
                blacklist=params.get('prefix_blacklist', None))
            params['sequences'].append(token_list)
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

        if current_forest.data_as_hashes:
            current_forest.hashes_to_indices()

        d_ref = lexicon.get_d(TYPE_REF, data_as_hashes=current_forest.data_as_hashes)
        d_ref_seealso = lexicon.get_d(TYPE_REF_SEEALSO, data_as_hashes=current_forest.data_as_hashes)
        costs = {}
        if 'link_cost_ref' in params:
            costs[d_ref] = params['link_cost_ref']
        if 'link_cost_ref_seealso' in params:
            costs[d_ref_seealso] = params['link_cost_ref_seealso']
        params['transformed_idx'] = True
        tree_dict = current_forest.get_tree_dict(idx=idx, max_depth=max_depth, context=context,
                                                 transform=params['transformed_idx'],
                                                 costs=costs, link_types=[d_ref, d_ref_seealso])
        vis_forest = Forest(tree_dict=tree_dict, lexicon=current_forest.lexicon,
                            data_as_hashes=current_forest.data_as_hashes, root_ids=current_forest.root_ids,
                            lexicon_roots=current_forest.lexicon_roots)
        params['data_as_hashes'] = vis_forest.data_as_hashes
        params['data_sequences'] = [[vis_forest.data, vis_forest.parents]]
        token_list = vis_forest.get_text_plain(blacklist=params.get('prefix_blacklist', None),
                                               transformed=params['transformed_idx'])
        params['sequences'] = [token_list]
    elif 'reroot_start' in params:
        if current_forest.data_as_hashes:
            current_forest.hashes_to_indices()

        tree_start = params.get('reroot_start', 0)
        tree_end = params.get('reroot_end', len(current_forest))
        max_depth = params.get('max_depth', 100)
        #context = params.get('context', 0)

        params['transformed_idx'] = True
        params['sequences'] = []
        params['data_sequences'] = []
        params['data_as_hashes'] = current_forest.data_as_hashes

        for i, [(tree_dict_children, candidate_heads), probs] \
                in enumerate(data_iterators.data_tuple_iterator_reroot(sequence_trees=current_forest, neg_samples=10,
                                                                       max_tries=10, max_depth=max_depth,
                                                                       link_cost_ref=params.get('link_cost_ref', None),
                                                                       link_cost_ref_seealso=params.get('link_cost_ref_seealso', -1),
                                                                       transform=params['transformed_idx'])):
            if i < tree_start:
                continue
            if 0 <= tree_end <= i:
                break

            tree_dict = {KEY_HEAD: candidate_heads[0], KEY_CHILDREN: tree_dict_children}
            vis_forest = Forest(tree_dict=tree_dict, lexicon=current_forest.lexicon,
                                data_as_hashes=current_forest.data_as_hashes,
                                root_ids=current_forest.root_ids,
                                lexicon_roots=current_forest.lexicon_roots)
            params['data_sequences'].append([vis_forest.data, vis_forest.parents])
            token_list = vis_forest.get_text_plain(blacklist=params.get('prefix_blacklist', None),
                                                   transformed=params['transformed_idx'])
            params['sequences'].append(token_list)

            canidates_forest = Forest(data=candidate_heads[1:],
                                      parents=np.zeros(shape=len(candidate_heads)-1, dtype=DTYPE_OFFSET),
                                      lexicon=current_forest.lexicon, data_as_hashes=current_forest.data_as_hashes,
                                      root_ids=current_forest.root_ids, lexicon_roots=current_forest.lexicon_roots
                                      )
            params['data_sequences'].append([canidates_forest.data, canidates_forest.parents])
            token_list = canidates_forest.get_text_plain(blacklist=params.get('prefix_blacklist', None),
                                                         transformed=params['transformed_idx'])
            params['sequences'].append(token_list)


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
        # TODO: rework! (add link_types and costs)
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


def calc_missing_embeddings(indices, forest, concat_mode, model_tree, max_depth=10, batch_size=100):
    global embeddings, embedding_indices
    if embedding_indices is None:
        new_indices = indices
    else:
        new_indices_candidates = indices
        mask_new = ~np.isin(new_indices_candidates, embedding_indices)
        new_indices = new_indices_candidates[mask_new]
    logging.debug('checked existing embeddings')

    _embeddings = []
    if tfidf_indices is not None:
        _mapping = {_idx: i for i, _idx in enumerate(tfidf_indices)}
        _indices_tfidf_data = np.array([_mapping[_idx] for _idx in new_indices])
        for start in range(0, len(new_indices), batch_size):
            current_indices = _indices_tfidf_data[start:start+batch_size]
            selected_tfidf_data = tfidf_data[current_indices]
            feed_dict = {model_tree.embeddings_placeholder: convert_sparse_matrix_to_sparse_tensor(selected_tfidf_data)}
            _embeddings.append(sess.run(model_tree.embeddings_all, feed_dict))
    else:
        tree_iterator = diter.tree_iterator(indices=new_indices, forest=forest, concat_mode=concat_mode,
                                            max_depth=max_depth)
        with model_tree.compiler.multiprocessing_pool():
            trees_compiled_iter = model_tree.compiler.build_loom_inputs(map(lambda x: [x], tree_iterator), ordered=True)
            for start in range(0, len(new_indices), batch_size):
                current_size = min(batch_size, len(new_indices) - start)
                current_trees = [trees_compiled_iter.next() for _ in range(current_size)]
                feed_dict = {model_tree.compiler.loom_input_tensor: [current_trees]}
                _embeddings.append(sess.run(model_tree.embeddings_all, feed_dict))

    if len(_embeddings) > 0:
        new_embeddings = np.concatenate(_embeddings)
        logging.debug('%i new embeddings calculated' % len(new_embeddings))
        if embeddings is None:
            embeddings = new_embeddings
            embedding_indices = new_indices
        else:
            embeddings = np.concatenate((embeddings, new_embeddings))
            embedding_indices = np.concatenate((embedding_indices, new_indices))
    else:
        logging.debug('no new embeddings calculated')


def calc_tuple_scores(root_id, root_ids_target, forest, concat_mode, model_tree, model_tuple, max_depth=10, batch_size=100):

    root_ids = np.concatenate(([root_id], root_ids_target))
    root_indices = forest.roots[root_ids]
    scoring_indices = root_indices + CONTEXT_ROOT_OFFEST

    calc_missing_embeddings(indices=scoring_indices, forest=forest, concat_mode=concat_mode, model_tree=model_tree,
                            max_depth=max_depth, batch_size=batch_size)

    embedding_indices_mapping = {idx: i for i, idx in enumerate(embedding_indices)}
    indices_scoring_embeddings = np.array([embedding_indices_mapping[idx] for idx in scoring_indices], dtype=np.int32)
    logging.debug('mappings calculated')

    scoring_embeddings = embeddings[indices_scoring_embeddings]
    _scores = []
    embedding_src = scoring_embeddings[0, :]
    for start in range(0, len(root_ids)-1, batch_size):
        current_embeddings = np.concatenate(([embedding_src], scoring_embeddings[start+1:start+batch_size+1, :]))
        feed_dict = {model_tuple.tree_model.embeddings_placeholder: current_embeddings,
                     model_tuple.values_gold: np.zeros(shape=current_embeddings.shape[0]-1)
                     #model_tuple.candidate_count: len(current_embeddings) - 1
                     }
        _scores.append(sess.run(model_tuple.values_predicted, feed_dict).flatten())
    if len(_scores) > 0:
        scores = np.concatenate(_scores)
    else:
        scores = np.zeros(shape=0, dtype=np.float32)
    logging.debug('scores calculated')

    # get true seealsos
    seealso_root_idx = forest.roots[root_id] + diter.SEEALSO_ROOT_OFFSET
    seealso_root_ids = diter.link_root_ids_iterator(indices=[seealso_root_idx], forest=forest, link_type=TYPE_REF_SEEALSO).next()
    if seealso_root_ids is None:
        seealso_root_ids = []

    return scores, seealso_root_ids


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


def current_root_ids():
    if tfidf_root_ids is not None:
        return tfidf_root_ids
    else:
        return np.arange(len(forest.roots), dtype=np.int32)


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
    global lexicon
    params = None
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
                forest_temp = Forest(forest=data_sequence, lexicon=lexicon, data_as_hashes=params.get('data_as_hashes', False),
                                     root_ids=root_ids)
                forest_temp.visualize(TEMP_FN_SVG + '.' + str(i), transformed=params.get('transformed_idx', False),
                                      token_list=params['sequences'][i])
            assert len(params['data_sequences']) > 0, 'empty data_sequences'
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
        if params is not None and params.get('clear_lexicon', 'false').lower() in ['true', '1']:
            lexicon = None
        raise InvalidUsage(e.message)
    if params.get('clear_lexicon', 'false').lower() in ['true', '1']:
        lexicon = None
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


@app.route("/api/roots", methods=['GET'])
def show_roots():
    try:
        start = time.time()
        logging.info('Show roots requested')
        params = get_params(request)
        init_forest(data_path)
        assert forest.lexicon_roots is not None, 'lexicon_roots is None. Load data with lexicon_roots via /api/load.'
        root_start = params.get('root_start', 0)
        all_root_ids = current_root_ids()
        root_end = params.get('root_end', len(all_root_ids))
        indices = np.arange(root_start, root_end, dtype=np.int32)
        root_ids = all_root_ids[indices]
        root_strings = [forest.lexicon_roots.get_s(_id, data_as_hashes=False) for _id in root_ids]

        params['root_ids'] = root_ids.tolist()
        params['root_strings'] = root_strings
        return_type = params.get('HTTP_ACCEPT', False) or 'application/json'
        json_data = json.dumps(filter_result(make_serializable(params)))
        response = Response(json_data, mimetype=return_type)

        logging.info("Time spent handling the request: %f" % (time.time() - start))
    except Exception as e:
        raise InvalidUsage(e.message)
    return response


@app.route("/api/tuple_scores", methods=['GET'])
def get_tuple_scores():
    try:
        start = time.time()
        logging.info('Tuple scoring requested')
        params = get_params(request)
        init_forest(data_path)

        root_id = params['root_id']
        all_root_ids = current_root_ids()
        root_ids_target_nbr = params.get('root_ids_target_nbr', len(all_root_ids))
        root_ids_target = params.get('root_ids_target', all_root_ids[np.arange(root_ids_target_nbr)])
        concat_mode = params.get('concat_mode', 'tree')
        max_depth = params.get('max_depth', 10)
        top = params.get('top', len(root_ids_target))
        batch_size = params.get('batch_size', 100)
        _scores, seealso_root_ids = calc_tuple_scores(root_id=root_id, root_ids_target=root_ids_target, forest=forest,
                                                      model_tree=model_tree, model_tuple=model_tuple,
                                                      concat_mode=concat_mode, max_depth=max_depth,
                                                      batch_size=batch_size)
        params['root_ids_seealso'] = seealso_root_ids
        params['root_ids_seealso_string'] = [forest.lexicon_roots.get_s(_id, data_as_hashes=False) for _id in seealso_root_ids]
        scores = _scores.flatten()


        indices_sorted = np.argsort(scores)[::-1][:top]

        params['tuple_scores'] = scores[indices_sorted]
        params['root_ids_target'] = np.array(root_ids_target)[indices_sorted].tolist()

        #if forest.lexicon_roots is not None:
        params['root_id_string'] = forest.lexicon_roots.get_s(root_id, data_as_hashes=False)
        params['root_ids_target_string'] = np.array([forest.lexicon_roots.get_s(_id, data_as_hashes=False) for _id in root_ids_target])[indices_sorted].tolist()

        #params['merged'] = [[params['tuple_scores'][i], params['root_ids_target'][i], params['root_ids_target_string'][i]] for i in range(len(indices_sorted))]
        params['merged'] = ["%10i: %.4f %s" % (params['root_ids_target'][i], params['tuple_scores'][i], params['root_ids_target_string'][i]) for i in range(len(indices_sorted))]

        return_type = params.get('HTTP_ACCEPT', False) or 'application/json'
        json_data = json.dumps(filter_result(make_serializable(params)))
        response = Response(json_data, mimetype=return_type)

        logging.info("Time spent handling the request: %f" % (time.time() - start))
    except Exception as e:
        raise InvalidUsage(e.message)
    return response


@app.route("/api/clear", methods=['POST'])
def clear_cached_embeddings():
    global embeddings, embedding_indices
    try:
        start = time.time()
        logging.info('Clear embeddings requested')
        embeddings = None
        embedding_indices = None
        logging.info("Time spent handling the request: %f" % (time.time() - start))
    except Exception as e:
        raise InvalidUsage(e.message)
    return "clearing embeddings successful"


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
    global forest, tfidf_root_ids
    if forest is None:
        assert data_path is not None, 'No data loaded. Use /api/load to load a corpus.'
        assert Forest.exist(data_path), 'Could not open corpus: %s' % data_path
        lexicon_root_fn = '%s.root.id' % data_path
        if Lexicon.exist(lexicon_root_fn, types_only=True):
            logging.info('load lexicon_roots from %s' % lexicon_root_fn)
            lexicon_roots = Lexicon(filename=lexicon_root_fn, load_vecs=False)
        else:
            lexicon_roots = None
        forest = Forest(filename=data_path, lexicon=lexicon, lexicon_roots=lexicon_roots)
        if tfidf_indices is not None:
            tfidf_root_ids = np.array([forest.root_mapping[root_idx] for root_idx in (tfidf_indices - CONTEXT_ROOT_OFFEST)])
        else:
            tfidf_root_ids = None


def main(data_source):
    global sess, model_tree, model_tuple, lexicon, data_path, forest, tfidf_data, tfidf_indices, tfidf_root_ids, embedding_indices, embeddings
    sess = None
    model_tree = None
    model_tuple = None
    lexicon = None
    forest = None
    data_path = None
    tfidf_data = None
    tfidf_indices = None
    tfidf_root_ids = None
    embedding_indices = None
    embeddings = None

    if data_source is None:
        logging.info('Start api without data source. Use /api/load before any other request.')
        return

    lexicon, checkpoint_fn, _ = get_lexicon(logdir=data_source, train_data_path=data_source, dont_dump=True)
    if checkpoint_fn:
        assert lexicon.vecs is None or lexicon.is_filled, \
            'lexicon: not all vecs for all types are set (len(types): %i, len(vecs): %i)' \
            % (len(lexicon), len(lexicon.vecs))

    # load model
    if checkpoint_fn:
        model_config = Config(logdir_continue=data_source)
        data_path = model_config.train_data_path

        with tf.Graph().as_default() as graph:
            with tf.device(tf.train.replica_device_setter(FLAGS.ps_tasks)):
                logging.debug('trainable lexicon entries: %i' % lexicon.len_var)
                logging.debug('fixed lexicon entries:     %i' % lexicon.len_fixed)

                assert model_config.model_type == 'tuple', 'only model_type=tuple implemented'
                model_tree, model_tuple, prepared_embeddings, tree_indices = create_models(
                    config=model_config, lexicon=lexicon, tree_count=1, tree_iterators={}, tree_indices=None,
                    logdir=data_source, use_inception_tree_model=True)

                if model_config.tree_embedder == 'tfidf':
                    _indices, _tfidf = zip(*[(tree_indices[m], prepared_embeddings[m]) for m in tree_indices.keys()])
                    tfidf_data = scipy.sparse.vstack(_tfidf)
                    logging.info('total tfidf shape: %s' % str(tfidf_data.shape))
                    tfidf_indices = np.concatenate(_indices)
                    logging.debug('number of tfidf_indices: %i' % len(tfidf_indices))

                # TODO: still ok?
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
                    logging.info('restore model from: %s ...' % checkpoint_fn)
                    saver.restore(sess, checkpoint_fn)

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
