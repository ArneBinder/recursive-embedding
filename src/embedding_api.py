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
import preprocessing
from lexicon import Lexicon
from sequence_trees import Forest, concatenate_structures
from config import Config
from constants import TYPE_REF_SEEALSO, DTYPE_OFFSET, DTYPE_IDX, KEY_HEAD, KEY_CHILDREN, KEY_CANDIDATES, \
    LOGGING_FORMAT, vocab_manual, IDENTITY_EMBEDDING, TYPE_PARAGRAPH, SEPARATOR, TYPE_LEXEME, MT_CANDIDATES, \
    OFFSET_ID, BASE_TYPES
import data_iterators
from data_iterators import OFFSET_CONTEXT_ROOT
import data_iterators as diter
from src.mytools import numpy_dump
from train_fold import get_lexicon, create_models, convert_sparse_matrix_to_sparse_tensor, init_model_type

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
model_main = None
lexicon = None
lexicon_dims = None
forest = None
data_path = None
tfidf_data = None
tfidf_indices = None
tfidf_root_ids = None
embedding_indices = None
embeddings = None
model_config = None
nbr_candidates = None

logger = logging.getLogger('')
logger.setLevel(logging.DEBUG)
logger_streamhandler = logging.StreamHandler()
logger_streamhandler.setLevel(logging.DEBUG)
logger_streamhandler.setFormatter(logging.Formatter(LOGGING_FORMAT))


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
    try:
        json.dumps(d)
    except TypeError:
        if type(d) == dict:
            for k in d:
                d[k] = make_serializable(d[k])
        elif type(d) == list:
            d = [make_serializable(d[i]) for i in range(len(d))]
        elif type(d) == tuple:
            d = tuple([make_serializable(d[i]) for i in range(len(d))])
        elif type(d) == np.ndarray:
            d = make_serializable(d.tolist())

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


def parse_iterator(sequences, sentence_processor, concat_mode, inner_concat_mode, root_label=TYPE_PARAGRAPH,
                   expand_lexicon=False):
    global lexicon
    init_nlp()
    for s in sequences:
        _forest = lexicon.read_data(reader=preprocessing.identity_reader,
                                    sentence_processor=sentence_processor,
                                    parser=nlp,
                                    reader_args={'content': s},
                                    concat_mode=concat_mode,
                                    inner_concat_mode=inner_concat_mode,
                                    expand_dict=expand_lexicon or len(lexicon) == 0,
                                    reader_roots_args={'root_label': root_label or TYPE_PARAGRAPH})
        yield _forest.forest


def get_tree_dicts_for_indices_from_forest(indices, current_forest, params, transform):

    context = params.get('context', 0)
    transform = transform or context > 0

    if current_forest.data_as_hashes:
        current_forest.hashes_to_indices()

    transformed = params.get('reroot', False) or transform

    blank_ids = set()
    blank_strings = set()
    for prefix in params.get('blank', ()):
        _ids, _id_strings = lexicon.get_ids_for_prefix(prefix)
        blank_ids.update(_ids)
        blank_strings.update(_id_strings)
    logging.info('blank %i types: %s' % (len(blank_ids), ', '.join(blank_strings)))
    blank_types = set([lexicon.get_d(s=s, data_as_hashes=False) for s in blank_strings])

    tree_dicts = []
    for tree_dict in data_iterators.tree_iterator(
            indices, current_forest, concat_mode=params.get('concat_mode', constants.CM_TREE), context=context,
            max_depth=params.get('max_depth', 10), transform=transform,
            link_cost_ref=params.get('link_cost_ref', None),
            link_cost_ref_seealso=params.get('link_cost_ref_seealso', None), reroot=params.get('reroot', False),
            max_size_plain=1000, keep_prob_blank=params.get('keep_prob_blank', 1.0), keep_prob_node=params.get('keep_prob_node', 1.0),
            blank_types=blank_types):
        tree_dicts.append(tree_dict)

    return tree_dicts, transformed


def get_or_calc_tree_dicts_or_forests(params):
    global data_path, lexicon

    if params.get('clear_lexicon', False):
        lexicon = Lexicon()

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
                                                       expand_lexicon=params.get('clear_lexicon', False),
                                                       root_label=params.get('root_label', None)))

    if 'data_sequences' in params:
        d_list, structure_list = zip(*params['data_sequences'])
        structure = concatenate_structures(structure_list)
        params['data_as_hashes'] = params.get('data_as_hashes', False)
        current_forest = Forest(data=np.concatenate(d_list), structure=structure,
                                lexicon=lexicon, data_as_hashes=params['data_as_hashes'],
                                root_ids=params.get('root_ids', None))
    else:
        init_forest(data_path)
        current_forest = forest
        params['data_as_hashes'] = current_forest.data_as_hashes
        params['root_ids'] = current_forest.root_data

    _forests = None
    #params['transformed_idx'] = False

    if 'indices_getter' in params:

        #if not os.path.isfile(fn):
        #    raise IOError('could not open idx_file=%s' % fn)
        #assert 'data_iterator' in params, 'parameter data_iterator is not given, can not iterate idx_file'
        #assert 'indices_getter' in params, 'parameter indices_getter is not given, can not iterate idx_file'
        indices_getter = getattr(data_iterators, params['indices_getter'])

        if current_forest.data_as_hashes:
            current_forest.hashes_to_indices()
        params['data_as_hashes'] = current_forest.data_as_hashes

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

        params['scores_gold'] = []

        tuple_start = params.get('tuple_start', 0)
        tuple_end = params.get('tuple_end', -1)

        fn = '%s.%s' % (data_path, params.get('idx_file', None))

        indices, indices_targets_unused = indices_getter(index_files=[fn], forest=current_forest)
        # set tree iterator
        tree_iter = data_iterators.tree_iterator(indices=indices, forest=current_forest, **tree_iterator_args)
        _forests = []
        for i, tree_dict in enumerate(tree_iter):
            if i < tuple_start:
                continue
            if 0 <= tuple_end <= i:
                break
            #for tree_dict in tree_dicts:
            vis_forest = Forest(tree_dict=tree_dict, lexicon=current_forest.lexicon,
                                data_as_hashes=current_forest.data_as_hashes, #root_ids=current_forest.root_data,
                                lexicon_roots=current_forest.lexicon_roots)

            _forests.append(vis_forest)

    elif 'root_start' in params:
        roots = current_forest.roots
        root_start = params.get('root_start', 0)
        root_end = params.get('root_end', len(roots))
        assert root_start <= root_end, 'ERROR: root_start=%i > root_end=%i' % (root_start, root_end)
        assert root_end <= len(roots), 'ERROR: root_end=%i > len(roots)=%i' % (root_end, len(roots))

        _forests = [current_forest.get_slice(root=root, show_links=params.get('show_links', True)) for root in roots[root_start:root_end]]
        params['indices'] = current_forest.roots[root_start:root_end]
    elif 'idx_start' in params:
        idx_start = params.get('idx_start', 0)
        idx_end = params.get('idx_end', len(current_forest))
        assert idx_start <= idx_end, 'ERROR: idx_start=%i > idx_end=%i' % (idx_start, idx_end)
        assert idx_end <= len(current_forest), 'ERROR: root_end=%i > len(roots)=%i' % (idx_end, len(current_forest))

        params['data_as_hashes'] = current_forest.data_as_hashes

        _forests = [current_forest.get_slice(indices=np.arange(start=idx_start, stop=idx_end))]
    elif 'idx' in params:
        params['tree_dicts'], params['transformed_idx'] = get_tree_dicts_for_indices_from_forest(
            indices=[params['idx']], current_forest=current_forest,params=params, transform=False)
        params['indices'] = [params['idx']]
    elif 'indices' in params:
        params['tree_dicts'], params['transformed_idx'] = get_tree_dicts_for_indices_from_forest(
            indices=params['indices'], current_forest=current_forest, params=params, transform=False)
    elif 'indices_prefixes' in params:
        ids = current_forest.lexicon.get_ids_for_prefixes_or_types(prefixes_or_types=params['indices_prefixes'],
                                                                   data_as_hashes=current_forest.data_as_hashes)
        mask = np.isin(current_forest.data, ids)
        indices = np.nonzero(mask)[0]
        params['indices'] = indices
        logger.info('create %i tree_dicts ...' % len(indices))
        params['tree_dicts'], params['transformed_idx'] = get_tree_dicts_for_indices_from_forest(indices=indices,
                                                                                                 current_forest=current_forest,
                                                                                                 params=params,
                                                                                                 transform=False)
    else:
        _forests = [current_forest]
        params['transformed_idx'] = False
        params['indices'] = [current_forest.roots[0]]

    #if return_forests and 'tree_dicts' in params:
    #    _forests = tree_dicts_to_forests(params['tree_dicts'], current_forest)

    if 'candidates_prefixes' in params:
        params['candidates_data'] = current_forest.lexicon.get_ids_for_prefixes_or_types(
            prefixes_or_types=params['candidates_prefixes'], data_as_hashes=current_forest.data_as_hashes)
    elif 'candidates' in params:
        # convert to lexemes if not already an url (starting with "http://") or a type (prefixed with a BASE_TYPE)
        params['candidates'] = [TYPE_LEXEME + SEPARATOR + unicode(c)
                                if not any([c.startswith(base_type) for base_type in ['http://'] + BASE_TYPES])
                                else unicode(c) for c in params['candidates']]
        params['candidates_data'] = [current_forest.lexicon.get_d(s=s, data_as_hashes=current_forest.data_as_hashes) for s in
                                     params['candidates']]
    if _forests is not None:
        params['forests'] = _forests
        params['sequences'] = [f.get_text_plain(blacklist=params.get('prefix_blacklist', None), transformed=params.get('transformed_idx', False)) for f in _forests]
        #params['data_sequences'], params['sequences'] = zip(*[([f.data, f.graph_out], f.get_text_plain(blacklist=params.get('prefix_blacklist', None), transformed=params.get('transformed_idx', False))) for f in _forests])
    #if params.get('calc_depths', False):
    #    params['depths'] = [f.depths for f in _forests]
    #if params.get('calc_depths_max', False):
    #    params['depths_max'] = np.array([np.max(f.depths) for f in _forests])
    return current_forest


def calc_embeddings(tree_dicts_or_forests, transformed, root_ids=None, max_depth=20):
    assert model_tree is not None, 'No model loaded. To load a model, use endpoint: /api/load?path=path_to_model'

    # TODO: rework! (add link_types and costs)
    batch = []
    logger.info('calculate %i embeddings ...' % len(tree_dicts_or_forests))
    for i, tree_dict_or_forest in enumerate(tree_dicts_or_forests):
        if isinstance(tree_dict_or_forest, dict):
            tree_dict = tree_dict_or_forest
        else:
            if isinstance(tree_dict_or_forest, Forest):
                tree = tree_dict_or_forest
            else:
                tree = Forest(forest=tree_dict_or_forest, lexicon=lexicon, root_ids=root_ids)

            #tree.visualize(filename='debug_%d.svg' % i, transformed=transformed)
            tree_dict = tree.get_tree_dict(idx=tree.roots[0], max_depth=max_depth, transform=not transformed)
        # add correct root as candidate (if HTUBatchedHead model is used)
        tree_dict[KEY_CANDIDATES] = [tree_dict[KEY_HEAD]]
        batch.append([tree_dict])

    if len(batch) > 0:
        logger.debug('compile batch ...')
        fdict = model_tree.build_feed_dict(batch)
        logger.debug('calculate embeddings for batch ...')
        embeddings_all = sess.run(model_tree.embeddings_all, feed_dict=fdict)
        embeddings = embeddings_all.reshape((-1, model_tree.tree_output_size))
    else:
        embeddings = np.zeros(shape=(0, model_tree.embedder.dimension_embeddings), dtype=np.float32)

    return embeddings


def get_or_calc_embeddings(params):
    if 'embeddings' in params:
        params['embeddings'] = np.array(params['embeddings'])
    else:
        if 'tree_dicts' not in params and 'forests' not in params:
            get_or_calc_tree_dicts_or_forests(params)

        params['embeddings'] = calc_embeddings(
            tree_dicts_or_forests=params.get('tree_dicts', None) or params['forests'],
            max_depth=int(params.get('max_depth', 20)), transformed=params['transformed_idx'])


def tree_dicts_to_forests(tree_dicts, current_forest):
    forests = [
        Forest(tree_dict=tree_dict, lexicon=current_forest.lexicon, data_as_hashes=current_forest.data_as_hashes,
               lexicon_roots=current_forest.lexicon_roots) for tree_dict in tree_dicts]
    return forests


def get_or_calc_scores(params):
    if 'scores' in params:
        params['scores'] = np.array(params['scores'])
    else:
        import model_fold

        if 'dump' in params:
            dump_dir = params['dump']
            if not os.path.exists(dump_dir):
                os.makedirs(dump_dir)
        else:
            dump_dir = None
        params['embeddings'] = []
        params['scores'] = []
        current_forest = get_or_calc_tree_dicts_or_forests(params)
        if 'candidates_data' in params:
            assert isinstance(model_tree.embedder, model_fold.TreeEmbedding_HTUBatchedHead), \
                'embedder of tree model has to be TreeEmbedding_HTUBatchedHead'
            # we modify the number of trees, so we remove other tree data as well
            if 'sequences' in params:
                del params['sequences']
            if 'data_sequences' in params:
                del params['data_sequences']

            if params.get('transformed_idx', False):
                params['candidates_data'] = lexicon.transform_indices(params['candidates_data'])
            if dump_dir is None:
                candidate_forest = Forest(data=params['candidates_data'],
                                          parents=np.zeros(len(params['candidates_data']), dtype=DTYPE_OFFSET),
                                          lexicon=lexicon)
                if 'tree_dicts' in params and 'forests' not in params:
                    params['forests'] = tree_dicts_to_forests(params['tree_dicts'], current_forest)
            new_forests = []
            heads = np.ones(len((params.get('tree_dicts', None) or params['forests'])), dtype=DTYPE_IDX) * -1
            batch = []
            for i, tree_dict_or_forest in enumerate((params.get('tree_dicts', None) or params['forests'])):
                if isinstance(tree_dict_or_forest, dict):
                    forest_dict = tree_dict_or_forest
                else:
                    forest_dict = tree_dict_or_forest.get_tree_dict(
                        idx=tree_dict_or_forest.roots[0], transform=not params.get('transformed_idx', False),
                        max_depth=params.get('max_depth', 10),
                        # TODO: check these parameters
                        context=0, costs={}, link_types=[])

                heads[i] = forest_dict[KEY_HEAD]
                if nbr_candidates is not None and nbr_candidates == len(params['candidates_data']):
                    forest_dict[KEY_CANDIDATES] = params['candidates_data']
                    batch.append([forest_dict])
                else:
                    for c in params['candidates_data']:
                        current_tree_dict = forest_dict.copy()
                        current_tree_dict[KEY_CANDIDATES] = [c]
                        batch.append([current_tree_dict])
                if dump_dir is None:
                    new_forests.append(params['forests'][i])
                    new_forests.append(candidate_forest)

            fdict_embeddings = model_tree.build_feed_dict(batch)
            embeddings_all = sess.run(model_tree.embeddings_all, feed_dict=fdict_embeddings)
            embeddings_shaped = embeddings_all.reshape((-1, model_tree.tree_output_size))

            fdict_scores = {model_main.tree_model.embeddings_all: embeddings_shaped,
                            model_main.values_gold: np.zeros(shape=(len(params['candidates_data']),), dtype=np.float32)}

            scores = sess.run(model_main.scores, feed_dict=fdict_scores)
            scores_shaped = scores.reshape((-1, len(params['candidates_data'])))

            # TODO: add normalization
            #if params.get('normalize_scores', True):
            #    current_scores = current_scores / np.sum(current_scores)

            if dump_dir is None:
                params['scores'] = []
                for current_scores in scores_shaped:
                    params['scores'].append(None)
                    params['scores'].append(current_scores)
                params['forests'] = new_forests
                if 'tree_dicts' in params:
                    del params['tree_dicts']
            else:
                numpy_dump(os.path.join(dump_dir, 'scores.head'), heads)
                numpy_dump(os.path.join(dump_dir, 'scores.value'), scores_shaped)
                numpy_dump(os.path.join(dump_dir, 'scores.candidate'), np.array(params['candidates_data']))
                if 'indices' in params:
                    numpy_dump(os.path.join(dump_dir, 'scores.idx'), np.array(params['indices']))
                    del params['indices']

            params['color_by_rank'] = True

        else:
            if params.get('transformed_idx', False):
                current_embeddings = calc_embeddings(params.get('tree_dicts', None) or params['forests'],
                                                     max_depth=int(params.get('max_depth', 20)),
                                                     transformed=params.get('transformed_idx', False))
                fdict = {model_main.tree_model.embeddings_all: current_embeddings,
                         model_main.values_gold: np.zeros(shape=(1,), dtype=np.float32)}
                current_scores = sess.run(model_main.scores, feed_dict=fdict).reshape((current_embeddings.shape[0]))
                if params.get('normalize_scores', False):
                    current_scores = current_scores / np.sum(current_scores)
                logger.info('calculated %i scores' % len(current_scores))

                if dump_dir is None:
                    forests = params.get('forests', None) or tree_dicts_to_forests(params['tree_dicts'], current_forest)
                    params['sequences'] = [[f.get_text_plain(start=f.roots[0], end=f.roots[0] + 1,
                                                             blacklist=params.get('prefix_blacklist', None),
                                                             transformed=params.get('transformed_idx', False))[0] for f in forests]]
                    params['data_sequences'] = [[[0] * len(params['sequences'][0]), [0] * len(params['sequences'][0])]]
                    #params['embeddings'].append(current_embeddings)
                    params['scores'].append(current_scores)
                else:
                    params['sequences'] = []
                    params['data_sequences'] = []
                    #numpy_dump(os.path.join(dump_dir, 'embeddings.concat'), current_embeddings)
                    numpy_dump(os.path.join(dump_dir, 'scores.value'), current_scores)
                    if 'indices' in params:
                        numpy_dump(os.path.join(dump_dir, 'scores.idx'), np.array(params['indices']))
                        del params['indices']
                params['transformed_idx'] = True
            else:
                assert not (params.get('transformed_idx', False) and params.get('reroot', False)), \
                    'can not construct reroot trees of an already transformed tree'
                params['reroot'] = True
                if 'forests' in params:
                    forests = params['forests']
                else:
                    forests = tree_dicts_to_forests(params['tree_dicts'], current_forest)

                for forest in forests:
                    # create for every node a tree_dict rooted by this node
                    # transform, if not already done
                    tree_dicts, transformed = get_tree_dicts_for_indices_from_forest(
                        indices=range(len(forest)), current_forest=forest, params=params,
                        transform=not params.get('transformed_idx', False))
                    current_embeddings = calc_embeddings(tree_dicts, max_depth=int(params.get('max_depth', 20)),
                                                         transformed=transformed)

                    fdict = {model_main.tree_model.embeddings_all: current_embeddings,
                             model_main.values_gold: np.zeros(shape=(1,), dtype=np.float32)}
                    current_scores = sess.run(model_main.scores, feed_dict=fdict).reshape((current_embeddings.shape[0]))
                    if params.get('normalize_scores', False):
                        current_scores = current_scores / np.sum(current_scores)

                    params['embeddings'].append(current_embeddings)
                    params['scores'].append(current_scores)
                if dump_dir is not None:
                    params['sequences'] = []
                    params['data_sequences'] = []
                    numpy_dump(os.path.join(dump_dir, 'scores.value'), np.array(params['scores']))
                    del params['embeddings']
                    del params['scores']
                    if 'indices' in params:
                        numpy_dump(os.path.join(dump_dir, 'scores.idx'), np.array(params['indices']))
                        del params['indices']


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
        def tree_iterator():
            for tree in diter.tree_iterator(indices=new_indices, forest=forest, concat_mode=concat_mode,
                                            max_depth=max_depth):
                yield [tree]
        with model_tree.compiler.multiprocessing_pool():
            trees_compiled_iter = model_tree.compiler.build_loom_inputs(tree_iterator(), ordered=True)
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
    scoring_indices = root_indices + OFFSET_CONTEXT_ROOT

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
    seealso_root_idx = forest.roots[root_id] + diter.OFFSET_SEEALSO_ROOT
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
        if 'dump' in params:
            _embeddings = params['embeddings']
            dump_dir = params['dump']
            numpy_dump(os.path.join(dump_dir, 'embeddings.context'), _embeddings[:, :-lexicon_dims - 1])
            numpy_dump(os.path.join(dump_dir, 'embeddings.head'), _embeddings[:, -lexicon_dims - 1:-1])
            numpy_dump(os.path.join(dump_dir, 'embeddings.id'), _embeddings[:, -1].astype(dtype=DTYPE_IDX))
            del params['embeddings']
            if 'indices' in params:
                numpy_dump(os.path.join(dump_dir, 'embeddings.idx'), np.array(params['indices']))
                del params['indices']
            response = Response("created %i embeddings (dumped to %s)" % (_embeddings.shape[0], dump_dir))
        else:
            return_type = params.get('HTTP_ACCEPT', False) or 'application/json'
            json_data = json.dumps(filter_result(make_serializable(params)))
            response = Response(json_data, mimetype=return_type)

        logging.info("Time spent handling the request: %f" % (time.time() - start))
    except Exception as e:
        raise InvalidUsage('%s: %s' % (type(e).__name__, e.message))
    return response


def create_visualization_response(params):
    mode = params.get('vis_mode', 'image')
    if mode == 'image':
        if 'root_ids' in params:
            root_ids = params['root_ids']
        else:
            root_ids = None
        forest_temp = None
        count = 0
        for i, forests_or_data_and_structure_or_tree_dicts in enumerate(params.get('forests', None) or params.get('data_sequences', None) or params['tree_dicts']):
            if isinstance(forests_or_data_and_structure_or_tree_dicts, Forest):
                forest_temp = forests_or_data_and_structure_or_tree_dicts
            elif isinstance(forests_or_data_and_structure_or_tree_dicts, tuple) or isinstance(forests_or_data_and_structure_or_tree_dicts, list):
                data, structure = forests_or_data_and_structure_or_tree_dicts
                forest_temp = Forest(data=data, structure=structure, lexicon=lexicon,
                                     data_as_hashes=params.get('data_as_hashes', False),
                                     #root_ids=root_ids
                                     )
            else:
                forest_temp = Forest(tree_dict=forests_or_data_and_structure_or_tree_dicts, lexicon=lexicon,
                                     data_as_hashes=params.get('data_as_hashes', False),
                                     #root_ids=root_ids
                                     )
            forest_temp.visualize(TEMP_FN_SVG + '.' + str(i), transformed=params.get('transformed_idx', False),
                                  token_list=params['sequences'][i] if 'sequences' in params else None,
                                  scores=params['scores'][i] if 'scores' in params else None,
                                  color_by_rank=params.get('color_by_rank', False))
            count += 1
        #assert len(params['data_sequences']) > 0, 'empty data_sequences'
        assert forest_temp is not None, 'no data to visualize'
        concat_visualizations_svg(TEMP_FN_SVG, count)

        response = send_file(TEMP_FN_SVG)
        # debug of
        #os.remove(TEMP_FN_SVG)
    elif mode == 'text':
        return_type = params.get('HTTP_ACCEPT', False) or 'application/json'
        json_data = json.dumps(make_serializable(filter_result(params)))
        response = Response(json_data, mimetype=return_type)
    else:
        raise ValueError('Unknown mode=%s. Use "image" (default) or "text".')
    return response


@app.route("/api/score", methods=['POST'])
def score():
    global lexicon
    params = None
    try:
        start = time.time()
        logging.info('Scores requested')
        params = get_params(request)
        get_or_calc_scores(params)
        if 'dump' in params:
            response = Response('data was dumped to: %s' % params['dump'])
        else:
            response = create_visualization_response(params)
        logging.info("Time spent handling the request: %f" % (time.time() - start))
    except Exception as e:
        if params is not None and params.get('clear_lexicon', False):
            lexicon = None
        raise InvalidUsage('%s: %s' % (type(e).__name__, e.message))
    if params.get('clear_lexicon', False):
        lexicon = None
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
        json_data = json.dumps(make_serializable(filter_result(params)))
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
        json_data = json.dumps(make_serializable(filter_result(params)))
        response = Response(json_data, mimetype=return_type)
        logging.info("Time spent handling the request: %f" % (time.time() - start))
    except Exception as e:
        raise InvalidUsage('%s: %s' % (type(e).__name__, e.message))

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
        json_data = json.dumps(make_serializable(filter_result(params)))
        response = Response(json_data, mimetype=return_type)
        logging.info("Time spent handling the request: %f" % (time.time() - start))
    except Exception as e:
        raise InvalidUsage('%s: %s' % (type(e).__name__, e.message))
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
        json_data = json.dumps(make_serializable(filter_result(params)))
        response = Response(json_data, mimetype=return_type)
        logging.info("Time spent handling the request: %f" % (time.time() - start))
    except Exception as e:
        raise InvalidUsage('%s: %s' % (type(e).__name__, e.message))

    return response


def set_datasequences_and_sequences_from_tree_dicts_or_forests(params, current_forest, keep_sequences=True):
    if 'forests' in params:
        forests = params['forests']
    else:
        forests = tree_dicts_to_forests(params['tree_dicts'], current_forest)
    data_sequences, sequences = zip(*[([f.data, f.graph_out], f.get_text_plain(
        blacklist=params.get('prefix_blacklist', None), transformed=params.get('transformed_idx', False))) for f
                                                          in forests])
    if keep_sequences and 'sequences' not in params:
        params['sequences'] = sequences
    params['data_sequences'] = data_sequences


@app.route("/api/visualize", methods=['POST'])
def visualize():
    global lexicon
    params = None
    try:
        start = time.time()
        logging.info('Visualizations requested')
        params = get_params(request)
        current_forest = get_or_calc_tree_dicts_or_forests(params)
        set_datasequences_and_sequences_from_tree_dicts_or_forests(params, current_forest)
        response = create_visualization_response(params)
        logging.info("Time spent handling the request: %f" % (time.time() - start))
    except Exception as e:
        if params is not None and params.get('clear_lexicon', False):
            lexicon = None
        raise InvalidUsage('%s: %s' % (type(e).__name__, e.message))
    if params.get('clear_lexicon', False):
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
            json_data = json.dumps(make_serializable(filter_result(params)))
            response = Response(json_data, mimetype=return_type)
        else:
            ValueError('Unknown mode=%s. Use "image" (default) or "text".')
        logging.info("Time spent handling the request: %f" % (time.time() - start))
    except Exception as e:
        raise InvalidUsage('%s: %s' % (type(e).__name__, e.message))
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
            json_data = json.dumps(make_serializable(filter_result(params)))
            response = Response(json_data, mimetype=return_type)
        else:
            ValueError('Unknown mode=%s. Use "image" (default) or "text".')
        logging.info("Time spent handling the request: %f" % (time.time() - start))
    except Exception as e:
        raise InvalidUsage('%s: %s' % (type(e).__name__, e.message))
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
        json_data = json.dumps(make_serializable(filter_result(params)))
        response = Response(json_data, mimetype=return_type)

        logging.info("Time spent handling the request: %f" % (time.time() - start))
    except Exception as e:
        raise InvalidUsage('%s: %s' % (type(e).__name__, e.message))
    return response


@app.route("/api/tuple_scores", methods=['GET'])
def get_tuple_scores():
    try:
        start = time.time()
        logging.info('Tuple scoring requested')
        params = get_params(request)
        init_forest(data_path)

        root_id_str = params.get('root_id_string', None)
        if root_id_str is not None:
            try:
                params['root_id'] = forest.lexicon_roots.get_d(root_id_str, data_as_hashes=False)
            except KeyError:
                raise InvalidUsage("root_id_string not found: %s" % root_id_str)
        root_id = params['root_id']
        if root_id_str is None:
            params['root_id_string'] = forest.lexicon_roots.get_s(root_id, data_as_hashes=False)
        all_root_ids = current_root_ids()
        root_ids_target_strings = params.get('root_ids_target_string', None)
        if root_ids_target_strings is not None:
            root_ids_target = []
            for root_id_target_string in root_ids_target_strings:
                try:
                    root_ids_target.append(forest.lexicon_roots.get_d(root_id_target_string, data_as_hashes=False))
                except KeyError:
                    raise InvalidUsage("root_id_target_string not found: %s" % root_id_target_string)
        else:
            root_ids_target_nbr = params.get('root_ids_target_nbr', len(all_root_ids))
            root_ids_target = params.get('root_ids_target', all_root_ids[np.arange(root_ids_target_nbr)])
        concat_mode = params.get('concat_mode', 'tree')
        max_depth = params.get('max_depth', 10)
        top = params.get('top', len(root_ids_target))
        batch_size = params.get('batch_size', 100)
        _scores, seealso_root_ids = calc_tuple_scores(root_id=root_id, root_ids_target=root_ids_target, forest=forest,
                                                      model_tree=model_tree, model_tuple=model_main,
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
        json_data = json.dumps(make_serializable(filter_result(params)))
        response = Response(json_data, mimetype=return_type)

        logging.info("Time spent handling the request: %f" % (time.time() - start))
    except Exception as e:
        raise InvalidUsage('%s: %s' % (type(e).__name__, e.message))
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
        raise InvalidUsage('%s: %s' % (type(e).__name__, e.message))
    return "clearing embeddings successful"


@app.route("/api/load", methods=['POST'])
def load_data_source():
    global nbr_candidates
    try:
        start = time.time()
        logging.info('Reload requested')
        params = get_params(request)
        path = params['path']
        nbr_candidates = params.get('nbr_candidates', None)
        main(path)

        logging.info("Time spent handling the request: %f" % (time.time() - start))
    except Exception as e:
        raise InvalidUsage('%s: %s' % (type(e).__name__, str(e)))
    if nbr_candidates is None:
        return "reload successful"
    else:
        return "reload successful. set nbr_candidates=%i" % nbr_candidates


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
    global nlp, lexicon

    if nlp is None:
        logging.info('load spacy ...')
        nlp = spacy.load('en')
        if lexicon is None:
            lexicon = Lexicon()
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
        if model_config is not None and model_config.model_type == MT_CANDIDATES:
            logger.warning('set (root) ids to IDENTITY')
            d_identity = forest.lexicon.get_d(s=vocab_manual[IDENTITY_EMBEDDING], data_as_hashes=forest.data_as_hashes)
            forest.data[forest.roots + OFFSET_ID] = d_identity
        if tfidf_indices is not None:
            tfidf_root_ids = np.array([forest.root_mapping[root_idx] for root_idx in (tfidf_indices - OFFSET_CONTEXT_ROOT)])
        else:
            tfidf_root_ids = None


def main(data_source):
    global sess, model_tree, model_main, lexicon, data_path, forest, tfidf_data, tfidf_indices, tfidf_root_ids, \
        embedding_indices, embeddings, model_config, nbr_candidates, lexicon_dims
    sess = None
    model_tree = None
    model_main = None
    lexicon = None
    lexicon_dims = None
    forest = None
    data_path = None
    tfidf_data = None
    tfidf_indices = None
    tfidf_root_ids = None
    embedding_indices = None
    embeddings = None
    model_config = None

    if data_source is None:
        logging.info('Start api without data source. Use /api/load before any other request.')
        return

    lexicon, checkpoint_fn, _ = get_lexicon(logdir=data_source, train_data_path=data_source, dont_dump=True)
    if lexicon.has_vecs:
        lexicon_dims = lexicon.dims
    if checkpoint_fn:
        assert lexicon.vecs is None or lexicon.is_filled, \
            'lexicon: not all vecs for all types are set (len(types): %i, len(vecs): %i)' \
            % (len(lexicon), len(lexicon.vecs))

    # load model
    if checkpoint_fn:
        import model_fold
        model_config = Config(logdir=data_source)
        data_path = model_config.train_data_path

        with tf.Graph().as_default() as graph:
            with tf.device(tf.train.replica_device_setter(FLAGS.ps_tasks)):
                logging.debug('trainable lexicon entries: %i' % lexicon.len_var)
                logging.debug('fixed lexicon entries:     %i' % lexicon.len_fixed)

                model_config.keep_prob = 1.0
                model_config.neg_samples = "0" if nbr_candidates is None else str(nbr_candidates - 1)

                tree_iterator, tree_iterator_args, indices_getter, load_parents, model_kwargs = init_model_type(model_config, logdir=data_source)
                model_tree, model_main, prepared_embeddings, compiled_trees = create_models(
                    config=model_config, lexicon=lexicon,
                    tree_iterators={},
                    tree_iterators_tfidf={},
                    indices={},
                    precompile=False,
                    model_kwargs=model_kwargs
                )

                if model_config.tree_embedder == 'tfidf':
                    raise NotImplementedError('tfidf model not implemented for embedding_api')
                    #_indices, _tfidf = zip(*[(tree_indices[m], prepared_embeddings[m]) for m in tree_indices.keys()])
                    #tfidf_data = scipy.sparse.vstack(_tfidf)
                    #logging.info('total tfidf shape: %s' % str(tfidf_data.shape))
                    #tfidf_indices = np.concatenate(_indices)
                    #logging.debug('number of tfidf_indices: %i' % len(tfidf_indices))

                #if FLAGS.external_lexicon or FLAGS.merge_nlp_lexicon:
                lexicon_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                 scope=model_fold.VAR_NAME_LEXICON_VAR) \
                               + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                   scope=model_fold.VAR_NAME_LEXICON_FIX)

                vars_all = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
                vars_without_embed = [v for v in vars_all if v not in lexicon_vars]
                if len(vars_without_embed) > 0:
                    saver = tf.train.Saver(var_list=vars_without_embed)
                else:
                    saver = None
                #else:
                #    saver = tf.train.Saver()

                sess = tf.Session()
                # Restore variables from disk.
                if saver:
                    logging.info('restore model from: %s ...' % checkpoint_fn)
                    saver.restore(sess, checkpoint_fn)

                #if FLAGS.external_lexicon or FLAGS.merge_nlp_lexicon:
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
    else:
        data_path = data_source


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    #mytools.logging_init()
    FLAGS._parse_flags()
    main(FLAGS.data_source)
    logging.info('Starting the API')
    app.run(host='0.0.0.0', port=5000)
