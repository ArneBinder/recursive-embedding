import json
import logging
import traceback

import numpy as np
import os
import sys

from sklearn.feature_extraction.text import TfidfTransformer
from scipy.sparse import csr_matrix

from constants import TYPE_REF, KEY_HEAD, KEY_CANDIDATES, DTYPE_OFFSET, DTYPE_IDX, TYPE_REF_SEEALSO, \
    TYPE_SECTION_SEEALSO, UNKNOWN_EMBEDDING, vocab_manual, KEY_CHILDREN, TYPE_DBPEDIA_RESOURCE, TYPE_CONTEXT, \
    TYPE_PARAGRAPH, TYPE_TITLE, TYPE_SENTENCE, TYPE_SECTION, LOGGING_FORMAT, CM_TREE, CM_AGGREGATE, \
    CM_SEQUENCE, TARGET_EMBEDDING, OFFSET_ID, TYPE_RELATEDNESS_SCORE, SEPARATOR, OFFSET_CONTEXT_ROOT, \
    OFFSET_SEEALSO_ROOT, OFFSET_RELATEDNESS_SCORE_ROOT, OFFSET_OTHER_ENTRY_ROOT, DTYPE_PROBS, BLANKED_EMBEDDING, \
    KEY_HEAD_CONCAT, RDF_BASED_FORMAT, REC_EMB_HAS_PARSE, REC_EMB_HAS_GLOBAL_ANNOTATION, SICK_VOCAB, \
    REC_EMB_GLOBAL_ANNOTATION, SICK_RELATEDNESS_SCORE, JSONLD_IDX, JSONLD_VALUE, DEBUG, NIF_SENTENCE, NIF_NEXT_SENTENCE, \
    NIF_NEXT_WORD, NIF_WORD, PADDING_EMBEDDING, NIF_CONTEXT, KEY_DEPTH, DATA_STATS_PATH, CONLL_EDGE, CONLL_POS, \
    CONLL_WORD
from sequence_trees import Forest, targets
from mytools import numpy_load

RECURSION_LIMIT_MIN = 1000
RECURSION_LIMIT_ADD = 100

## different sample methods for reroot_wrapper
## take all
SAMPLE_METHOD_UNIFORM = 'U'
SAMPLE_METHOD_UNIFORM_ALL = 'UA'
SAMPLE_METHOD_FREQUENCY = 'F'
SAMPLE_METHOD_FREQUENCY_ALL = 'FA'
SAMPLE_METHOD_NEAREST = 'N'
SAMPLE_METHOD_NEAREST_ALL = 'NA'


logger = logging.getLogger('data_iterators')
logger.setLevel(logging.DEBUG)
logger_streamhandler = logging.StreamHandler()
logger_streamhandler.setLevel(logging.DEBUG)
logger_streamhandler.setFormatter(logging.Formatter(LOGGING_FORMAT))
logger.addHandler(logger_streamhandler)
logger.propagate = False


def index_iterator(index_files):
    """
    yields index values from plain numpy arrays
    :param index_files: a list of file names of dumped numpy arrays
    :return: index values
    """
    for file_name in index_files:
        indices = np.load(file_name)
        for idx in indices:
            yield idx


def load_indices(index_files):
    """
    yields index values from plain numpy arrays
    :param index_files: a list of file names of dumped numpy arrays
    :return: index values
    """
    return [np.load(file_name) for file_name in index_files]


def index_np(index_files):
    """
    yields index values from plain numpy arrays
    :param index_files: a list of file names of dumped numpy arrays
    :return: index values
    """
    indices = []
    for file_name in index_files:
        indices.append(np.load(file_name))
    return np.concatenate(indices)


def root_id_to_idx_offsets_iterator(indices, mapping, offsets=(2, 3)):
    """
    map each index in indices via a list/map and add the offsets
    :param indices: the indices to map and add the offsets to
    :param mapping: the mapping list/map
    :param offsets: offsets that are added to the mapped indices
    :return: for every index in indices and every offset in offsets, yield the mapped and shifted (by offset) new index
    """
    for idx in indices:
        idx_mapped = mapping[idx]
        yield [idx] + [o + idx_mapped for o in offsets]


def root_id_to_idx_offsets(indices, mapping, offsets=(2, 3)):
    """
    map each index in indices via a list/map and add the offsets
    :param indices: the indices to map and add the offsets to
    :param mapping: the mapping list/map
    :param offsets: offsets that are added to the mapped indices
    :return: for every index in indices and every offset in offsets, yield the mapped and shifted (by offset) new index
    """
    #_offsets = (0,) + tuple(offsets)
    #root_indices = mapping[indices]
    #for idx in indices:
    #    idx_mapped = mapping[idx]
    #    yield [idx] + [o + idx_mapped for o in offsets]
    #return [root_indices] + [ for  in (0,) + tuple(offsets)]
    return [mapping[indices] + o for o in offsets]


def root_id_to_idx_offsets_np(indices, mapping, offsets=(2, 3)):
    """
    map each index in indices via a list/map and add the offsets
    :param indices: the indices to map and add the offsets to
    :param mapping: the mapping list/map
    :param offsets: offsets that are added to the mapped indices
    :return: for every index in indices and every offset in offsets, yield the mapped and shifted (by offset) new index
    """
    indices_mapped = mapping[indices]
    indices_mapped_offset = [indices_mapped + offset for offset in offsets]
    return indices_mapped_offset


def link_root_ids_iterator(indices, forest, link_type=TYPE_REF_SEEALSO):
    """
    For every index in indices and with regard to sequence_trees, yield all root ids referenced via link_type
    :param indices: indices to sequence_trees.data
    :param forest: all trees
    :param link_type: One of TYPE_REF_SEEALSO or TYPE_REF. Defaults to TYPE_REF_SEEALSO.
    :return: lists of root ids that are referenced from indices
    """
    data_unknown = forest.lexicon.get_d(vocab_manual[UNKNOWN_EMBEDDING],
                                        data_as_hashes=forest.data_as_hashes)
    d_link_type = forest.lexicon.get_d(link_type, data_as_hashes=forest.data_as_hashes)
    n_trees = 0
    n_links = 0
    for idx in indices:
        target_root_ids = []
        # allow multiple link edges
        for child_idx in forest.get_children(idx):
            child_data = forest.data[child_idx]
            # allow only one type of link edge
            assert child_data == d_link_type, \
                'link_data (%s, data=%i, idx=%i) is not as expected (%s). parent: %s, data=%i, idx=%i' \
                % (forest.lexicon.get_s(child_data, data_as_hashes=forest.data_as_hashes),
                   child_data, child_idx,
                   forest.lexicon.get_s(d_link_type, data_as_hashes=forest.data_as_hashes),
                   forest.lexicon.get_s(forest.data[idx], data_as_hashes=forest.data_as_hashes),
                   forest.data[idx], idx)
            child_indices_target = forest.get_children(child_idx)
            assert len(child_indices_target) == 1, \
                'link has more or less then one targets: %i' % len(child_indices_target)

            target_id_idx = child_indices_target[0]
            target_id_data = forest.data[target_id_idx]
            # TODO: should not be possible
            if target_id_data == data_unknown:
                logger.warning('target_id_data is UNKNOWN (target_id_idx: %i), skip' % target_id_idx)
                continue
            target_root_id = forest.root_id_mapping.get(target_id_data, None)
            if target_root_id is None:
                continue
            target_root_ids.append(target_root_id)

        if len(target_root_ids) > 0:
            yield target_root_ids
            n_trees += 1
            n_links += len(target_root_ids)
        else:
            yield None
    logger.info('found %i trees with %i links in total (%s)' % (n_trees, n_links, link_type))


def debug_get_child_ids(d):
    return [c['h'] for c in d['c']]


def tree_iterator(indices, forest, concat_mode=CM_TREE, max_depth=9999, context=0, transform=True,
                  link_cost_ref=None, link_cost_ref_seealso=1, reroot=False, max_size_plain=1000,
                  keep_prob_blank=1.0, keep_prob_node=1.0, blank_types=(), add_heads_types=set(), additional_heads=0,
                  **unused):
    """
    create trees rooted at indices
    :param indices:
    :param forest:
    :param concat_mode:
    :param max_depth:
    :param context:
    :param transform:
    :param link_cost_ref:
    :param link_cost_ref_seealso:
    :param unused:
    :return:
    """
    try:
        if reroot:
            assert concat_mode == CM_TREE, 'reroot requires concat_mode==%s, but found concat_mode: %s' \
                                           % (CM_TREE, concat_mode)

        # set fixed nbr of heads for flat models (WORD(real head), DEP, POS)
        # this _can_ be different from additional_heads
        nbr_heads_flat = 3
        default_add_heads_prefixes = [CONLL_EDGE + u'=', CONLL_POS + u'=']
        other_add_heads_types = ()
        assert additional_heads < nbr_heads_flat, 'additional_heads [%i] has to be smaller then nbr_heads_flat [%i]' \
                                                  % (additional_heads, nbr_heads_flat)
        if len(add_heads_types) > 0:
            # collect all default add_heads types
            default_add_heads_types = []
            for p in default_add_heads_prefixes:
                # dont transform
                default_add_heads_types.extend(forest.lexicon.get_ids_for_prefix(prefix=p, add_separator=False)[0])
            other_add_heads_types = add_heads_types - set(default_add_heads_types)


        #sys.setrecursionlimit(max(RECURSION_LIMIT_MIN, max_depth + context + RECURSION_LIMIT_ADD))
        sys.setrecursionlimit(1000)

        lexicon = forest.lexicon
        costs = {}
        link_types = []

        # check, if TYPE_REF and TYPE_REF_SEEALSO are in lexicon
        if TYPE_REF in lexicon.strings:
            data_ref = lexicon.get_d(TYPE_REF, data_as_hashes=forest.data_as_hashes)
            link_types.append(data_ref)
            if link_cost_ref is not None:
                costs[data_ref] = link_cost_ref
        if TYPE_REF_SEEALSO in lexicon.strings:
            data_ref_seealso = lexicon.get_d(TYPE_REF_SEEALSO, data_as_hashes=forest.data_as_hashes)
            costs[data_ref_seealso] = link_cost_ref_seealso
            link_types.append(data_ref_seealso)

        data_padding = lexicon.get_d(s=vocab_manual[PADDING_EMBEDDING], data_as_hashes=forest.data_as_hashes)
        data_flat_root = lexicon.get_d(REC_EMB_HAS_PARSE, data_as_hashes=forest.data_as_hashes)
        data_unknown = lexicon.get_d(vocab_manual[UNKNOWN_EMBEDDING], data_as_hashes=forest.data_as_hashes)
        if transform:
            data_flat_root = lexicon.transform_idx(idx=data_flat_root)
            data_unknown = lexicon.transform_idx(idx=data_unknown)
            data_padding = lexicon.transform_idx(idx=data_padding)
        # ATTENTION: don't transform data_blanking, it will be transformed in get_tree_dict / get_tree_dict_rooted
        data_blanking = lexicon.get_d(s=vocab_manual[BLANKED_EMBEDDING], data_as_hashes=forest.data_as_hashes)

        # do not remove TYPE_ANCHOR (nif:Context), as it is used for aggregation
        if not RDF_BASED_FORMAT:
            remove_types_naive_str = [TYPE_REF_SEEALSO, TYPE_REF, TYPE_DBPEDIA_RESOURCE, TYPE_SECTION_SEEALSO, TYPE_PARAGRAPH,
                                      TYPE_TITLE, TYPE_SECTION, TYPE_SENTENCE]
        else:
            remove_types_naive_str = [REC_EMB_HAS_PARSE, NIF_WORD, NIF_NEXT_WORD, NIF_SENTENCE, NIF_NEXT_SENTENCE]
        remove_types_naive = [lexicon.get_d(s, data_as_hashes=forest.data_as_hashes) for s in
                              remove_types_naive_str if s in forest.lexicon]

        n = 0

        if concat_mode == CM_TREE:
            depths = []
            for idx in indices:
                tree_context = forest.get_tree_dict(idx=idx, max_depth=max_depth, context=context, transform=transform or reroot,
                                                    costs=costs, link_types=link_types, data_blank=data_blanking,
                                                    keep_prob_blank=keep_prob_blank, keep_prob_node=keep_prob_node,
                                                    blank_types=blank_types, go_back=reroot,
                                                    add_heads_types=add_heads_types,
                                                    add_heads_dummies=[data_padding] * additional_heads,
                                                    return_strings=DEBUG, return_depth=DATA_STATS_PATH is not None)
                yield tree_context
                n += 1
                if DATA_STATS_PATH is not None:
                    depths.append(tree_context[KEY_DEPTH])
                # measure progress in percent
                #if n % x == 0:
                #    progress = n / x
                #    logger.debug('%i%%' % progress)
            if DATA_STATS_PATH is not None:
                if not os.path.exists(DATA_STATS_PATH):
                    os.makedirs(DATA_STATS_PATH)
                depths = np.array(depths)
                fn = os.path.join(DATA_STATS_PATH, 'depth')
                _i = 0
                while os.path.exists('%s.%i.np' % (fn, _i)):
                    _i += 1
                fn = '%s.%i.np' % (fn, _i)
                logger.debug('dump depths to: %s' % fn)
                depths.dump(fn)
                logger.debug('dumped.')

        elif concat_mode == CM_AGGREGATE:
            lengths = []
            all_data_string = []
            for idx in indices:
                # ATTENTION: idx has to be added to forest.pos_to_component_mapping before!
                component_idx = forest.pos_to_component_mapping[idx]
                idx_end = forest.pos_end(component_idx=component_idx)
                data_span_cleaned = forest.get_data_span_cleaned(idx_start=idx, idx_end=idx_end,
                                                                 link_types=link_types,
                                                                 remove_types=remove_types_naive, transform=transform)

                if DEBUG:
                    data_span_cleaned_not_transformed = forest.get_data_span_cleaned(idx_start=idx, idx_end=idx_end,
                                                                                     link_types=link_types,
                                                                                     remove_types=remove_types_naive,
                                                                                     transform=False)
                    assert len(data_span_cleaned) == len(data_span_cleaned_not_transformed), 'length mismatch'
                    data_string_cleaned = [forest.lexicon.get_s(d, data_as_hashes=forest.data_as_hashes) for d in
                                           data_span_cleaned_not_transformed]
                    all_data_string.append(data_string_cleaned)
                    tree_context_string = {KEY_HEAD: data_flat_root,
                                            KEY_CHILDREN: [{KEY_HEAD: data_string_cleaned[i],
                                                            KEY_CHILDREN: [],
                                                            KEY_HEAD_CONCAT: data_string_cleaned[i+1:i+1+additional_heads]}
                                                           for i in range(0, len(data_string_cleaned), nbr_heads_flat)]
                                            }
                    _idx = targets(forest.graph_out, idx)[0]
                    _idx = targets(forest.graph_out, _idx)[0]
                    _tree_context = forest.get_tree_dict(idx=_idx, max_depth=max_depth, context=context,
                                                        transform=transform or reroot,
                                                        costs=costs, link_types=link_types, data_blank=data_blanking,
                                                        keep_prob_blank=keep_prob_blank, keep_prob_node=keep_prob_node,
                                                        blank_types=blank_types, go_back=reroot,
                                                        add_heads_types=add_heads_types,
                                                        add_heads_dummies=[data_padding] * additional_heads,
                                                        return_strings=DEBUG)
                    debug_ids_tree = debug_get_child_ids(_tree_context)

                assert len(data_span_cleaned) % nbr_heads_flat == 0, \
                    'idx:%i: len(data_span_cleaned) [%i] is not a multiple of nbr_heads_flat [%i]' \
                    % (component_idx, len(data_span_cleaned), nbr_heads_flat)
                data_span_cleaned = data_span_cleaned.reshape((len(data_span_cleaned) / nbr_heads_flat, nbr_heads_flat))

                if len(other_add_heads_types) > 0:
                    data_span_cleaned = data_span_cleaned[:,:additional_heads]

                    # use types of "first" entries of data_span_cleaned and select these entries in data[idx:idx_end]
                    # sanity check
                    assert len(set(data_span_cleaned[:, 0]) & set(data_span_cleaned[:, 1:].flatten())) == 0, 'type overlap detected'
                    mask = np.isin(forest.lexicon.transform_indices(forest.data[idx:idx_end]), data_span_cleaned[:, 0])
                    indices_main = np.arange(len(mask), dtype=DTYPE_IDX)[mask] + idx
                    assert len(indices_main) == len(data_span_cleaned), \
                        'len(indices_main)[%i] != len(data_span_cleaned)[%i]' \
                        % (len(indices_main), len(data_span_cleaned))
                    other_add_heads = np.ones(len(indices_main), dtype=DTYPE_IDX) * data_padding
                    for i, idx_main in enumerate(indices_main):
                        indices_main_children = targets(forest.graph_out, idx_main)
                        datas_main_child = forest.data[indices_main_children]
                        for data_child in datas_main_child:
                            if data_child in other_add_heads_types:
                                other_add_heads[i] = forest.lexicon.transform_idx(data_child) if transform else data_child
                                break
                    data_span_cleaned = np.append(data_span_cleaned,
                                                  other_add_heads.reshape((len(other_add_heads), 1)),
                                                  axis=1)
                else:
                    data_span_cleaned = data_span_cleaned[:, :additional_heads+1]
                if DATA_STATS_PATH is not None:
                    lengths.append(len(data_span_cleaned))
                # cut tokens in the front to resemble behaviour of max_depth
                data_span_cleaned = data_span_cleaned[max(len(data_span_cleaned) - max_size_plain, 0):]
                tree_context = {KEY_HEAD: data_flat_root,
                                KEY_CHILDREN: [{KEY_HEAD: data_span_cleaned[i][0],
                                                KEY_CHILDREN: [],
                                                KEY_HEAD_CONCAT:  data_span_cleaned[i][1:]}
                                               for i in range(len(data_span_cleaned))]
                                }

                yield tree_context
                n += 1
            if DATA_STATS_PATH is not None:
                if not os.path.exists(DATA_STATS_PATH):
                    os.makedirs(DATA_STATS_PATH)
                lengths = np.array(lengths)
                fn = os.path.join(DATA_STATS_PATH, 'length')
                _i = 0
                while os.path.exists('%s.%i.np' % (fn, _i)):
                    _i += 1
                fn = '%s.%i.np' % (fn, _i)
                logger.debug('dump lengths to: %s' % fn)
                lengths.dump(fn)

                fn_tokens = '%s.%i.json' % (os.path.join(DATA_STATS_PATH, 'tokens'), _i)
                logger.debug('dump tokens to: %s' % fn_tokens)
                with open(fn_tokens, 'w') as f:
                    json.dump(all_data_string, f)
                logger.debug('dumped.')

        elif concat_mode == CM_SEQUENCE:
            # DEPRECATED
            raise NotImplementedError('concat_mode=%s is deprecated. Use "%s" or "%s" instead.' % (concat_mode, CM_TREE, CM_AGGREGATE))
        elif concat_mode == 'dummy_dbpedianif1000':
            for idx in indices:
                yield {'h': 12, 'c': [{'h': 14, 'c': [{'h': 16, 'c': [{'h': 1, 'c': [{'h': 1, 'c': [{'h': 1, 'c': [{'h': -1952, 'c': [{'h': -1300, 'c': [{'h': 15, 'c': []}]}, {'h': -23, 'c': [{'h': -12238, 'c': [{'h': -15, 'c': []}, {'h': -12237, 'c': []}, {'h': -3650, 'c': []}, {'h': -1045, 'c': []}]}]}, {'h': -23, 'c': [{'h': -712, 'c': [{'h': -10, 'c': []}, {'h': -517, 'c': [{'h': 15, 'c': []}]}]}]}, {'h': -19, 'c': []}]}]}, {'h': -1275, 'c': [{'h': -2472, 'c': [{'h': -42, 'c': []}, {'h': -4600, 'c': []}]}, {'h': -21, 'c': []}, {'h': -32626, 'c': []}, {'h': -6, 'c': [{'h': -9978, 'c': [{'h': -15, 'c': []}, {'h': -2037, 'c': [{'h': -4600, 'c': []}]}, {'h': -4600, 'c': []}, {'h': -8127, 'c': []}, {'h': 15, 'c': []}, {'h': -1, 'c': []}, {'h': -66750, 'c': []}, {'h': -8, 'c': []}]}]}, {'h': -19, 'c': []}]}]}, {'h': -5279, 'c': [{'h': 0, 'c': []}, {'h': -1300, 'c': []}, {'h': -119, 'c': [{'h': -14, 'c': [{'h': -15, 'c': []}, {'h': -9665, 'c': [{'h': 0, 'c': []}, {'h': 15, 'c': []}, {'h': 0, 'c': []}]}, {'h': -10, 'c': []}, {'h': -1838, 'c': [{'h': -361, 'c': []}, {'h': -477, 'c': []}, {'h': -104, 'c': []}, {'h': -464, 'c': [{'h': -5244, 'c': [{'h': -20136, 'c': []}]}, {'h': -79, 'c': [{'h': -1503, 'c': []}]}]}]}]}]}, {'h': -19, 'c': []}]}]}]}]}]}
                n += 1
        elif concat_mode == 'dummy_unknown':
            for idx in indices:
                yield {KEY_HEAD: data_flat_root, KEY_CHILDREN: [{KEY_HEAD: data_unknown, KEY_CHILDREN: []}] * 7}
                n += 1
        else:
            raise ValueError('unknown concat_mode=%s' % concat_mode)
    except Exception as e:
        logger.error('exception occurred in tree_iterator:')
        traceback.print_exc()
        raise e


def get_nearest_neighbor_samples_batched(reference_indices_transformed, candidate_indices_transformed, embedder,
                                         session, nbr):
    feed_dict = {embedder.reference_indices: reference_indices_transformed,
                 embedder.candidate_indices: candidate_indices_transformed}
    all_sims = session.run(embedder.reference_vs_candidate, feed_dict)
    res = {}
    max_indices_indices = np.argpartition(all_sims, -(nbr + 1))[:,-(nbr + 1):]
    for i, ref_idx in enumerate(reference_indices_transformed):
        max_indices = candidate_indices_transformed[max_indices_indices[i]]
        max_indices[max_indices == ref_idx] = max_indices[0]
        max_indices[0] = ref_idx
        res[ref_idx] = max_indices
    return res


def reroot_wrapper(tree_iter, neg_samples, forest, indices, indices_mapping=None,
                   transform=True, debug=False,
                   sample_method='', embedder=None, session=None, **kwargs):
    try:
        d_target = forest.lexicon.get_d(s=vocab_manual[TARGET_EMBEDDING], data_as_hashes=forest.data_as_hashes)
        nearest_neighbors_transformed = {}
        _sm_split = sample_method.split('.')
        sample_method = _sm_split[0]
        sample_method_value = float('0.'+_sm_split[1]) if len(_sm_split) > 1 else 1.0
        logger.debug('default sample_method=%s' % str(sample_method))
        sample_method_backup = SAMPLE_METHOD_FREQUENCY_ALL
        if sample_method in [SAMPLE_METHOD_NEAREST, SAMPLE_METHOD_NEAREST_ALL]:
            if embedder is None and session is None:
                logger.warning('embedder or session not available, but required for sample_method=nearest. Use "%s" instead.'
                               % str(sample_method_backup))
                sample_method = sample_method_backup
        if sample_method in [SAMPLE_METHOD_UNIFORM_ALL, SAMPLE_METHOD_FREQUENCY_ALL, SAMPLE_METHOD_NEAREST_ALL]:
            data_indices_all, lexicon_indices_all = indices_mapping[None]
            indices_mapping = None
            assert data_indices_all is not None, 'not class-wise sampling requires data_indices_all, but it is None'
        else:
            assert indices_mapping is not None, 'class-wise sampling requires indices_mapping, but it is None'
            data_indices_all = None
            lexicon_indices_all = None

        # pre-calculate nearest neighbors in batches
        if sample_method == SAMPLE_METHOD_NEAREST_ALL:
            lexicon_indices_transformed = np.array(forest.lexicon.transform_indices(indices=lexicon_indices_all))
            bs = 20
            logger.debug('calculate nearest neighbours (batch_size: %i; #batches: %i)...'
                         % (bs, len(lexicon_indices_transformed) // bs + 1))
            for start in range(0, len(lexicon_indices_transformed), bs):
                #logger.debug('calc batch #%i ...' % (start // bs))
                lex_indices_batch = lexicon_indices_transformed[start:start+bs]
                new_nn = get_nearest_neighbor_samples_batched(lex_indices_batch, lexicon_indices_transformed, embedder,
                                                              session, nbr=neg_samples)
                nearest_neighbors_transformed.update(new_nn)
            logger.debug('calculate nearest neighbours finished')

        #d_identity = forest.lexicon.get_d(s=vocab_manual[IDENTITY_EMBEDDING], data_as_hashes=forest.data_as_hashes)
        for tree in tree_iter(forest=forest, indices=indices, transform=True, **kwargs):
            #samples = np.random.choice(forest.data, size=neg_samples + 1)
            head_transformed_back, was_reverted = forest.lexicon.transform_idx_back(tree[KEY_HEAD])
            # get selected data
            if indices_mapping is None:
                data_indices, lexicon_indices = data_indices_all, lexicon_indices_all
            else:
                data_indices, lexicon_indices = indices_mapping[head_transformed_back]

            # use all lexicon_indices as samples (exhaustive sampling without repetitions), if its number matches neg_samples
            if sample_method in [SAMPLE_METHOD_UNIFORM, SAMPLE_METHOD_UNIFORM_ALL]:
                # ATTENTION: lexicon_indices might get shuffled
                samples = lexicon_indices
                if neg_samples + 1 < len(samples):
                    np.random.shuffle(samples)
                    samples = samples[:neg_samples + 1]
                elif neg_samples + 1 > len(samples):
                    raise Exception('not enough lexicon_indices (%i) for uniform sampling with (neg_samples+1)=%i'
                                    % (len(samples), neg_samples + 1))

                # swap head to front
                samples[samples == head_transformed_back] = samples[0]
                samples[0] = head_transformed_back
                samples = forest.lexicon.transform_indices(samples)
            elif sample_method in [SAMPLE_METHOD_FREQUENCY, SAMPLE_METHOD_FREQUENCY_ALL]:
                # sample only from selected data
                sample_indices = np.random.choice(data_indices, size=neg_samples + 1)
                samples = forest.data[sample_indices]
                # replace samples that equal the head/root: sample replacement element
                rep = np.random.randint(len(lexicon_indices) - 1)
                # map replacement element from class index to lexicon index, if lexicon_indices are given
                if lexicon_indices[rep] == head_transformed_back:
                    rep = lexicon_indices[-1]
                else:
                    rep = lexicon_indices[rep]

                samples[samples == head_transformed_back] = rep
                # set all IDs to TARGET. That should affect only IDs mentioned under links, ID mentions under roots are
                # replaced by IDENTITY in train_fold.execute_run.
                samples[samples < 0] = d_target
                samples[0] = head_transformed_back
                samples = forest.lexicon.transform_indices(samples)
            elif sample_method in [SAMPLE_METHOD_NEAREST, SAMPLE_METHOD_NEAREST_ALL]:
                if tree[KEY_HEAD] not in nearest_neighbors_transformed:
                    assert len(lexicon_indices) > neg_samples, \
                        'not enough data_indices (%i) to get neg_samples=%i nearest neighbors' % (data_indices, neg_samples)
                    lexicon_indices_transformed = np.array(forest.lexicon.transform_indices(indices=lexicon_indices))
                    new_nn = get_nearest_neighbor_samples_batched([tree[KEY_HEAD]], lexicon_indices_transformed, embedder,
                                                                  session, nbr=neg_samples)
                    nearest_neighbors_transformed.update(new_nn)
                samples = nearest_neighbors_transformed[tree[KEY_HEAD]]
            else:
                raise Exception('unknown sample_method: %s' % str(sample_method))

            tree[KEY_CANDIDATES] = samples
            tree[KEY_HEAD] = None
            yield tree
    except Exception as e:
        logger.error('exception occurred in reroot_wrapper:')
        traceback.print_exc()
        raise e


def embeddings_tfidf(aggregated_trees, d_unknown, vocabulary=None):
    """
    trees --> tf-idf embeddings

    :param aggregated_trees: trees in bag-of-words (i.e. created with concat_mode=aggregate)
    :return:
    """

    # * create id-list versions of articles
    #   --> use data_single_iterator_dbpedianif_context with concat_mode='aggregate'
    #   --> use heads (keys) of root child
    # * create occurrence count matrix
    #   --> create sparse matrix with counts as csr_matrix (Compressed Sparse Row matrix) with entries [doc_idx, lex_idx]
    #       for iterative alogorithm, see: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html#scipy.sparse.csr_matrix
    #   --> use TfidfTransformer (see http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html#from-occurrences-to-frequencies)

    # create sparse count matrix
    indptr = [0]
    indices = []
    data = []
    if vocabulary is None:
        # add unknown as first entry
        vocabulary = {d_unknown: 0}
        expand = True
    else:
        expand = False
        assert d_unknown in vocabulary, 'd_unknown=%i not in predefined vocabulary'
    i_unknown = vocabulary[d_unknown]

    positions = [0]
    n = 0
    # get id-list versions of articles
    for tree_context_iter in aggregated_trees:
        for tree_context in tree_context_iter:
            d = [node[KEY_HEAD] for node in tree_context[KEY_CHILDREN]]
            for term in d:
                if expand:
                    index = vocabulary.setdefault(term, len(vocabulary))
                else:
                    index = vocabulary.get(term, i_unknown)
                indices.append(index)
                data.append(1)
            indptr.append(len(indices))
            n += 1
        positions.append(n)

    # if vocabulary was given, add dummy document that contains all vocabulary entries to get correct shape.
    # Because we cut with positions array, it is not impact the output.
    if not expand:
        for index in vocabulary.values():
            indices.append(index)
            data.append(1)
        indptr.append(len(indices))

    counts = csr_matrix((data, indices, indptr), dtype=int)
    logger.debug('shape of count matrix: %s%s' % (str(counts.shape), ' (dummy document was added)' if not expand else ''))

    # transform to tf-idf,
    # see http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html#from-occurrences-to-frequencies
    #tf_transformer = TfidfTransformer(use_idf=False).fit(counts)
    #tf_idf = tf_transformer.transform(counts)
    tfidf_transformer = TfidfTransformer()
    tf_idf = tfidf_transformer.fit_transform(counts)
    return [tf_idf[positions[i]:positions[i+1], :] for i in range(len(positions)-1)], vocabulary


def indices_to_sparse(indices, length, dtype=np.float32):
    return csr_matrix((np.ones(len(indices), dtype=dtype), (np.zeros(len(indices), dtype=dtype), indices)),
                      shape=(1, length))


def get_classes_ids(indices_classes_root, classes_all_ids, forest):
    classes_mapping = {cd: i for i, cd in enumerate(classes_all_ids)}

    unknown_id = forest.lexicon.get_d(vocab_manual[UNKNOWN_EMBEDDING], data_as_hashes=False)
    classes_ids = []
    for i, classes_root_idx in enumerate(indices_classes_root):
        classes_indices = forest.get_children(classes_root_idx)
        # exclude mesh ids that occurs less then min_count (these were mapped to UNKNOWN in merge_batches)
        current_class_ids_mapped = [classes_mapping[m_id] for m_id in forest.data[classes_indices]
                                    if m_id != unknown_id and m_id in classes_mapping]
        # convert list of indices
        classes_csr = indices_to_sparse(current_class_ids_mapped, len(classes_all_ids))
        classes_ids.append(classes_csr)
    return classes_ids


def indices_value(index_files, forest, nbr_embeddings_in=2, meta_getter_args={}, meta_value_getter=None,
                  sort_indices=False, **unused):
    # get root indices from files
    indices_per_file = load_indices(index_files)
    indices = np.concatenate(indices_per_file)
    if sort_indices or DEBUG:
        logger.debug('sort indices from %s' % str(index_files))
        if len(indices_per_file) != 1:
            logger.warning('sort indices of multiple files, set sizes to None (%s)' % str(index_files))
            sizes = [None for _indices in indices_per_file]
        else:
            sizes = [len(indices)]
        indices = np.sort(indices)
    else:
        sizes = [len(_indices) * nbr_embeddings_in for _indices in indices_per_file]
    assert RDF_BASED_FORMAT, 'old format (not RDF based) is deprecated'
    assert nbr_embeddings_in >= 1, 'nbr_embeddings_in has to be at least 1, but is %s' % str(nbr_embeddings_in)
    assert meta_value_getter is not None, 'meta_value_getter is None'
    meta_args_default = {'stop_types': (REC_EMB_HAS_PARSE,), 'index_types': (REC_EMB_HAS_PARSE,)}
    for k in meta_args_default:
        meta_getter_args[k] = meta_getter_args.get(k, ()) + meta_args_default[k]
    meta_args_default.update(meta_getter_args)
    values = np.empty_like(indices, dtype=np.float32)
    all_indices = np.empty(shape=len(indices) * nbr_embeddings_in, dtype=DTYPE_IDX)
    for root_i_offset in range(nbr_embeddings_in):
        for i, root_i in enumerate(indices):
            idx_start = forest.roots[root_i + root_i_offset]
            meta = forest.get_tree_dict_string(idx=idx_start, **meta_args_default)
            idx_parse = int(meta[REC_EMB_HAS_PARSE][0][JSONLD_IDX])
            all_indices[nbr_embeddings_in * i + root_i_offset] = idx_parse
            forest.pos_to_component_mapping[idx_parse] = root_i + root_i_offset
            if root_i_offset == 0:
                values[i] = meta_value_getter(meta)

    logger.debug('indices created')
    # shift original score range [1.0..5.0] into range [0.0..1.0]
    return all_indices, values, sizes


def indices_multiclass(index_files, forest, classes_all_ids, nbr_embeddings_in=1, meta_getter_args={},
                       meta_class_indices_getter=None, fixed_offsets=False, sort_indices=False, **unused):
    # get root indices from files
    indices_per_file = load_indices(index_files)
    indices = np.concatenate(indices_per_file)
    if sort_indices or DEBUG:
        logger.debug('sort indices from %s' % str(index_files))
        if len(indices_per_file) != 1:
            logger.warning('sort indices of multiple files, set sizes to None (%s)' % str(index_files))
            sizes = [None for _indices in indices_per_file]
        else:
            sizes = [len(indices)]
        indices = np.sort(indices)
    else:
        sizes = [len(_indices) * nbr_embeddings_in for _indices in indices_per_file]
    assert RDF_BASED_FORMAT, 'old format (not RDF based) is deprecated'
    assert nbr_embeddings_in >= 1, 'nbr_embeddings_in has to be at least 1, but is %s' % str(nbr_embeddings_in)
    assert meta_class_indices_getter is not None, 'meta_class_indices_getter is None'
    meta_args_default = {'stop_types': (REC_EMB_HAS_PARSE,), 'index_types': (REC_EMB_HAS_PARSE,)}
    for k in meta_args_default:
        meta_getter_args[k] = meta_getter_args.get(k, ()) + meta_args_default[k]
    meta_args_default.update(meta_getter_args)
    #relatedness_scores = np.empty_like(indices, dtype=np.float32)
    classes_mapping = {cd: i for i, cd in enumerate(classes_all_ids)}
    unknown_id = forest.lexicon.get_d(vocab_manual[UNKNOWN_EMBEDDING], data_as_hashes=False)
    classes_ids = []
    indices_context_root = np.empty(shape=len(indices) * nbr_embeddings_in, dtype=DTYPE_IDX)
    fixed_offset_parse = None
    fixed_offsets_class = None
    for root_i_offset in range(nbr_embeddings_in):
        for i, root_i in enumerate(indices):
            idx_start = forest.roots[root_i + root_i_offset]

            if not fixed_offsets or fixed_offset_parse is None or fixed_offsets_class is None:
                meta = forest.get_tree_dict_string(idx=idx_start, **meta_args_default)
                if fixed_offsets:
                    fixed_offset_parse = int(meta[REC_EMB_HAS_PARSE][0][JSONLD_IDX]) - idx_start
                    fixed_offsets_class = np.array(meta_class_indices_getter(meta), dtype=DTYPE_IDX) - idx_start
            if fixed_offsets:
                idx_parse = idx_start + fixed_offset_parse
            else:
                idx_parse = int(meta[REC_EMB_HAS_PARSE][0][JSONLD_IDX])
            indices_context_root[nbr_embeddings_in * i + root_i_offset] = idx_parse
            forest.pos_to_component_mapping[idx_parse] = root_i + root_i_offset
            if root_i_offset == 0:
                if fixed_offsets:
                    current_class_ids = forest.data[fixed_offsets_class + idx_start]
                else:
                    current_class_ids = [forest.data[_idx] for _idx in meta_class_indices_getter(meta)]
                # exclude mesh ids that occurs less then min_count (these were mapped to UNKNOWN in merge_batches)
                current_class_ids_mapped = [classes_mapping[c_id] for c_id in current_class_ids
                                            if c_id != unknown_id and c_id in classes_mapping]
                # convert list of indices
                classes_csr = indices_to_sparse(current_class_ids_mapped, len(classes_all_ids))
                classes_ids.append(classes_csr)

    logger.debug('indices created')
    return indices_context_root, classes_ids, sizes


def indices_reroot(index_files, **unused):
    # get root indices from files
    indices = index_iterator(index_files)
    return np.array(list(indices), dtype=DTYPE_IDX), None, [len(indices) for indices in index_files]


def indices_as_ids(index_files, **unused):
    indices = np.fromiter(index_iterator(index_files), dtype=np.int32)
    return indices, None


def batch_iter_default(forest_indices, indices_targets, nbr_embeddings_in, shuffle=True):
    """
    For every index in forest_indices, yield it and the respective values of indices_targets
    :param forest_indices: indices to the forest
    :param indices_targets: the target values, e.g. sparse class probabilities for every entry in forest_indices
    :param shuffle: if True, shuffle the indices
    :return:
    """
    assert len(forest_indices) % nbr_embeddings_in == 0, \
        'number of forest_indices is not a multiple of tree_count=%i' % nbr_embeddings_in
    indices = np.arange(len(indices_targets))
    if shuffle:
        np.random.shuffle(indices)
    for i in indices:
        yield [forest_indices[i * nbr_embeddings_in + j] for j in range(nbr_embeddings_in)], indices_targets[i]


def batch_iter_fixed_probs(forest_indices, number_of_samples):
    """
        For every _dummy_ index in forest_indices, yield its index and an array of probabilities of
        number_of_samples + 1 entries where only the first is one and all other are zero.
        :param forest_indices: dummy indices (just length is important)
        :param number_of_samples: number of classes/additional trees with probability of zero
        :return:
        """
    probs = np.zeros(shape=number_of_samples + 1, dtype=DTYPE_PROBS)
    probs[0] = 1
    for idx in forest_indices:
        yield [idx], probs


