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
    NIF_NEXT_WORD, NIF_WORD, PADDING_EMBEDDING
from sequence_trees import Forest
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

#OFFSET_CONTEXT_ROOT = 2
#OFFSET_SEEALSO_ROOT = 3
#OFFSET_RELATEDNESS_SCORE_ROOT = 3
#OFFSET_OTHER_ENTRY_ROOT = 7

logger = logging.getLogger('data_iterators')
logger.setLevel(logging.DEBUG)
logger_streamhandler = logging.StreamHandler()
logger_streamhandler.setLevel(logging.DEBUG)
logger_streamhandler.setFormatter(logging.Formatter(LOGGING_FORMAT))
logger.addHandler(logger_streamhandler)
logger.propagate = False


# TODO: move sampling to do_epoch
def data_tuple_iterator_reroot(sequence_trees, neg_samples, index_files=[], indices=None, max_depth=100,
                               link_cost_ref=None, link_cost_ref_seealso=-1, #transform=True,
                               **unused):
    """
    Maps: index (index files) --> ((children, candidate_heads), probs)
    First candidate_head is the original head

    :param sequence_trees:
    :param neg_samples:
    :param index_files:
    :param indices:
    :param max_depth:
    :param link_cost_ref:
    :param link_cost_ref_seealso:
    :param transform:
    :param unused:
    :return:
    """
    raise NotImplementedError('data_tuple_iterator_reroot is deprecated. use tree_iterator')
    logger.debug('size of data: %i' % len(sequence_trees))
    logger.debug('size of lexicon: %i' % len(sequence_trees.lexicon))
    assert max_depth > 0, 'can not produce candidates for zero depth trees (single nodes)'

    lexicon = sequence_trees.lexicon
    link_ids = []
    costs = {}
    if TYPE_REF in lexicon.strings:
        data_ref = lexicon.get_d(TYPE_REF, data_as_hashes=sequence_trees.data_as_hashes)
        link_ids.append(data_ref)
        if link_cost_ref is not None:
            costs[data_ref] = link_cost_ref
    if TYPE_REF_SEEALSO in lexicon.strings:
        data_ref_seealso = lexicon.get_d(TYPE_REF_SEEALSO, data_as_hashes=sequence_trees.data_as_hashes)
        link_ids.append(data_ref_seealso)
        costs[data_ref_seealso] = link_cost_ref_seealso
    #root_types = []
    #if TYPE_PMID in lexicon.strings:
    #    d_pmid = lexicon.get_d(TYPE_PMID, data_as_hashes=sequence_trees.data_as_hashes)
    #    root_types.append(d_pmid)
    #if TYPE_DBPEDIA_RESOURCE in lexicon.strings:
    #    d_dbpedia_resource = lexicon.get_d(TYPE_DBPEDIA_RESOURCE, data_as_hashes=sequence_trees.data_as_hashes)
    #    root_types.append(d_dbpedia_resource)
    #data_identity = lexicon.get_d(vocab_manual[IDENTITY_EMBEDDING], data_as_hashes=sequence_trees.data_as_hashes)

    if len(index_files) > 0:
        indices = np.concatenate([numpy_load(fn) for fn in index_files])

    # take all, if indices is not set
    if indices is None:
        indices = np.arange(len(sequence_trees))# range(len(sequence_trees))
    logger.info('size of used indices: %i' % len(indices))
    # try maximal every one twice
    max_tries = neg_samples
    count = 0
    for idx in indices:
        if idx in link_ids:
            continue
        #candidate_ids = []
        candidate_data = []
        try_count = 0
        while len(candidate_data) < neg_samples and try_count < max_tries:
            idx_cand = np.random.randint(len(sequence_trees), size=1)[0]
            data_cand = sequence_trees.data[idx_cand]
            if data_cand != sequence_trees.data[idx] \
                    and data_cand not in link_ids:# \
                    #and idx_cand not in candidate_ids:#\
                    #and sequence_trees.data[idx_cand] not in sequence_trees.root_id_mapping:
                #if data_cand in sequence_trees.root_id_mapping:
                #    data_cand = data_identity
                #if transform:
                #    data_cand = lexicon.transform_idx(idx=data_cand, root_id_pos=sequence_trees.root_id_pos)
                candidate_data.append(data_cand)
            else:
                try_count += 1

        if try_count == max_tries:
            logger.warning('not enough samples: %i, required: %i. skip idx=%i' % (len(candidate_data), neg_samples, idx))
            continue
        tree = sequence_trees.get_tree_dict_rooted(idx=idx, max_depth=max_depth, #transform=transform,
                                                   costs=costs, link_types=link_ids)

        #if transform:
        candidate_data = [lexicon.transform_idx(idx=d) for d in candidate_data]

        children = tree[KEY_CHILDREN]
        if len(children) > 0:
            candidate_data = [tree[KEY_HEAD]] + candidate_data
            probs = np.zeros(shape=len(candidate_data), dtype=int)
            probs[0] = 1
            yield [(children, candidate_data), probs]
            count += 1
    logger.info('use %i trees for training' % count)


# deprecated
def get_tree_naive(idx_start, idx_end, forest, data_aggregator, concat_mode=CM_SEQUENCE, link_types=[], remove_types=[]):

    data = np.zeros(idx_end - idx_start + 1, dtype=forest.data.dtype)
    data[:-1] = forest.data[idx_start:idx_end]
    ## append 'nif:context'
    data[-1] = data_aggregator

    ## remove entries
    indices_remove = []
    ## remove link entries
    for link_type in link_types:
        indices_remove.append(np.where(data == link_type)[0] + 1)
    ## remove other entries of specified types
    for remove_type in remove_types:
        indices_remove.append(np.where(data == remove_type)[0])
    indices_remove_np = np.sort(np.concatenate(indices_remove))
    mask = np.ones(data.shape, dtype=bool)
    mask[indices_remove_np] = False
    data = data[mask]

    #d_unknown = forest.lexicon.get_d(vocab_manual[UNKNOWN_EMBEDDING], data_as_hashes=forest.data_as_hashes)
    #data = np.ones(shape=idx_end-idx_start, dtype=forest.data.dtype) * d_unknown

    if concat_mode == CM_SEQUENCE:
        parents = np.ones(len(data), dtype=DTYPE_OFFSET)
        parents[-1] = 0
    elif concat_mode == CM_AGGREGATE:
        #parents = np.zeros(len(data), dtype=DTYPE_OFFSET)
        #for i in range(len(parents)-1):
        #    parents[i] = len(parents) - i - 1
        parents = np.arange(len(data))[::-1]
    else:
        raise ValueError('unknown concat_mode=%s' % concat_mode)

    return Forest(data=data, parents=parents, lexicon=forest.lexicon)


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


def tree_iterator(indices, forest, concat_mode=CM_TREE, max_depth=9999, context=0, transform=True,
                  link_cost_ref=None, link_cost_ref_seealso=1, reroot=False, max_size_plain=1000,
                  keep_prob_blank=1.0, keep_prob_node=1.0, blank_types=(), add_heads_types=(), additional_heads=0,
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
        # use if and warning instead of assertion because of dummy concat modes
        #if reroot and concat_mode != CM_TREE:
        #    logger.warning('reroot requires concat_mode==%s, but found concat_mode: %s' % (CM_TREE, concat_mode))
        if reroot:
            assert concat_mode == CM_TREE, 'reroot requires concat_mode==%s, but found concat_mode: %s' \
                                           % (CM_TREE, concat_mode)

        # enable tree dict caching
        #forest._dicts = {}

        #sys.setrecursionlimit(max(RECURSION_LIMIT_MIN, max_depth + context + RECURSION_LIMIT_ADD))
        sys.setrecursionlimit(1000)
        #print(resource.getrlimit(resource.RLIMIT_STACK))

        lexicon = forest.lexicon
        costs = {}
        link_types = []
        #root_types = []
        # ATTENTION: don't transform data_blanking, it will be transformed in get_tree_dict / get_tree_dict_rooted
        data_blanking = lexicon.get_d(s=vocab_manual[BLANKED_EMBEDDING], data_as_hashes=forest.data_as_hashes)
        data_padding = lexicon.get_d(s=vocab_manual[PADDING_EMBEDDING], data_as_hashes=forest.data_as_hashes)

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

        #if TYPE_PMID in lexicon.strings:
        #    d_pmid = lexicon.get_d(TYPE_PMID, data_as_hashes=forest.data_as_hashes)
        #    root_types.append(d_pmid)
        #if TYPE_DBPEDIA_RESOURCE in lexicon.strings:
        #    d_dbpedia_resource = lexicon.get_d(TYPE_DBPEDIA_RESOURCE, data_as_hashes=forest.data_as_hashes)
        #    root_types.append(d_dbpedia_resource)

        data_nif_context = lexicon.get_d(TYPE_CONTEXT, data_as_hashes=forest.data_as_hashes)
        data_nif_context_transformed = data_nif_context
        data_unknown_transformed = lexicon.get_d(vocab_manual[UNKNOWN_EMBEDDING], data_as_hashes=forest.data_as_hashes)
        if transform:
            data_nif_context_transformed = lexicon.transform_idx(idx=data_nif_context)
            data_unknown_transformed = lexicon.transform_idx(idx=data_unknown_transformed)
            data_padding = lexicon.transform_idx(idx=data_padding)

        # do not remove TYPE_ANCHOR (nif:Context), as it is used for aggregation
        if not RDF_BASED_FORMAT:
            remove_types_naive_str = [TYPE_REF_SEEALSO, TYPE_REF, TYPE_DBPEDIA_RESOURCE, TYPE_SECTION_SEEALSO, TYPE_PARAGRAPH,
                                      TYPE_TITLE, TYPE_SECTION, TYPE_SENTENCE]
        else:
            remove_types_naive_str = [REC_EMB_HAS_PARSE, NIF_WORD, NIF_NEXT_WORD, NIF_SENTENCE, NIF_NEXT_SENTENCE]
        remove_types_naive = [lexicon.get_d(s, data_as_hashes=forest.data_as_hashes) for s in
                              remove_types_naive_str if s in forest.lexicon]

        n = 0

        #logger.debug('create trees with concat_mode=%s' % concat_mode)
        #x = (1 + (len(indices) / 100))
        if concat_mode == CM_TREE:
            for idx in indices:
                tree_context = forest.get_tree_dict(idx=idx, max_depth=max_depth, context=context, transform=transform or reroot,
                                                    costs=costs, link_types=link_types, data_blank=data_blanking,
                                                    keep_prob_blank=keep_prob_blank, keep_prob_node=keep_prob_node,
                                                    blank_types=blank_types, go_back=reroot,
                                                    add_heads_types=add_heads_types,
                                                    add_heads_dummies=[data_padding] * additional_heads,
                                                    return_strings=DEBUG)
                yield tree_context
                n += 1
                # measure progress in percent
                #if n % x == 0:
                #    progress = n / x
                #    logger.debug('%i%%' % progress)

        elif concat_mode == CM_AGGREGATE:

            for idx in indices:
                # ATTENTION: idx has to be added to forest.pos_to_component_mapping before!
                idx_end = forest.pos_end(idx)
                data_span_cleaned = forest.get_data_span_cleaned(idx_start=idx, idx_end=idx_end,
                                                                 link_types=link_types,
                                                                 remove_types=remove_types_naive, transform=transform)
                if DEBUG:
                    data_span_cleaned_not_transformed = forest.get_data_span_cleaned(idx_start=idx, idx_end=idx_end,
                                                                                     link_types=link_types,
                                                                                     remove_types=remove_types_naive,
                                                                                     transform=False)
                    data_string_cleaned = [forest.lexicon.get_s(d, data_as_hashes=forest.data_as_hashes) for d in
                                           data_span_cleaned_not_transformed]
                if len(data_span_cleaned) > max_size_plain * (additional_heads + 1):
                    logger.warning('len(data_span_cleaned)==%i > max_size_plain==%i. Cut tokens to max_size_plain. root-idx: %i'
                                   % (len(data_span_cleaned) / (additional_heads + 1), max_size_plain, idx))
                data_span_cleaned = data_span_cleaned[:max_size_plain * (additional_heads + 1)]
                tree_context = {KEY_HEAD: data_nif_context_transformed,
                                KEY_CHILDREN: [{KEY_HEAD: data_span_cleaned[i], KEY_CHILDREN: [], KEY_HEAD_CONCAT: data_span_cleaned[i+1:i+1+additional_heads]} for i in range(0, len(data_span_cleaned), additional_heads + 1)]}
                if DEBUG:
                    tree_context_string = {KEY_HEAD: data_nif_context_transformed,
                                    KEY_CHILDREN: [{KEY_HEAD: data_string_cleaned[i], KEY_CHILDREN: [],
                                                    KEY_HEAD_CONCAT: data_string_cleaned[i + 1:i + 1 + additional_heads]} for
                                                   i in range(0, len(data_string_cleaned), additional_heads + 1)]}

                yield tree_context
                n += 1
                # measure progress in percent
                #if n % x == 0:
                #    progress = n / x
                #    logger.debug('%i%%' % progress)
            # statistics
            #sizes_np = np.array(sizes)
            #sizes_fn = 'sizes_plain.npy'
            #if os.path.exists(sizes_fn):
            #    sizes_np_loaded = np.load(sizes_fn)
            #    sizes_np = np.concatenate((sizes_np_loaded, sizes_np))
            #    logger.debug('append sizes to: %s' % sizes_fn)
            #else:
            #    logger.debug('write sizes to: %s' % sizes_fn)
            #sizes_np.dump(sizes_fn)

        elif concat_mode == CM_SEQUENCE:
            # DEPRECATED
            raise NotImplementedError('concat_mode=%s is deprecated. Use "%s" or "%s" instead.' % (concat_mode, CM_TREE, CM_AGGREGATE))
            logger.warning('concat_mode=%s is deprecated. Use "%s" or "%s" instead.' % (concat_mode, CM_TREE, CM_AGGREGATE))
            # TODO: remove
            # ATTENTION: works only if idx points to a data_nif_context and leafs are sequential and in order, especially
            # root_ids occur only directly after link_types
            for idx in indices:
                # follow to first element of sequential data
                context_child_offset = forest.get_children(idx)[0]
                # find last element
                idx_end = idx + context_child_offset
                for idx_end in range(idx + context_child_offset, len(forest)):
                    if forest.data[idx_end] == data_root:
                        break

                f = get_tree_naive(idx_start=idx+context_child_offset, idx_end=idx_end, forest=forest,
                                   concat_mode=concat_mode, link_types=link_types,
                                   remove_types=remove_types_naive, data_aggregator=data_nif_context)
                #f.set_children_with_parents()
                tree_context = f.get_tree_dict(max_depth=max_depth, context=context, transform=transform)
                yield tree_context
                n += 1
        elif concat_mode == 'dummy_dbpedianif1000':
            for idx in indices:
                yield {'h': 12, 'c': [{'h': 14, 'c': [{'h': 16, 'c': [{'h': 1, 'c': [{'h': 1, 'c': [{'h': 1, 'c': [{'h': -1952, 'c': [{'h': -1300, 'c': [{'h': 15, 'c': []}]}, {'h': -23, 'c': [{'h': -12238, 'c': [{'h': -15, 'c': []}, {'h': -12237, 'c': []}, {'h': -3650, 'c': []}, {'h': -1045, 'c': []}]}]}, {'h': -23, 'c': [{'h': -712, 'c': [{'h': -10, 'c': []}, {'h': -517, 'c': [{'h': 15, 'c': []}]}]}]}, {'h': -19, 'c': []}]}]}, {'h': -1275, 'c': [{'h': -2472, 'c': [{'h': -42, 'c': []}, {'h': -4600, 'c': []}]}, {'h': -21, 'c': []}, {'h': -32626, 'c': []}, {'h': -6, 'c': [{'h': -9978, 'c': [{'h': -15, 'c': []}, {'h': -2037, 'c': [{'h': -4600, 'c': []}]}, {'h': -4600, 'c': []}, {'h': -8127, 'c': []}, {'h': 15, 'c': []}, {'h': -1, 'c': []}, {'h': -66750, 'c': []}, {'h': -8, 'c': []}]}]}, {'h': -19, 'c': []}]}]}, {'h': -5279, 'c': [{'h': 0, 'c': []}, {'h': -1300, 'c': []}, {'h': -119, 'c': [{'h': -14, 'c': [{'h': -15, 'c': []}, {'h': -9665, 'c': [{'h': 0, 'c': []}, {'h': 15, 'c': []}, {'h': 0, 'c': []}]}, {'h': -10, 'c': []}, {'h': -1838, 'c': [{'h': -361, 'c': []}, {'h': -477, 'c': []}, {'h': -104, 'c': []}, {'h': -464, 'c': [{'h': -5244, 'c': [{'h': -20136, 'c': []}]}, {'h': -79, 'c': [{'h': -1503, 'c': []}]}]}]}]}]}, {'h': -19, 'c': []}]}]}]}]}]}
                n += 1
        elif concat_mode == 'dummy_unknown':
            for idx in indices:
                yield {KEY_HEAD: data_nif_context_transformed, KEY_CHILDREN: [{KEY_HEAD: data_unknown_transformed, KEY_CHILDREN: []}] * 7}
                n += 1
        else:
            raise ValueError('unknown concat_mode=%s' % concat_mode)
        #logger.debug('created %i trees' % n)
    except Exception as e:
        logger.error('exception occurred in tree_iterator:')
        traceback.print_exc()
        raise e


# DEPRECATED use get_nearest_neighbor_samples_batched
def get_nearest_neighbor_samples(idx_transformed, all_candidate_indices, lexicon, embedder, session, nbr):
    all_candidate_indices_transformed = np.array(lexicon.transform_indices(indices=all_candidate_indices))
    feed_dict = {embedder.reference_indices: [idx_transformed],
                 embedder.candidate_indices: all_candidate_indices_transformed}
    all_sims = session.run(embedder.reference_vs_candidate, feed_dict)
    max_indices_indices = np.argpartition(all_sims[0], -(nbr + 1))[-(nbr + 1):]
    max_indices = all_candidate_indices_transformed[max_indices_indices]
    max_indices[max_indices == idx_transformed] = max_indices[0]
    max_indices[0] = idx_transformed
    return max_indices


def get_nearest_neighbor_samples_batched(reference_indices_transformed, candidate_indices_transformed, embedder,
                                         session, nbr):
    #logger.debug('calc sims (batch_size: %i) ...' % len(reference_indices_transformed))
    feed_dict = {embedder.reference_indices: reference_indices_transformed,
                 embedder.candidate_indices: candidate_indices_transformed}
    all_sims = session.run(embedder.reference_vs_candidate, feed_dict)
    res = {}
    #logger.debug('calc nearest neighbor indices (batch_size: %i) ...' % len(reference_indices_transformed))
    max_indices_indices = np.argpartition(all_sims, -(nbr + 1))[:,-(nbr + 1):]
    #logger.debug('calc nearest neighbors (batch_size: %i) ...' % len(reference_indices_transformed))
    for i, ref_idx in enumerate(reference_indices_transformed):
        max_indices = candidate_indices_transformed[max_indices_indices[i]]
        max_indices[max_indices == ref_idx] = max_indices[0]
        max_indices[0] = ref_idx
        res[ref_idx] = max_indices
    return res


def reroot_wrapper(tree_iter, neg_samples, forest, indices, indices_mapping=None, #data_indices_all=None,
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
        for tree in tree_iter(forest=forest, indices=indices, reroot=True, transform=True, **kwargs):
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
                #samples[0] = tree[KEY_HEAD]
            elif sample_method in [SAMPLE_METHOD_FREQUENCY, SAMPLE_METHOD_FREQUENCY_ALL]:
                # sample only from selected data
                sample_indices = np.random.choice(data_indices, size=neg_samples + 1)
                samples = forest.data[sample_indices]
                # replace samples that equal the head/root: sample replacement element
                rep = np.random.randint(len(lexicon_indices) - 1)
                # map replacement element from class index to lexicon index, if lexicon_indices are given
                #if lexicon_indices is not None:
                if lexicon_indices[rep] == head_transformed_back:
                    rep = lexicon_indices[-1]
                else:
                    rep = lexicon_indices[rep]
                #else:
                #    # otherwise just use as index to lexicon
                #    if rep == head_transformed_back:
                #        rep = len(lexicon_indices) - 1

                samples[samples == head_transformed_back] = rep
                # set all IDs to TARGET. That should affect only IDs mentioned under links, ID mentions under roots are
                # replaced by IDENTITY in train_fold.execute_run.
                samples[samples < 0] = d_target
                samples[0] = head_transformed_back
                samples = forest.lexicon.transform_indices(samples)
                #samples[0] = tree[KEY_HEAD]
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


def get_scores_list_from_indices(indices, forest, type_prefix):
    # scores_ids = forest.data[list(indices_score)]
    scores_ids = forest.data[indices]
    scores_list = []
    l = len(type_prefix + SEPARATOR)
    for rel_score_id in scores_ids:
        rel_score_str = forest.lexicon.get_s(rel_score_id, data_as_hashes=forest.data_as_hashes)
        rel_score = float(rel_score_str[l:])
        scores_list.append(rel_score)
    return scores_list


def get_context_roots(root_indices, forest):
    return forest.roots[root_indices] + OFFSET_CONTEXT_ROOT


def get_and_merge_other_context_roots(indices_context_root, indices_other, forest):
    other_root_ids = forest.data[indices_other]
    other_root_indices = []
    for other_root_id in other_root_ids:
        other_root_index = forest.root_id_pos.get(other_root_id, None)
        assert other_root_index is not None, 'linked root: %i not found in root_id_pos' % other_root_id
        other_root_indices.append(other_root_index)
    other_indices_context_root = np.array(other_root_indices, dtype=DTYPE_IDX) + OFFSET_CONTEXT_ROOT
    # re-pack indices: [..., sentence_A_index_i, ...], [..., sentence_A_index_i, ...]
    #                  -> [..., sentence_A_index_i, sentence_B_index_i, ...]
    all_indices = np.concatenate((np.array(indices_context_root).reshape((1, -1)),
                                  other_indices_context_root.reshape((1, -1)))).T.reshape((-1))
    return all_indices


def indices_to_sparse(indices, length, dtype=np.float32):
    return csr_matrix((np.ones(len(indices), dtype=dtype), (np.zeros(len(indices), dtype=dtype), indices)),
                      shape=(1, length))


def indices_dbpedianif(index_files, forest, **unused):
    if RDF_BASED_FORMAT:
        raise NotImplementedError('indices_dbpedianif not implemented for RDF_BASED_FORMAT')

    # get root indices from files
    indices = index_iterator(index_files)
    # map to context and seealso indices

    indices_mapped = root_id_to_idx_offsets_iterator(indices, mapping=forest.roots,
                                                     offsets=np.array([OFFSET_CONTEXT_ROOT, OFFSET_SEEALSO_ROOT]))
    # unzip (produces lists)
    root_ids, indices_context_root, indices_seealso_root = zip(*indices_mapped)
    root_ids_seealsos_iterator = link_root_ids_iterator(indices=indices_seealso_root, forest=forest,
                                                        link_type=TYPE_REF_SEEALSO)

    #root_ids_seealsos_list = []
    indices_sealso_contexts_lists = []
    #root_ids_list = []
    indices_context_root_list = []

    root_id_prefix_exclude = 'http://dbpedia.org/resource/List_of_'

    # do not use root_id, etc., if root_ids_seealsos is empty
    for i, root_ids_seealsos in enumerate(root_ids_seealsos_iterator):
        if root_ids_seealsos is not None:
            skip = False
            curren_root_ids_seealsos = []
            for root_id_seealso in root_ids_seealsos:
                root_id_str = forest.lexicon_roots.get_s(root_id_seealso, data_as_hashes=forest.data_as_hashes)
                if root_id_str[:len(root_id_prefix_exclude)] == root_id_prefix_exclude:
                    skip = True
                else:
                    curren_root_ids_seealsos.append(root_id_seealso)
            if not skip:
                root_id_str = forest.lexicon_roots.get_s(root_ids[i], data_as_hashes=forest.data_as_hashes)
                if root_id_str[:len(root_id_prefix_exclude)] != root_id_prefix_exclude:
                    #root_ids_seealsos_list.append(curren_root_ids_seealsos)
                    indices_sealso_contexts_lists.append(forest.roots[curren_root_ids_seealsos] + OFFSET_CONTEXT_ROOT)
                    #root_ids_list.append(root_ids[i])
                    indices_context_root_list.append(indices_context_root[i])

    #root_ids_set = set(root_ids_list)
    indices_context_set = set(indices_context_root_list)
    #added_root_ids = []
    added_indices_context_root = []

    #for ls in root_ids_seealsos_list:
    for ls in indices_sealso_contexts_lists:
        #for root_id_seealso in ls:
        for idx_seealso_context in ls:
            #if root_id_seealso not in root_ids_set and root_id_seealso not in added_root_ids:
            if idx_seealso_context not in indices_context_set and idx_seealso_context not in added_indices_context_root:
                #added_root_ids.append(root_id_seealso)
                #idx_seealso_context = forest.roots[root_id_seealso] + CONTEXT_ROOT_OFFEST
                added_indices_context_root.append(idx_seealso_context)

    #root_ids_list.extend(added_root_ids)
    indices_context_root_list.extend(added_indices_context_root)
    #root_ids_seealsos_list.extend([[]] * len(added_indices_context_root))
    indices_sealso_contexts_lists.extend([[]] * len(added_indices_context_root))
    logger.debug('selected %i root_ids (filtered; source + target trees)' % len(indices_context_root_list))

    #return np.array(root_ids_list), np.array(indices_context_root_list), root_ids_seealsos_list
    return np.array(indices_context_root_list), indices_sealso_contexts_lists, [len(np.load(ind_f)) for ind_f in index_files]


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


def indices_sick(index_files, forest, rdf_based_format=RDF_BASED_FORMAT, meta_getter=None, **unused):

    # get root indices from files
    indices_per_file = load_indices(index_files)
    indices = np.concatenate(indices_per_file)
    if rdf_based_format:
        if meta_getter is None:
            meta_getter = lambda x: float(x[REC_EMB_HAS_GLOBAL_ANNOTATION][0][REC_EMB_GLOBAL_ANNOTATION][0]
                                          [SICK_RELATEDNESS_SCORE][0][JSONLD_VALUE])
        nbr_embeddings_in = 2
        assert nbr_embeddings_in >=1, 'nbr_embeddings_in has to be at least 1, but is %s' % str(nbr_embeddings_in)
        relatedness_scores = np.empty_like(indices, dtype=np.float32)
        all_indices = np.empty(shape=len(indices) * nbr_embeddings_in, dtype=DTYPE_IDX)
        for j in range(nbr_embeddings_in):
            for i, idx in enumerate(indices):
                meta = forest.get_tree_dict_string(idx=forest.roots[idx+j], stop_types=(REC_EMB_HAS_PARSE),
                                                   index_types=(REC_EMB_HAS_PARSE))
                all_indices[nbr_embeddings_in * i + j] = int(meta[REC_EMB_HAS_PARSE][0][JSONLD_IDX])
                if j == 0:
                    relatedness_scores[i] = meta_getter(meta)
    else:
        indices_score = forest.roots[indices] + OFFSET_RELATEDNESS_SCORE_ROOT + 1
        relatedness_scores_list = get_scores_list_from_indices(indices_score, forest, type_prefix=TYPE_RELATEDNESS_SCORE)
        relatedness_scores = np.array(relatedness_scores_list, dtype=np.float32)

        indices_context_root = get_context_roots(indices, forest)
        indices_other = forest.roots[indices] + OFFSET_OTHER_ENTRY_ROOT + 1
        all_indices = get_and_merge_other_context_roots(indices_context_root, indices_other, forest)

    logger.debug('indices created')
    # shift original score range [1.0..5.0] into range [0.0..1.0]
    return all_indices, (relatedness_scores - 1) / 4.0, [len(indices) * 2 for indices in indices_per_file]


def indices_multiclass(index_files, forest, classes_all_ids, classes_root_offset, other_offset=None, nbr_embeddings_in=1,
                       rdf_based_format=RDF_BASED_FORMAT, meta_class_indices_getter=None, meta_args={}, fixed_offsets=False, **unused):
    # get root indices from files
    indices_per_file = load_indices(index_files)
    indices = np.concatenate(indices_per_file)
    if rdf_based_format:
        assert nbr_embeddings_in >= 1, 'nbr_embeddings_in has to be at least 1, but is %s' % str(nbr_embeddings_in)
        assert meta_class_indices_getter is not None, 'meta_getter is None'
        meta_args_default = {'stop_types': (REC_EMB_HAS_PARSE,), 'index_types': (REC_EMB_HAS_PARSE,)}
        for k in meta_args_default:
            meta_args[k] = meta_args.get(k, ()) + meta_args_default[k]
        meta_args_default.update(meta_args)
        #relatedness_scores = np.empty_like(indices, dtype=np.float32)
        classes_mapping = {cd: i for i, cd in enumerate(classes_all_ids)}
        unknown_id = forest.lexicon.get_d(vocab_manual[UNKNOWN_EMBEDDING], data_as_hashes=False)
        classes_ids = []
        indices_context_root = np.empty(shape=len(indices) * nbr_embeddings_in, dtype=DTYPE_IDX)
        fixed_offset_parse = None
        fixed_offsets_class = None
        for root_i_offset in range(nbr_embeddings_in):
            if DEBUG:
                indices = np.sort(indices)
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
                forest.pos_to_component_mapping[idx_parse] = root_i
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

    else:
        indices_context_root = forest.roots[indices] + OFFSET_CONTEXT_ROOT
        indices_classes_root = forest.roots[indices] + classes_root_offset
        classes_ids = get_classes_ids(indices_classes_root, classes_all_ids, forest)

        if other_offset is not None:
            indices_other = forest.roots[indices] + other_offset
            indices_context_root = get_and_merge_other_context_roots(indices_context_root=indices_context_root,
                                                                     indices_other=indices_other, forest=forest)
    logger.debug('indices created')
    return indices_context_root, classes_ids, [len(indices) * nbr_embeddings_in for indices in indices_per_file]


def indices_reroot(index_files, **unused):
    # get root indices from files
    indices = index_iterator(index_files)
    return np.array(list(indices), dtype=DTYPE_IDX), None, [len(indices) for indices in index_files]


def indices_dbpedianif_dummy(forest, **unused):

    #CONTEXT_ROOT_OFFEST = 2
    #SEEALSO_ROOT_OFFSET = 3
    #indices_mapped = root_id_to_idx_offsets_iterator(indices=np.arange(len(forest.roots), dtype=DTYPE_IDX), mapping=forest.roots,
    #                                                 offsets=np.array([CONTEXT_ROOT_OFFEST]))
    ## unzip (produces lists)
    #root_ids, indices_context_root = zip(*indices_mapped)
    root_ids = np.arange(len(forest.roots))
    indices_context_root = forest.roots + OFFSET_CONTEXT_ROOT
    logger.debug('found %i root_ids' % len(root_ids))
    #return np.array(root_ids), np.array(indices_context_root), None
    #return root_ids, indices_context_root, None
    return indices_context_root, None


def indices_as_ids(index_files, **unused):
    indices = np.fromiter(index_iterator(index_files), dtype=np.int32)
    #return indices, indices, None
    return indices, None


# DEPRECATED
def data_tuple_iterator_dbpedianif(index_files, sequence_trees, concat_mode=CM_TREE,
                                   max_depth=9999, context=0, transform=True, offset_context=2,
                                   offset_seealso=3, link_cost_ref=None, link_cost_ref_seealso=1,
                                   bag_of_seealsos=True, root_strings=None,
                                   **unused):

    sys.setrecursionlimit(max(RECURSION_LIMIT_MIN, max_depth + context + RECURSION_LIMIT_ADD))

    lexicon = sequence_trees.lexicon
    costs = {}
    data_ref = lexicon.get_d(TYPE_REF, data_as_hashes=sequence_trees.data_as_hashes)
    data_ref_seealso = lexicon.get_d(TYPE_REF_SEEALSO, data_as_hashes=sequence_trees.data_as_hashes)
    data_root_seealso = lexicon.get_d(TYPE_SECTION_SEEALSO, data_as_hashes=sequence_trees.data_as_hashes)
    data_unknown = lexicon.get_d(vocab_manual[UNKNOWN_EMBEDDING], data_as_hashes=sequence_trees.data_as_hashes)
    if transform:
        #data_ref_transformed = sequence_trees.lexicon.transform_idx(data_ref)
        data_ref_seealso_transformed = sequence_trees.lexicon.transform_idx(data_ref_seealso)
        data_root_seealso_transformed = sequence_trees.lexicon.transform_idx(data_root_seealso)
    else:
        #data_ref_transformed = data_ref
        data_ref_seealso_transformed = data_ref_seealso
        data_root_seealso_transformed = data_root_seealso

    # do not remove TYPE_ANCHOR (nif:Context), as it is used for aggregation
    remove_types_naive_str = [TYPE_REF_SEEALSO, TYPE_REF, TYPE_DBPEDIA_RESOURCE, TYPE_SECTION_SEEALSO, TYPE_PARAGRAPH,
                              TYPE_TITLE, TYPE_SECTION, TYPE_SENTENCE]
    remove_types_naive = [lexicon.get_d(s, data_as_hashes=sequence_trees.data_as_hashes) for s in remove_types_naive_str]

    if link_cost_ref is not None:
        costs[data_ref] = link_cost_ref
    costs[data_ref_seealso] = link_cost_ref_seealso
    n = 0
    for file_name in index_files:
        indices = np.load(file_name)
        for root_id in indices:
            idx_root = sequence_trees.roots[root_id]
            idx_context_root = idx_root + offset_context
            idx_seealso_root = idx_root + offset_seealso
            children = []
            seealso_root_ids = []
            for c_idx in sequence_trees.get_children(idx_seealso_root):
                seealso_root_ids = []
                seealso_offset = sequence_trees.get_children(idx_seealso_root + c_offset)[0]
                seealso_idx = idx_seealso_root + c_offset + seealso_offset
                seealso_data_id = sequence_trees.data[seealso_idx]
                if seealso_data_id == data_unknown:
                    continue
                seealso_root_id = sequence_trees.root_id_mapping.get(seealso_data_id, None)
                if seealso_root_id is None:
                    continue
                if concat_mode == CM_TREE:
                    idx_root_seealso = sequence_trees.roots[seealso_root_id] + offset_context
                    tree_seealso = sequence_trees.get_tree_dict(idx=idx_root_seealso, max_depth=max_depth-2,
                                                                context=context, transform=transform,
                                                                costs=costs,
                                                                link_types=[data_ref, data_ref_seealso])
                else:
                    f_seealso = get_tree_naive(root=seealso_root_id, forest=sequence_trees, concat_mode=concat_mode,
                                               lexicon=lexicon, link_types=[data_ref, data_ref_seealso],
                                               remove_types=remove_types_naive)
                    #f_seealso.set_children_with_parents()
                    tree_seealso = f_seealso.get_tree_dict(max_depth=max_depth-2, context=context, transform=transform)
                children.append({KEY_HEAD: data_ref_seealso_transformed, KEY_CHILDREN: [tree_seealso]})
                seealso_root_ids.append(seealso_root_id)
            if len(children) > 0:
                if concat_mode == CM_TREE:
                    tree_context = sequence_trees.get_tree_dict(idx=idx_context_root, max_depth=max_depth,
                                                                context=context, transform=transform,
                                                                costs=costs,
                                                                link_types=[data_ref, data_ref_seealso])
                else:
                    f = get_tree_naive(root=root_id, forest=sequence_trees, concat_mode=concat_mode, lexicon=lexicon,
                                       link_types=[data_ref, data_ref_seealso], remove_types=remove_types_naive)
                    #f.set_children_with_parents()
                    tree_context = f.get_tree_dict(max_depth=max_depth, context=context, transform=transform)
                if bag_of_seealsos:
                    yield [[tree_context, {KEY_HEAD: data_root_seealso_transformed, KEY_CHILDREN: children}],
                           np.ones(shape=2, dtype=int)]
                    n += 1
                else:
                    for child in children:
                        # use fist child (^= the context) of tree_seealso
                        yield [[tree_context, child[KEY_CHILDREN][0]], np.ones(shape=2, dtype=int)]
                        n += 1

                # if debug is enabled, show root_id_strings and seealsos
                if root_strings is not None:
                    root_id = sequence_trees.data[idx_root + OFFSET_ID] - len(lexicon)
                    logger.debug('root: %s -> [%s]' % (root_strings[root_id], ', '.join([root_strings[root_id] for root_id in seealso_root_ids])))

                #if n >= n_max:
                #    break
        #if n >= n_max:
        #    break
    logger.info('created %i tree tuples' % n)


def batch_iter_naive(number_of_samples, forest_indices, forest_indices_targets, idx_forest_to_idx_trees, sampler=None):
    """
    batch iterator for tree tuple settings. yields the correct forest index and number_of_samples negative samples for
    every tree rooted at the respective forest_indices and the respective probabilities [1, 0, 0, ...] e.g.
    [1] + [0] * number_of_samples.
    :param number_of_samples: number of negative samples
    :param forest_indices: indices for forest that root the used trees
    :param forest_indices_targets: lists containing indices to trees that are true (positive) targets for every index
                                    in forest_indices
    :param idx_forest_to_idx_trees: mapping to convert forest_indices to indices of compiled trees
    :param sampler:
    :return: indices to forest (source, correct_target, number_of_samples negative samples), probabilities
    """
    if sampler is None:
        def sampler(idx_target):
            # sample from [1, len(dataset_indices)-1] (inclusive); size +1 to hold the correct target and +1 to hold
            # the origin/reference (idx)
            sample_indices = np.random.random_integers(len(forest_indices) - 1, size=number_of_samples + 1 + 1)
            # replace sampled correct target with 0 (0 is located outside the sampled values, so statistics remain correct)
            # (e.g. do not sample the correct target)
            sample_indices[sample_indices == idx_forest_to_idx_trees[idx_target]] = 0
            return sample_indices

    indices = np.arange(len(forest_indices))
    np.random.shuffle(indices)
    for i in indices:
        idx = forest_indices[i]
        for idx_target in forest_indices_targets[i]:
            sample_indices = sampler(idx_target)
            # set the first to the correct target
            sample_indices[1] = idx_forest_to_idx_trees[idx_target]
            # set the 0th to the origin/reference
            sample_indices[0] = idx_forest_to_idx_trees[idx]

            # convert candidates to ids
            candidate_indices = forest_indices[sample_indices[1:]]
            ix = np.isin(candidate_indices, forest_indices_targets[i])
            probs = np.zeros(shape=len(candidate_indices), dtype=DTYPE_PROBS)
            probs[ix] = 1

            yield forest_indices[sample_indices], probs


# DEPRECATED
def batch_iter_nearest(number_of_samples, forest_indices, forest_indices_targets, sess, tree_model,
                       highest_sims_model, dataset_trees, tree_model_batch_size, idx_forest_to_idx_trees):
    raise NotImplementedError('batch_iter_nearest is depreacted')

    _tree_embeddings = []
    feed_dict = {}
    if isinstance(tree_model, model_fold.DummyTreeModel):
        for start in range(0, dataset_trees.shape[0], tree_model_batch_size):
            feed_dict[tree_model.prepared_embeddings_placeholder] = convert_sparse_matrix_to_sparse_tensor(dataset_trees[start:start+tree_model_batch_size])
            current_tree_embeddings = sess.run(tree_model.embeddings_all, feed_dict)
            _tree_embeddings.append(current_tree_embeddings)
    else:
        for batch in td.group_by_batches(dataset_trees, tree_model_batch_size):
            feed_dict[tree_model.compiler.loom_input_tensor] = batch
            current_tree_embeddings = sess.run(tree_model.embeddings_all, feed_dict)
            _tree_embeddings.append(current_tree_embeddings)
    dataset_trees_embedded = np.concatenate(_tree_embeddings)
    logger.debug('calculated %i embeddings ' % len(dataset_trees_embedded))

    s = dataset_trees_embedded.shape[0]
    # calculate cosine sim for all combinations by tree-index ([0..tree_count-1])
    normed = pp.normalize(dataset_trees_embedded, norm='l2')
    logger.debug('normalized %i embeddings' % s)

    current_device = get_ith_best_device(1)
    with tf.device(current_device):
        logger.debug('calc nearest on device: %s' % str(current_device))
        neg_sample_indices = np.zeros(shape=(s, number_of_samples), dtype=np.int32)

        # initialize normed embeddings
        sess.run(highest_sims_model.normed_embeddings_init,
                 feed_dict={highest_sims_model.normed_embeddings_placeholder: normed})
        for i in range(s):
            current_sims = sess.run(highest_sims_model.sims,
                                    {
                                        highest_sims_model.reference_idx: i,
                                    })
            current_sims[i] = 0
            current_indices = np.argpartition(current_sims, -number_of_samples)[-number_of_samples:]
            neg_sample_indices[i, :] = current_indices

    # TODO: clear normed_embeddings (or move to second gpu?)
    #sess.run(highest_sims_model.normed_embeddings_init,
    #         feed_dict={highest_sims_model.normed_embeddings_placeholder: normed})
    logger.debug('created nearest indices')

    def sampler(idx_target):
        sample_indices = np.zeros(shape=number_of_samples + 1 + 1, dtype=np.int32)
        sample_indices[2:] = neg_sample_indices[idx_forest_to_idx_trees[idx_target]]
        return sample_indices

    for sample_indices, probs in batch_iter_naive(number_of_samples, forest_indices, forest_indices_targets,
                                                  idx_forest_to_idx_trees, sampler=sampler):
        yield sample_indices, probs


#def batch_iter_reroot(forest_indices, number_of_samples, data_transformed):
#    for idx in forest_indices:
#        samples = np.random.choice(data_transformed, size=number_of_samples+1)
#        samples[0] = data_transformed[idx]
#
#        #samples = forest.lexicon.transform_indices(samples)
#
#        probs = np.zeros(shape=number_of_samples + 1, dtype=DT_PROBS)
#        probs[samples == samples[0]] = 1
#
#        yield [idx], probs, samples


# DEPRECATED
def batch_iter_all(forest_indices, forest_indices_targets, batch_size):
    for i in range(len(forest_indices)):
        ix = np.isin(forest_indices, forest_indices_targets[i])
        probs = np.zeros(shape=len(ix), dtype=DTYPE_PROBS)
        probs[ix] = 1
        for start in range(0, len(probs), batch_size):
            # do not yield, if it is not full (end of the dataset)
            if start+batch_size > len(probs):
                continue
            sampled_indices = np.arange(start - 1, start + batch_size)
            sampled_indices[0] = i
            current_probs = probs[start:start + batch_size]
            yield forest_indices[sampled_indices], current_probs, None


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
        #yield [indices_forest_to_tree[forest_indices[i]]], indices_targets[i]
        #yield [forest_indices[i]], indices_targets[i]
        yield [forest_indices[i * nbr_embeddings_in + j] for j in range(nbr_embeddings_in)], indices_targets[i]


def batch_iter_fixed_probs(forest_indices, number_of_samples):
    """
        For every _dummy_ index in forest_indices, yield its index and an array of probabilities of
        number_of_samples + 1 entries where only the first is one and all other are zero.
        :param forest_indices: dummy indices (just length is important)
        :param number_of_samples: number of classes/additional trees with probability of zero
        :return:
        """
    #indices = np.arange(len(forest_indices))
    probs = np.zeros(shape=number_of_samples + 1, dtype=DTYPE_PROBS)
    probs[0] = 1
    for idx in forest_indices:
        #probs = np.zeros(shape=number_of_samples + 1, dtype=DT_PROBS)
        #probs[0] = 1
        yield [idx], probs


# deprecated. use batch_iter instead
def batch_iter_simtuple_dep(forest_indices, indices_targets, nbr_embeddings_in, shuffle=True):
    """
    For every index in forest_indices, yield it and the respective values of indices_targets
    :param forest_indices: indices to the forest
    :param indices_targets: a tuple containing (context_indices_targets, similarity_scores)
    :param shuffle: if True, shuffle the indices
    :return:
    """
    assert len(forest_indices) % nbr_embeddings_in == 0, \
        'number of forest_indices is not a multiple of tree_count=%i' % nbr_embeddings_in
    indices = np.arange(len(indices_targets))
    if shuffle:
        np.random.shuffle(indices)
    for i in indices:
        #yield [indices_forest_to_tree[forest_indices[i]]], indices_targets[i]
        #yield [forest_indices[i*2], forest_indices[i*2+1]], indices_targets[i]
        yield [forest_indices[i * nbr_embeddings_in + j] for j in range(nbr_embeddings_in)], indices_targets[i]


# not used
def load_sim_tuple_indices(filename, extensions=None):
    if extensions is None:
        extensions = ['']
    probs = []
    indices = []
    for ext in extensions:
        if not os.path.isfile(filename + ext):
            raise IOError('file not found: %s' % filename + ext)
        logger.debug('load idx file: %s' % filename + ext)
        _loaded = np.load(filename + ext).T
        if _loaded.dtype.kind == 'f':
            n = (len(_loaded) - 1) / 2
            _correct = _loaded[0].astype(int)
            _indices = _loaded[1:-n].astype(int)
            _probs = _loaded[-n:]
        else:
            n = (len(_loaded) - 1)
            _correct = _loaded[0]
            _indices = _loaded[1:]
            _probs = np.zeros(shape=(n, len(_correct)), dtype=np.float32)
        if len(indices) > 0:
            if not np.array_equal(indices[0][0], _correct):
                raise ValueError
        else:
            indices.append(_correct.reshape((1, len(_correct))))
            probs.append(np.ones(shape=(1, len(_correct)), dtype=np.float32))
        probs.append(_probs)
        indices.append(_indices)

    return np.concatenate(indices).T, np.concatenate(probs).T


#not used
def data_tuple_iterator(index_files, sequence_trees, root_idx=None, shuffle=False, extensions=None,
                        split=False, head_dropout=False, merge_prob_idx=None, subtree_head_ids=None, count=None,
                        merge=False, max_depth=9999, context=0, transform=True, **unused):
    lexicon = sequence_trees.lexicon

    # use this to enable full head dropout
    def set_head_neg(tree):
        tree[KEY_HEAD] -= len(lexicon)
        for c in tree[KEY_CHILDREN]:
            set_head_neg(c)

    if merge_prob_idx is not None:
        assert subtree_head_ids is not None and type(subtree_head_ids) == list, \
            'merge_prob_idx is given (%i), but subtree_head_ids is not a list' % merge_prob_idx
        assert root_idx is not None, 'merge_prob_idx is given (%i), but root_idx is not set' % merge_prob_idx
        assert not shuffle, 'merge_prob_idx is given (%i), but SHUFFLE is enabled' % merge_prob_idx
        assert not split, 'merge_prob_idx is given (%i), but SPLIT is enabled' % merge_prob_idx
    n_last = None
    for sim_index_file in index_files:
        indices, probabilities = load_sim_tuple_indices(sim_index_file, extensions)
        n = len(indices[0])
        assert n_last is None or n_last == n, 'all (eventually merged) index tuple files have to contain the ' \
                                              'same amount of tuple entries, but entries in %s ' \
                                              '(with extensions=%s) deviate with %i from %i' \
                                              % (sim_index_file, str(extensions), n, n_last)
        n_last = n
        if count is None:
            count = n
        _trees_merged = []
        _probs_merged = np.zeros(shape=(0,))
        for idx in range(len(indices)):
            index_tuple = indices[idx]
            _trees = [sequence_trees.get_tree_dict(idx=i, max_depth=max_depth, context=context, transform=transform) for i in index_tuple]
            _probs = probabilities[idx]

            if merge_prob_idx is not None:
                for i in range(n):
                    _trees[i][KEY_HEAD] = subtree_head_ids[i]
                new_root = {KEY_HEAD: root_idx, KEY_CHILDREN: _trees}
                _trees = [new_root]
                _probs = [_probs[merge_prob_idx]]
            else:
                if root_idx is not None:
                    _trees[0][KEY_HEAD] = root_idx
                # unify heads
                for i in range(1, n):
                    _trees[i][KEY_HEAD] = _trees[0][KEY_HEAD]

            if head_dropout:
                for t in _trees:
                    set_head_neg(t)

            if shuffle:
                perm = np.random.permutation(n)
                [_trees, _probs] = [[_trees[i] for i in perm], np.array([_probs[i] for i in perm])]
            if split:
                for i in range(1, n):
                    yield [[_trees[0], _trees[i]], np.array([_probs[0], _probs[i]])]
            elif merge:
                _trees_merged.extend(_trees)
                _probs_merged = np.concatenate((_probs_merged, _probs))
                if len(_trees_merged) >= count:
                    yield [_trees_merged, _probs_merged]
                    _trees_merged = []
                    _probs_merged = np.zeros((0,))
            else:
                yield [_trees, _probs]
