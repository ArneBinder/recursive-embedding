import logging
import numpy as np
import os
import sys
import resource

from sklearn.feature_extraction.text import TfidfTransformer
from scipy.sparse import csr_matrix

from constants import TYPE_REF, KEY_HEAD, KEY_CANDIDATES, DTYPE_OFFSET, DTYPE_IDX, TYPE_REF_SEEALSO, \
    TYPE_SECTION_SEEALSO, UNKNOWN_EMBEDDING, vocab_manual, KEY_CHILDREN, TYPE_DBPEDIA_RESOURCE, TYPE_CONTEXT, \
    TYPE_PARAGRAPH, TYPE_TITLE, TYPE_SENTENCE, TYPE_SECTION, LOGGING_FORMAT, IDENTITY_EMBEDDING, CM_TREE, CM_AGGREGATE, \
    CM_SEQUENCE, TYPE_PMID, TARGET_EMBEDDING, OFFSET_ID
from sequence_trees import Forest
from mytools import numpy_load

RECURSION_LIMIT_MIN = 1000
RECURSION_LIMIT_ADD = 100

CONTEXT_ROOT_OFFEST = 2
SEEALSO_ROOT_OFFSET = 3
MESH_ROOT_OFFSET = 3

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
        for child_offset in forest.get_children(idx):
            child_data = forest.data[idx + child_offset]
            # allow only one type of link edge
            assert child_data == d_link_type, \
                'link_data (%s, data=%i, idx=%i) is not as expected (%s). parent: %s, data=%i, idx=%i' \
                % (forest.lexicon.get_s(child_data, data_as_hashes=forest.data_as_hashes),
                   child_data, idx + child_offset,
                   forest.lexicon.get_s(d_link_type, data_as_hashes=forest.data_as_hashes),
                   forest.lexicon.get_s(forest.data[idx], data_as_hashes=forest.data_as_hashes),
                   forest.data[idx], idx)
            child_offset_target = forest.get_children(idx + child_offset)
            assert len(child_offset_target) == 1, 'link has more or less then one targets: %i' % len(child_offset_target)

            target_id_idx = idx + child_offset + child_offset_target[0]
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
                  link_cost_ref=None, link_cost_ref_seealso=1, reroot=False, max_size_plain=1000, root_ids_set = None,
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
    if reroot:
        assert concat_mode == CM_TREE, 'reroot requires concat_mode==%s, but found concat_mode: %s' % (CM_TREE, concat_mode)

    #sys.setrecursionlimit(max(RECURSION_LIMIT_MIN, max_depth + context + RECURSION_LIMIT_ADD))
    sys.setrecursionlimit(1000)
    #print(resource.getrlimit(resource.RLIMIT_STACK))

    lexicon = forest.lexicon
    costs = {}
    link_types = []
    #root_types = []

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
    data_root = lexicon.get_d(TYPE_DBPEDIA_RESOURCE, data_as_hashes=forest.data_as_hashes)

    # do not remove TYPE_ANCHOR (nif:Context), as it is used for aggregation
    remove_types_naive_str = [TYPE_REF_SEEALSO, TYPE_REF, TYPE_DBPEDIA_RESOURCE, TYPE_SECTION_SEEALSO, TYPE_PARAGRAPH,
                              TYPE_TITLE, TYPE_SECTION, TYPE_SENTENCE]
    remove_types_naive = [lexicon.get_d(s, data_as_hashes=forest.data_as_hashes) for s in
                          remove_types_naive_str]

    n = 0

    logger.debug('create trees with concat_mode=%s' % concat_mode)

    if concat_mode == CM_TREE:
        for idx in indices:
            if reroot:
                tree_context = forest.get_tree_dict_rooted(idx=idx, max_depth=max_depth, #transform=transform,
                                                           costs=costs, link_types=link_types)
            else:
                tree_context = forest.get_tree_dict(idx=idx, max_depth=max_depth, context=context, transform=transform,
                                                    costs=costs, link_types=link_types)
            yield tree_context
            n += 1
    elif concat_mode == CM_AGGREGATE:
        # ATTENTION: works only if idx points to a data_nif_context CONTEXT_ROOT_OFFEST behind the root and leafs are
        # sequential and in order, especially root_ids occur in data only directly after link_types
        #sizes = []
        for idx in indices:
            # follow to first element of sequential data
            context_child_offset = forest.get_children(idx)[0]
            idx_start = idx + context_child_offset

            root_pos = idx - CONTEXT_ROOT_OFFEST
            root_idx = forest.root_mapping[root_pos]
            # get position of next root or end of sequence data
            if root_idx == len(forest.roots) - 1:
                idx_end = len(forest)
            else:
                idx_end = forest.roots[root_idx+1]

            data_span_cleaned = forest.get_data_span_cleaned(idx_start=idx_start, idx_end=idx_end,
                                                             link_types=link_types,
                                                             remove_types=remove_types_naive, transform=transform)
            if max_size_plain > 0:
                if len(data_span_cleaned) > max_size_plain:
                    logger.warning('len(data_span_cleaned)==%i > max_size_plain==%i. Cut tokens to max_size_plain.'
                                   % (len(data_span_cleaned), max_size_plain))
                #sizes.append([root_idx, len(data_span_cleaned)])
                data_span_cleaned = data_span_cleaned[:max_size_plain]
            tree_context = {KEY_HEAD: data_nif_context_transformed,
                            KEY_CHILDREN: [{KEY_HEAD: d, KEY_CHILDREN: []} for d in data_span_cleaned]}
            yield tree_context
            n += 1
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
            f.set_children_with_parents()
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
    logger.info('created %i trees' % n)


def reroot_wrapper(tree_iter, neg_samples, forest, nbr_indices, indices_mapping, transform=True, debug=False, **kwargs):
    logger.debug('select %i new root indices (selected data size: %i)' % (nbr_indices, len(indices_mapping)))
    if debug:
        logger.warning('use %i FIXED indices (debug: True)' % nbr_indices)
        assert nbr_indices <= len(indices_mapping), 'nbr_indices (%i) is higher then selected data size (%i)' \
                                                    % (nbr_indices, len(indices_mapping))
        _indices = np.arange(nbr_indices, dtype=DTYPE_IDX)
    else:
        _indices = np.random.randint(len(indices_mapping), size=nbr_indices)
    indices = indices_mapping[_indices]
    d_target = forest.lexicon.get_d(s=vocab_manual[TARGET_EMBEDDING], data_as_hashes=forest.data_as_hashes)
    #d_identity = forest.lexicon.get_d(s=vocab_manual[IDENTITY_EMBEDDING], data_as_hashes=forest.data_as_hashes)
    for tree in tree_iter(forest=forest, indices=indices, reroot=True, **kwargs):
        #samples = np.random.choice(forest.data, size=neg_samples + 1)
        # sample only from selected data
        sample_indices = np.random.choice(indices_mapping, size=neg_samples + 1)
        samples = forest.data[sample_indices]
        # replace samples that equal the head/root
        rep = np.random.randint(len(forest.lexicon) - 1)
        if rep == tree[KEY_HEAD]:
            rep = len(forest.lexicon) - 1
        samples[samples == tree[KEY_HEAD]] = rep
        # set all IDs to TARGET. That should affect only IDs mentioned under links, ID mentions under roots are
        # replaced by IDENTITY in train_fold.execute_run.
        samples[samples < 0] = d_target

        samples = forest.lexicon.transform_indices(samples)
        samples[0] = tree[KEY_HEAD]
        tree[KEY_CANDIDATES] = samples
        tree[KEY_HEAD] = None
        yield tree


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

    counts = csr_matrix((data, indices, indptr), dtype=int)
    logger.debug('shape of count matrix: %s' % str(counts.shape))

    # transform to tf-idf
    tf_transformer = TfidfTransformer(use_idf=False).fit(counts)
    tf_idf = tf_transformer.transform(counts)
    return [tf_idf[positions[i]:positions[i+1], :] for i in range(len(positions)-1)], vocabulary


def indices_dbpedianif(index_files, forest, **unused):

    # get root indices from files
    indices = index_iterator(index_files)
    # map to context and seealso indices

    indices_mapped = root_id_to_idx_offsets_iterator(indices, mapping=forest.roots,
                                                     offsets=np.array([CONTEXT_ROOT_OFFEST, SEEALSO_ROOT_OFFSET]))
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
                    indices_sealso_contexts_lists.append(forest.roots[curren_root_ids_seealsos] + CONTEXT_ROOT_OFFEST)
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


def indices_to_sparse(indices, length, dtype=np.float32):
    return csr_matrix((np.ones(len(indices), dtype=dtype), (np.zeros(len(indices), dtype=dtype), indices)),
                      shape=(1, length))


def indices_bioasq(index_files, forest, classes_ids, **unused):

    classes_mapping = {cd: i for i, cd in enumerate(classes_ids)}

    unknown_id = forest.lexicon.get_d(vocab_manual[UNKNOWN_EMBEDDING], data_as_hashes=False)

    # get root indices from files
    indices = index_iterator(index_files)

    # map to context and mesh indices
    indices_mapped = root_id_to_idx_offsets_iterator(indices, mapping=forest.roots,
                                                     offsets=(CONTEXT_ROOT_OFFEST, MESH_ROOT_OFFSET))
    # unzip (produces lists)
    root_ids, indices_context_root, indices_mesh_root = zip(*indices_mapped)

    mesh_ids = []
    for i, mesh_root_idx in enumerate(indices_mesh_root):
        mesh_indices = forest.get_children(mesh_root_idx) + mesh_root_idx
        # exclude mesh ids that occurs less then min_count (these were mapped to UNKNOWN in merge_batches)
        current_mesh_ids_mapped = [classes_mapping[m_id] for m_id in forest.data[mesh_indices] if m_id != unknown_id]
        # convert list of indices
        mesh_csr = indices_to_sparse(current_mesh_ids_mapped, len(classes_ids))
        mesh_ids.append(mesh_csr)

    return np.array(indices_context_root), mesh_ids, [len(np.load(ind_f)) for ind_f in index_files]


def indices_reroot(index_files, **unused):
    # get root indices from files
    indices = index_iterator(index_files)
    return np.array(list(indices), dtype=DTYPE_IDX), None, [len(np.load(ind_f)) for ind_f in index_files]


def indices_dbpedianif_dummy(forest, **unused):

    #CONTEXT_ROOT_OFFEST = 2
    #SEEALSO_ROOT_OFFSET = 3
    #indices_mapped = root_id_to_idx_offsets_iterator(indices=np.arange(len(forest.roots), dtype=DTYPE_IDX), mapping=forest.roots,
    #                                                 offsets=np.array([CONTEXT_ROOT_OFFEST]))
    ## unzip (produces lists)
    #root_ids, indices_context_root = zip(*indices_mapped)
    root_ids = np.arange(len(forest.roots))
    indices_context_root = forest.roots + CONTEXT_ROOT_OFFEST
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
            for c_offset in sequence_trees.get_children(idx_seealso_root):
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
                    f_seealso.set_children_with_parents()
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
                    f.set_children_with_parents()
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