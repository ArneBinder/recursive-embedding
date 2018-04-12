import logging
import numpy as np
import os
import sys

from sklearn.feature_extraction.text import TfidfTransformer
from scipy.sparse import csr_matrix

from constants import TYPE_REF, KEY_HEAD, DTYPE_OFFSET, TYPE_REF_SEEALSO, TYPE_SECTION_SEEALSO, UNKNOWN_EMBEDDING, \
    vocab_manual, KEY_CHILDREN, TYPE_ROOT, TYPE_ANCHOR, TYPE_PARAGRAPH, TYPE_TITLE, TYPE_SENTENCE, TYPE_SECTION, \
    LOGGING_FORMAT, IDENTITY_EMBEDDING
from sequence_trees import Forest
from mytools import numpy_load

RECURSION_LIMIT_MIN = 1000
RECURSION_LIMIT_ADD = 100

logger = logging.getLogger('data_iterators')
logger.setLevel(logging.DEBUG)
logger_streamhandler = logging.StreamHandler()
logger_streamhandler.setLevel(logging.DEBUG)
logger_streamhandler.setFormatter(logging.Formatter(LOGGING_FORMAT))
logger.addHandler(logger_streamhandler)
logger.propagate = False


def data_tuple_iterator_reroot(sequence_trees, neg_samples, index_files=[], indices=None, max_depth=100,
                               link_cost_ref=None, link_cost_ref_seealso=-1, transform=True, **unused):
    logger.debug('size of data: %i' % len(sequence_trees))
    logger.debug('size of lexicon: %i' % len(sequence_trees.lexicon))
    assert max_depth > 0, 'can not produce candidates for zero depth trees (single nodes)'

    lexicon = sequence_trees.lexicon
    data_ref = lexicon.get_d(TYPE_REF, data_as_hashes=sequence_trees.data_as_hashes)
    data_ref_seealso = lexicon.get_d(TYPE_REF_SEEALSO, data_as_hashes=sequence_trees.data_as_hashes)
    link_ids = [data_ref, data_ref_seealso]
    #data_identity = lexicon.get_d(vocab_manual[IDENTITY_EMBEDDING], data_as_hashes=sequence_trees.data_as_hashes)
    costs = {}
    if link_cost_ref is not None:
        costs[data_ref] = link_cost_ref
    costs[data_ref_seealso] = link_cost_ref_seealso

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
        tree = sequence_trees.get_tree_dict_rooted(idx=idx, max_depth=max_depth, transform=transform,
                                                   costs=costs, link_types=[data_ref, data_ref_seealso])

        if transform:
            candidate_data = [lexicon.transform_idx(idx=d, root_id_pos=sequence_trees.root_id_pos) for d in candidate_data]

        children = tree[KEY_CHILDREN]
        if len(children) > 0:
            candidate_data = [tree[KEY_HEAD]] + candidate_data
            probs = np.zeros(shape=len(candidate_data), dtype=int)
            probs[0] = 1
            yield [(children, candidate_data), probs]
            count += 1
    logger.info('use %i trees for training' % count)


def get_tree_naive(root, forest, lexicon, concat_mode='sequence', content_offset=2, link_types=[], remove_types=[]):
    idx_root = forest.roots[root]
    idx_context = idx_root + content_offset
    if root < len(forest.roots) - 1:
        idx_next_root = forest.roots[root + 1]
    else:
        idx_next_root = len(forest)

    child_offset = forest.get_children(idx_context)[0]
    idx_content_root = idx_context + child_offset
    data = np.zeros(idx_next_root-idx_content_root+1, dtype=forest.data.dtype)
    data[:-1] = forest.data[idx_content_root:idx_next_root]
    # append 'nif:context'
    data[-1] = forest.data[idx_context]

    # remove entries
    indices_remove = []
    # remove link entries
    for link_type in link_types:
        indices_remove.append(np.where(data == link_type)[0] + 1)
    # remove other entries of specified types
    for remove_type in remove_types:
        indices_remove.append(np.where(data == remove_type)[0])
    indices_remove_np = np.sort(np.concatenate(indices_remove))
    mask = np.ones(data.shape, dtype=bool)
    mask[indices_remove_np] = False
    data = data[mask]

    if concat_mode == 'sequence':
        parents = np.ones(len(data), dtype=DTYPE_OFFSET)
        parents[-1] = 0
    elif concat_mode == 'aggregate':
        parents = np.zeros(len(data), dtype=DTYPE_OFFSET)
        for i in range(len(parents)-1):
            parents[i] = len(parents) - i - 1
    else:
        raise ValueError('unknown concat_mode=%s' % concat_mode)
    return Forest(data=data, parents=parents, lexicon=lexicon)


def data_single_iterator_dbpedianif_context(index_files, sequence_trees, concat_mode='tree',
                                            max_depth=9999, context=0, transform=True, offset_context=2,
                                            link_cost_ref=None, link_cost_ref_seealso=1,
                                            **unused):
    # TODO: test!

    lexicon = sequence_trees.lexicon
    costs = {}
    data_ref = lexicon.get_d(TYPE_REF, data_as_hashes=sequence_trees.data_as_hashes)
    data_ref_seealso = lexicon.get_d(TYPE_REF_SEEALSO, data_as_hashes=sequence_trees.data_as_hashes)

    # do not remove TYPE_ANCHOR (nif:Context), as it is used for aggregation
    remove_types_naive_str = [TYPE_REF_SEEALSO, TYPE_REF, TYPE_ROOT, TYPE_SECTION_SEEALSO, TYPE_PARAGRAPH,
                              TYPE_TITLE, TYPE_SECTION, TYPE_SENTENCE]
    remove_types_naive = [lexicon.get_d(s, data_as_hashes=sequence_trees.data_as_hashes) for s in
                          remove_types_naive_str]

    if link_cost_ref is not None:
        costs[data_ref] = link_cost_ref
    costs[data_ref_seealso] = link_cost_ref_seealso
    n = 0
    for file_name in index_files:
        indices = np.load(file_name)
        for root_id in indices:
            idx_root = sequence_trees.roots[root_id]
            idx_context_root = idx_root + offset_context
            if concat_mode == 'tree':
                tree_context = sequence_trees.get_tree_dict(idx=idx_context_root, max_depth=max_depth,
                                                            context=context, transform=transform,
                                                            costs=costs,
                                                            link_types=[data_ref, data_ref_seealso])
            else:
                f = get_tree_naive(root=root_id, forest=sequence_trees, concat_mode=concat_mode, lexicon=lexicon,
                                   link_types=[data_ref, data_ref_seealso], remove_types=remove_types_naive)
                f.set_children_with_parents()
                tree_context = f.get_tree_dict(max_depth=max_depth, context=context, transform=transform)
            yield tree_context
            n += 1
    logger.info('created %i tree tuples' % n)


def data_single_iterator_dbpedianif_context_tfidf(*args, **kwargs):
    # TODO: test!
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
    vocabulary = {}
    # get id-list versions of articles
    for tree_context in data_single_iterator_dbpedianif_context(*args, concat_mode='aggregate', **kwargs):
        d = tree_context[KEY_CHILDREN].keys()
        for term in d:
            index = vocabulary.setdefault(term, len(vocabulary))
            indices.append(index)
            data.append(1)
        indptr.append(len(indices))
    counts = csr_matrix((data, indices, indptr), dtype=int)
    logger.debug('shape of count matrix: %s' % str(counts.shape))

    # transform to tf-idf
    tf_transformer = TfidfTransformer(use_idf=False).fit(counts)
    tf_idf = tf_transformer.transform(counts)
    return tf_idf


def data_tuple_iterator_dbpedianif(index_files, sequence_trees, concat_mode='tree',
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
    remove_types_naive_str = [TYPE_REF_SEEALSO, TYPE_REF, TYPE_ROOT, TYPE_SECTION_SEEALSO, TYPE_PARAGRAPH,
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
                if concat_mode == 'tree':
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
                if concat_mode == 'tree':
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
                    root_id = sequence_trees.data[idx_root + 1] - len(lexicon)
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