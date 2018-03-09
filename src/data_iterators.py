import logging
import copy
import numpy as np
import os
import sys

from constants import TYPE_REF, KEY_HEAD, DTYPE_OFFSET, TYPE_REF_SEEALSO, TYPE_SECTION_SEEALSO, UNKNOWN_EMBEDDING, \
    vocab_manual, KEY_CHILDREN
from sequence_trees import Forest

RECURSION_LIMIT_MIN = 1000
RECURSION_LIMIT_ADD = 100


def data_tuple_iterator_reroot(sequence_trees, neg_samples, indices=None, max_tries=10, max_depth=100, **unused):
    logging.debug('size of data: %i' % len(sequence_trees))
    # take all, if indices is not set
    if indices is None:
        indices = range(len(sequence_trees))
    for idx in indices:
        candidate_ids = []
        try_count = 0
        while len(candidate_ids) < neg_samples and try_count < max_tries:
            idx_cand = np.random.randint(len(sequence_trees), size=1)[0]
            if idx_cand != idx and sequence_trees.data[idx_cand] != sequence_trees.data[idx] and idx_cand not in candidate_ids:
                candidate_ids.append(idx_cand)
            else:
                try_count += 1

        if try_count == max_tries:
            logging.warning('not enough samples: %i, required: %i' % (len(candidate_ids), neg_samples))
            continue
        _trees = [sequence_trees.get_tree_dict_rooted(idx, max_depth=max_depth, transform=True)]
        for idx_cand in candidate_ids:
            cand_tree = copy.deepcopy(_trees[0])
            cand_tree[KEY_HEAD] = sequence_trees.data[idx_cand]
            _trees.append(cand_tree)
        _probs = np.zeros(neg_samples + 1)
        _probs[0] = 1.
        yield [_trees, _probs]


def get_tree_naive(root, forest, lexicon, concat_mode='sequence', content_offset=2):
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


def data_tuple_iterator_dbpedianif_bag_of_seealsos(index_files, sequence_trees, concat_mode='tree',
                                                   max_depth=9999, context=0, transform=True, offset_context=2,
                                                   offset_seealso=3, link_cost_ref=None, link_cost_ref_seealso=1,
                                                   **unused):
    # DEBUG: TODO: remove!
    #n_max = 4

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

    if link_cost_ref is not None:
        costs[data_ref] = link_cost_ref
    costs[data_ref_seealso] = link_cost_ref_seealso
    n = 0
    for file_name in index_files:
        indices = np.load(file_name)
        for root in indices:
            idx_root = sequence_trees.roots[root]
            idx_context_root = idx_root + offset_context
            idx_seealso_root = idx_root + offset_seealso

            children = []
            for c_offset in sequence_trees.get_children(idx_seealso_root):
                seealso_offset = sequence_trees.get_children(idx_seealso_root + c_offset)[0]
                seealso_idx = idx_seealso_root + c_offset + seealso_offset
                seealso_data_id = sequence_trees.data[seealso_idx]
                if seealso_data_id == data_unknown:
                    continue
                seealso_root = sequence_trees.root_id_mapping.get(seealso_data_id, None)
                if seealso_root is None:
                    continue
                if concat_mode == 'tree':
                    idx_root_seealso = sequence_trees.roots[seealso_root] + offset_context
                    tree_seealso = sequence_trees.get_tree_dict(idx=idx_root_seealso, max_depth=max_depth-2,
                                                                context=context, transform=transform,
                                                                costs=costs,
                                                                link_types=[data_ref, data_ref_seealso])
                else:
                    f_seealso = get_tree_naive(seealso_root, sequence_trees, concat_mode=concat_mode, lexicon=lexicon)
                    f_seealso.set_children_with_parents()
                    tree_seealso = f_seealso.get_tree_dict(max_depth=max_depth-2, context=context, transform=transform)
                children.append({KEY_HEAD: data_ref_seealso_transformed, KEY_CHILDREN: [tree_seealso]})
            if len(children) > 0:
                if concat_mode == 'tree':
                    tree_context = sequence_trees.get_tree_dict(idx=idx_context_root, max_depth=max_depth,
                                                                context=context, transform=transform,
                                                                costs=costs,
                                                                link_types=[data_ref, data_ref_seealso])
                else:
                    f = get_tree_naive(root, sequence_trees, concat_mode=concat_mode, lexicon=lexicon)
                    f.set_children_with_parents()
                    tree_context = f.get_tree_dict(max_depth=max_depth, context=context, transform=transform)
                yield [[tree_context, {KEY_HEAD: data_root_seealso_transformed, KEY_CHILDREN: children}],
                       np.ones(shape=2, dtype=int)]
                n += 1

                #if n >= n_max:
                #    break
        #if n >= n_max:
        #    break
    logging.info('created %i tree tuples' % n)


def load_sim_tuple_indices(filename, extensions=None):
    if extensions is None:
        extensions = ['']
    probs = []
    indices = []
    for ext in extensions:
        if not os.path.isfile(filename + ext):
            raise IOError('file not found: %s' % filename + ext)
        logging.debug('load idx file: %s' % filename + ext)
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