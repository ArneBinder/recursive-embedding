import numpy as np
import logging
import os

import preprocessing
import sequence_node_candidates_pb2
import sequence_node_pb2
import sequence_node_sequence_pb2

import constants


def load(fn):
    logging.debug('load data and parents ...')
    data = np.load('%s.data' % fn)
    parents = np.load('%s.parent' % fn)
    return data, parents


def dump(fn, data=None, parents=None):
    if data is not None:
        logging.debug('dump data ...')
        data.dump('%s.data' % fn)
    if parents is not None:
        logging.debug('dump parents ...')
        parents.dump('%s.parent' % fn)


def exist(fn):
    return os.path.isfile('%s.data' % fn) and os.path.isfile('%s.parent' % fn)


def get_root(parents, idx):
    i = idx
    while parents[i] != 0:
        i += parents[i]
    return i


def children_and_roots(seq_parents):
    # assume, all parents are inside this array!
    # collect children
    # children = [[] for _ in xrange(len(seq_parents))]
    children = {}
    roots = []
    for i, p in enumerate(seq_parents):
        if p == 0:  # is it a root?
            roots.append(i)
            continue
        p_idx = i + p
        chs = children.get(p_idx, [])
        chs.append(-p)
        children[p_idx] = chs
    return children, roots


def children_and_roots_flat(seq_parents):
    # assume, all parents are inside this array!

    root_count = 0
    children_counts = np.zeros(shape=len(seq_parents), dtype=np.int32)
    for i, p in enumerate(seq_parents):
        if p == 0:  # is it a root?
            root_count += 1
            continue
        p_idx = i + p
        children_counts[p_idx] += 1
    children_offsets = np.zeros(shape=len(seq_parents), dtype=np.int32)
    cc = 0
    for i, c in enumerate(children_counts):
        children_offsets[i] = cc
        cc += c

    roots = np.zeros(shape=root_count, dtype=np.int32)
    root_count = 0
    children_indices = np.zeros(shape=cc, dtype=np.int32)
    current_offsets = np.zeros(shape=len(seq_parents), dtype=np.int32)
    for i, p in enumerate(seq_parents):
        if p == 0:  # is it a root?
            roots[root_count] = i
            root_count += 1
            continue
        p_idx = i + p
        children_indices[children_offsets[p_idx] + current_offsets[p_idx]] = -p
        current_offsets[p_idx] += 1
    return (children_counts, children_offsets, children_indices), roots


def get_children(children, idx):
    children_counts, children_offsets, children_indices = children
    c = children_counts[idx]
    o = children_offsets[idx]
    return children_indices[o:o+c]


def get_descendant_indices(children, root):
    leafs = [root]
    if root in children:
        for c in children[root]:
            leafs.extend(get_descendant_indices(children, c + root))
    return leafs


def get_descendant_indices_flat(children, root):
    leafs = [root]
    for c in get_children(children, root):
        leafs.extend(get_descendant_indices(children, c + root))
    return leafs


def calc_depths_and_child_indices(x):  # (out_path, offset, max_depth)):
    (parents, max_depth, child_idx_offset) = x
    # parents = np.load(out_path + '.parent.batch' + str(offset))
    # calc children and roots
    children, roots = children_and_roots(parents)
    # calc depth for every position
    depth = calc_seq_depth(children, roots, parents)

    # depth.dump(out_path + '.depth.batch' + str(offset))

    idx_tuples = []

    for idx in children.keys():
        for (child_offset, child_steps_to_root) in get_all_children_rec(idx, children, max_depth):
            idx_tuples.append((idx + child_idx_offset, child_offset, child_steps_to_root))

    # np.array(idx_tuples).dump(out_path + '.children.batch' + str(offset))

    return depth, np.array(idx_tuples)


def calc_seq_depth(children, roots, seq_parents):
    # ATTENTION: int16 restricts the max sentence count per tree to 32767
    depth = -np.ones(len(seq_parents), dtype=np.int16)
    for root in roots:
        calc_depth(children, seq_parents, depth, root)
    return depth


# unused
def children_indices_and_roots(seq_parents):
    # assume, all parents are inside this array!
    # collect children
    children = {}
    roots = []
    for i, p in enumerate(seq_parents):
        if p == 0:  # is it a root?
            roots.append(i)
            continue
        p_idx = i + p
        chs = children.get(p_idx, [])
        chs.append(i)
        children[p_idx] = chs
    return children, roots


# unused
def get_all_children_rec(idx, children, max_depth, current_depth=0, max_depth_only=False):
    # if idx not in children or max_depth == 0:
    if idx not in children or max_depth == 0:
        return []
    result = []
    for child in children[idx]:
        if max_depth_only is None or current_depth + 1 == max_depth:
            result.append((child, current_depth + 1))
        result.extend(get_all_children_rec(idx + child, children, max_depth - 1, current_depth + 1, max_depth_only))
    return result


# unused
# depth has to be an array pre-initialized with negative int values
def calc_depth_rec(children, depth, idx):
    if idx not in children:
        depth[idx] = 0
    else:
        max_depth = -1
        for child in children[idx]:
            if depth[child] < 0:
                calc_depth_rec(children, depth, child)
            if depth[child] > max_depth:
                max_depth = depth[child]
        depth[idx] = max_depth + 1


def calc_depth(children, parents, depth, start):
    idx = start
    children_idx = list()
    if start in children:
        # if len(children[start]) > 0:
        children_idx.append(0)
        while len(children_idx) > 0:
            current_child_idx = children_idx.pop()
            # go down
            # idx = children[idx][current_child_idx]
            idx += children[idx][current_child_idx]
            # not already calculated?
            if depth[idx] < 0:
                # no children --> depth == 0
                if idx not in children:
                    # if len(children[idx]) == 0:
                    depth[idx] = 0
                else:
                    # calc children
                    children_idx.append(current_child_idx)
                    children_idx.append(0)
                    continue

            parent_depth = depth[idx + parents[idx]]
            # update parent, if this path is longer
            if parent_depth < depth[idx] + 1:
                depth[idx + parents[idx]] = depth[idx] + 1

            # go up
            idx += parents[idx]

            # go only to next child, if it exists
            if current_child_idx + 1 < len(children[idx]):
                children_idx.append(current_child_idx + 1)
            # otherwise, go up again
            else:
                idx += parents[idx]
    else:
        depth[start] = 0


# Build a sequence_tree from a data and a parents sequence.
# All roots are children of a headless node.
def build_sequence_tree(seq_data, children, root, seq_tree=None, max_depth=9999):
    # assume, all parents are inside this array!

    """Recursively build a tree of SequenceNode_s"""

    def build(seq_node, pos, max_depth):
        seq_node.head = seq_data[pos]
        if pos in children and max_depth > 0:
            for child_offset in children[pos]:
                build(seq_node.children.add(), pos + child_offset, max_depth - 1)

    if seq_tree is None:
        seq_tree = sequence_node_pb2.SequenceNode()
    build(seq_tree, root, max_depth)

    return seq_tree


def build_sequence_tree_flat(seq_data, children, root, seq_tree, max_depth=9999):
    # assume, all parents are inside this array!

    """Recursively build a tree of SequenceNode_s"""

    def build(seq_node, pos, max_depth):
        seq_node.head = seq_data[pos]
        if max_depth > 0:
            for child_offset in get_children(children, pos):
                build(seq_node.children.add(), pos + child_offset, max_depth - 1)

    if seq_tree is None:
        seq_tree = sequence_node_pb2.SequenceNode()
    build(seq_tree, root, max_depth)

    return seq_tree


def create_seq_tree_seq(child_tuple, seq_data, children, max_depth, sample_count, all_depths_collected):
    idx = child_tuple[0]
    idx_child = child_tuple[0] + child_tuple[1]
    path_len = child_tuple[2]

    max_candidate_depth = max_depth - path_len
    seq_tree_seq = sequence_node_sequence_pb2.SequenceNodeSequence()
    # seq_tree_seq.idx_correct = 0

    # add correct tree
    build_sequence_tree_with_candidate(seq_data=seq_data, children=children, root=idx, insert_idx=idx_child,
                                       candidate_idx=idx_child, max_depth=max_depth,
                                       max_candidate_depth=max_candidate_depth, seq_tree=seq_tree_seq.trees.add())
    # add samples
    for _ in range(sample_count):
        candidate_idx = np.random.choice(all_depths_collected[max_candidate_depth])
        build_sequence_tree_with_candidate(seq_data=seq_data, children=children, root=idx, insert_idx=idx_child,
                                           candidate_idx=candidate_idx, max_depth=max_depth,
                                           max_candidate_depth=max_candidate_depth, seq_tree=seq_tree_seq.trees.add())
    # pp.pprint(seq_tree_seq)
    # print('')
    return seq_tree_seq


def build_sequence_tree_dict(seq_data, children, root, max_depth=9999):
    # assume, all parents are inside this array!

    """Recursively build a tree of SequenceNode_s"""

    def build(pos, max_depth):
        seq_node = {'head': seq_data[pos], 'children': []}
        if pos in children and max_depth > 0:
            for child_offset in children[pos]:
                seq_node['children'].append(build(pos + child_offset, max_depth - 1))
        return seq_node

    return build(root, max_depth)


def build_sequence_tree_with_candidate(seq_data, children, root, insert_idx, candidate_idx, max_depth,
                                       max_candidate_depth, seq_tree=None):
    """Recursively build a tree of SequenceNode_s"""

    def build(seq_node, pos, max_depth, inserted):
        seq_node.head = seq_data[pos]
        if pos in children and max_depth > 0:
            for child_offset in children[pos]:
                if pos + child_offset == insert_idx and not inserted:
                    build(seq_node.children.add(), candidate_idx, max_candidate_depth, True)
                else:
                    build(seq_node.children.add(), pos + child_offset, max_depth - 1, inserted)

    if seq_tree is None:
        seq_tree = sequence_node_pb2.SequenceNode()

    build(seq_tree, root, max_depth, False)
    return seq_tree


def build_sequence_tree_with_candidates(seq_data, parents, children, root, insert_idx, candidate_indices, seq_tree=None,
                                        max_depth=999):
    # assume, all parents and candidate_indices are inside this array!

    # create path from insert_idx to root
    candidate_path = [insert_idx]
    candidate_parent = insert_idx + parents[insert_idx]
    while candidate_parent != root:
        candidate_path.append(candidate_parent)
        candidate_parent = candidate_parent + parents[candidate_parent]

    """Recursively build a tree of SequenceNode_s"""

    def build(seq_node, pos, max_depth):
        if pos == insert_idx and len(candidate_path) == 0:
            for candidate_idx in [pos] + candidate_indices:
                build_sequence_tree(seq_data, children, candidate_idx, seq_node.candidates.add(), max_depth - 1)
        else:
            seq_node.head = seq_data[pos]
            if pos in children and max_depth > 0:
                candidate_child = candidate_path.pop()
                for child_offset in children[pos]:
                    if candidate_child == pos + child_offset:
                        x = seq_node.children_candidate
                        build(x, pos + child_offset, max_depth - 1)
                    else:
                        build_sequence_tree(seq_data, children, pos + child_offset, seq_node.children.add(),
                                            max_depth - 1)

    if seq_tree is None:
        seq_tree = sequence_node_candidates_pb2.SequenceNodeCandidates()
    build(seq_tree, root, max_depth)

    return seq_tree


def build_sequence_tree_dict_from_parse(seq_graph, max_depth=9999):
    seq_data, seq_parents = seq_graph
    children, roots = children_and_roots(seq_parents)
    return build_sequence_tree_dict(seq_data, children, roots[0], max_depth)


def sequence_node_to_sequence_trees(seq_tree):
    current_data = []
    current_parents = []
    children_roots = []
    for child in seq_tree['children']:
        child_data, child_parents = sequence_node_to_sequence_trees(child)
        current_data.extend(child_data)
        current_parents.extend(child_parents)
        children_roots.append(len(current_data) - 1)
    for child_root in children_roots:
        current_parents[child_root] = len(current_data) - child_root
    # head is the last element
    current_data.append(seq_tree['head'])
    current_parents.append(0)

    return current_data, current_parents


def convert_data(data, converter, lex_size, new_idx_unknown):
    logging.info('convert data ...')
    count_unknown = 0
    for i, d in enumerate(data):
        new_idx = converter[d]
        if 0 <= new_idx < lex_size:
            data[i] = new_idx
        # set to UNKNOWN
        else:
            data[i] = new_idx_unknown  # 0 #new_idx_unknown #mapping[constants.UNKNOWN_EMBEDDING]
            count_unknown += 1
    logging.info('set ' + str(count_unknown) + ' of ' + str(len(data)) + ' data points to UNKNOWN')
    return data


def _compare_tree_dicts(tree1, tree2):
    if tree1['head'] != tree2['head']:
        return tree1['head'] - tree2['head']
    c1 = tree1['children']
    c2 = tree2['children']
    if len(c1) != len(c2):
        return len(c1) - len(c2)
    for i in range(len(c1)):
        _comp = _compare_tree_dicts(c1[i], c2[i])
        if _comp != 0:
            return _comp
    return 0


class Forest(object):
    def __init__(self, filename=None, data=None, parents=None, forest=None, tree_dict=None):
        self._children = None
        self._roots = None
        self._depths = None
        self._depths_collected = None
        self._dicts = {}
        self._filename = filename
        if filename is not None:
            if exist(filename):
                self.set_forest(*load(filename))
            else:
                raise IOError('could not load sequence_trees from "%s"' % filename)
        elif data is not None and parents is not None:
            assert len(data) == len(parents), 'sizes of data and parents arrays differ: len(data)==%i != len(parents)==%i' % (len(data), len(parents))
            self.set_forest(data, parents)
        elif forest is not None:
            if type(forest) == np.ndarray:
                assert forest.shape[0] == 2, 'Wrong shape: %s. trees array has to contain exactly the parents and data arrays: shape=(2, None)' % str(forest.shape)
                self._forest = forest
            else:
                assert len(forest) == 2, 'Wrong shape: %i. Trees array has to contain exactly the parents and data arrays: shape=(2, None)' % str(
                    forest.shape)
                assert len(forest[0]) == len(forest[1]), 'sizes of data and parents arrays differ: len(trees[0])==%i != len(trees[1])==%i' % (len(forest[0]), len(forest[1]))
                self._forest = np.array(forest, dtype=np.int32)
                self._data = forest[0]
                self._parents = forest[1]
                #raise TypeError('trees has to be a numpy.ndarray')
        elif tree_dict is not None:
            _data, _parents = sequence_node_to_sequence_trees(tree_dict)
            self.set_forest(data=np.array(_data), parents=np.array(_parents))
        else:
            raise ValueError(
                'Not enouth arguments to instantiate SequenceTrees object. Please provide a filename or data and parent arrays.')
        #self._sorted = np.zeros(len(self), dtype=bool)

    def set_forest(self, data, parents):
        # return np.concatenate((data, parents)).reshape(2, len(data))
        self._forest = np.empty(shape=(2, len(data)), dtype=data.dtype)
        self._forest[0] = data
        self._forest[1] = parents

    def dump(self, filename):
        dump(fn=filename, data=self.data, parents=self.parents)
        self._filename = filename

    def reload(self):
        assert self._filename is not None, 'no filename set'
        self.set_forest(*load(self._filename))
        self._depths = None
        self._depths_collected = None
        self._children = None
        self._roots = None

    # deprecated
    def write_tuple_idx_data(self, sizes, factor=1, out_path_prefix='', root_scores=None, root_index_converter=None):
        if len(out_path_prefix) > 0:
            out_path_prefix = '.' + out_path_prefix
        if root_index_converter is None:
            root_index_converter = range(sum(sizes) * factor)
        start = 0
        for idx, end in enumerate(np.cumsum(sizes)):
            current_sim_tuples = [(self.roots[(i / factor) * 2],
                                   self.roots[root_index_converter[i] * 2 + 1],
                                   0.0 if root_scores is None else root_scores[i]) for i in range(start * factor, end * factor)]
            logging.info('write sim_tuple_indices to: %s.idx.%i%s ...' % (self._filename, idx, out_path_prefix))
            np.array(current_sim_tuples).dump('%s.idx.%i%s' % (self._filename, idx, out_path_prefix))
            start = end

    def convert_data(self, converter, lex_size, new_idx_unknown):
        convert_data(data=self.data, converter=converter, lex_size=lex_size, new_idx_unknown=new_idx_unknown)
        self._dicts = None

    def indices_to_forest(self, indices):
        return np.array(map(lambda idx: self.forest.T[idx], indices)).T

    def trees(self, root_indices=None):
        if root_indices is None:
            root_indices = self.roots
        for i in root_indices:
            descendant_indices = sorted(get_descendant_indices(self.children, i))
            # new_subtree = zip(*[(data[idx], parents[idx]) for idx in descendant_indices])
            yield self.indices_to_forest(descendant_indices)

    def descendant_indices(self, root):
        return get_descendant_indices(self.children, root)

    def _set_depths(self, indices, current_depth, child_offset=0):
        # depths are counted starting from roots!
        for i in indices:
            idx = i + child_offset
            self._depths[idx] = current_depth
            if idx in self.children:
                self._set_depths(self.children[idx], current_depth+1, idx)

    def trees_equal(self, root1, root2):
        _cmp = _compare_tree_dicts(self.get_tree_dict(root1), self.get_tree_dict(root2))
        return _cmp == 0

    # TODO: check!
    def sample_all(self, sample_count=1, retry_count=10):
        logging.info('create negative samples for all data points ...')
        sampled_sim_tuples = []
        max_depth = np.max(self.depths)
        # sample for every depth only from trees with this depth
        for current_depth, indices in enumerate(self.depths_collected):
            if current_depth == max_depth:
                # add all leafs
                #for idx in indices:
                #    # pad to (sample_count + 1)
                #    sampled_sim_tuples.append([idx] * (sample_count + 1))
                continue

            for idx in indices:
                idx_data = self.data[idx]
                current_sampled_indices = [idx]
                for _ in range(retry_count):
                    candidate_indices = np.random.choice(indices, 100 * sample_count)
                    for candidate_idx in candidate_indices:
                        if idx_data != self.data[candidate_idx] and not self.trees_equal(idx, candidate_idx):
                            current_sampled_indices.append(candidate_idx)
                        if len(current_sampled_indices) > sample_count:
                            break
                    if len(current_sampled_indices) > sample_count:
                        break
                if len(current_sampled_indices) <= sample_count:
                    logging.warning('sampled less candidates (%i) than sample_count (%i) for idx=%i. Skip it.'
                                    % (len(current_sampled_indices) - 1, sample_count, idx))
                else:
                    sampled_sim_tuples.append(current_sampled_indices)
        return sampled_sim_tuples

    def get_tree_dict(self, idx, max_depth=9999):
        """
        Build a _sorted_ (children) dict version of the subtree of this sequence_tree rooted at idx.
        :param idx: root of the subtree
        :param max_depth: stop if this depth is exceeded
        :return: the dict version of the subtree
        """
        if idx in self._dicts:
            return self._dicts[idx]
        seq_node = {'head': self.data[idx], 'children': []}
        if idx in self.children:
            for child_offset in self.children[idx]:
                seq_node['children'].append(self.get_tree_dict(idx + child_offset, max_depth=max_depth - 1))
        seq_node['children'].sort(cmp=_compare_tree_dicts)

        self._dicts[idx] = seq_node
        return self._dicts[idx]

    # COMPATIBILITY: to maintain order for FLAT_LSTM models
    def get_tree_dict_unsorted(self, idx=None, max_depth=9999, with_parent=False):
        """
        Build a _sorted_ (children) dict version of the subtree of this sequence_tree rooted at idx.
        :param idx: root of the subtree
        :param max_depth: stop if this depth is exceeded
        :return: the dict version of the subtree
        """
        if idx is None:
            idx = self.roots[0]
        seq_node = {'head': self.data[idx], 'children': []}
        if idx in self.children and max_depth > 0:
            for child_offset in self.children[idx]:
                seq_node['children'].append(self.get_tree_dict_unsorted(idx=idx + child_offset, max_depth=max_depth - 1,
                                                                        with_parent=with_parent))
        if with_parent and self.parents[idx] != 0 and max_depth > 0:
            seq_node['children'].append(self.get_tree_dict_parent(idx, max_depth-1))
        return seq_node

    def get_tree_dict_rooted(self, idx, max_depth=9999):
        result = self.get_tree_dict_unsorted(idx, max_depth=max_depth)
        current_dict_tree = result
        current_id = idx
        while self.parents[current_id] != 0 and max_depth > 0:
            parent_id = current_id + self.parents[current_id]
            new_parent_child = {'head': self.data[parent_id], 'children': []}
            for c in self.children[parent_id]:
                c_id = parent_id + c
                if c_id != current_id:
                    new_parent_child['children'].append(self.get_tree_dict_unsorted(c_id, max_depth=max_depth-1))
            current_dict_tree['children'].append(new_parent_child)
            current_dict_tree = new_parent_child
            current_id = parent_id
            max_depth -= 1
        return result

    def get_tree_dict_parent(self, idx, max_depth=9999):
        if self.parents[idx] == 0:
            return None
        previous_id = idx
        current_id = idx + self.parents[idx]
        result = {'head': self.data[current_id], 'children': []}
        current_dict_tree = result
        while max_depth > 0:
            # add other children
            for c in self.children[current_id]:
                c_id = current_id + c
                if c_id != previous_id:
                    current_dict_tree['children'].append(self.get_tree_dict_unsorted(c_id, max_depth=max_depth-1))
            # go up
            if self.parents[current_id] != 0:
                previous_id = current_id
                current_id = current_id + self.parents[current_id]
                new_parent_child = {'head': self.data[current_id], 'children': []}
                current_dict_tree['children'].append(new_parent_child)
                current_dict_tree = new_parent_child
                max_depth -= 1
            else:
                break
        return result

    def __str__(self):
        return self._forest.__str__()

    def __len__(self):
        return len(self.data)

    @property
    def data(self):
        return self._forest[0]

    @property
    def parents(self):
        return self._forest[1]

    @property
    def forest(self):
        return self._forest

    @property
    def children(self):
        if self._children is None:
            self._children, self._roots = children_and_roots(self.parents)
        return self._children

    @property
    def roots(self):
        if self._roots is None:
            self._roots = np.where(self.parents == 0)[0]
        return self._roots

    @property
    def depths(self):
        if self._depths is None:
            self._depths = np.zeros(shape=self.data.shape, dtype=np.int32)
            self._set_depths(self.roots, 0)
            self._depths_collected = None
        return self._depths

    @property
    def depths_collected(self):
        if self._depths_collected is None:
            self._depths_collected = []
            m = np.max(self.depths)
            for depth in range(m + 1):
                self._depths_collected.append(np.where(self.depths == depth)[0])
        return self._depths_collected
