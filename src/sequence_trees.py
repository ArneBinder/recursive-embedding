#from __future__ import unicode_literals
import numpy as np
import logging
import os

import pydot
from scipy.sparse import csr_matrix, csc_matrix, dok_matrix

from constants import DTYPE_HASH, DTYPE_COUNT, DTYPE_IDX, DTYPE_OFFSET, DTYPE_DEPTH, KEY_HEAD, KEY_CHILDREN, \
    LOGGING_FORMAT, SEPARATOR, vocab_manual, TYPE_LEXEME, TYPE_REF, TYPE_REF_SEEALSO, TARGET_EMBEDDING, BASE_TYPES, \
    OFFSET_ID, UNKNOWN_EMBEDDING, OFFSET_CONTEXT_ROOT, LINK_TYPES, KEY_HEAD_CONCAT, JSONLD_ID, JSONLD_TYPE, JSONLD_IDX, \
    JSONLD_DATA, JSONLD_VALUE, KEY_HEAD_STRING, KEY_DEPTH
from mytools import numpy_load, numpy_dump, numpy_exists

FE_DATA = 'data'
FE_PARENTS = 'parent'
FE_DATA_HASHES = 'data.hash'
FE_CHILDREN = 'child'
FE_CHILDREN_POS = 'child.pos'
FE_ROOT_ID = 'root.id'
FE_ROOT_POS = 'root.pos'
FE_GRAPH_IN = 'graph.in'
FE_GRAPH_OUT = 'graph.out'

MAX_DEPTH = 9999

logger = logging.getLogger('sequence_trees')
logger.setLevel(logging.DEBUG)


def targets(g, idx):
    return g.indices[g.indptr[idx]:g.indptr[idx + 1]]


def graph_in_from_parents(parents):
    logger.debug('create graph_in from parents')
    mask = (parents != 0)
    indices = (parents + np.arange(len(parents), dtype=DTYPE_IDX))[mask]
    # indptr = np.arange(len(indices) + 1, dtype=DTYPE_IDX)
    indptr = np.concatenate(([0], np.add.accumulate(mask, dtype=DTYPE_IDX)))
    data = np.ones(shape=len(indices), dtype=bool)
    graph = csr_matrix((data, indices, indptr), shape=(len(parents), len(parents)))
    return graph


def graph_out_from_children_dict(children, size, return_dok=False):
    m = dok_matrix((size, size), dtype=bool)
    for _from, _to in children.items():
        # (row, col)
        m[_to, _from] = True
    if return_dok:
        return m
    graph = m.tocsc()
    return graph


def get_path(g, data, idx_start, stop_data):
    assert g.shape[0] == g.shape[1] == len(data), 'shape mismatch: %s vs. %i' % (str(g.shape), len(data))
    d = data[idx_start]
    path = []
    while d != stop_data:
        path.append(idx_start)
        _indices = targets(g, idx_start)
        assert len(_indices) == 1, 'wrong nbr of indices [%i], expected 1.' % len(_indices)
        idx_start = _indices[0]
        d = data[idx_start]
    return path


def get_lca_from_paths(paths, root):
    assert len(paths) > 0, 'no paths available'
    idx = -1
    try:
        while True:
            v = None
            for p in paths:
                if v is not None and p[idx] != v:
                    raise IndexError
                v = p[idx]
            idx -= 1
    except IndexError:
        pass

    idx += 1
    if idx == 0:
        return root
    return paths[0][idx]


def _sequence_node_to_sequence_trees(seq_tree):
    current_data = []
    current_parents = []
    children_roots = []
    for child in seq_tree[KEY_CHILDREN]:
        child_data, child_parents = _sequence_node_to_sequence_trees(child)
        current_data.extend(child_data)
        current_parents.extend(child_parents)
        children_roots.append(len(current_data) - 1)
    for child_root in children_roots:
        current_parents[child_root] = len(current_data) - child_root
    # head is the last element
    current_data.append(seq_tree[KEY_HEAD])
    current_parents.append(0)

    return current_data, current_parents


def _convert_data(data, converter, lex_size, new_idx_unknown):
    logger.info('convert data ...')
    count_unknown = 0
    for i, d in enumerate(data):
        new_idx = converter[d]
        if 0 <= new_idx < lex_size:
            data[i] = new_idx
        # set to UNKNOWN
        else:
            data[i] = new_idx_unknown  # 0 #new_idx_unknown #mapping[constants.UNKNOWN_EMBEDDING]
            count_unknown += 1
    logger.info('set ' + str(count_unknown) + ' of ' + str(len(data)) + ' data points to UNKNOWN')
    return data


def _compare_tree_dicts(tree1, tree2):
    if tree1[KEY_HEAD] != tree2[KEY_HEAD]:
        return tree1[KEY_HEAD] - tree2[KEY_HEAD]
    c1 = tree1[KEY_CHILDREN]
    c2 = tree2[KEY_CHILDREN]
    if len(c1) != len(c2):
        return len(c1) - len(c2)
    for i in range(len(c1)):
        _comp = _compare_tree_dicts(c1[i], c2[i])
        if _comp != 0:
            return _comp
    return 0


def concatenate_graphs(graphs):
    m_type = None
    datas = []
    indices = []
    indptrs = []
    offset_shape = 0
    offset_data = 0
    for g in graphs:
        assert m_type is None or m_type == type(g), 'type of previous sparse matrix does not match current: %s != %s' % (str(m_type), str(type(g)))
        m_type = type(g)
        datas.append(g.data)
        indices.append(g.indices + offset_shape)
        offset_shape += g.shape[0]
        indptrs.append(g.indptr[1:] + offset_data)
        offset_data += len(g.data)
    assert m_type is not None, 'can not concatenate empty list of graphs'
    _indptr = np.concatenate([[0]] + indptrs)
    _indices = np.concatenate(indices)
    _data = np.concatenate(datas)

    if m_type == csr_matrix:
        return csr_matrix((_data, _indices, _indptr), shape=(offset_shape, offset_shape))
    elif m_type == csc_matrix:
        return csc_matrix((_data, _indices, _indptr), shape=(offset_shape, offset_shape))
    else:
        raise NotImplementedError('concatenation for matrix type "%s" not implemented' % m_type)


def concatenate_structures(structure_list):
    assert len(structure_list) > 0, 'can not concatenate empty structure_list'
    if isinstance(structure_list[0], csc_matrix) or isinstance(structure_list[0], csr_matrix):
        return concatenate_graphs(structure_list)
    elif isinstance(structure_list[0], np.ndarray):
        return np.concatenate(structure_list)
    elif isinstance(structure_list[0], list) or isinstance(structure_list[0], np.ndarray):
        parents = []
        for p in structure_list:
            parents.extend(p)
        return parents
    else:
        raise AssertionError('Unknown structure type: %s. Can not concatenate.' % type(structure_list[0]))


def empty_graph_from_graph(graph, size):
    if isinstance(graph, csr_matrix):
        return csr_matrix((size, size), dtype=graph.dtype)
    elif isinstance(graph, csc_matrix):
        return csc_matrix((size, size), dtype=graph.dtype)
    else:
        raise AssertionError('unknown graph type: %s' % str(type(graph)))


def slice_graph(graph, indices):
    return graph[indices].transpose()[indices].transpose()


class Forest(object):
    def __init__(self, filename=None, data=None, parents=None, forest=None, tree_dict=None, lexicon=None,
                 data_as_hashes=False, root_ids=None, root_pos=None, load_root_ids=True, load_root_pos=True,
                 lexicon_roots=None, transformed_indices=False, graph_in=None, graph_out=None, structure=None):
        self.reset_cache_values()
        self._as_hashes = data_as_hashes
        self._lexicon = lexicon
        self._lexicon_roots = lexicon_roots
        self._data = None
        self._root_pos = None
        self._root_data = None
        self._graph_in = None
        self._graph_out = None
        self._nbr_out = None
        self._nbr_in = None
        self._pos_ids = None
        self._pos_to_component_mapping = {}

        if filename is not None:
            self.load(filename=filename, load_root_ids=load_root_ids, load_root_pos=load_root_pos)
        elif (data is not None and (graph_in is not None or graph_out is not None or structure is not None or parents is not None)) \
                or forest is not None:
            self.set_forest(data=data, parents=parents, forest=forest, graph_in=graph_in, graph_out=graph_out, #children=children, children_pos=children_pos,
                            structure=structure, data_as_hashes=data_as_hashes, root_data=root_ids, root_pos=root_pos)
        elif tree_dict is not None:
            _data, _parents = _sequence_node_to_sequence_trees(tree_dict)
            if transformed_indices:
                assert not data_as_hashes, 'can not transform indices back if data_as_hashes'
                for i in range(len(_data)):
                    _data[i], _ = self.lexicon.transform_idx_back(_data[i])
            self.set_forest(data=_data, parents=_parents, data_as_hashes=data_as_hashes, root_data=root_ids)
        else:
            raise ValueError(
                'Not enouth arguments to instantiate Forest object. Please provide a filename or data and (parents or children) arrays.')

    def copy(self, copy_lexicon=True, copy_root_pos=True, copy_graph_in=True, copy_graph_out=True,
             copy_lexicon_roots=True, copy_root_data=True, lexicon_copy_vecs=True, lexicon_copy_ids_fixed=True):
        logger.debug('copy forest ...')
        return Forest(data=self.data.copy(),
                      root_pos=self._root_pos.copy() if copy_root_pos and self._root_pos is not None else None,
                      root_ids=self._root_data.copy() if copy_root_data and self._root_data is not None else None,
                      lexicon=self.lexicon.copy(copy_vecs=lexicon_copy_vecs, copy_ids_fixed=lexicon_copy_ids_fixed) if copy_lexicon else None,
                      lexicon_roots=self.lexicon_roots.copy() if copy_lexicon_roots else None,
                      graph_in=self._graph_in.copy() if (copy_graph_in and self._graph_in is not None) else None,
                      graph_out=self._graph_out.copy() if (copy_graph_out and self._graph_out is not None) else None,
                      )

    def reset_cache_values(self):
        self._depths = None
        self._depths_collected = None
        self._dicts = None
        self._root_id_pos = None
        self._root_data_mapping = None
        self._root_mapping = None
        self._root_id_set = None
        self._nbr_in = None
        self._nbr_out = None
        self._root_strings = None

    def load(self, filename, load_root_ids=True, load_root_pos=True,
             load_graph_in=True, load_graph_out=True):
        logger.debug('load data and parents from %s ...' % filename)
        self._data = numpy_load('%s.%s' % (filename, FE_DATA), assert_exists=False)
        self._as_hashes = False
        if self._data is None:
            self._data = numpy_load('%s.%s' % (filename, FE_DATA_HASHES), assert_exists=False)
            self._as_hashes = True
        if self._data is None:
            raise IOError('data file (%s.%s or %s.%s) not found' % (filename, FE_DATA, filename, FE_DATA_HASHES))
        if load_graph_in:
            self._graph_in = numpy_load('%s.%s' % (filename, FE_GRAPH_IN), assert_exists=False)
            if self._graph_in is None:
                logger.warning('no graph-in data found')
        else:
            self._graph_in = None
        if load_graph_out:
            self._graph_out = numpy_load('%s.%s' % (filename, FE_GRAPH_OUT), assert_exists=False)
            if self._graph_out is None:
                logger.warning('no graph-out data found')
        else:
            self._graph_out = None
        if self._graph_out is None and self._graph_in is None:
            parents = numpy_load('%s.%s' % (filename, FE_PARENTS), assert_exists=False)
            self._graph_in = graph_in_from_parents(parents)

        assert self._graph_in is not None or self._graph_out is not None, \
            'no structure data (graph_in or graph_out) found'
        if load_root_ids:
            self._root_data = numpy_load('%s.%s' % (filename, FE_ROOT_ID), assert_exists=False)
        else:
            self._root_data = None
        if load_root_pos:
            self._root_pos = numpy_load('%s.%s' % (filename, FE_ROOT_POS), assert_exists=False)
        else:
            self._root_pos = None

    def set_forest(self, data=None, parents=None, forest=None, graph_in=None, graph_out=None, structure=None,
                   data_as_hashes=False, root_data=None, root_pos=None):
        self._as_hashes = data_as_hashes
        if self.data_as_hashes:
            data_dtype = DTYPE_HASH
        else:
            data_dtype = DTYPE_IDX
        if forest is not None:
            assert len(forest) == 2, 'Wrong array count: %i. Trees array has to contain exactly the parents and data ' \
                                     'arrays: len=2' % str(len(forest))
            data = forest[0]
            graph_out = forest[1]
        if not isinstance(data, np.ndarray) or not data.dtype == data_dtype:
            data = np.array(data, dtype=data_dtype)
        self._data = data
        self._graph_in = graph_in
        if self._graph_in is not None and not isinstance(self._graph_in, csr_matrix):
            self._graph_in = csr_matrix(self._graph_in)
        self._graph_out = graph_out
        if self._graph_out is not None and not isinstance(self._graph_out, csc_matrix):
            self._graph_out = csc_matrix(self._graph_out)
        if self._graph_in is None and self._graph_out is None:
            if parents is not None:
                assert len(data) == len(parents), \
                    'sizes of data and parents arrays differ: len(data)==%i != len(parents)==%i' % (len(data), len(parents))
                if not isinstance(parents, np.ndarray) or not parents.dtype == DTYPE_OFFSET:
                    parents = np.array(parents, dtype=DTYPE_OFFSET)
                self._graph_in = graph_in_from_parents(parents)
            elif structure is not None:
                if isinstance(structure, csc_matrix):
                    self._graph_out = structure
                elif isinstance(structure, csr_matrix):
                    self._graph_in = structure
                elif isinstance(structure, np.ndarray) and structure.dtype == DTYPE_OFFSET:
                    self._graph_in = graph_in_from_parents(structure)
                elif isinstance(structure, list) and len(data) == len(structure):
                    self._graph_in = graph_in_from_parents(np.array(structure, dtype=DTYPE_OFFSET))
                else:
                    raise AssertionError('Unknown structure for graph. type: %s' % type(structure))
        assert self._graph_in is not None or self._graph_out is not None, \
            'either graph_in (%s) or graph_out (%s) have to be set (True iff array is set, in brackets)' \
            % (str(self._graph_in is not None), str(self._graph_out is not None))

        if root_data is not None:
            if not isinstance(root_data, np.ndarray) or not root_data.dtype == data_dtype:
                root_data = np.array(root_data, dtype=data_dtype)
            else:
                assert root_data.dtype == data_dtype, 'root_ids has wrong dtype (%s), expected: %s' \
                                                      % (root_data.dtype, data_dtype)
        self._root_data = root_data
        self._root_pos = root_pos

    def set_lexicon(self, lexicon):
        self._lexicon = lexicon

    def set_lexicon_roots(self, lexicon_roots):
        self._lexicon_roots = lexicon_roots
        self._root_strings = None

    def split_lexicon_to_lexicon_and_lexicon_roots(self):
        assert self.data_as_hashes, 'can not split lexicon to lexicon and lexicon_roots if data_as_hashes == False'
        mask_no_ids = np.ones_like(self.data, dtype=bool)
        mask_no_ids[self.pos_ids] = False
        hashes = self.data[mask_no_ids]
        u, c = np.unique(hashes, return_counts=True)
        hashes_ids = self.data[self.pos_ids]
        lex = self.lexicon.create_subset_with_hashes(u, add_vocab_manual=True)
        lex_ids = self.lexicon.create_subset_with_hashes(hashes_ids)
        self.set_lexicon(lex)
        self.set_lexicon_roots(lex_ids)

    def set_root_data_by_offset(self):
        self._root_data = self.data[self.roots + OFFSET_ID]
        self._root_strings = None

    def dump(self, filename, save_root_ids=True, save_root_pos=True, save_graph_in=True, save_graph_out=True):
        logger.debug('dump data ...')
        if self.data_as_hashes:
            numpy_dump('%s.%s' % (filename, FE_DATA_HASHES), self.data)
        else:
            numpy_dump('%s.%s' % (filename, FE_DATA), self.data)
        if save_root_ids and self._root_data is not None:
            numpy_dump('%s.%s' % (filename, FE_ROOT_ID), self._root_data)
        if save_root_pos and self._root_pos is not None:
            numpy_dump('%s.%s' % (filename, FE_ROOT_POS), self._root_pos)
        logger.debug('dump structure ...')
        if save_graph_in and self._graph_in is not None:
            numpy_dump('%s.%s' % (filename, FE_GRAPH_IN), self._graph_in)
        if save_graph_out and self._graph_out is not None:
            numpy_dump('%s.%s' % (filename, FE_GRAPH_OUT), self._graph_out)

    @staticmethod
    def exist(filename):
        data_exist = numpy_exists('%s.%s' % (filename, FE_DATA)) \
                     or numpy_exists('%s.%s' % (filename, FE_DATA_HASHES))
        structure_exist = numpy_exists('%s.%s' % (filename, FE_PARENTS)) \
                          or numpy_exists('%s.%s' % (filename, FE_GRAPH_IN)) \
                          or numpy_exists('%s.%s' % (filename, FE_GRAPH_OUT))

        return data_exist and structure_exist

    def hashes_to_indices(self):
        assert self.lexicon is not None, 'no lexicon available'
        assert self.data_as_hashes, 'data consists already of indices'
        if self.lexicon_roots is not None:
            indices_root_ids = self.roots + OFFSET_ID
            link_types = self.link_types
            if len(link_types) > 0:
                indices_link_types = np.concatenate([np.where(self.data == lt)[0] for lt in link_types])
                indices_links = self.get_child_positions_batched(indices=indices_link_types)
            else:
                indices_links = []

            mask_no_ids = np.ones(len(self.data), dtype=bool)
            mask_no_ids[indices_root_ids] = False
            mask_no_ids[indices_links] = False

            # mark ids as negative and shift by 1 (no double zero!)
            self._data[indices_root_ids] = - (1 + self.lexicon_roots.convert_data_hashes_to_indices(
                self.data[indices_root_ids], convert_dtype=False))
            if len(indices_links) > 0:
                self._data[indices_links] = - (1 + self.lexicon_roots.convert_data_hashes_to_indices(
                    self.data[indices_links], convert_dtype=False))
            # all other tokens, etc.
            self._data[mask_no_ids] = self.lexicon.convert_data_hashes_to_indices(
                self.data[mask_no_ids], convert_dtype=False)
            self._data = self.data.astype(DTYPE_IDX)

            self.set_root_data_by_offset()
        else:
            self._data = self.lexicon.convert_data_hashes_to_indices(self.data)
        self._as_hashes = False

    def convert_data(self, converter, new_idx_unknown):
        assert self.lexicon is not None, 'lexicon is not set'
        _convert_data(data=self.data, converter=converter, lex_size=len(self.lexicon), new_idx_unknown=new_idx_unknown)
        self._dicts = None

    def get_descendant_indices(self, root, seen=None, show_links=True):
        if seen is None:
            seen = set()
        leafs = [root]
        seen.add(root)
        # do not follow links
        if self.nbr_out[root] > 0 and (show_links or (self.data[root] not in self.link_types)):
            for c in targets(self.graph_out, root):
                if c not in seen:
                    leafs.extend(self.get_descendant_indices(c, seen=seen, show_links=show_links))
        # because of graph structure indices could double
        return np.unique(leafs)

    def get_slice(self, root=None, indices=None, root_exclude=None):
        if indices is None:
            idx_start = self.roots[root]
            idx_end = self.pos_end(component_idx=root)
            indices = np.arange(idx_start, idx_end)
        if root_exclude is not None:
            assert root != root_exclude, 'root==root_exclude (%i)' % root
            assert root_exclude in indices, 'root_exclude=%i is not in indices' % root_exclude
            indices_exclude = np.sort(self.get_descendant_indices(root_exclude))
            mask = np.isin(indices, indices_exclude, invert=True)
            indices = indices[mask]
        return Forest(data=self.data[indices].copy(), graph_out=slice_graph(self.graph_out, indices).copy(),
                      lexicon=self.lexicon,
                      lexicon_roots=self.lexicon_roots,
                      data_as_hashes=self.data_as_hashes)

    def _set_depths(self, indices, current_depth):
        self._depths[indices] = current_depth
        child_positions = self.get_child_positions_batched(indices)
        if len(child_positions) > 0:
            self._set_depths(child_positions, current_depth + 1)

    # not used
    def trees_equal(self, root1, root2):
        _cmp = _compare_tree_dicts(self.get_tree_dict_cached(root1), self.get_tree_dict_cached(root2))
        return _cmp == 0

    # not used
    def get_tree_dict_cached(self, idx):
        """
        DOES NOT WORK WITH max_depth!
        Build a _sorted_ (children) dict version of the subtree of this sequence_tree rooted at idx.
        Does _not_ maintain order of data elements.
        :param idx: root of the subtree
        :param max_depth: stop if this depth is exceeded    DOES NOT WORK WITH CACHING!
        :return: the dict version of the subtree
        """
        if idx in self._dicts:
            return self._dicts[idx]
        seq_node = {KEY_HEAD: self.data[idx], KEY_CHILDREN: []}
        if self.nbr_out[idx] > 0:
            for child_idx in targets(self.graph_out, idx):
                seq_node[KEY_CHILDREN].append(self.get_tree_dict_cached(child_idx))
        seq_node[KEY_CHILDREN].sort(cmp=_compare_tree_dicts)

        self._dicts[idx] = seq_node
        return self._dicts[idx]

    def get_tree_dict(self, idx, visited=None, max_depth=MAX_DEPTH, context=0, transform=False, costs={}, link_types=[],
                      link_content_offset=OFFSET_CONTEXT_ROOT, data_blank=None, keep_prob_blank=1.0, keep_prob_node=1.0,
                      revert=False, blank_types=(), go_back=False, add_heads_types=(), add_heads_dummies=(),
                      return_strings=False, return_depth=False):
        """
        Build a dict version of the subtree of this sequence_tree rooted at idx.
        Maintains order of data elements.

        :param link_types: TODO
        :param idx: root of the subtree
        :param max_depth: stop if this depth is exceeded
        :param context depth of context tree (walk up parents) to add to all nodes
        :param transform transform data ids (does not work for hashes) to ids regarding fixed / variable embeddings;
                         values indicating fixed embeddings are negative
        :param costs contains mappings from data ids or hashes to costs that are subtracted, if following a link
                          with this id or hash. NOTE: Link following requires max_depth != MAX_DEPTH.
        :param link_content_offset If a link is followed, this offset is added to the root position of the target, e.g.
                                   link_content_offset=2 means, that the subtree at target_root_position+2 is inserted
                                   beneath the link
        :return: the dict version of the subtree
        """

        visited = set(visited or [])
        visited.add(idx)
        data_head = self.data[idx]
        assert context == 0 or transform, 'context > 0, but transform is disabled.'

        cost = costs.get(data_head, 1)

        # blank node dropout
        # do not blank heads and links
        if len(visited) > 1 and \
                ((keep_prob_blank < 1.0 and data_head not in link_types and keep_prob_blank < np.random.uniform())):
            data_head = data_blank

        seq_node = {KEY_CHILDREN: []}

        if return_strings:
            seq_node = {KEY_HEAD_STRING: self.lexicon.get_s(d=data_head, data_as_hashes=self.data_as_hashes),
                        KEY_CHILDREN: []}
        if transform:
            data_head = self.lexicon.transform_idx(data_head, revert=revert)

        seq_node[KEY_HEAD] = data_head
        current_targets = targets(self.graph_out, idx)
        if len(add_heads_dummies) > 0:
            current_additional_heads = [self.data[ad] for ad in current_targets if self.data[ad] in add_heads_types]
            if len(current_additional_heads) > 0:
                # TODO: transform current_additional_heads??? (add_heads_dummies are already transformed)
                #seq_node[KEY_HEAD_CONCAT] = current_additional_heads
                if transform:
                    current_additional_heads = self.lexicon.transform_indices(current_additional_heads)
                seq_node[KEY_HEAD_CONCAT] = current_additional_heads + add_heads_dummies[len(current_additional_heads):]
            else:
                seq_node[KEY_HEAD_CONCAT] = add_heads_dummies
            assert len(seq_node[KEY_HEAD_CONCAT]) == len(add_heads_dummies), \
                'nbr of current_additional_heads [%i] does not match expected additional heads [%i]' \
                % (len(seq_node[KEY_HEAD_CONCAT]), len(add_heads_dummies))
        else:
            current_additional_heads = ()
        # ATTENTION: allows cost of 0!
        if self.nbr_out[idx] > 0 and 0 <= cost <= max_depth:
            for target in current_targets:
                # if the child is a link ...
                if data_head in link_types:
                    # ... and the target tree exists: jump to target root, ...
                    #if self.data[target] in self.root_id_pos:
                    target_root_pos = self.root_id_pos.get(self.data[target], None)
                    if target_root_pos is not None:
                        target = target_root_pos + link_content_offset
                    else:
                        # ... otherwise add the TARGET element NO: if costs prohibit link following, this is never reached. otherwise this should not be used
                        #d_target = self.lexicon.get_d(s=vocab_manual[TARGET_EMBEDDING], data_as_hashes=False)
                        #seq_node[KEY_CHILDREN].append({KEY_HEAD: self.lexicon.transform_idx(d_target) if transform else d_target, KEY_CHILDREN: []})
                        continue

                if visited is not None and target in visited:
                    continue

                data_target = self.data[target]
                # full node dropout
                if (keep_prob_node < 1.0 and keep_prob_node < np.random.uniform()) or data_target in blank_types:
                    continue
                # skip children that are already added to heads
                if data_target in current_additional_heads:
                    continue

                seq_node[KEY_CHILDREN].append(self.get_tree_dict(idx=target,
                                                                 visited=visited,
                                                                 max_depth=max_depth - cost,
                                                                 context=context,
                                                                 transform=transform or context > 0,
                                                                 costs=costs,
                                                                 link_types=link_types,
                                                                 data_blank=data_blank,
                                                                 keep_prob_blank=keep_prob_blank,
                                                                 keep_prob_node=keep_prob_node,
                                                                 blank_types=blank_types,
                                                                 add_heads_types=add_heads_types,
                                                                 add_heads_dummies=add_heads_dummies,
                                                                 go_back=go_back,
                                                                 return_strings=return_strings,
                                                                 return_depth=return_depth))
        if go_back:
            if self.nbr_in[idx] > 0 and 0 <= cost <= max_depth:
                for target in targets(self.graph_in, idx):
                    if visited is not None and target in visited:
                        continue
                    data_target = self.data[target]
                    # full node dropout
                    if (keep_prob_node < 1.0 and keep_prob_node < np.random.uniform()) or data_target in blank_types:
                        continue
                    # TODO: handle link_types (see above)?
                    seq_node[KEY_CHILDREN].append(self.get_tree_dict(idx=target,
                                                                     revert=True,
                                                                     visited=visited,
                                                                     max_depth=max_depth - cost,
                                                                     context=context,
                                                                     transform=transform or context > 0,
                                                                     costs=costs,
                                                                     link_types=link_types,
                                                                     data_blank=data_blank,
                                                                     keep_prob_blank=keep_prob_blank,
                                                                     keep_prob_node=keep_prob_node,
                                                                     blank_types=blank_types,
                                                                     add_heads_types=add_heads_types,
                                                                     add_heads_dummies=add_heads_dummies,
                                                                     go_back=go_back,
                                                                     return_strings=return_strings,
                                                                     return_depth=return_depth))
        if context > 0:
            for target_back in targets(self.graph_in, idx):
                if visited is not None and target_back in visited:
                    continue
                # full node dropout
                if keep_prob_node < 1.0 and keep_prob_node < np.random.uniform():
                    continue
                seq_node[KEY_CHILDREN].append(self.get_tree_dict(idx=target_back,
                                                                 revert=True,
                                                                 visited=visited,
                                                                 max_depth=context - cost,
                                                                 context=context,
                                                                 transform=transform or context > 0,
                                                                 costs=costs,
                                                                 link_types=link_types,
                                                                 data_blank=data_blank,
                                                                 keep_prob_blank=keep_prob_blank,
                                                                 keep_prob_node=keep_prob_node,
                                                                 add_heads_types=add_heads_types,
                                                                 add_heads_dummies=add_heads_dummies,
                                                                 blank_types=blank_types,
                                                                 return_strings=return_strings))

        if return_depth:
            max_child_depth = max([c[KEY_DEPTH] for c in seq_node[KEY_CHILDREN]] + [0])
            seq_node[KEY_DEPTH] = max_child_depth + 1
        return seq_node


    def get_data_span_cleaned(self, idx_start, idx_end, link_types, remove_types=(), transform=False):
        data = self.data[idx_start:idx_end]
        ## remove entries
        indices_remove = []
        ## remove link entries
        for link_type in link_types:
            indices_remove.append(np.where(data == link_type)[0] + 1)
        ## remove other entries of specified types
        mask = np.ones(data.shape, dtype=bool)
        if len(remove_types) > 0:
            for remove_type in remove_types:
                indices_remove.append(np.where(data == remove_type)[0])

            indices_remove_np = np.sort(np.concatenate(indices_remove))
            mask[indices_remove_np] = False
        if transform:
            return np.array(self.lexicon.transform_indices(indices=data[mask]), dtype=DTYPE_IDX)
        else:
            return data[mask]

    def get_tree_dict_string(self, idx, stop_types=(), index_types=(), data_types=()):
        # TODO: use constants
        res = {}
        if idx - OFFSET_ID in self.roots:
            res[JSONLD_ID] = self.get_text_plain_idx(idx)
            return res
        data_str = self.lexicon.get_s(d=self.data[idx], data_as_hashes=self.data_as_hashes)
        target_indices = targets(self.graph_out, idx)
        # ATTENTION: may cause unintended result if uri contains "=" (see nif:Context instances)
        type_parts = data_str.split(u'=')
        res[JSONLD_TYPE] = type_parts[0]
        if res[JSONLD_TYPE] in index_types:
            res[JSONLD_IDX] = int(idx)
        if res[JSONLD_TYPE] in data_types:
            res[JSONLD_DATA] = int(self.data[idx])
        if res[JSONLD_TYPE] in stop_types:
            return res
        if len(type_parts) > 1:
            res[JSONLD_VALUE] = u'='.join(type_parts[1:])
        targed_elements = [self.get_tree_dict_string(idx=idx_target, stop_types=stop_types, index_types=index_types,
                                                     data_types=data_types) for idx_target in target_indices]
        target_dict = {}
        for t_elem in targed_elements:
            if JSONLD_TYPE in t_elem:
                _t = t_elem[JSONLD_TYPE]
                del t_elem[JSONLD_TYPE]
                l = target_dict.setdefault(_t, [])
                if len(t_elem) > 0:
                    l.append(t_elem)
            elif JSONLD_ID in t_elem:
                res[JSONLD_ID] = t_elem[JSONLD_ID]

        res.update(target_dict)

        return res

    @staticmethod
    def meta_matches(a, b, operation):
        assert a.lexicon == b.lexicon or b.lexicon is None, 'lexica do not match, can not %s.' % operation
        assert a.data.dtype == b.data.dtype, 'dtype of data arrays do not match, can not %s.' % operation
        assert a.data_as_hashes == b.data_as_hashes, 'data_as_hash do not match, can not %s.' % operation
        assert len(a.pos_to_component_mapping) == 0, '%s not implemented for pos_to_start_mapping containting entries' % operation
        assert len(b.pos_to_component_mapping) == 0, '%s not implemented for pos_to_start_mapping containting entries' % operation
        if a._graph_in is not None:
            assert b._graph_in is not None, 'if graph_in array of first forest is set, graph_in of second forest ' \
                                        'have to be set, too, can not %s.' % operation
        if a._graph_out is not None:
            assert b._graph_out is not None, 'if graph_out array of first forest is set, graph_out of second forest ' \
                                        'have to be set, too, can not %s.' % operation
        if a._root_data is not None:
            assert b._root_data is not None, 'root_ids of first forest is set, but not of the second, can not %s.' \
                                            % operation
        if a._root_pos is not None:
            assert b._root_pos is not None, 'root positions of first forest are set, but not of the second, can not %s.' \
                                            % operation
        if a._lexicon_roots is not None:
            assert b._lexicon_roots is not None, 'lexicon_roots of first forest are set, but not of the second, can not %s.' \
                                            % operation

    # TODO(graph): remove! (use Forest.concatenate instead)
    def extend(self, others):
        raise NotImplementedError('forest.extend is deprecated. use concat.')
        if type(others) != list:
            others = [others]
        for other in others:
            Forest.meta_matches(self, other, 'extend')
        self.reset_cache_values()
        if self._children is not None and self._children_pos is not None:
            new_children = np.concatenate([f._children for f in [self] + others])
            new_children_pos_list = []
            offset = 0
            for forest in [self] + others:
                new_children_pos_list.append(forest._children_pos + offset)
                offset += len(forest._children)
            new_children_pos = np.concatenate(new_children_pos_list)
        else:
            new_children = None
            new_children_pos = None
        if self._root_data is not None:
            new_root_ids = np.concatenate([f._root_data for f in [self] + others])
        else:
            new_root_ids = None
        if self._parents is not None:
            new_parents = np.concatenate([f._parents for f in [self] + others])
        else:
            new_parents = None
        if self._root_pos is not None:
            new_root_pos_list = []
            offset = 0
            for forest in [self] + others:
                new_root_pos_list.append(forest._root_pos + offset)
                offset += len(forest)
            new_root_pos = np.concatenate(new_root_pos_list)
        else:
            new_root_pos = None

        # add lexicon_roots
        # TODO: Test this!
        if self._lexicon_roots is not None:
            for f in others:
                for s in f._lexion_roots.strings:
                    assert s not in self._lexicon_roots.strings, 'root string (%s) already in lexicon_roots' % s
                    self._lexicon_roots.strings.add(s)
        self._lexicon_roots.clear_cached_values()

        self.set_forest(data=np.concatenate([f.data for f in [self] + others]),
                        parents=new_parents,
                        children=new_children,
                        children_pos=new_children_pos,
                        root_data=new_root_ids,
                        root_pos=new_root_pos)

    @staticmethod
    def concatenate(forests):
        assert len(forests) > 0, 'can not concatenate empty list of forests'
        for f in forests[1:]:
            Forest.meta_matches(forests[0], f, 'concatenate')
        if forests[0]._graph_in is not None:
            new_graph_in = concatenate_graphs((f._graph_in for f in forests))
        else:
            new_graph_in = None
        if forests[0]._graph_out is not None:
            new_graph_out = concatenate_graphs((f._graph_out for f in forests))
        else:
            new_graph_out = None

        # should _not_ work if lexicon_roots are merged!
        #if forests[0]._root_data is not None:
        #    new_root_data = np.concatenate([f._root_data for f in forests])
        #else:
        #    new_root_data = None
        if forests[0]._root_pos is not None:
            new_root_pos_list = []
            offset = 0
            for forest in forests:
                new_root_pos_list.append(forest._root_pos + offset)
                offset += len(forest)
            new_root_pos = np.concatenate(new_root_pos_list)
        else:
            new_root_pos = None
        # add lexicon_roots.
        # TODO: Test this!
        if forests[0]._lexicon_roots is not None:
            logger.warning('Concatenate Forests with lexicon_roots reuses and appends to the lexicon_roots of the '
                           'first Forest.')
            for f in forests[1:]:
                for s in f._lexicon_roots.strings:
                    assert s not in forests[0]._lexicon_roots.strings, 'root string (%s) already in lexicon_roots' % s
                    forests[0]._lexicon_roots.strings.add(s)
            forests[0]._lexicon_roots.clear_cached_values()

        return Forest(data=np.concatenate([f.data for f in forests]),
                      data_as_hashes=forests[0].data_as_hashes,
                      #lexicon=forests[0].lexicon,
                      lexicon_roots=forests[0]._lexicon_roots,
                      #root_ids=new_root_data,
                      root_pos=new_root_pos,
                      graph_in=new_graph_in,
                      graph_out=new_graph_out)

    def visualize(self, filename, start=0, end=None, transformed=False, token_list=None, scores=None,
                  color_by_rank=False, edge_source_blacklist=()):
        if end is None:
            end = len(self)
        if scores is not None:
            assert len(scores) == end - start, 'number of scores (%i) does not match sequence length (%i)' \
                                               % (len(scores), end - start)
            # calculate ranks
            indices_sorted = scores.argsort()[::-1]
            ranks = np.empty_like(indices_sorted)
            ranks[indices_sorted] = np.arange(len(scores))
            # merge identical ranks
            for i, idx in enumerate(indices_sorted):
                if i > 0 and scores[indices_sorted[i-1]] == scores[idx]:
                    ranks[idx] = ranks[indices_sorted[i-1]]

        graph = pydot.Dot(graph_type='digraph', rankdir='LR', bgcolor='transparent')
        if token_list is None:
            token_list = self.get_text_plain(start=start, end=end, transformed=transformed)
        if len(token_list) > 0:
            nodes = []
            for i, l in enumerate(token_list):
                fixed = l.endswith('-FIX')
                if fixed:
                    l = l[:-len('-FIX')]

                reverted = l.endswith('-REV')
                if reverted:
                    l = l[:-len('-REV')]

                penwidth = 1
                # if scores are given ...
                if scores is not None:
                    l += '\n%f\n#%i' % (scores[i], ranks[i] + 1)
                    if color_by_rank:
                        if np.max(ranks) == 0:
                            c = 0.
                        else:
                            c = 1. - ranks[i] / float(np.max(ranks))
                    else:
                        c = scores[i]
                    # score 0 -> 255, 0, 0
                    # score 1 -> 255, 255, 255
                    color = '#{:02x}{:02x}{:02x}'.format(255, int(255 * c), int(255 * c))
                    if scores[i] == np.max(scores):
                        penwidth = 3

                else:
                    if fixed:
                        color = "dodgerblue"
                        #color = '#{:02x}{:02x}{:02x}'.format(255 - int(255 * scores[i]), 255 - int(255 * scores[i]), 255)
                    else:
                        color = "limegreen"
                        #color = '#{:02x}{:02x}{:02x}'.format(255 - int(255 * scores[i]), 255, 255 - int(255 * scores[i]))

                if reverted:
                    nodes.append(pydot.Node(i, label=l, style="filled",
                                            fillcolor='%s;0.5:white;0.5' % color,
                                            gradientangle=135,
                                            penwidth=penwidth))
                else:
                    nodes.append(pydot.Node(i, label=l, style="filled", fillcolor=color, penwidth=penwidth))

            for node in nodes:
                graph.add_node(node)

            # add invisible edges for horizontal alignment
            last_node = nodes[0]
            for node in nodes[1:]:
                graph.add_edge(pydot.Edge(last_node, node, weight=100, style='invis'))
                last_node = node

            for i in range(len(nodes)):
                target_indices = targets(self.graph_in, i)
                for target_index in target_indices:
                    if target_index < 0 or target_index >= len(nodes):
                        target_index = i
                    if token_list[target_index][1:-1] not in edge_source_blacklist:
                        graph.add_edge(pydot.Edge(nodes[i], nodes[target_index], dir='back'))

        logger.debug('graph created. write to file: %s ...' % filename)
        # print(graph.to_string())
        graph.write_svg(filename, encoding='utf-8')

    @staticmethod
    def filter_and_shorten_label(l, blacklist=[], do_filter=True):
        if do_filter:
            for b in blacklist:
                if l.startswith(b):
                    return None

            for embedding_prefix in BASE_TYPES:
                if l.startswith(embedding_prefix + SEPARATOR) and len(l) > len(embedding_prefix + SEPARATOR):
                    return l[len(embedding_prefix + SEPARATOR):]
                elif l.startswith(embedding_prefix) and len(l) > len(embedding_prefix):
                    return l[len(embedding_prefix):]
            return l
        else:
            return l

    def get_text_plain_idx(self, idx):
        d = self.data[idx]
        if self.lexicon_roots is not None \
                and (d < 0 or (self.data_as_hashes and self.lexicon_roots.get_s(d, data_as_hashes=True) != vocab_manual[
            UNKNOWN_EMBEDDING])):
            s = self.lexicon_roots.get_s(d if self.data_as_hashes else -d - 1, self.data_as_hashes)
        else:
            s = self.lexicon.get_s(d, self.data_as_hashes)
        return s

    def get_text_plain(self, blacklist=None, start=0, end=None, transformed=False):
        assert self.lexicon is not None, 'lexicon is not set'
        if end is None:
            end = len(self)
        result = []
        if len(self.data) > 0:
            for d in self.data[start:end]:
                reverted = False

                if transformed:
                    d, reverted = self.lexicon.transform_idx_back(d)
                    s = self.lexicon.get_s(d, self.data_as_hashes)
                else:
                    if self.lexicon_roots is not None \
                            and (d < 0 or (self.data_as_hashes and self.lexicon_roots.get_s(d, data_as_hashes=True) != vocab_manual[UNKNOWN_EMBEDDING])):
                        s = 'ID:%s' % self.lexicon_roots.get_s(d if self.data_as_hashes else -d - 1, self.data_as_hashes)
                    else:
                        s = self.lexicon.get_s(d, self.data_as_hashes)

                l = "'%s'" % Forest.filter_and_shorten_label(s, blacklist, do_filter=blacklist is not None)
                if l is not None:
                    if reverted:
                        l += '-REV'
                    if self.lexicon.is_fixed(d):
                        l += '-FIX'
                    result.append(l)
        return result

    # deprecated, use self.nbr_out[idx] > 0 directly
    def has_children(self, idx):
        #logger.warning('self.has_children(idx) is deprecated, use self.nbr_out[idx] > 0')
        return self.nbr_out[idx] > 0

    # deprecated, use targets(self.graph_out, idx) directly
    def get_children(self, idx):
        #logger.warning('self.get_children(idx) is deprecated, use self.targets(self.graph_out, idx)')
        return targets(self.graph_out, idx)

    def get_child_positions_batched(self, indices):
        if len(indices) == 0:
            return []
        targets_list = (targets(self.graph_out, idx) for idx in indices)
        positions = np.concatenate(targets_list)
        return positions

    # deprecated, use self.nbr_out[indices] directly
    def get_children_counts(self, indices):
        #logger.warning('self.get_children_counts is deprecated, use self.nbr_out[indices]')
        return self.nbr_out[indices]

    def pos_end(self, idx=None, component_idx=None):
        if component_idx is None:
            component_idx = self.pos_to_component_mapping[idx]
        next_component_idx = component_idx + 1
        if next_component_idx == len(self.roots):
            return len(self)
        else:
            return self.pos_start[next_component_idx]

    def __str__(self):
        return self._data.__str__()

    def __len__(self):
        return len(self.data)

    @property
    def data(self):
        return self._data

    @property
    def forest(self):
        return self.data, self.graph_out

    @property
    def roots(self):
        """
        Holds the indices (regarding the data sequence) of all roots.
        :return: a plain numpy array containing all root positions
        """
        if self._root_pos is None:
            self._root_pos = np.where(self.nbr_in == 0)[0].astype(DTYPE_IDX)
        return self._root_pos

    @property
    def root_data(self):
        """
        Holds data for all root ids. As hashes, if self.data_as_hashes, or as NEGATIVE VALUES SHIFTED BY ONE to avoid
        collision with sequence data. Can be used with lexicon_roots to get the string representation of the root id.
        :return: a plain numpy array holding root data entries referencing elements in lexicon_roots
        """
        if self._root_data is None:
            self.set_root_data_by_offset()
        return self._root_data

    @property
    def root_strings(self):
        assert self.lexicon_roots is not None, 'lexicon_roots not available'
        if self._root_strings is None:
            _root_data = self.root_data
            if max(_root_data) < 0:
                self._root_strings = np.array([self.lexicon_roots.get_s(idx, data_as_hashes=False) for idx in -self.root_data - 1])
            else:
                self._root_strings = np.array([self.lexicon_roots.get_s(h, data_as_hashes=True) for h in self.root_data])

        return self._root_strings

    @property
    def depths(self):
        if self._depths is None:
            logger.debug('forest: calculate depths')
            self._depths = np.zeros(shape=self.data.shape, dtype=DTYPE_DEPTH)
            self._set_depths(self.roots, 0)
            self._depths_collected = None
        return self._depths

    @property
    def depths_collected(self):
        if self._depths_collected is None:
            logger.debug('forest: create depths_collected')
            self._depths_collected = []
            m = np.max(self.depths)
            for depth in range(m + 1):
                self._depths_collected.append(np.where(self.depths == depth)[0])
        return self._depths_collected

    @property
    def lexicon(self):
        return self._lexicon

    @property
    def lexicon_roots(self):
        """
        Get the lexicon responsible for root data.
        ATTENTION: The order of its string entries does not have to reflect the order of the roots!
        :return: the root data lexicon
        """
        return self._lexicon_roots

    @property
    def data_as_hashes(self):
        return self._as_hashes

    @property
    def root_id_pos(self):
        """
        Maps from root_data to the position of the associated root in the data sequence.
        :return: the mapping
        """
        if self._root_id_pos is None:
            #assert self._root_data is not None, 'root_ids not set'
            #if self._root_data is not None:
            logger.debug('forest: create root_id_pos from root_ids (%i)' % len(self.root_data))
            if len(self.roots) != len(self.root_data):
                logger.warning('number of roots (%d) does not match number of root_ids (%d). Set root_id_pos to {}.'
                               % (len(self.roots), len(self.root_data)))
                self._root_id_pos = {}
            else:
                self._root_id_pos = {v: self.roots[i] for i, v in enumerate(self.root_data)}
            #else:
            #    self._root_id_pos = {}
        return self._root_id_pos

    #not used
    @property
    def root_id_set(self):
        """
        A set representation of root_data.
        :return: A set containing the values of root_data or an empty set, if no root_data is available
        """
        if self._root_id_set is None:
            #if self._root_data is not None:
            logger.debug('forest: create root_id_set from root_ids (%i)' % len(self.root_data))
            self._root_id_set = set(self.root_data)
            #else:
            #    self._root_id_set = set()
        return self._root_id_set

    @property
    def root_id_mapping(self):
        """
        Maps from root_data to root index
        :return: a mapping as a dict or an empty dict if no root_data is available
        """
        if self._root_data_mapping is None:
            #assert self._root_data is not None, 'root_ids not set'
            #if self._root_data is not None:
            logger.debug('forest: create root_id_mapping from root_ids (%i)' % len(self.root_data))
            self._root_data_mapping = {v: i for i, v in enumerate(self.root_data)}
            #else:
            #    self._root_data_mapping = {}
        return self._root_data_mapping

    @property
    def root_mapping(self):
        """
        Maps from root positions regarding the data sequence to indices in the roots array
        :return:
        """
        if self._root_mapping is None:
            logger.debug('forest: create root_id_pos from root_ids (%i)' % len(self.roots))
            self._root_mapping = {pos: i for i, pos in enumerate(self.roots)}
        return self._root_mapping

    @property
    def link_types(self):
        """
        Get the data of the link types available in the associated lexicon.
        :return: data of link types
        """
        assert self.lexicon is not None, 'can not get link types if lexicon is not available'
        return self.lexicon.get_link_types(data_as_hashes=self.data_as_hashes)

    @property
    def graph_in(self):
        if self._graph_in is None:
            if self._graph_out is not None:
                logger.debug('create graph_in from graph_out')
                self._graph_in = csr_matrix(self._graph_out)
        return self._graph_in

    @property
    def graph_out(self):
        if self._graph_out is None:
            logger.debug('create graph_out from graph_in')
            self._graph_out = csc_matrix(self.graph_in)
        return self._graph_out

    @property
    def nbr_out(self):
        if self._nbr_out is None:
            self._nbr_out = self.graph_out.indptr[1:] - self.graph_out.indptr[:-1]
        return self._nbr_out

    @property
    def nbr_in(self):
        if self._nbr_in is None:
            self._nbr_in = self.graph_in.indptr[1:] - self.graph_in.indptr[:-1]
        return self._nbr_in

    @property
    def pos_ids(self):
        if self._pos_ids is None:
            self._pos_ids = self.roots + OFFSET_ID
        return self._pos_ids

    @property
    def pos_to_component_mapping(self):
        return self._pos_to_component_mapping

    @property
    def pos_start(self):
        return self.roots


