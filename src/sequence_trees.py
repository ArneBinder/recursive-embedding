#from __future__ import unicode_literals
import numpy as np
import logging
import os

import pydot

from constants import DTYPE_HASH, DTYPE_COUNT, DTYPE_IDX, DTYPE_OFFSET, DTYPE_DEPTH, KEY_HEAD, KEY_CHILDREN, \
    LOGGING_FORMAT, SEPARATOR, vocab_manual
from mytools import numpy_load, numpy_dump, numpy_exists

FE_DATA = 'data'
FE_PARENTS = 'parent'
FE_DATA_HASHES = 'data.hash'
FE_CHILDREN = 'child'
FE_CHILDREN_POS = 'child.pos'
#FE_ROOTS = 'root'
FE_ROOT_ID = 'root.id'
FE_ROOT_POS = 'root.pos'

MAX_DEPTH = 9999

logger = logging.getLogger('sequence_trees')
logger.setLevel(logging.DEBUG)
logger_streamhandler = logging.StreamHandler()
logger_streamhandler.setLevel(logging.INFO)
logger_streamhandler.setFormatter(logging.Formatter(LOGGING_FORMAT))
logger.addHandler(logger_streamhandler)


def _get_root(parents, idx):
    i = idx
    while parents[i] != 0:
        i += parents[i]
    return i


def _children_and_roots(seq_parents):
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


# unused
def _calc_depth(children, parents, depth, start):
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


class Forest(object):
    def __init__(self, filename=None, data=None, parents=None, forest=None, tree_dict=None, lexicon=None, children=None,
                 children_pos=None, data_as_hashes=False, root_ids=None, root_pos=None,
                 load_parents=True, load_children=True, load_root_ids=True, load_root_pos=True,
                 transformed_indices=False):
        self.reset_cache_values()
        self._as_hashes = data_as_hashes
        self._lexicon = None
        self._data = None
        self._parents = None
        self._children = None
        self._children_pos = None
        self._root_pos = None

        if lexicon is not None:
            self._lexicon = lexicon
        if filename is not None:
            self.load(filename=filename, load_parents=load_parents, load_children=load_children,
                      load_root_ids=load_root_ids, load_root_pos=load_root_pos)
        elif (data is not None and parents is not None) or forest is not None:
            self.set_forest(data=data, parents=parents, forest=forest, children=children, children_pos=children_pos,
                            data_as_hashes=data_as_hashes, root_ids=root_ids, root_pos=root_pos)
        elif tree_dict is not None:
            _data, _parents = _sequence_node_to_sequence_trees(tree_dict)
            if transformed_indices:
                assert not data_as_hashes, 'can not transform indices back if data_as_hashes'
                for i in range(len(_data)):
                    _data[i], _ = self.lexicon.transform_idx_back(_data[i])
            self.set_forest(data=_data, parents=_parents, data_as_hashes=data_as_hashes, root_ids=root_ids)
        else:
            raise ValueError(
                'Not enouth arguments to instantiate Forest object. Please provide a filename or data and parent arrays.')

    def reset_cache_values(self):
        self._children_dict = None
        self._depths = None
        self._depths_collected = None
        self._dicts = {}
        self._root_id_pos = None
        self._root_id_mapping = None

    def load(self, filename, load_parents=True, load_children=True, load_root_ids=True, load_root_pos=True):
        logger.debug('load data and parents from %s ...' % filename)
        #if os.path.exists('%s.%s' % (filename, FE_DATA)):
        self._data = numpy_load('%s.%s' % (filename, FE_DATA), assert_exists=False)
        self._as_hashes = False
        if self._data is None:
            self._data = numpy_load('%s.%s' % (filename, FE_DATA_HASHES), assert_exists=False)
            self._as_hashes = True
        if self._data is None:
            raise IOError('data file (%s.%s or %s.%s) not found' % (filename, FE_DATA, filename, FE_DATA_HASHES))
        if load_parents:
            #self._parents = np.load('%s.%s' % (filename, FE_PARENTS))
            self._parents = numpy_load('%s.%s' % (filename, FE_PARENTS), assert_exists=False)
        else:
            self._parents = None
        #if load_children and os.path.exists('%s.%s' % (filename, FE_CHILDREN)) and os.path.exists(
        #        '%s.%s' % (filename, FE_CHILDREN_POS)):
        if load_children:
            #self._children = np.load('%s.%s' % (filename, FE_CHILDREN))
            #self._children_pos = np.load('%s.%s' % (filename, FE_CHILDREN_POS))
            self._children = numpy_load('%s.%s' % (filename, FE_CHILDREN), assert_exists=False)
            self._children_pos = numpy_load('%s.%s' % (filename, FE_CHILDREN_POS), assert_exists=False)
        else:
            self._children = None
            self._children_pos = None
        assert self._parents is not None or (self._children is not None and self._children_pos is not None), \
            'no structure data (parents or children) loaded'
        #if load_root_ids and os.path.exists('%s.%s' % (filename, FE_ROOT_ID)):
        if load_root_pos:
            #self._root_ids = np.load('%s.%s' % (filename, FE_ROOT_ID))
            self._root_ids = numpy_load('%s.%s' % (filename, FE_ROOT_ID), assert_exists=False)
        else:
            self._root_ids = None
        #if load_root_pos and os.path.exists('%s.%s' % (filename, FE_ROOT_POS)):
        if load_root_pos:
            #self._root_pos = np.load('%s.%s' % (filename, FE_ROOT_POS))
            self._root_pos = numpy_load('%s.%s' % (filename, FE_ROOT_POS), assert_exists=False)
        else:
            self._root_pos = None

    def set_forest(self, data=None, parents=None, forest=None, children=None, children_pos=None,
                   data_as_hashes=False, root_ids=None, root_pos=None):
        self._as_hashes = data_as_hashes
        if self.data_as_hashes:
            data_dtype = DTYPE_HASH
        else:
            data_dtype = DTYPE_IDX
        if forest is not None:
            assert len(forest) == 2, 'Wrong array count: %i. Trees array has to contain exactly the parents and data ' \
                                     'arrays: len=2' % str(len(forest))
            data = forest[0]
            parents = forest[1]
        assert len(data) == len(parents), \
            'sizes of data and parents arrays differ: len(data)==%i != len(parents)==%i' % (len(data), len(parents))
        if not isinstance(data, np.ndarray) or not data.dtype == data_dtype:
            data = np.array(data, dtype=data_dtype)
        if not isinstance(parents, np.ndarray) or not parents.dtype == DTYPE_OFFSET:
            parents = np.array(parents, dtype=DTYPE_OFFSET)
        self._data = data
        self._parents = parents

        if children is not None and children_pos is not None:
            if not isinstance(children, np.ndarray) or not children.dtype == DTYPE_OFFSET:
                children = np.array(children, dtype=DTYPE_OFFSET)
            if not isinstance(children_pos, np.ndarray) or not children_pos.dtype == DTYPE_IDX:
                children_pos = np.array(children_pos, dtype=DTYPE_IDX)
        self._children = children
        self._children_pos = children_pos
        if root_ids is not None:
            if not isinstance(root_ids, np.ndarray):
                root_ids = np.array(root_ids, dtype=data_dtype)
            else:
                assert root_ids.dtype == data_dtype, 'root_ids has wrong dtype (%s), expected: %s' \
                                                     % (root_ids.dtype, data_dtype)
        self._root_ids = root_ids
        self._root_pos = root_pos
        assert self._parents is not None or \
               (self._children_pos is not None and self._children is not None and self._root_pos is not None), \
            'either parents (%s) or (children (%s), children positions (%s) and root positions (%s)) have to be set ' \
            '(True iff array is set, in brackets)' \
            % (str(self._parents is not None), str(self._children_pos is not None),
               str(self._children is not None), str(self._root_pos is not None))

    def set_lexicon(self, lexicon):
        self._lexicon = lexicon

    def set_root_ids(self, root_ids):
        assert len(root_ids) == len(self.roots), 'wrong amount of root ids=%i (amount of roots=%i)' \
                                                 % (len(root_ids), len(self.roots))
        if self.data_as_hashes:
            assert root_ids.dtype == DTYPE_HASH, 'wrong dtype of new root_ids=%s (expected: %s)' \
                                                 % (str(root_ids.dtype), str(DTYPE_HASH))
        else:
            assert root_ids.dtype == DTYPE_IDX, 'wrong dtype of new root_ids=%s (expected: %s)' \
                                                % (str(root_ids.dtype), str(DTYPE_IDX))
        self._root_ids = root_ids

    def dump(self, filename, save_parents=True, save_children=True, save_root_ids=True, save_root_pos=True):
        logger.debug('dump data ...')
        if self.data_as_hashes:
            #self.data.dump('%s.%s' % (filename, FE_DATA_HASHES))
            numpy_dump('%s.%s' % (filename, FE_DATA_HASHES), self.data)
        else:
            #self.data.dump('%s.%s' % (filename, FE_DATA))
            numpy_dump('%s.%s' % (filename, FE_DATA), self.data)
        logger.debug('dump parents ...')
        if save_parents and self.parents is not None:
            #self.parents.dump('%s.%s' % (filename, FE_PARENTS))
            numpy_dump('%s.%s' % (filename, FE_PARENTS), self.parents)

        if save_children and self._children is not None and self._children_pos is not None:
            #self._children.dump('%s.%s' % (filename, FE_CHILDREN))
            #self._children_pos.dump('%s.%s' % (filename, FE_CHILDREN_POS))
            numpy_dump('%s.%s' % (filename, FE_CHILDREN), self._children)
            numpy_dump('%s.%s' % (filename, FE_CHILDREN_POS), self._children_pos)

        if save_root_ids and self._root_ids is not None:
            #self._root_ids.dump('%s.%s' % (filename, FE_ROOT_ID))
            numpy_dump('%s.%s' % (filename, FE_ROOT_ID), self._root_ids)

        if save_root_pos and self._root_pos is not None:
            #self._root_pos.dump('%s.%s' % (filename, FE_ROOT_POS))
            numpy_dump('%s.%s' % (filename, FE_ROOT_POS), self._root_pos)

    @staticmethod
    def exist(filename):
        #data_exist = os.path.exists('%s.%s' % (filename, FE_DATA)) \
        #             or os.path.exists('%s.%s' % (filename, FE_DATA_HASHES))
        data_exist = numpy_exists('%s.%s' % (filename, FE_DATA)) \
                     or numpy_exists('%s.%s' % (filename, FE_DATA_HASHES))

        #structure_exist = os.path.exists('%s.%s' % (filename, FE_PARENTS)) \
        #                  or (os.path.exists('%s.%s' % (filename, FE_CHILDREN))
        #                      and os.path.exists('%s.%s' % (filename, FE_CHILDREN_POS)))
        structure_exist = numpy_exists('%s.%s' % (filename, FE_PARENTS)) \
                          or (numpy_exists('%s.%s' % (filename, FE_CHILDREN))
                              and numpy_exists('%s.%s' % (filename, FE_CHILDREN_POS)))

        return data_exist and structure_exist

    def hashes_to_indices(self, id_offset_mapping={}):
        assert self.lexicon is not None, 'no lexicon available'
        assert self.data_as_hashes, 'data consists already of indices'

        self._data = self.lexicon.convert_data_hashes_to_indices(self.data, id_offset_mapping)
        if self._root_ids is not None:
            self._root_ids = self.lexicon.convert_data_hashes_to_indices(self._root_ids, id_offset_mapping)
        self._as_hashes = False

    def convert_data(self, converter, new_idx_unknown):
        assert self.lexicon is not None, 'lexicon is not set'
        _convert_data(data=self.data, converter=converter, lex_size=len(self.lexicon), new_idx_unknown=new_idx_unknown)
        self._dicts = None

    def get_descendant_indices(self, root):
        leafs = [root]
        if self.has_children(root):
            for c in self.get_children(root):
                leafs.extend(self.get_descendant_indices(c + root))
        return leafs

    def trees(self, root_indices=None):
        if root_indices is None:
            root_indices = self.roots
        for i in root_indices:
            descendant_indices = sorted(self.get_descendant_indices(i))
            # TODO: check this!
            yield self.data[descendant_indices], self.parents[descendant_indices]

    def _set_depths(self, indices, current_depth, child_offset=0):
        # depths are counted starting from roots!
        for i in indices:
            idx = i + child_offset
            self._depths[idx] = current_depth
            if self.has_children(idx):
                self._set_depths(self.get_children(idx), current_depth + 1, idx)

    def trees_equal(self, root1, root2):
        _cmp = _compare_tree_dicts(self.get_tree_dict_cached(root1), self.get_tree_dict_cached(root2))
        return _cmp == 0

    # TODO: check!
    def sample_all(self, sample_count=1, retry_count=10):
        if self.data_as_hashes:
            raise NotImplementedError('sample_all not implemented for data hashes')
        logger.info('create negative samples for all data points ...')
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
                    logger.warning('sampled less candidates (%i) than sample_count (%i) for idx=%i. Skip it.'
                                    % (len(current_sampled_indices) - 1, sample_count, idx))
                else:
                    sampled_sim_tuples.append(current_sampled_indices)
        return sampled_sim_tuples

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
        if self.has_children(idx):
            for child_offset in self.get_children(idx):
                seq_node[KEY_CHILDREN].append(self.get_tree_dict_cached(idx + child_offset))
        seq_node[KEY_CHILDREN].sort(cmp=_compare_tree_dicts)

        self._dicts[idx] = seq_node
        return self._dicts[idx]

    def get_tree_dict(self, idx=None, max_depth=MAX_DEPTH, context=0, transform=False, costs={}, link_types=[],
                      link_content_offset=2):
        """
        Build a dict version of the subtree of this sequence_tree rooted at idx.
        Maintains order of data elements.

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
        if idx is None:
            idx = self.roots[0]
        data_head = self.data[idx]

        # TODO: DEBUG
        #s = self.lexicon.get_s(data_head, data_as_hashes=self.data_as_hashes)
        # DEBUG end

        cost = costs.get(data_head, 1)

        if transform:
            seq_node = {KEY_HEAD: self.lexicon.transform_idx(data_head, root_id_pos=self.root_id_pos), KEY_CHILDREN: []}
        else:
            seq_node = {KEY_HEAD: data_head, KEY_CHILDREN: []}

        # ATTENTION: allows cost of 0!
        if self.has_children(idx) and 0 <= cost <= max_depth:
            for child_offset in self.get_children(idx):
                target_idx = idx + child_offset
                if data_head in link_types and self.data[target_idx] in self.root_id_pos:
                    target_idx = self.root_id_pos[self.data[target_idx]] + link_content_offset
                    #logger.debug('follow link (%s)' % s)

                seq_node[KEY_CHILDREN].append(self.get_tree_dict(idx=target_idx,
                                                                 max_depth=max_depth - cost,
                                                                 context=context, transform=transform,
                                                                 costs=costs,
                                                                 link_types=link_types))
        if self.parents[idx] != 0 and context > 0:
            seq_node[KEY_CHILDREN].append(self.get_tree_dict_parent(idx=idx, max_depth=context-cost,
                                                                    transform=transform, costs=costs,
                                                                    link_types=link_types))
        return seq_node

    def get_tree_dict_rooted(self, idx, max_depth=9999, transform=False, costs={}, link_types=[]):
        result = self.get_tree_dict(idx, max_depth=max_depth, transform=transform)
        cost = costs.get(self.data[idx], 1)
        if self.parents[idx] != 0 and max_depth > 0:
            result[KEY_CHILDREN].append(self.get_tree_dict_parent(idx, max_depth-cost, transform=transform, costs=costs,
                                                                  link_types=link_types))
        return result

    def get_tree_dict_parent(self, idx, max_depth=9999, transform=False, costs={}, link_types=[]):
        assert self.lexicon is not None, 'lexicon is not set'

        if self.parents[idx] == 0:
            return None
        previous_id = idx
        current_id = idx + self.parents[idx]
        #data_head = self.lexicon.reverse_idx(self.data[current_id])
        #if transform:
        data_head = self.lexicon.transform_idx(self.data[current_id], revert=True, root_id_pos=self.root_id_pos)
        result = {KEY_HEAD: data_head, KEY_CHILDREN: []}
        current_dict_tree = result
        while max_depth > 0:
            current_d = self.data[current_id]
            current_cost = costs.get(current_d, 1)

            # add other children
            for c in self.get_children(current_id):
                c_id = current_id + c
                if c_id != previous_id:
                    current_dict_tree[KEY_CHILDREN].append(
                        self.get_tree_dict(c_id, max_depth=max_depth - current_cost, transform=transform, costs=costs,
                                           link_types=link_types))
            # go up
            if self.parents[current_id] != 0:
                previous_id = current_id
                current_id = current_id + self.parents[current_id]
                #data_head = self.lexicon.reverse_idx(self.data[current_id])
                #if transform:
                data_head = self.lexicon.transform_idx(self.data[current_id], revert=True, root_id_pos=self.root_id_pos)
                new_parent_child = {KEY_HEAD: data_head, KEY_CHILDREN: []}
                current_dict_tree[KEY_CHILDREN].append(new_parent_child)
                current_dict_tree = new_parent_child
                max_depth -= current_cost
            else:
                break
        return result

    @staticmethod
    def meta_matches(a, b, operation):
        assert a.lexicon == b.lexicon or b.lexicon is None, 'lexica do not match, can not %s.' % operation
        assert a.data.dtype == b.data.dtype, 'dtype of data arrays do not match, can not %s.' % operation
        assert a.parents.dtype == b.parents.dtype, 'dtype of parent arrays do not match, can not %s.' % operation
        assert a.data_as_hashes == b.data_as_hashes, 'data_as_hash do not match, can not %s.' % operation
        if a._children is not None and a._children_pos is not None:
            assert b._children is not None and b._children_pos is not None, 'if children arrays of first forest ' \
                                                                                'are set, the children arrays of second ' \
                                                                                'forest have to be set, too, can not ' \
                                                                                '%s.' % operation
        if a._parents is not None:
            assert b._parents is not None, 'parent array of first forest is set, but not of second, can not %s.' \
                                           % operation
        if a._root_ids is not None:
            assert b._root_ids is not None, 'root_ids of first forest is set, but not of the second, can not %s.' \
                                            % operation
        if a._root_pos is not None:
            assert b._root_pos is not None, 'root positions of first forest are set, but not of the second, can not %s.' \
                                            % operation

    def extend(self, others):
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
        if self._root_ids is not None:
            new_root_ids = np.concatenate([f._root_ids for f in [self] + others])
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
        self.set_forest(data=np.concatenate([f.data for f in [self] + others]),
                        parents=new_parents,
                        children=new_children,
                        children_pos=new_children_pos,
                        root_ids=new_root_ids,
                        root_pos=new_root_pos)

    @staticmethod
    def concatenate(forests):
        assert len(forests) > 0, 'can not concatenate empty list of forests'
        for f in forests[1:]:
            Forest.meta_matches(forests[0], f, 'concatenate')
        if forests[0]._children is not None and forests[0]._children_pos is not None:
            new_children = np.concatenate([f._children for f in forests])
            new_children_pos_list = []
            offset = 0
            for forest in forests:
                new_children_pos_list.append(forest._children_pos + offset)
                offset += len(forest._children)
            new_children_pos = np.concatenate(new_children_pos_list)
        else:
            new_children = None
            new_children_pos = None

        if forests[0]._root_ids is not None:
            new_root_ids = np.concatenate([f._root_ids for f in forests])
        else:
            new_root_ids = None
        if forests[0]._parents is not None:
            new_parents = np.concatenate([f.parents for f in forests])
        else:
            new_parents = None
        if forests[0]._root_pos is not None:
            new_root_pos_list = []
            offset = 0
            for forest in forests:
                new_root_pos_list.append(forest._root_pos + offset)
                offset += len(forest)
            new_root_pos = np.concatenate(new_root_pos_list)
        else:
            new_root_pos = None
        return Forest(data=np.concatenate([f.data for f in forests]),
                      parents=new_parents,
                      children=new_children,
                      children_pos=new_children_pos,
                      data_as_hashes=forests[0].data_as_hashes,
                      lexicon=forests[0].lexicon,
                      root_ids=new_root_ids,
                      root_pos=new_root_pos)

    def visualize(self, filename, start=0, end=None, transformed=False):
        if end is None:
            end = len(self)
        assert self.lexicon is not None, 'lexicon is not set'

        graph = pydot.Dot(graph_type='digraph', rankdir='LR', bgcolor='transparent')
        if len(self) > 0:
            nodes = []
            for i, d in enumerate(self.data[start:end]):
                reverted = False
                if transformed:
                    d, reverted = self.lexicon.transform_idx_back(d)
                s = self.lexicon.get_s(d, self.data_as_hashes)
                if self.data_as_hashes:
                    d = self.lexicon.mapping[self.lexicon.strings[s]]
                if self.lexicon.is_fixed(d):
                    color = "dodgerblue"
                else:
                    color = "limegreen"
                l = Forest.filter_and_shorten_label(s, do_filter=True)

                if reverted:
                    nodes.append(pydot.Node(i, label="'" + l + "'", style="filled",
                                            fillcolor='%s;0.5:white;0.5' % color,
                                            gradientangle=135))
                else:
                    nodes.append(pydot.Node(i, label="'" + l + "'", style="filled", fillcolor=color))

            for node in nodes:
                graph.add_node(node)

            # add invisible edges for alignment
            last_node = nodes[0]
            for node in nodes[1:]:
                graph.add_edge(pydot.Edge(last_node, node, weight=100, style='invis'))
                last_node = node

            for i in range(len(nodes)):
                target_index = i + self.parents[i+start]
                if target_index < 0 or target_index >= len(nodes):
                    target_index = i
                graph.add_edge(pydot.Edge(nodes[i], nodes[target_index], dir='back'))

        logger.debug('graph created. write to file ...')
        # print(graph.to_string())
        graph.write_svg(filename, encoding='utf-8')

    @staticmethod
    def filter_and_shorten_label(l, blacklist=[], do_filter=True):
        if do_filter:
            for b in blacklist:
                if l.startswith(b):
                    return None

            for embedding_prefix in vocab_manual.values():
                if l.startswith(embedding_prefix + SEPARATOR):
                    return l[len(embedding_prefix + SEPARATOR):]
            return l
        else:
            return l

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
                l = Forest.filter_and_shorten_label(s, blacklist, do_filter=blacklist is not None)
                if l is not None:
                    if reverted:
                        l += '_REV'
                    result.append(l)
        return result

    def has_children(self, idx):
        c_pos = self._children_pos[idx]
        return self._children[c_pos] > 0

    def get_children(self, idx):
        child_pos = self._children_pos[idx]
        c = self._children[child_pos]
        return self._children[child_pos+1:child_pos+1+c]

    def get_children_counts(self, indices):
        children_pos = self._children_pos[indices]
        return self._children[children_pos]

    def set_parents_with_children(self):
        logger.warning('set_parents_with_children ...')
        assert self._children is not None and self._children_pos is not None, 'children arrays are None, can not ' \
                                                                              'create parents'
        self._parents = np.zeros(shape=len(self), dtype=DTYPE_OFFSET)
        for p_idx, c_pos in enumerate(self._children_pos):
            c_count = self._children[c_pos]
            for c_offset in self._children[c_pos+1:c_pos+1+c_count]:
                c_idx = p_idx + c_offset
                self._parents[c_idx] = -c_offset

    def set_children_with_parents(self):
        #logger.warning('set_children_with_parents ...')
        assert self._parents is not None, 'parents are None, can not create children arrays'
        children_dict, _ = _children_and_roots(self.parents)

        _children_pos = np.zeros(shape=len(self), dtype=DTYPE_IDX)
        # initialize with maximal size possible (if forest is a list)
        _children = np.zeros(shape=2 * len(self), dtype=DTYPE_OFFSET)

        pos = 0
        for idx in range(len(self)):
            _children_pos[idx] = pos
            if idx in children_dict:
                current_children = children_dict[idx]
                l = len(current_children)
                _children[pos] = l
                _children[pos + 1:pos + 1 + l] = current_children
                pos += l
            else:
                _children[pos] = 0
            pos += 1

        self._children = _children[:pos]
        self._children_pos = _children_pos

    def __str__(self):
        return self._data.__str__()

    def __len__(self):
        return len(self.data)

    @property
    def data(self):
        return self._data

    @property
    def parents(self):
        if self._parents is None:
            self.set_parents_with_children()
        return self._parents

    @property
    def forest(self):
        return self.data, self.parents

    @property
    def roots(self):
        if self._root_pos is None:
            self._root_pos = np.where(self.parents == 0)[0].astype(DTYPE_IDX)
        return self._root_pos

    @property
    def depths(self):
        if self._depths is None:
            self._depths = np.zeros(shape=self.data.shape, dtype=DTYPE_DEPTH)
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

    @property
    def lexicon(self):
        return self._lexicon

    @property
    def data_as_hashes(self):
        return self._as_hashes

    @property
    def root_id_pos(self):
        if self._root_id_pos is None:
            #assert self._root_ids is not None, 'root_ids not set'
            if self._root_ids is not None:
                self._root_id_pos = {v: self.roots[i] for i, v in enumerate(self._root_ids)}
            else:
                self._root_id_pos = {}
        return self._root_id_pos

    @property
    def root_id_mapping(self):
        if self._root_id_mapping is None:
            #assert self._root_ids is not None, 'root_ids not set'
            if self._root_ids is not None:
                self._root_id_mapping = {v: i for i, v in enumerate(self._root_ids)}
            else:
                self._root_id_mapping = {}
        return self._root_id_mapping
