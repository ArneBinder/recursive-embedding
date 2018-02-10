#from __future__ import unicode_literals
import numpy as np
import logging
import os

import pydot

from constants import DTYPE_HASH, DTYPE_COUNT, DTYPE_IDX, DTYPE_OFFSET, DTYPE_DEPTH
import constants


def load(fn):
    logging.debug('load data and parents ...')
    data = np.load('%s.data' % fn)
    parents = np.load('%s.parent' % fn)
    return data, parents


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
    children_counts = np.zeros(shape=len(seq_parents), dtype=DTYPE_COUNT)
    for i, p in enumerate(seq_parents):
        if p == 0:  # is it a root?
            root_count += 1
            continue
        p_idx = i + p
        children_counts[p_idx] += 1
    children_offsets = np.zeros(shape=len(seq_parents), dtype=DTYPE_IDX)
    cc = 0
    for i, c in enumerate(children_counts):
        children_offsets[i] = cc
        cc += c

    roots = np.zeros(shape=root_count, dtype=DTYPE_IDX)
    root_count = 0
    children_indices = np.zeros(shape=cc, dtype=DTYPE_IDX)
    current_offsets = np.zeros(shape=len(seq_parents), dtype=DTYPE_IDX)
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


# unused
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


# unused
def calc_seq_depth(children, roots, seq_parents):
    # ATTENTION: int16 restricts the max sentence count per tree to 32767
    depth = -np.ones(len(seq_parents), dtype=DTYPE_DEPTH)
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


# unused
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


def tree_from_sorted_parent_triples(sorted_parent_triples, root_id,
                                    see_also_refs,
                                    root_type="http://dbpedia.org/resource",
                                    anchor_type="http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#Context",
                                    terminal_types=None,
                                    see_also_link_type="http://dbpedia.org/ontology/seeAlsoWikiPageWikiLink"
                                    #see_also_type="http://dbpedia.org/ontology/wikiPageWikiLink"
                                    ):
    """
    Constructs a tree from triples.
    :param sorted_parent_triples: list of triples (uri, uri_type, uri_parent)
                                  sorted by ?parent_beginIndex DESC(?parent_endIndex) ?beginIndex DESC(?endIndex))
    :param lexicon: the lexicon used to convert the uri strings into integer ids
    :param root_id: uri string added as direct root child, e.g. "http://dbpedia.org/resource/Damen_Group"
    :param root_type: uri string used as root data, e.g. "http://dbpedia.org/resource"
    :param anchor_type: the tree represented in sorted_parent_triples will be anchored via this uri string to the
                        root_type node, e.g. "http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#Context"
    :param terminal_types: uri strings that are considered as terminals, i.e. are used as roots of parsed string trees
    :return: the tree data strings, parents, position mappings ({str(uri): offset}) and list of terminal types
    """
    if terminal_types is None:
        terminal_types = [u'http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#Paragraph',
                          u'http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#Title']
    #ids_terminal_types = [lexicon[unicode(terminal_type)] for terminal_type in terminal_types]
    root_type_str = unicode(root_type)
    root_id_str = unicode(root_id)
    anchor_type_str = unicode(anchor_type)
    see_also_link_type_str = unicode(see_also_link_type)
    temp_data = [root_type_str, root_id_str, anchor_type_str]
    temp_parents = [0, -1, -2]
    for see_also_ref, in see_also_refs:
        temp_data.append(see_also_link_type_str)
        temp_parents.append(-len(temp_parents))
        temp_data.append(unicode(see_also_ref))
        temp_parents.append(-1)
    pre_len = len(temp_data)
    positions = {}
    parent_uris = {}
    terminal_parent_positions = {}
    terminal_types_list = []

    for uri, uri_type, uri_parent in sorted_parent_triples:
        uri_str = unicode(uri)
        uri_type_str = unicode(uri_type)
        uri_parent_str = unicode(uri_parent)
        if len(temp_data) == pre_len:
            positions[uri_parent_str] = 2
        parent_uris[uri_str] = uri_parent_str
        positions[uri_str] = len(temp_data)
        #id_uri_type = lexicon[uri_type_str]
        #if id_uri_type in ids_terminal_types:
        if uri_type_str in terminal_types:
            terminal_types_list.append(uri_type_str)
            terminal_parent_positions[uri_str] = positions[parent_uris[uri_str]]
        else:
            temp_data.append(uri_type_str)
            temp_parents.append(positions[uri_parent_str] - len(temp_parents))
        #positions[unicode(uri)] = len(temp_data)

    return temp_data, temp_parents, terminal_parent_positions, terminal_types_list


FE_DATA = 'data'
FE_PARENTS = 'parent'
FE_DATA_HASHES = 'data.hash'
FE_CHILDREN = 'child'
FE_CHILDREN_OFFSET = 'child.offset'


class Forest(object):
    def __init__(self, filename=None, data=None, parents=None, forest=None, tree_dict=None, lexicon=None, children=None,
                 children_offset=None, data_as_hashes=False):
        self.reset_cache_values()
        self._as_hashes = data_as_hashes
        self._lexicon = None
        self._data = None
        self._parents = None
        self._children = None
        self._children_pos = None
        if lexicon is not None:
            self._lexicon = lexicon
        if (data is not None and parents is not None) or forest is not None or filename is not None:
            self.set_forest(filename=filename, data=data, parents=parents, forest=forest,
                            children=children, children_offset=children_offset, data_as_hashes=data_as_hashes)
        elif tree_dict is not None:
            _data, _parents = sequence_node_to_sequence_trees(tree_dict)
            self.set_forest(data=_data, parents=_parents, data_as_hashes=data_as_hashes)
        else:
            raise ValueError(
                'Not enouth arguments to instantiate SequenceTrees object. Please provide a filename or data and parent arrays.')
        #self._sorted = np.zeros(len(self), dtype=bool)

    def reset_cache_values(self):
        self._children_dict = None
        self._roots = None
        self._depths = None
        self._depths_collected = None
        self._dicts = {}

    def set_forest(self, filename=None, data=None, parents=None, forest=None, children=None, children_offset=None,
                   data_as_hashes=False):
        if filename is not None:
            logging.debug('load data and parents from %s ...' % filename)
            if os.path.exists('%s.%s' % (filename, FE_DATA)):
                data = np.load('%s.%s' % (filename, FE_DATA))
            elif os.path.exists('%s.%s' % (filename, FE_DATA_HASHES)):
                data = np.load('%s.%s' % (filename, FE_DATA_HASHES))
                data_as_hashes = True
            else:
                raise IOError('data file (%s.%s or %s.%s) not found' % (filename, FE_DATA, filename, FE_DATA_HASHES))
            parents = np.load('%s.%s' % (filename, FE_PARENTS))
            if os.path.exists('%s.%s' % (filename, FE_CHILDREN)) and os.path.exists('%s.%s' % (filename, FE_CHILDREN_OFFSET)):
                children = np.load('%s.%s' % (filename, FE_CHILDREN))
                children_offset = np.load('%s.%s' % (filename, FE_CHILDREN_OFFSET))
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

        if children is not None and children_offset is not None:
            if not isinstance(children, np.ndarray) or not children.dtype == DTYPE_OFFSET:
                children = np.array(children, dtype=DTYPE_OFFSET)
            if not isinstance(children_offset, np.ndarray) or not children_offset.dtype == DTYPE_IDX:
                children_offset = np.array(children_offset, dtype=DTYPE_IDX)
            self._children = children
            self._children_pos = children_offset

    def set_lexicon(self, lexicon):
        self._lexicon = lexicon

    def dump(self, filename):
        logging.debug('dump data ...')
        if self.data_as_hashes:
            self.data.dump('%s.%s' % (filename, FE_DATA_HASHES))
        else:
            self.data.dump('%s.%s' % (filename, FE_DATA))
        logging.debug('dump parents ...')
        self.parents.dump('%s.%s' % (filename, FE_PARENTS))

        if self._children is not None and self._children_pos is not None:
            self._children.dump('%s.%s' % (filename, FE_CHILDREN))
            self._children_pos.dump('%s.%s' % (filename, FE_CHILDREN_OFFSET))

    @staticmethod
    def exist(filename):
        a = (os.path.exists('%s.%s' % (filename, FE_DATA)) or os.path.exists('%s.%s' % (filename, FE_DATA_HASHES)))
        b = os.path.exists('%s.%s' % (filename, FE_PARENTS))
        return a and b

    def reload(self, filename):
        self.reset_cache_values()
        self.set_forest(*load(filename))

    def hashes_to_indices(self):
        assert self.lexicon is not None, 'no lexicon available'
        assert self.data_as_hashes, 'data consists already of indices'
        self._data = self.lexicon.convert_data_hashes_to_indices(self.data)
        self._as_hashes = False

    def convert_data(self, converter, new_idx_unknown):
        #assert not self.data_as_hashes, 'can not convert data hashes'
        assert self.lexicon is not None, 'lexicon is not set'
        #assert self.lexicon.frozen, 'lexicon has to be frozen to convert data'
        convert_data(data=self.data, converter=converter, lex_size=len(self.lexicon), new_idx_unknown=new_idx_unknown)
        self._dicts = None

    def indices_to_forest(self, indices):
        # return np.array(map(lambda idx: self.forest.T[idx], indices)).T
        return np.array(map(lambda idx: self.data[idx], indices)), \
               np.array(map(lambda idx: self.parents[idx], indices))

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
        _cmp = _compare_tree_dicts(self.get_tree_dict_cached(root1), self.get_tree_dict_cached(root2))
        return _cmp == 0

    # TODO: check!
    def sample_all(self, sample_count=1, retry_count=10):
        if self.data_as_hashes:
            raise NotImplementedError('sample_all not implemented for data hashes')
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
        seq_node = {'head': self.data[idx], 'children': []}
        if idx in self.children:
            for child_offset in self.children[idx]:
                seq_node['children'].append(self.get_tree_dict_cached(idx + child_offset))
        seq_node['children'].sort(cmp=_compare_tree_dicts)

        self._dicts[idx] = seq_node
        return self._dicts[idx]

    def transform_data(self, idx):
        assert self.lexicon is not None, 'lexicon is not set'
        idx_trans = self.lexicon.ids_fixed_dict.get(idx, None)
        if idx_trans is not None:
            return -idx_trans
        idx_trans = self.lexicon.ids_var_dict.get(idx, None)
        if idx_trans is not None:
            return idx_trans
        raise ValueError('idx=%i not in ids_fixed and not in ids_var' % idx)

    def get_tree_dict(self, idx=None, max_depth=9999, context=0, transform=False):
        """
        Build a dict version of the subtree of this sequence_tree rooted at idx.
        Maintains order of data elements.

        :param idx: root of the subtree
        :param max_depth: stop if this depth is exceeded
        :param context depth of context tree (walk up parents) to add to all nodes
        :return: the dict version of the subtree
        """
        if idx is None:
            idx = self.roots[0]
        data_head = self.data[idx]
        if transform:
            data_head = self.transform_data(data_head)
        seq_node = {'head': data_head, 'children': []}
        if idx in self.children and max_depth > 0:
            for child_offset in self.children[idx]:
                seq_node['children'].append(self.get_tree_dict(idx=idx + child_offset, max_depth=max_depth - 1,
                                                               context=context, transform=transform))
        if self.parents[idx] != 0 and context > 0:
            seq_node['children'].append(self.get_tree_dict_parent(idx, context-1, transform=transform))
        return seq_node

    def get_tree_dict_rooted(self, idx, max_depth=9999, transform=False):
        result = self.get_tree_dict(idx, max_depth=max_depth, transform=transform)
        if self.parents[idx] != 0 and max_depth > 0:
            result['children'].append(self.get_tree_dict_parent(idx, max_depth-1, transform=transform))
        return result

    def get_tree_dict_parent(self, idx, max_depth=9999, transform=False):
        assert self.lexicon is not None, 'lexicon is not set'
        if self.parents[idx] == 0:
            return None
        previous_id = idx
        current_id = idx + self.parents[idx]
        data_head = self.data[current_id] + len(self.lexicon) / 2
        if transform:
            data_head = self.transform_data(data_head)
        result = {'head': data_head, 'children': []}
        current_dict_tree = result
        while max_depth > 0:
            # add other children
            for c in self.children[current_id]:
                c_id = current_id + c
                if c_id != previous_id:
                    current_dict_tree['children'].append(self.get_tree_dict(c_id, max_depth=max_depth - 1, transform=transform))
            # go up
            if self.parents[current_id] != 0:
                previous_id = current_id
                current_id = current_id + self.parents[current_id]
                data_head = self.data[current_id] + len(self.lexicon) / 2
                if transform:
                    data_head = self.transform_data(data_head)
                new_parent_child = {'head': data_head, 'children': []}
                current_dict_tree['children'].append(new_parent_child)
                current_dict_tree = new_parent_child
                max_depth -= 1
            else:
                break
        return result

    def children_dict_to_arrays(self):
        _children_pos = np.zeros(shape=len(self), dtype=constants.DTYPE_IDX)
        # initialize with maximal size possible (if forest is a list)
        _children = np.zeros(shape=2*len(self), dtype=constants.DTYPE_OFFSET)

        pos = 0
        for idx in range(len(self)):
            if idx in self.children:
                _children_pos[idx] = pos
                current_children = self.children[idx]
                l = len(current_children)
                _children[pos] = l
                #self._children[pos + 1:pos + 1 + l] = np.array(current_children, dtype=self._children.dtype)
                _children[pos + 1:pos + 1 + l] = current_children
                pos += l + 1
            else:
                _children_pos[idx] = -1

        self._children = _children[:pos]
        self._children_pos = _children_pos

    def children_array_to_dict(self, clear_arrays=False):
        assert self._children is not None and self._children_pos is not None, '_children or _children_pos arrays not set'
        self._children_dict = {}
        for idx, pos in enumerate(self._children_pos):
            if pos >= 0:
                l = self._children[pos]
                self._children_dict[idx] = self._children[pos+1:pos+1+l].tolist()
        if clear_arrays:
            self._children = None
            self._children_pos = None

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
                offset += len(forest._children_pos)
            new_children_pos = np.concatenate(new_children_pos_list)
        else:
            new_children = None
            new_children_pos = None
        self.set_forest(data=np.concatenate([f.data for f in [self] + others]),
                        parents=np.concatenate([f.parents for f in [self] + others]),
                        children=new_children,
                        children_offset=new_children_pos)

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
                offset += len(forest._children_pos)
            new_children_pos = np.concatenate(new_children_pos_list)
        else:
            new_children = None
            new_children_pos = None
        return Forest(data=np.concatenate([f.data for f in forests]),
                      parents=np.concatenate([f.parents for f in forests]),
                      children=new_children,
                      children_offset=new_children_pos,
                      data_as_hashes=forests[0].data_as_hashes,
                      lexicon=forests[0].lexicon)

    def visualize(self, filename, start=0, end=None):
        if end is None:
            end = len(self)
        assert self.lexicon is not None, 'lexicon is not set'

        graph = pydot.Dot(graph_type='digraph', rankdir='LR', bgcolor='transparent')
        if len(self) > 0:
            nodes = []
            for i, d in enumerate(self.data[start:end]):
                s = self.lexicon.get_s(d, self.data_as_hashes)
                if self.data_as_hashes:
                    d = self.lexicon.mapping[self.lexicon.strings[s]]
                if self.lexicon.is_fixed(d):
                    color = "dodgerblue"
                else:
                    color = "limegreen"
                l = Forest.filter_and_shorten_label(s, do_filter=True)
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

        logging.debug('graph created. write to file ...')
        # print(graph.to_string())
        graph.write_svg(filename)

    @staticmethod
    def filter_and_shorten_label(l, blacklist=[], do_filter=True):
        if do_filter:
            for b in blacklist:
                if l.startswith(b + constants.SEPARATOR):
                    return None

            for embedding_prefix in constants.vocab_manual.values():
                if l.startswith(embedding_prefix + constants.SEPARATOR):
                    return l[len(embedding_prefix + constants.SEPARATOR):]
            return l
        else:
            return l

    def get_text_plain(self, blacklist=None, start=0, end=None):
        assert self.lexicon is not None, 'lexicon is not set'
        if end is None:
            end = len(self)
        result = []
        if len(self.data) > 0:
            for d in self.data[start:end]:
                s = self.lexicon.get_s(d, self.data_as_hashes)
                l = Forest.filter_and_shorten_label(s, blacklist, do_filter=blacklist is not None)
                if l is not None:
                    result.append(l)
        return result

    def __str__(self):
        return self._data.__str__()

    def __len__(self):
        return len(self.data)

    @property
    def data(self):
        return self._data

    @property
    def parents(self):
        return self._parents

    @property
    def forest(self):
        return self.data, self.parents

    @property
    def children(self):
        if self._children_dict is None:
            if self._children is not None and self._children_pos is not None:
                self.children_array_to_dict()
            else:
                self._children_dict, self._roots = children_and_roots(self.parents)
        return self._children_dict

    @property
    def roots(self):
        if self._roots is None:
            self._roots = np.where(self.parents == 0)[0]
        return self._roots

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
