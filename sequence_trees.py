import numpy as np

import preprocessing
import sequence_node_candidates_pb2
import sequence_node_pb2
import sequence_node_sequence_pb2

import constants


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
        if not max_depth_only or current_depth + 1 == max_depth:
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


def identity_reader(content):
    yield content


# unused
def build_sequence_tree_from_str(str_, sentence_processor, parser, data_maps, concat_mode=constants.default_concat_mode,
                                 inner_concat_mode=constants.default_inner_concat_mode, expand_dict=True,
                                 seq_tree=None):
    seq_data, seq_parents, _ = preprocessing.read_data(identity_reader, sentence_processor, parser, data_maps,
                                                       reader_args={'content': str_}, concat_mode=concat_mode,
                                                       inner_concat_mode=inner_concat_mode, expand_dict=expand_dict)
    children, roots = children_and_roots(seq_parents)
    return build_sequence_tree(seq_data, children, roots[0], seq_tree)


# unused
def build_sequence_tree_from_parse(seq_graph, seq_tree=None):
    seq_data, seq_parents = seq_graph
    children, roots = children_and_roots(seq_parents)
    return build_sequence_tree(seq_data, children, roots[0], seq_tree)


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
