from __future__ import print_function

import fnmatch
import logging
import ntpath
import os

import numpy as np

import constants
import sequence_node_candidates_pb2
import sequence_node_pb2
import sequence_node_sequence_pb2
import tools


def merge_sentence_data(sen_data, sen_parents, sen_offsets, sen_a):
    root_offset = 0
    result_data = list()
    result_parents = list()
    l = len(sen_data)
    for i in range(l):
        # set root
        if sen_parents[i] == 0:
            root_offset = len(result_data)
        # add main data
        result_data.append(sen_data[i])
        # shift parent indices
        parent_idx = sen_parents[i] + i
        shift = tools.get_default(sen_offsets, parent_idx - 1, 0) - tools.get_default(sen_offsets, i - 1, 0)
        # add (shifted) main parent
        result_parents.append(sen_parents[i] + shift)
        # insert additional data
        a_data, a_parents = sen_a[i]
        if len(a_data) > 0:
            result_data.extend(a_data)
            result_parents.extend(a_parents)

    return result_data, result_parents, [root_offset]


def concat_roots(parents, root_offsets, root_parents=None, concat_mode='tree'):

    if not concat_mode:
        return parents, root_offsets
    elif concat_mode == 'aggregate':
        for i in range(len(root_offsets)):
            parents[root_offsets[i]] = len(parents) - root_offsets[i] - 1
        return parents, [len(parents) - 1]
    elif concat_mode == 'tree':
        assert root_parents and len(root_parents) == len(root_offsets), \
            'length of root_parents does not match length of root_offsets='+str(len(root_offsets))
        new_roots = []
        for i in range(len(root_parents)):
            if root_parents[i] != 0:
                parents[root_offsets[i]] = root_offsets[i + root_parents[i]] - root_offsets[i]
            else:
                new_roots.append(root_offsets[i])

        return parents, new_roots
    # connect roots consecutively
    elif concat_mode == 'sequence':
        for i in range(len(root_offsets) - 1):
            parents[root_offsets[i]] = root_offsets[i+1] - root_offsets[i]
        return parents, [len(parents) - 1]

    else:
        raise NameError('unknown concat_mode: ' + concat_mode)


# embeddings for:
# word
def process_sentence2(sentence, parsed_data, data_maps, dict_unknown=None, concat_mode='tree'):
    sen_data = []
    sen_parents = []
    root_offsets = []
    root_parents = []
    for i in range(sentence.start, sentence.end):
        token = parsed_data[i]
        # add word embedding
        sen_data.append(tools.getOrAdd(data_maps, token.orth_, dict_unknown))

        parent_offset = token.head.i - i
        root_parents.append(parent_offset)
        # save root offset
        root_offsets.append(len(sen_parents))
        # set as root
        sen_parents.append(0)

    if concat_mode == 'aggregate':
        sen_parents.append(0)
        sen_data.append(tools.getOrAdd(data_maps, constants.vocab_manual[constants.AGGREGATOR_EMBEDDING], dict_unknown))
    sen_parents, root_offsets = concat_roots(sen_parents, root_offsets, root_parents, concat_mode)

    return sen_data, sen_parents, root_offsets


# embeddings for:
# word, edge
def process_sentence3(sentence, parsed_data, data_maps, dict_unknown=None, concat_mode=None):
    sen_data = []
    sen_parents = []
    root_offsets = []
    root_parents = []
    for i in range(sentence.start, sentence.end):

        # get current token
        token = parsed_data[i]
        parent_offset = token.head.i - i
        root_parents.append(parent_offset)
        # save root offset
        root_offsets.append(len(sen_parents))
        # add word embedding
        sen_data.append(tools.getOrAdd(data_maps, token.orth_, dict_unknown))
        sen_parents.append(0)
        # add edge type embedding
        sen_data.append(tools.getOrAdd(data_maps, token.dep_, dict_unknown))
        sen_parents.append(-1)

    if concat_mode == 'aggregate':
        sen_parents.append(0)
        sen_data.append(tools.getOrAdd(data_maps, constants.vocab_manual[constants.AGGREGATOR_EMBEDDING], dict_unknown))
    sen_parents, root_offsets = concat_roots(sen_parents, root_offsets, root_parents, concat_mode)

    return sen_data, sen_parents, root_offsets


# embeddings for:
# word, word embedding, edge, edge embedding
def process_sentence4(sentence, parsed_data, data_maps, dict_unknown=None, concat_mode=None):
    sen_data = []
    sen_parents = []
    root_offsets = []
    root_parents = []
    for i in range(sentence.start, sentence.end):
        # get current token
        token = parsed_data[i]
        parent_offset = token.head.i - i
        root_parents.append(parent_offset)
        # save root offset
        root_offsets.append(len(sen_parents))
        # add word embedding
        sen_data.append(tools.getOrAdd(data_maps, token.orth_, dict_unknown))
        sen_parents.append(0)
        # add word embedding embedding
        sen_data.append(tools.getOrAdd(data_maps, constants.vocab_manual[constants.TOKEN_EMBEDDING], dict_unknown))
        sen_parents.append(-1)
        # add edge type embedding
        sen_data.append(tools.getOrAdd(data_maps, token.dep_, dict_unknown))
        sen_parents.append(-2)
        # add edge type embedding embedding
        sen_data.append(tools.getOrAdd(data_maps, constants.vocab_manual[constants.EDGE_EMBEDDING], dict_unknown))
        sen_parents.append(-1)

    if concat_mode == 'aggregate':
        sen_parents.append(0)
        sen_data.append(tools.getOrAdd(data_maps, constants.vocab_manual[constants.AGGREGATOR_EMBEDDING], dict_unknown))
    sen_parents, root_offsets = concat_roots(sen_parents, root_offsets, root_parents, concat_mode)

    return sen_data, sen_parents, root_offsets


# embeddings for:
# words, edges, entity type (if !=0)
def process_sentence5(sentence, parsed_data, data_maps, dict_unknown=None, concat_mode=None):
    sen_data = []
    sen_parents = []
    root_offsets = []
    root_parents = []
    for i in range(sentence.start, sentence.end):

        # get current token
        token = parsed_data[i]
        parent_offset = token.head.i - i
        root_parents.append(parent_offset)
        # save root offset
        root_offsets.append(len(sen_parents))
        # add word embedding
        sen_data.append(tools.getOrAdd(data_maps, token.orth_, dict_unknown))
        sen_parents.append(0)
        # add edge type embedding
        sen_data.append(tools.getOrAdd(data_maps, token.dep_, dict_unknown))
        sen_parents.append(-1)

        if token.ent_type != 0 and (token.head == token or token.head.ent_type != token.ent_type):
            sen_data.append(tools.getOrAdd(data_maps, token.ent_type_, dict_unknown))
            sen_parents.append(-2)

    if concat_mode == 'aggregate':
        sen_parents.append(0)
        sen_data.append(tools.getOrAdd(data_maps, constants.vocab_manual[constants.AGGREGATOR_EMBEDDING], dict_unknown))
    sen_parents, root_offsets = concat_roots(sen_parents, root_offsets, root_parents, concat_mode)

    return sen_data, sen_parents, root_offsets


# embeddings for:
# words, word type, edges, edge type, entity type (if !=0), entity type type
def process_sentence6(sentence, parsed_data, data_maps, dict_unknown=None, concat_mode=None):
    sen_data = []
    sen_parents = []
    root_offsets = []
    root_parents = []
    for i in range(sentence.start, sentence.end):

        # get current token
        token = parsed_data[i]
        parent_offset = token.head.i - i
        root_parents.append(parent_offset)
        # save root offset
        root_offsets.append(len(sen_parents))
        # add word embedding
        sen_data.append(tools.getOrAdd(data_maps, token.orth_, dict_unknown))
        sen_parents.append(0)

        # add word type type embedding
        sen_data.append(tools.getOrAdd(data_maps, constants.vocab_manual[constants.TOKEN_EMBEDDING], dict_unknown))
        sen_parents.append(-1)
        # add edge type embedding
        sen_data.append(tools.getOrAdd(data_maps, token.dep_, dict_unknown))
        sen_parents.append(-2)
        # add edge type type embedding
        sen_data.append(tools.getOrAdd(data_maps, constants.vocab_manual[constants.EDGE_EMBEDDING], dict_unknown))
        sen_parents.append(-1)

        if token.ent_type != 0 and (token.head == token or token.head.ent_type != token.ent_type):
            sen_data.append(tools.getOrAdd(data_maps, token.ent_type_, dict_unknown))
            sen_parents.append(-2)
            sen_data.append(tools.getOrAdd(data_maps, constants.vocab_manual[constants.ENTITY_EMBEDDING], dict_unknown))
            sen_parents.append(-1)

    if concat_mode == 'aggregate':
        sen_parents.append(0)
        sen_data.append(tools.getOrAdd(data_maps, constants.vocab_manual[constants.AGGREGATOR_EMBEDDING], dict_unknown))
    sen_parents, root_offsets = concat_roots(sen_parents, root_offsets, root_parents, concat_mode)

    return sen_data, sen_parents, root_offsets


# embeddings for:
# words, edges, entity type (if !=0),
# lemma (if different), pos-tag
def process_sentence7(sentence, parsed_data, data_maps, dict_unknown=None, concat_mode=None):

    sen_data = []
    sen_parents = []
    root_offsets = []
    root_parents = []
    for i in range(sentence.start, sentence.end):

        # get current token
        token = parsed_data[i]
        parent_offset = token.head.i - i
        root_parents.append(parent_offset)
        # save root offset
        root_offset = len(sen_parents)
        root_offsets.append(root_offset)
        # add word embedding
        sen_data.append(tools.getOrAdd(data_maps, token.orth_, dict_unknown))
        sen_parents.append(0)

        # add edge type embedding
        sen_data.append(tools.getOrAdd(data_maps, token.dep_, dict_unknown))
        sen_parents.append(root_offset - len(sen_parents))
        # add pos-tag type embedding
        sen_data.append(tools.getOrAdd(data_maps, token.tag_, dict_unknown))
        sen_parents.append(root_offset - len(sen_parents))

        # add entity type embedding
        if token.ent_type != 0 and (token.head == token or token.head.ent_type != token.ent_type):
            sen_data.append(tools.getOrAdd(data_maps, token.ent_type_, dict_unknown))
            sen_parents.append(root_offset - len(sen_parents))
        # add lemma type embedding
        if token.lemma != token.orth:
            sen_data.append(tools.getOrAdd(data_maps, token.lemma_, dict_unknown))
            sen_parents.append(root_offset - len(sen_parents))

    if concat_mode == 'aggregate':
        sen_parents.append(0)
        sen_data.append(tools.getOrAdd(data_maps, constants.vocab_manual[constants.AGGREGATOR_EMBEDDING], dict_unknown))
    sen_parents, root_offsets = concat_roots(sen_parents, root_offsets, root_parents, concat_mode)

    return sen_data, sen_parents, root_offsets


# embeddings for:
# words, word type, edges, edge type, entity type (if !=0), entity type type,
# lemma (if different), lemma type, pos-tag, pos-tag type
def process_sentence8(sentence, parsed_data, data_maps, dict_unknown=None, concat_mode=None):
    sen_data = []
    sen_parents = []
    root_offsets = []
    root_parents = []
    for i in range(sentence.start, sentence.end):

        # get current token
        token = parsed_data[i]
        parent_offset = token.head.i - i
        root_parents.append(parent_offset)
        # save root offset
        root_offset = len(sen_parents)
        root_offsets.append(root_offset)
        # add word embedding
        sen_data.append(tools.getOrAdd(data_maps, token.orth_, dict_unknown))
        sen_parents.append(0)

        # add word type type embedding
        sen_data.append(tools.getOrAdd(data_maps, constants.vocab_manual[constants.TOKEN_EMBEDDING], dict_unknown))
        sen_parents.append(root_offset - len(sen_parents))
        # add edge type embedding
        sen_data.append(tools.getOrAdd(data_maps, token.dep_, dict_unknown))
        sen_parents.append(root_offset - len(sen_parents))
        # add edge type type embedding
        sen_data.append(tools.getOrAdd(data_maps, constants.vocab_manual[constants.EDGE_EMBEDDING], dict_unknown))
        sen_parents.append(-1)
        # add pos-tag type embedding
        sen_data.append(tools.getOrAdd(data_maps, token.tag_, dict_unknown))
        sen_parents.append(root_offset - len(sen_parents))
        # add pos-tag type type embedding
        sen_data.append(tools.getOrAdd(data_maps, constants.vocab_manual[constants.POS_EMBEDDING], dict_unknown))
        sen_parents.append(-1)

        # add entity type embedding
        if token.ent_type != 0 and (token.head == token or token.head.ent_type != token.ent_type):
            sen_data.append(tools.getOrAdd(data_maps, token.ent_type_, dict_unknown))
            sen_parents.append(root_offset - len(sen_parents))
            # add entity type type embedding
            sen_data.append(
                tools.getOrAdd(data_maps, constants.vocab_manual[constants.ENTITY_EMBEDDING], dict_unknown))
            sen_parents.append(-1)
        # add lemma type embedding
        if token.lemma != token.orth:
            sen_data.append(tools.getOrAdd(data_maps, token.lemma_, dict_unknown))
            sen_parents.append(root_offset - len(sen_parents))
            # add lemma type type embedding
            sen_data.append(
                tools.getOrAdd(data_maps, constants.vocab_manual[constants.LEMMA_EMBEDDING], dict_unknown))
            sen_parents.append(-1)

    if concat_mode == 'aggregate':
        sen_parents.append(0)
        sen_data.append(tools.getOrAdd(data_maps, constants.vocab_manual[constants.AGGREGATOR_EMBEDDING], dict_unknown))
    sen_parents, root_offsets = concat_roots(sen_parents, root_offsets, root_parents, concat_mode)

    return sen_data, sen_parents, root_offsets


def dummy_str_reader():
    yield u'I like RTRC!'


# DEPRICATED use identity_reader instead
def string_reader(content):
    yield content.decode('utf-8')
    #yield content.decode('utf-8')


def identity_reader(content):
    yield content


def get_root(parents, idx):
    i = idx
    while parents[i] != 0:
        i += parents[i]
    return i


def read_data(reader, sentence_processor, parser, data_maps, args={}, batch_size=1000, concat_mode='sequence', inner_concat_mode='tree', expand_dict=True, calc_depths=False):

    # ids (dictionary) of the data points in the dictionary
    seq_data = list()
    # offsets of the parents
    seq_parents = list()

    # init as list containing an empty dummy array with dtype=int16 to allow numpy concatenation even if empty
    depth_list = [np.ndarray(shape=(0,), dtype=np.int16)]

    if expand_dict:
        unknown_default = None
    else:
        unknown_default = constants.vocab_manual[constants.UNKNOWN_EMBEDDING]

    logging.debug('start read_data ...')
    sen_count = 0
    for parsed_data in parser.pipe(reader(**args), n_threads=4, batch_size=batch_size):
        #prev_root = None
        temp_roots = []
        start_idx = len(seq_data)
        for sentence in parsed_data.sents:
            processed_sen = sentence_processor(sentence, parsed_data, data_maps, unknown_default, inner_concat_mode)
            # skip not processed sentences (see process_sentence)
            if processed_sen is None:
                continue

            sen_data, sen_parents, root_offsets = processed_sen

            sen_roots = [offset + len(seq_data) for offset in root_offsets]

            seq_parents += sen_parents
            seq_data += sen_data
            temp_roots += sen_roots

            sen_count += 1

        # add aggregator node
        if concat_mode == 'aggregate':
            seq_parents.append(0)
            seq_data.append(tools.getOrAdd(data_maps, constants.vocab_manual[constants.AGGREGATOR_EMBEDDING],
                                           unknown_default))
        seq_parents, seq_roots = concat_roots(seq_parents, temp_roots, concat_mode=concat_mode)

        #if not concat_mode or concat_mode == 'sequence':
        #    # connect roots consecutively
        #    prev_root = temp_roots[0]
        #    for temp_root in temp_roots[1:]:
        #        seq_parents[prev_root] = temp_root - prev_root
        #        prev_root = temp_root
        #elif concat_mode == 'aggregate':
        #    # connect roots to artificial AGGREGATOR node
        #    aggregator_id = tools.getOrAdd(data_maps, constants.vocab_manual[constants.AGGREGATOR_EMBEDDING],
        #                                   unknown_default)
        #    for temp_root in temp_roots:
        #        seq_parents[temp_root] = len(seq_data) - temp_root
        #    seq_data.append(aggregator_id)
        #    seq_parents.append(0)
        #else:
        #    raise NameError('unknown concat_mode: ' + concat_mode)

        if calc_depths:
            # get current parents
            current_seq_parents = np.array(seq_parents[start_idx:])

            #logging.info('calc children and roots ...')
            children, roots = children_and_roots(current_seq_parents)
            # calc depth for every position
            #logging.info('calc depths ...')
            depth = calc_seq_depth(children, roots, current_seq_parents)
            depth_list.append(depth)

    logging.debug('sentences read: '+str(sen_count))
    data = np.array(seq_data)
    parents = np.array(seq_parents)
    #root = prev_root

    return data, parents, np.concatenate(depth_list) #, root #, np.array(seq_edges)#, dep_map


def calc_depths_and_child_indices(x):#(out_path, offset, max_depth)):
    (parents, max_depth, child_idx_offset) = x
    #parents = np.load(out_path + '.parent.batch' + str(offset))
    # calc children and roots
    children, roots = children_and_roots(parents)
    # calc depth for every position
    depth = calc_seq_depth(children, roots, parents)

    #depth.dump(out_path + '.depth.batch' + str(offset))

    idx_tuples = []

    for idx in children.keys():
        for (child_offset, child_steps_to_root) in get_all_children_rec(idx, children, max_depth):
            idx_tuples.append((idx+child_idx_offset, child_offset, child_steps_to_root))

    #np.array(idx_tuples).dump(out_path + '.children.batch' + str(offset))

    return depth, np.array(idx_tuples)


def calc_seq_depth(children, roots, seq_parents):
    # ATTENTION: int16 restricts the max sentence count per tree to 32767
    depth = -np.ones(len(seq_parents), dtype=np.int16)
    for root in roots:
        calc_depth(children, seq_parents, depth, root)
    return depth

def addMissingEmbeddings(seq_data, embeddings):
    # get current count of embeddings
    l = embeddings.shape[0]
    # get all ids without embedding
    new_embedding_ids = sorted(list({elem for elem in seq_data if elem >= l}))
    # no unknown embeddings
    if len(new_embedding_ids) == 0:
        return

    #logging.info('new_embedding_ids: ' + str(new_embedding_ids))
    #logging.info('new_embedding_ids[0]: ' + str(new_embedding_ids[0]))
    #logging.info('l: ' + str(l))
    # check integrity
    assert new_embedding_ids[0] == l, str(new_embedding_ids[0]) + ' != ' + str(l)
    assert new_embedding_ids[-1] == l + len(new_embedding_ids) - 1, str(new_embedding_ids[-1]) + ' != ' + str(l + len(new_embedding_ids) - 1)

    # get mean of existing embeddings
    #mean = np.mean(embeddings, axis=1)

    new_embeddings = np.lib.pad(embeddings, ((0, len(new_embedding_ids)), (0, 0)), 'mean')
    return new_embeddings, len(new_embedding_ids)


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


def children_and_roots(seq_parents):
    # assume, all parents are inside this array!
    # collect children
    #children = [[] for _ in xrange(len(seq_parents))]
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


def get_all_children_rec(idx, children, max_depth, current_depth=0, max_depth_only=False):
    #if idx not in children or max_depth == 0:
    if idx not in children or max_depth == 0:
        return []
    result = []
    for child in children[idx]:
        if not max_depth_only or current_depth + 1 == max_depth:
            result.append((child, current_depth + 1))
        result.extend(get_all_children_rec(idx + child, children, max_depth-1, current_depth + 1, max_depth_only))
    return result


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
    #if len(children[start]) > 0:
        children_idx.append(0)
        while len(children_idx) > 0:
            current_child_idx = children_idx.pop()
            # go down
            #idx = children[idx][current_child_idx]
            idx += children[idx][current_child_idx]
            # not already calculated?
            if depth[idx] < 0:
                # no children --> depth == 0
                if idx not in children:
                #if len(children[idx]) == 0:
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
def build_sequence_tree(seq_data, children, root, seq_tree, max_depth=9999):
    # assume, all parents are inside this array!
    # collect children

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


def build_sequence_tree_with_candidate(seq_data, children, root, insert_idx, candidate_idx, max_depth, max_candidate_depth, seq_tree = None):
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


def build_sequence_tree_with_candidates(seq_data, parents, children, root, insert_idx, candidate_indices, seq_tree=None, max_depth=999):
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
                        build_sequence_tree(seq_data, children, pos + child_offset, seq_node.children.add(), max_depth - 1)

    if seq_tree is None:
        seq_tree = sequence_node_candidates_pb2.SequenceNodeCandidates()
    build(seq_tree, root, max_depth)

    return seq_tree


def build_sequence_tree_from_str(str_, sentence_processor, parser, data_maps, seq_tree=None, concat_mode=None, expand_dict=True):
    seq_data, seq_parents, _ = read_data(identity_reader, sentence_processor, parser, data_maps,
                                            args={'content': str_}, concat_mode=concat_mode, expand_dict=expand_dict)
    children, roots = children_and_roots(seq_parents)
    return build_sequence_tree(seq_data, children, roots[0], seq_tree)


def build_sequence_tree_from_parse(seq_graph, seq_tree=None):
    seq_data, seq_parents = seq_graph
    children, roots = children_and_roots(seq_parents)
    return build_sequence_tree(seq_data, children, roots[0], seq_tree)


def calc_depths_collected(out_filename, parent_dir, max_depth, seq_depths):
    depths_collected_files = fnmatch.filter(os.listdir(parent_dir),
                                            ntpath.basename(out_filename) + '.depth*.collected')
    if len(depths_collected_files) < max_depth:
        logging.info('collect depth indices in depth_maps ...')
        # depth_maps_files = fnmatch.filter(os.listdir(parent_dir), ntpath.basename(out_filename) + '.depth.*')
        depth_map = {}

        for idx, current_depth in enumerate(seq_depths):
            # if not os.path.isfile(out_filename+'.depth.'+str(current_depth)):
            try:
                depth_map[current_depth].append(idx)
            except KeyError:
                depth_map[current_depth] = [idx]

        # fill missing depths
        real_max_depth = max(depth_map.keys())
        for current_depth in range(real_max_depth + 1):
            if current_depth not in depth_map:
                depth_map[current_depth] = []

        depths_collected = np.array([], dtype=np.int16)
        for current_depth in reversed(sorted(depth_map.keys())):
            # if appending an empty list to an empty depths_collected, the dtype will change to float!
            if len(depth_map[current_depth]) > 0:
                depths_collected = np.append(depths_collected, depth_map[current_depth])
            if current_depth < max_depth:
                np.random.shuffle(depths_collected)
                depths_collected.dump(out_filename + '.depth' + str(current_depth) + '.collected')
                logging.info('depth: ' + str(current_depth) + ', size: ' + str(
                    len(depth_map[current_depth])) + ', collected_size: ' + str(len(depths_collected)))


def batch_file_count(total_count, batch_size):
    return total_count / batch_size + (total_count % batch_size > 0)


def rearrange_children_indices(out_filename, parent_dir, max_depth, max_articles, batch_size):
    # not yet used
    #child_idx_offset = 0
    ##
    children_depth_batch_files = fnmatch.filter(os.listdir(parent_dir),
                                                ntpath.basename(out_filename) + '.children.depth*.batch*')
    children_depth_files = fnmatch.filter(os.listdir(parent_dir), ntpath.basename(out_filename) + '.children.depth*')
    if len(children_depth_batch_files) < batch_file_count(max_articles, batch_size) and len(children_depth_files) < max_depth:
        for offset in range(0, max_articles, batch_size):
            current_depth_batch_files = fnmatch.filter(os.listdir(parent_dir),
                                                       ntpath.basename(out_filename) + '.children.depth*.batch' + str(
                                                           offset))
            # skip, if already processed
            if len(current_depth_batch_files) < max_depth:
                logging.info('read child indices for offset=' + str(offset) + ' ...')
                current_idx_tuples = np.load(out_filename + '.children.batch' + str(offset))
                # add offset
                #current_idx_tuples += np.array([child_idx_offset, 0, 0])
                logging.info(len(current_idx_tuples))
                logging.info('get depths ...')
                children_depths = current_idx_tuples[:, 2]
                logging.info('argsort ...')
                sorted_indices = np.argsort(children_depths)
                logging.info('find depth changes ...')
                depth_changes = []
                for idx, sort_idx in enumerate(sorted_indices):
                    current_depth = children_depths[sort_idx]
                    if idx == len(sorted_indices) - 1 or current_depth != children_depths[sorted_indices[idx + 1]]:
                        logging.info('new depth: ' + str(current_depth) + ' ends before index pos: ' + str(idx + 1))
                        depth_changes.append((idx + 1, current_depth))
                prev_end = 0
                for (end, current_depth) in depth_changes:
                    size = end - prev_end
                    logging.info('size: ' + str(size))
                    current_indices = np.zeros(shape=(size, 2), dtype=int)
                    for idx in range(size):
                        current_indices[idx] = current_idx_tuples[sorted_indices[prev_end + idx]][:2]
                    logging.info('dump children indices with distance (path length from root to child): ' + str(
                        current_depth) + ' ...')
                    current_indices.dump(out_filename + '.children.depth' + str(current_depth) + '.batch' + str(offset))
                    prev_end = end
                # remove processed batch file
                os.remove(out_filename + '.children.batch' + str(offset))
            # not yet used
            #seq_data = np.load(out_filename + '.parent.batch' + str(offset))
            #child_idx_offset += len(seq_data)
            ##


def collected_shuffled_child_indices(out_filename, max_depth, dump=False):
    logging.info('create shuffled child indices ...')
    # children_depth_files = fnmatch.filter(os.listdir(parent_dir), ntpath.basename(out_filename) + '.children.depth*')
    collected_child_indices = np.zeros(shape=(0, 3), dtype=np.int32)
    for current_depth in range(1, max_depth + 1):
        if not os.path.isfile(out_filename + '.children.depth' + str(current_depth) + '.collected'):
            #logging.info('load: ' + out_filename + '.children.depth' + str(current_depth))
            current_depth_indices = np.load(out_filename + '.children.depth' + str(current_depth))
            current_depth_indices = np.pad(current_depth_indices, ((0, 0), (0, 1)),
                                           'constant', constant_values=((0, 0), (0, current_depth)))
            collected_child_indices = np.append(collected_child_indices, current_depth_indices, axis=0)
            np.random.shuffle(collected_child_indices)
            if dump:
                # TODO: re-add! (crashes, files to big? --> cpickle size constraint! (2**32 -1))
                collected_child_indices.dump(out_filename + '.children.depth' + str(current_depth) + '.collected')
            logging.info('depth: ' + str(current_depth) + ', collected_size: ' + str(len(collected_child_indices)))
        else:
            collected_child_indices = np.load(out_filename + '.children.depth' + str(current_depth) + '.collected')
    return collected_child_indices


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


def merge_numpy_batch_files(batch_file_name, parent_dir, expected_count=None, overwrite=False):
    logging.info('concatenate batches: '+batch_file_name)
    #out_fn = batch_file_name.replace('.batch*', '', 1)
    if os.path.isfile(batch_file_name) and not overwrite:
        return np.load(batch_file_name)
    batch_file_names = fnmatch.filter(os.listdir(parent_dir), batch_file_name + '.batch*')
    batch_file_names = sorted(batch_file_names, key=lambda s: int(s[len(batch_file_name + '.batch'):]))
    if len(batch_file_names) == 0:
        return None
    if expected_count is not None and len(batch_file_names) != expected_count:
        return None
    l = []
    for fn in batch_file_names:
        l.append(np.load(os.path.join(parent_dir, fn)))
    concatenated = np.concatenate(l, axis=0)

    concatenated.dump(os.path.join(parent_dir, batch_file_name))
    for fn in batch_file_names:
        os.remove(os.path.join(parent_dir, fn))
    return concatenated


def sequence_node_to_arrays(seq_tree):
    current_data = []
    current_parents = []
    children_roots = []
    for child in seq_tree['children']:
        child_data, child_parents = sequence_node_to_arrays(child)
        current_data.extend(child_data)
        current_parents.extend(child_parents)
        children_roots.append(len(current_data) - 1)
    for child_root in children_roots:
        current_parents[child_root] = len(current_data) - child_root
    # head is the last element
    current_data.append(seq_tree['head'])
    current_parents.append(0)

    return current_data, current_parents
