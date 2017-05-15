from __future__ import print_function
import numpy as np
import pickle
import os

from tools import fn_timer, getOrAdd
import tools
import constants
import sequence_node_pb2


@fn_timer
def get_word_embeddings(vocab):
    #vecs = np.ndarray(shape=(len(vocab)+1, vocab.vectors_length), dtype=np.float32)
    vecs = np.ndarray(shape=(len(vocab), vocab.vectors_length), dtype=np.float32)
    #vecs[constants.NOT_IN_WORD_DICT] = np.zeros(vocab.vectors_length)
    m = {}
    #i = 1
    i = 0
    for lexeme in vocab:
        m[lexeme.orth] = i
        vecs[i] = lexeme.vector
        i += 1
    # add manual vocab
    for k in constants.vocab_manual.keys():
        tools.getOrAdd(m, k)
    return vecs, m


# embeddings for:
# word
def process_sentence2(sentence, parsed_data, data_maps, dict_unknown=None):
    sen_data = list()
    sen_parents = list()
    root_offset = (sentence.root.i - sentence.start)
    for i in range(sentence.start, sentence.end):

        token = parsed_data[i]
        parent_offset = token.head.i - i
        # add word embedding
        sen_data.append(getOrAdd(data_maps, token.orth, dict_unknown))
        sen_parents.append(parent_offset)

    return sen_data, sen_parents, root_offset

# embeddings for:
# word, edge
def process_sentence3(sentence, parsed_data, data_maps, dict_unknown=None):
    sen_data = list()
    sen_parents = list()
    root_offset = (sentence.root.i - sentence.start) * 2
    for i in range(sentence.start, sentence.end):

        # get current token
        token = parsed_data[i]
        parent_offset = token.head.i - i
        # add word embedding
        sen_data.append(getOrAdd(data_maps, token.orth, dict_unknown))
        sen_parents.append(parent_offset * 2)
        # add edge type embedding
        sen_data.append(getOrAdd(data_maps, token.dep, dict_unknown))
        sen_parents.append(-1)

    return sen_data, sen_parents, root_offset

# embeddings for:
# word, word embedding, edge, edge embedding
def process_sentence4(sentence, parsed_data, data_maps, dict_unknown=None):
    sen_data = list()
    sen_parents = list()
    root_offset = (sentence.root.i - sentence.start) * 4
    for i in range(sentence.start, sentence.end):

        token = parsed_data[i]
        parent_offset = token.head.i - i
        # add word embedding
        sen_data.append(getOrAdd(data_maps, token.orth, dict_unknown))
        sen_parents.append(parent_offset * 4)
        # add word embedding embedding
        sen_data.append(getOrAdd(data_maps, constants.WORD_EMBEDDING, dict_unknown))
        sen_parents.append(-1)
        # add edge type embedding
        sen_data.append(getOrAdd(data_maps, token.dep, dict_unknown))
        sen_parents.append(-2)
        # add edge type embedding embedding
        sen_data.append(getOrAdd(data_maps, constants.EDGE_EMBEDDING, dict_unknown))
        sen_parents.append(-1)

    return sen_data, sen_parents, root_offset

# embeddings for:
# words, edges, entity type (if !=0)
def process_sentence5(sentence, parsed_data, data_maps, dict_unknown=None):
    sen_data = list()
    sen_parents = list()
    sen_a = list()
    sen_offsets = list()

    last_offset = 0
    for i in range(sentence.start, sentence.end):
        token = parsed_data[i]
        parent_offset = token.head.i - i
        # add word embedding
        sen_data.append(getOrAdd(data_maps, token.orth, dict_unknown))
        sen_parents.append(parent_offset)
        # additional data for this token
        a_data = list()
        a_parents = list()
        # add edge type embedding
        a_data.append(getOrAdd(data_maps, token.dep, dict_unknown))
        a_parents.append(-1)
        # add entity type
        if token.ent_type != 0:
            a_data.append(getOrAdd(data_maps, token.ent_type, dict_unknown))
            a_parents.append(-2)
        sen_a.append((a_data, a_parents))
        # count additional data for every main data point
        current_offset = last_offset + len(a_data)
        sen_offsets.append(current_offset)
        last_offset = current_offset

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

    return result_data, result_parents, root_offset


# embeddings for:
# words, word type, edges, edge type, entity type (if !=0), entity type type
def process_sentence6(sentence, parsed_data, data_maps, dict_unknown=None):
    sen_data = list()
    sen_parents = list()
    sen_a = list()
    sen_offsets = list()

    last_offset = 0
    for i in range(sentence.start, sentence.end):
        token = parsed_data[i]
        parent_offset = token.head.i - i
        # add word embedding
        sen_data.append(getOrAdd(data_maps, token.orth, dict_unknown))
        sen_parents.append(parent_offset)
        # additional data for this token
        a_data = list()
        a_parents = list()

        # add word type type embedding
        a_data.append(getOrAdd(data_maps, constants.WORD_EMBEDDING, dict_unknown))
        a_parents.append(-1)
        # add edge type embedding
        a_data.append(getOrAdd(data_maps, token.dep, dict_unknown))
        a_parents.append(-2)
        # add edge type type embedding
        a_data.append(getOrAdd(data_maps, constants.EDGE_EMBEDDING, dict_unknown))
        a_parents.append(-1)
        # add entity type
        if token.ent_type != 0:
            a_data.append(getOrAdd(data_maps, token.ent_type, dict_unknown))
            a_parents.append(-4)
            a_data.append(getOrAdd(data_maps, constants.ENTITY_TYPE_EMBEDDING, dict_unknown))
            a_parents.append(-1)
        sen_a.append((a_data, a_parents))
        # count additional data for every main data point
        current_offset = last_offset + len(a_data)
        sen_offsets.append(current_offset)
        last_offset = current_offset

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

    return result_data, result_parents, root_offset


# embeddings for:
# words, edges, entity type (if !=0),
# lemma (if different), pos-tag
def process_sentence7(sentence, parsed_data, data_maps, dict_unknown=None):
    sen_data = list()
    sen_parents = list()
    sen_a = list()
    sen_offsets = list()

    last_offset = 0
    for i in range(sentence.start, sentence.end):
        token = parsed_data[i]
        parent_offset = token.head.i - i
        # add word embedding
        sen_data.append(getOrAdd(data_maps, token.orth, dict_unknown))
        sen_parents.append(parent_offset)
        # additional data for this token
        a_data = list()
        a_parents = list()

        # add edge type embedding
        a_data.append(getOrAdd(data_maps, token.dep, dict_unknown))
        a_parents.append(-len(a_data))
        # add pos-tag type embedding
        a_data.append(getOrAdd(data_maps, token.tag, dict_unknown))
        a_parents.append(-len(a_data))

        # add entity type embedding
        if token.ent_type != 0:
            a_data.append(getOrAdd(data_maps, token.ent_type, dict_unknown))
            a_parents.append(-len(a_data))
        # add lemma type embedding
        if token.lemma != token.orth:
            a_data.append(getOrAdd(data_maps, token.lemma, dict_unknown))
            a_parents.append(-len(a_data))
        sen_a.append((a_data, a_parents))
        # count additional data for every main data point
        current_offset = last_offset + len(a_data)
        sen_offsets.append(current_offset)
        last_offset = current_offset

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

    return result_data, result_parents, root_offset


# embeddings for:
# words, word type, edges, edge type, entity type (if !=0), entity type type,
# lemma (if different), lemma type, pos-tag, pos-tag type
def process_sentence8(sentence, parsed_data, data_maps, dict_unknown=None):
    sen_data = list()
    sen_parents = list()
    sen_a = list()
    sen_offsets = list()

    last_offset = 0
    for i in range(sentence.start, sentence.end):
        token = parsed_data[i]
        parent_offset = token.head.i - i
        # add word embedding
        sen_data.append(getOrAdd(data_maps, token.orth, dict_unknown))
        sen_parents.append(parent_offset)
        # additional data for this token
        a_data = list()
        a_parents = list()

        # add word type type embedding
        a_data.append(getOrAdd(data_maps, constants.WORD_EMBEDDING, dict_unknown))
        a_parents.append(-len(a_data))
        # add edge type embedding
        a_data.append(getOrAdd(data_maps, token.dep, dict_unknown))
        a_parents.append(-len(a_data))
        # add edge type type embedding
        a_data.append(getOrAdd(data_maps, constants.EDGE_EMBEDDING, dict_unknown))
        a_parents.append(-1)
        # add pos-tag type embedding
        a_data.append(getOrAdd(data_maps, token.tag, dict_unknown))
        a_parents.append(-len(a_data))
        # add pos-tag type type embedding
        a_data.append(getOrAdd(data_maps, constants.POS_TAG_EMBEDDING, dict_unknown))
        a_parents.append(-1)

        # add entity type embedding
        if token.ent_type != 0:
            a_data.append(getOrAdd(data_maps, token.ent_type, dict_unknown))
            a_parents.append(-len(a_data))
            # add entity type type embedding
            a_data.append(getOrAdd(data_maps, constants.ENTITY_TYPE_EMBEDDING, dict_unknown))
            a_parents.append(-1)
        # add lemma type embedding
        if token.lemma != token.orth:
            a_data.append(getOrAdd(data_maps, token.lemma, dict_unknown))
            a_parents.append(-len(a_data))
            # add lemma type type embedding
            a_data.append(getOrAdd(data_maps, constants.LEMMA_EMBEDDING, dict_unknown))
            a_parents.append(-1)
        sen_a.append((a_data, a_parents))
        # count additional data for every main data point
        current_offset = last_offset + len(a_data)
        sen_offsets.append(current_offset)
        last_offset = current_offset

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

    return result_data, result_parents, root_offset


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


def read_data(reader, sentence_processor, parser, data_maps, args={}, tree_mode=None, expand_dict=True):

    # ids of the dictionaries to query the data point referenced by seq_data
    # at the moment there is just one: WORD_EMBEDDING
    #seq_types = list()
    # ids (dictionary) of the data points in the dictionary specified by seq_types
    seq_data = list()
    # ids (dictionary) of relations to the heads (parents)
    #seq_edges = list()
    # ids (sequence) of the heads (parents)
    seq_parents = list()
    prev_root = None

    #roots = list()

    if expand_dict:
        unknown_default = None
    else:
        unknown_default = constants.UNKNOWN_EMBEDDING

    for parsed_data in parser.pipe(reader(**args), n_threads=4, batch_size=1000):
        prev_root = None
        for sentence in parsed_data.sents:
            processed_sen = sentence_processor(sentence, parsed_data, data_maps, unknown_default)
            # skip not processed sentences (see process_sentence)
            if processed_sen is None:
                continue

            sen_data, sen_parents, root_offset = processed_sen

            current_root = len(seq_data) + root_offset

            seq_parents += sen_parents
            seq_data += sen_data

            if prev_root is not None:
                seq_parents[prev_root] = current_root - prev_root
            prev_root = current_root
        #roots.append(prev_root)

    data = np.array(seq_data)
    parents = np.array(seq_parents)
    #root = prev_root

    # overwrite structure, if a special mode is set
    if tree_mode is not None:
        # collect first elements of the trees
        prev_root = -1
        first_in_tree = list()
        for idx in range(len(data)):
            root = get_root(parents, idx)
            if root != prev_root:
                first_in_tree.append(idx)
                prev_root = root
        # remove false first element and add dummy at the end
        # as a result (first_in_tree -1) point to the last elements
        if len(first_in_tree) > 0:
            del first_in_tree[0]
            first_in_tree.append(len(data))

        if tree_mode == 'sequence':
            parents = np.ones(parents.shape, dtype=parents.dtype)
            for idx in first_in_tree:
                parents[idx-1] = 0
            #if len(parents) > 0:
            #    parents[-1] = 0
                #root = len(parents) - 1
            #else:
            #    root = 0
        elif tree_mode == 'aggregate':
            TERMINATOR_id = getOrAdd(data_maps, constants.TERMINATOR_EMBEDDING, unknown_default)
            parents = np.array([], dtype=np.int32)
            last_first = 0
            count = 0
            print(first_in_tree)
            for idx in first_in_tree:
                print('idx: ' + str(idx))
                #parents[idx-1] = 0
                data = np.insert(data, idx+count, TERMINATOR_id)

                size = idx - last_first + 1
                print('idx: ' + str(idx) + ', size: ' + str(size))
                parents = np.append(parents, np.array(list(reversed(range(size)))))
                last_first = idx
                count += 1
            #data = np.append(data, TERMINATOR_id)
            #parents = np.array(list(reversed(range(len(data)))))
            #root = max(len(data) - 1, 0)
        else:
            raise NameError('unknown tree_mode: '+tree_mode)

    return data, parents#, root #, np.array(seq_edges)#, dep_map


def addMissingEmbeddings(seq_data, embeddings):
    # get current count of embeddings
    l = embeddings.shape[0]
    # get all ids without embedding
    new_embedding_ids = sorted(list({elem for elem in seq_data if elem >= l}))
    # no unknown embeddings
    if len(new_embedding_ids) == 0:
        return

    #print('new_embedding_ids: ' + str(new_embedding_ids))
    #print('new_embedding_ids[0]: ' + str(new_embedding_ids[0]))
    #print('l: ' + str(l))
    # check integrity
    assert new_embedding_ids[0] == l, str(new_embedding_ids[0]) + ' != ' + str(l)
    assert new_embedding_ids[-1] == l + len(new_embedding_ids) - 1, str(new_embedding_ids[-1]) + ' != ' + str(l + len(new_embedding_ids) - 1)

    # get mean of existing embeddings
    #mean = np.mean(embeddings, axis=1)

    new_embeddings = np.lib.pad(embeddings, ((0, len(new_embedding_ids)), (0, 0)), 'mean')
    return new_embeddings, len(new_embedding_ids)


def children_and_roots(seq_parents):
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


# Build a sequence_tree from a data and a parents sequence.
# All roots are children of a headless node.
def build_sequence_tree(seq_data, children, root, seq_tree = None):
    # assume, all parents are inside this array!
    # collect children
    #children, roots = children_and_roots(seq_parents)

    """Recursively build a tree of SequenceNode_s"""
    def build(seq_node, seq_data, children, pos):
        seq_node.head = seq_data[pos]
        if pos in children:
            for child_pos in children[pos]:
                build(seq_node.children.add(), seq_data, children, child_pos)

    if seq_tree is None:
        seq_tree = sequence_node_pb2.SequenceNode()
    build(seq_tree, seq_data, children, root)

    #seq_trees = []
    #seq_tree = sequence_node_pb2.SequenceNode()
    #for root in roots:
    #    root_tree = sequence_node_pb2.SequenceNode() # seq_tree.children.add() #sequence_node_pb2.SequenceNode()
    #    build(root_tree, seq_data, children, root)

    return seq_tree


def build_sequence_tree_from_str(str_, sentence_processor, parser, data_maps, seq_tree=None, tree_mode=None, expand_dict=True):
    seq_data, seq_parents = read_data(identity_reader, sentence_processor, parser, data_maps,
                                            args={'content': str_}, tree_mode=tree_mode, expand_dict=expand_dict)
    children, roots = children_and_roots(seq_parents)
    return build_sequence_tree(seq_data, children, roots[0], seq_tree)


def create_or_read_dict(fn, vocab):
    if os.path.isfile(fn+'.vecs'):
        print('load vecs from file: '+fn + '.vecs ...')
        v = np.load(fn+'.vecs')
        print('read mapping from file: ' + fn + '.mapping ...')
        m = pickle.load(open(fn+'.mapping', "rb"))
        print('vecs.shape: ' + str(v.shape))
        print('len(mapping): ' + str(len(m)))
    else:
        out_dir = os.path.abspath(os.path.join(fn, os.pardir))
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        print('extract word embeddings from spaCy ...')
        v, m = get_word_embeddings(vocab)
        print('vecs.shape: ' + str(v.shape))
        print('len(mapping): ' + str(len(m)))
        print('dump vecs to: ' + fn + '.vecs ...')
        v.dump(fn + '.vecs')
        print('dump mappings to: ' + fn + '.mapping ...')
        with open(fn + '.mapping', "wb") as f:
            pickle.dump(m, f)
    return v, m





