from __future__ import print_function
import csv
from sys import maxsize
import numpy as np

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
    return vecs, m


def process_sentence(sentence, parsed_data, max_forest_count, data_embedding_maps):

    # see read_data
    sen_data = list()
    sen_types = list()
    sen_parents = list()
    sen_edges = list()
    for i in range(sentence.start, sentence.end):
        # count roots (head points to self) and temp roots (head points to future token)
        forest_count = [parsed_data[j].head.i == j or parsed_data[j].head.i > i for j in
                        range(sentence.start, i + 1)].count(True)
        if forest_count > max_forest_count:
            return None

        data_type = constants.WORD_EMBEDDING
        sen_types.append(data_type)
        token = parsed_data[i]
        sen_parents.append(token.head.i - i)
        sen_edges.append(token.dep)
        try:
            x = data_embedding_maps[data_type][token.orth]
        # word doesnt occur in dictionary
        except KeyError:
            x = constants.NOT_IN_WORD_DICT
        sen_data.append(x)

        # collect dependency labels for human readable mapping
        # dep_map[token.dep] = token.dep_
    return sen_data, sen_types, sen_parents, sen_edges


def process_sentence2(sentence, parsed_data, data_maps):
    sen_data = list()
    sen_parents = list()
    root_offset = (sentence.root.i - sentence.start) * 2
    for i in range(sentence.start, sentence.end):

        token = parsed_data[i]
        parent_offset = token.head.i - i
        # add word embedding
        sen_data.append(getOrAdd(data_maps, token.orth))
        sen_parents.append(parent_offset * 2)
        # add edge type embedding
        sen_data.append(getOrAdd(data_maps, token.dep))
        sen_parents.append(-1)

    return sen_data, sen_parents, root_offset


def process_sentence3(sentence, parsed_data, data_maps):
    sen_data = list()
    sen_parents = list()
    root_offset = (sentence.root.i - sentence.start) * 2
    for i in range(sentence.start, sentence.end):

        # get current token
        token = parsed_data[i]
        parent_offset = token.head.i - i
        # add word embedding
        sen_data.append(getOrAdd(data_maps, token.orth))
        sen_parents.append(parent_offset * 2)
        # add edge type embedding
        sen_data.append(getOrAdd(data_maps, token.dep))
        sen_parents.append(-1)

    return sen_data, sen_parents, root_offset


def process_sentence5(sentence, parsed_data, data_maps):
    sen_data = list()
    sen_parents = list()
    sen_a = list()
    sen_offsets = list()

    last_offset = 0
    for i in range(sentence.start, sentence.end):
        token = parsed_data[i]
        parent_offset = token.head.i - i
        # add word embedding
        sen_data.append(getOrAdd(data_maps, token.orth))
        sen_parents.append(parent_offset)
        # additional data for this token
        a_data = list()
        a_parents = list()
        # add edge type embedding
        a_data.append(getOrAdd(data_maps, token.dep))
        a_parents.append(-1)
        # add entity type
        if token.ent_type != 0:
            a_data.append(getOrAdd(data_maps, token.ent_type))
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


def process_sentence4(sentence, parsed_data, data_maps):
    sen_data = list()
    sen_parents = list()
    root_offset = (sentence.root.i - sentence.start) * 4
    for i in range(sentence.start, sentence.end):

        token = parsed_data[i]
        parent_offset = token.head.i - i
        # add word embedding
        sen_data.append(getOrAdd(data_maps, token.orth))
        sen_parents.append(parent_offset * 4)
        # add word embedding embedding
        sen_data.append(getOrAdd(data_maps, constants.WORD_EMBEDDING))
        sen_parents.append(-1)
        # add edge type embedding
        sen_data.append(getOrAdd(data_maps, token.dep))
        sen_parents.append(-2)
        # add edge type embedding embedding
        sen_data.append(getOrAdd(data_maps, constants.EDGE_EMBEDDING))
        sen_parents.append(-1)

    return sen_data, sen_parents, root_offset


def dummy_str_reader():
    yield u'I like RTRC!'


def string_reader(content):
    yield content.decode('utf-8')


def articles_from_csv_reader(filename, max_articles=100):
    csv.field_size_limit(maxsize)
    print('parse', max_articles, 'articles...')
    with open(filename, 'rb') as csvfile:
        reader = csv.DictReader(csvfile, fieldnames=['article-id', 'content'])
        i = 0
        for row in reader:
            if i >= max_articles:
                break
            if (i * 100) % max_articles == 0:
                # sys.stdout.write("progress: %d%%   \r" % (i * 100 / max_rows))
                # sys.stdout.flush()
                print('read article:', row['article-id'], '... ', i * 100 / max_articles, '%')
            i += 1
            yield row['content'].decode('utf-8')


#@fn_timer
def read_data(reader, nlp, data_maps, max_forest_count=10, max_sen_length=75, args={}):

    # ids of the dictionaries to query the data point referenced by seq_data
    # at the moment there is just one: WORD_EMBEDDING
    seq_types = list()
    # ids (dictionary) of the data points in the dictionary specified by seq_types
    seq_data = list()
    # ids (dictionary) of relations to the heads (parents)
    seq_edges = list()
    # ids (sequence) of the heads (parents)
    seq_parents = list()

    offset = 0
    for parsed_data in nlp.pipe(reader(**args), n_threads=4, batch_size=1000):
        skipped_count = 0
        prev_root = None
        for sentence in parsed_data.sents:
            # skip too long sentences
            if len(sentence) > max_sen_length:
                skipped_count += len(sentence)
                continue
            processed_sen = process_sentence(sentence, parsed_data, max_forest_count, data_maps)
            # skip not processed sentences (see process_sentence)
            if processed_sen is None:
                skipped_count += len(sentence)
                continue

            sen_data, sen_types, sen_parents, sen_edges = processed_sen
            seq_parents += sen_parents
            seq_edges += sen_edges
            seq_data += sen_data
            seq_types += sen_types

            if prev_root is not None:
                seq_parents[prev_root] = sentence.root.i - skipped_count + offset - prev_root
            prev_root = sentence.root.i - skipped_count + offset
        offset = len(seq_data)

    if constants.EDGE_EMBEDDING not in data_maps:
        data_maps[constants.EDGE_EMBEDDING] = {}
        data_maps[constants.EDGE_EMBEDDING][constants.INTER_TREE] = constants.INTER_TREE

    # collect edge labels
    dep_map = data_maps[constants.EDGE_EMBEDDING]
    for i in range(len(seq_edges)):
        edge = seq_edges[i]
        if edge not in dep_map:
            dep_map[edge] = len(dep_map)
            seq_edges[i] = dep_map[edge]
        else:
            seq_edges[i] = dep_map[edge]

    return np.array(seq_data), np.array(seq_types), np.array(seq_parents), np.array(seq_edges)#, dep_map


def read_data2(reader, sentence_processor, parser, data_maps, max_forest_count=10, max_sen_length=75, args={}):

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

    for parsed_data in parser.pipe(reader(**args), n_threads=4, batch_size=1000):
        prev_root = None
        for sentence in parsed_data.sents:
            # skip too long sentences
            if len(sentence) > max_sen_length:
                continue
            processed_sen = sentence_processor(sentence, parsed_data, data_maps)
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

    return np.array(seq_data), np.array(seq_parents), prev_root #, np.array(seq_edges)#, dep_map


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


def build_sequence_tree_from_str(str, sentence_processor, parser, data_maps, seq_tree = None):
    seq_data, seq_parents, root = read_data2(string_reader, sentence_processor, parser, data_maps,
                                             args={'content': str})
    children, roots = children_and_roots(seq_parents)
    return build_sequence_tree(seq_data, children, root, seq_tree)





