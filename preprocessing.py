from __future__ import print_function
import csv
from sys import maxsize
import numpy as np

from tools import fn_timer
import constants


@fn_timer
def get_word_embeddings(vocab):
    vecs = np.ndarray(shape=(len(vocab)+1, vocab.vectors_length), dtype=np.float32)
    m_human = [u'UNKNOWN']
    vecs[0] = np.zeros(vocab.vectors_length)
    m = {}
    i = 1
    for lexeme in vocab:
        m_human.append(lexeme.orth_)
        m[lexeme.orth] = i
        vecs[i] = lexeme.vector
        i += 1
    return vecs, m, m_human


def process_sentence(sentence, parsed_data, max_forest_count, dep_map, data_embedding_maps):

    # see read_data
    sen_data = list()
    sen_types = list()
    sen_heads = list()
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
        sen_heads.append(token.head.i - i)
        sen_edges.append(token.dep)
        try:
            x = data_embedding_maps[data_type][token.orth]
        # word doesnt occur in dictionary
        except KeyError:
            x = 0
        sen_data.append(x)

        # collect dependency labels for human readable mapping
        dep_map[token.dep] = token.dep_
    return sen_data, sen_types, sen_heads, sen_edges


def dummy_str_reader():
    yield u'I like RTRC!'


def articles_from_csv_reader(filename, max_articles=100):
    csv.field_size_limit(maxsize)
    print('parse', max_articles, 'articles')
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


@fn_timer
def read_data(reader, nlp, data_embedding_maps, max_forest_count=10, max_sen_length=75, args={}):

    dep_map = {}

    # ids of the dictionaries to query the data point referenced by seq_data
    # at the moment there is just one: WORD_EMBEDDING
    seq_types = list()
    # ids (dictionary) of the data points in the dictionary specified by seq_types
    seq_data = list()
    # ids (dictionary) of relations to the heads (parents)
    seq_edges = list()
    # ids (sequence) of the heads (parents)
    seq_heads = list()

    offset = 0
    for parsed_data in nlp.pipe(reader(**args), n_threads=4, batch_size=1000):
        skipped_count = 0
        prev_root = None
        for sentence in parsed_data.sents:
            # skip too long sentences
            if len(sentence) > max_sen_length:
                skipped_count += len(sentence)
                continue
            processed_sen = process_sentence(sentence, parsed_data, max_forest_count, dep_map, data_embedding_maps)
            # skip not processed sentences (see process_sentence)
            if processed_sen is None:
                skipped_count += len(sentence)
                continue

            sen_data, sen_types, sen_heads, sen_edges = processed_sen
            seq_heads += sen_heads
            seq_edges += sen_edges
            seq_data += sen_data
            seq_types += sen_types

            if prev_root is not None:
                seq_heads[prev_root] = sentence.root.i - skipped_count + offset - prev_root
            prev_root = sentence.root.i - skipped_count + offset
        offset = len(seq_data)

    # re-map edge labels
    # retain 0 for inter tree "edge"
    mapping = dict(zip(dep_map.iterkeys(), range(1, len(dep_map)+1)))
    for i in range(len(seq_edges)):
        seq_edges[i] = mapping[seq_edges[i]]
    dep_map = {mapping[key]: value for key, value in dep_map.iteritems()}
    dep_map[0] = u'INTTREE'

    return (seq_data, seq_types, seq_heads, seq_edges), dep_map


