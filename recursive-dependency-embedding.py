from __future__ import print_function
import csv
from collections import defaultdict
from sys import exit
from sys import maxsize
from spacy.en import English

from tools import fn_timer

data_dir = '/media/arne/DATA/DEVELOPING/ML/data/'
article_count = 1


def process_token(token, plain_tokens=list()):
    return token.lemma_


def process_sentence(sentence, parsed_data, offset, max_forest_count):
    # print('sent:\t', sentence)
    # print('len:\t', len(sentence))
    # print('len2:\t', sentence.end - sentence.start)

    # print('i\t', parsed_data[sentence.root.i])
    # print('root\t', sentence.root)
    # assert (parsed_data[sentence.root.i] != sentence.root)

    sen_vecs = list()
    sen_heads = list()
    sen_edges = list()
    sen_forest_count = list()
    # print('start:\t', parsed_data[sentence.start])
    # print('end:\t', parsed_data[sentence.end-1])
    for i in range(sentence.start, sentence.end):
        # count roots (head points to self) and temp roots (head points to future token)
        forest_count = [parsed_data[j].head.i == j or parsed_data[j].head.i > i for j in
                        range(sentence.start, i + 1)].count(True)
        if forest_count > max_forest_count:
            return None
        sen_heads.append(parsed_data[i].head.i - offset)
        sen_edges.append(parsed_data[i].dep)
        sen_vecs.append(parsed_data[i].vector)
        sen_forest_count.append(forest_count)

    return sen_vecs, sen_heads, sen_edges, sen_forest_count


csv.field_size_limit(maxsize)


@fn_timer
def read_data_csv(filename, max_rows=100, max_forest_count=10, max_sen_length=75):
    print('parse', max_rows, 'articles')
    parser = English()
    plain_tokens = list()
    seq_vecs = list()
    seq_edges = list()
    seq_heads = list()
    seq_forest_count = list()

    sen_lengths = defaultdict(int)
    with open(filename, 'rb') as csvfile:
        reader = csv.DictReader(csvfile, fieldnames=['article-id', 'content'])
        article_count = 0
        for row in reader:
            if article_count >= max_rows:
                break
            if (article_count * 100) % max_rows == 0:
                # sys.stdout.write("progress: %d%%   \r" % (article_count * 100 / max_rows))
                # sys.stdout.flush()
                print('parse article:', row['article-id'], '... ', article_count * 100 / max_rows, '%')
            content = row['content'].decode('utf-8')

            # parsing
            parsed_data = parser(content)

            offset = 0
            prev_root = None
            # prev_offset = 0
            for sentence in parsed_data.sents:
                if len(sentence) > max_sen_length:
                    offset += len(sentence)

                    continue

                sen_lengths[len(sentence)] += 1
                processed_sen = process_sentence(sentence, parsed_data, offset, max_forest_count)
                if processed_sen is not None:
                    sen_vecs, sen_heads, sen_edges, sen_forest_count = processed_sen
                    seq_heads += sen_heads
                    seq_edges += sen_edges
                    seq_vecs += sen_vecs
                    seq_forest_count += sen_forest_count
                    if prev_root is not None:
                        seq_heads[prev_root] = sentence.root.i - offset
                    prev_root = sentence.root.i - offset
                    # prev_offset = offset
                # if sentence was skipped
                else:
                    offset += len(sentence)

            article_count += 1
    print('sentence lengths:', sen_lengths)
    return plain_tokens, seq_vecs, seq_heads, seq_edges, seq_forest_count


plain_tokens, seq_vecs, seq_heads, seq_edges, seq_forest_count = read_data_csv(
    data_dir + 'corpora/documents_utf8_filtered_20pageviews.csv', article_count)

print('max:', max(seq_forest_count))
# seq_forest_count.sort(reverse=True)

d = defaultdict(int)
for c in seq_forest_count:
    d[c] += 1
print('forest counts:', d)

