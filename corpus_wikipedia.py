from __future__ import print_function
import csv
import os
from sys import maxsize
import tensorflow as tf
import numpy as np

import spacy
import preprocessing
import tools

tf.flags.DEFINE_string(
    'corpus_data_input_train', '/home/arne/devel/ML/data/corpora/documents_utf8_filtered_20pageviews.csv', # '/home/arne/devel/ML/data/corpora/SICK/sick_train/SICK_train.txt',
    'The path to the SICK train data file.')
tf.flags.DEFINE_string(
    'corpus_data_input_test', '/home/arne/devel/ML/data/corpora/SICK/sick_test_annotated/SICK_test_annotated.txt',
    'The path to the SICK test data file.')
tf.flags.DEFINE_string(
    'corpus_data_output_dir', 'data/corpora/sick',
    'The path to the output data files (samples, embedding vectors, mappings).')
tf.flags.DEFINE_string(
    'corpus_data_output_fn', 'SICK',
    'Base filename of the output data files (samples, embedding vectors, mappings).')
tf.flags.DEFINE_string(
    'dict_filename', 'data/nlp/spacy/dict',
    'The path to the output data files (samples, embedding vectors, mappings).')
tf.flags.DEFINE_integer(
    'corpus_size', -1,
    'How many samples to write. Use a negative dummy value to set no limit.')
tf.flags.DEFINE_string(
    'sentence_processor', 'process_sentence3',
    'How long to make the expression embedding vectors.')
tf.flags.DEFINE_string(
    'tree_mode',
    None,
    #'aggregate',
    #'sequence',
    'How to structure the tree. '
     + '"sequence" -> parents point to next token, '
     + '"aggregate" -> parents point to an added, artificial token (TERMINATOR) in the end of the token sequence,'
     + 'None -> use parsed dependency tree')

FLAGS = tf.flags.FLAGS


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
            content = row['content'].decode('utf-8')
            # cut the title (is separated by two spaces from main content)
            yield content.split('  ', 1)[1]


@tools.fn_timer
def convert_wikipedia(in_filename, out_filename, sentence_processor, parser, mapping, max_articles=10000, tree_mode=None):
    seq_data, seq_parents = preprocessing.read_data(articles_from_csv_reader,
                                                    sentence_processor, parser, mapping,
                                                    args={
                                                        'filename': in_filename,
                                                        'max_articles': max_articles},
                                                    tree_mode=tree_mode)
    print('len(seq_data): ' + str(len(seq_data)))
    print('calc children ...')
    children, roots = preprocessing.children_and_roots(seq_parents)
    print('calc depths ...')
    depth = -np.ones(len(seq_data), dtype=np.int)
    for idx in range(len(seq_data)):
        if depth[idx] < 0:
            preprocessing.calc_depth(children, depth, idx)
    print('dummy')


if __name__ == '__main__':
    print('load spacy ...')
    nlp = spacy.load('en')
    nlp.pipeline = [nlp.tagger, nlp.entity, nlp.parser]

    vecs, mapping = preprocessing.create_or_read_dict(FLAGS.dict_filename, nlp.vocab)

    sentence_processor = getattr(preprocessing, FLAGS.sentence_processor)
    out_dir = os.path.abspath(os.path.join(FLAGS.corpus_data_output_dir, sentence_processor.func_name))
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    out_path = os.path.join(out_dir, FLAGS.corpus_data_output_fn)
    if FLAGS.tree_mode is not None:
        out_path = out_path + '_' + FLAGS.tree_mode

    print('parse train data ...')
    convert_wikipedia(FLAGS.corpus_data_input_train,
                      out_path + '.train',
                      sentence_processor,
                      nlp,
                      mapping,
                      tree_mode=FLAGS.tree_mode)

    #print('parse train data ...')
    #convert_sick(FLAGS.corpus_data_input_train,
    #             out_path + '.train',
    #             sentence_processor,
    #             nlp,
    #             mapping,
    #             FLAGS.corpus_size,
    #             FLAGS.tree_mode)

    #print('parse test data ...')
    #convert_sick(FLAGS.corpus_data_input_test,
    #             out_path + '.test',
    #             sentence_processor,
    #             nlp,
    #             mapping,
    #             FLAGS.corpus_size,
    #             FLAGS.tree_mode)

    #print('len(mapping): ' + str(len(mapping)))

    #corpus.write_dict(out_path, mapping, nlp.vocab, constants.vocab_manual)