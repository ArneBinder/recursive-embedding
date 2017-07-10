from __future__ import print_function

import csv
import logging
import os
import sys
from sys import maxsize

import tensorflow as tf

import constants
import corpus
import preprocessing

tf.flags.DEFINE_string(
    'corpus_data_input_train', '/home/arne/devel/ML/data/corpora/WIKIPEDIA/documents_utf8_filtered_20pageviews.csv',
    # '/home/arne/devel/ML/data/corpora/WIKIPEDIA/wikipedia-23886057.csv',#'/home/arne/devel/ML/data/corpora/WIKIPEDIA/documents_utf8_filtered_20pageviews.csv', # '/home/arne/devel/ML/data/corpora/SICK/sick_train/SICK_train.txt',
    'The path to the SICK train data file.')
tf.flags.DEFINE_string(
    'corpus_data_output_dir', '/media/arne/WIN/Users/Arne/ML/data/corpora/wikipedia',  # 'data/corpora/wikipedia',
    'The path to the output data files (samples, embedding vectors, mappings).')
tf.flags.DEFINE_string(
    'corpus_data_output_fn', 'WIKIPEDIA',
    'Base filename of the output data files (samples, embedding vectors, mappings).')
tf.flags.DEFINE_string(
    'init_dict_filename', None,
    # '/media/arne/WIN/Users/Arne/ML/data/corpora/wikipedia/process_sentence7/WIKIPEDIA_articles1000_maxdepth10',#None, #'data/nlp/spacy/dict',
    'The path to embedding and mapping files (without extension) to reuse them for the new corpus.')
tf.flags.DEFINE_integer(
    'max_articles', 10000,
    'How many articles to read.')
tf.flags.DEFINE_integer(
    'article_batch_size', 2500,  # 250,
    'How many articles to process in one batch.')
tf.flags.DEFINE_integer(
    'article_offset', 0,
    'How many articles to skip.')
tf.flags.DEFINE_integer(
    'max_depth', 10,
    'The maximal depth of the sequence trees.')
tf.flags.DEFINE_integer(
    'count_threshold', 2,
    'Change data types which occur less then count_threshold times to UNKNOWN')
# tf.flags.DEFINE_integer(
#    'sample_count', 14,
#    'Amount of samples per tree. This excludes the correct tree.')
tf.flags.DEFINE_string(
    'sentence_processor', 'process_sentence8',  # 'process_sentence8',#'process_sentence3',
    'Defines which NLP features are taken into the embedding trees.')
tf.flags.DEFINE_string(
    'concat_mode',
    #'sequence',
    constants.default_concat_mode,
    'How to concatenate the trees returned by one parser call (e.g. trees in one document). '
    + '"sequence" -> roots point to next root, '
    + '"aggregate" -> roots point to an added, artificial token (AGGREGATOR) in the end of the token sequence'
      'None -> do not concat at all')
tf.flags.DEFINE_string(
    'inner_concat_mode',
    #'tree',
    constants.default_inner_concat_mode,
    'How to concatenate the trees returned for one token. '
    '"tree" -> use dependency parse tree'
    '"sequence" -> roots point to next root, '
    '"aggregate" -> roots point to an added, artificial token (AGGREGATOR) in the end of the token sequence'
    'None -> do not concat at all')

FLAGS = tf.flags.FLAGS


def articles_from_csv_reader(filename, max_articles=100, skip=0):
    csv.field_size_limit(maxsize)
    logging.info('parse ' + str(max_articles) + ' articles...')
    with open(filename, 'rb') as csvfile:
        reader = csv.DictReader(csvfile, fieldnames=['article-id', 'content'])
        i = 0
        for row in reader:
            if skip > 0:
                skip -= 1
                continue
            if i >= max_articles:
                break
            if (i * 10) % max_articles == 0:
                # sys.stdout.write("progress: %d%%   \r" % (i * 100 / max_rows))
                # sys.stdout.flush()
                logging.info('read article: ' + row['article-id'] + '... ' + str(i * 100 / max_articles) + '%')
            i += 1
            content = row['content'].decode('utf-8')
            # cut the title (is separated by two spaces from main content)
            yield content.split('  ', 1)[1]


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
    sentence_proc = getattr(preprocessing, FLAGS.sentence_processor)
    out_dir = os.path.abspath(os.path.join(FLAGS.corpus_data_output_dir, sentence_proc.func_name))
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    out_path = os.path.join(out_dir, FLAGS.corpus_data_output_fn)
    if FLAGS.inner_concat_mode is not None:
        out_path = out_path + '_' + FLAGS.inner_concat_mode

    # out_path = out_path + '_maxdepth' + str(FLAGS.max_depth)
    out_path = out_path + '_articles' + str(FLAGS.max_articles)
    out_path = out_path + '_offset' + str(FLAGS.article_offset)

    logging.info('output base file name: ' + out_path)

    nlp = None
    nlp = corpus.convert_texts(in_filename=FLAGS.corpus_data_input_train,
                               out_filename=out_path,
                               init_dict_filename=FLAGS.init_dict_filename,
                               reader=articles_from_csv_reader,
                               sentence_processor=sentence_proc,
                               parser=nlp,
                               max_articles=FLAGS.max_articles,
                               max_depth=FLAGS.max_depth,
                               batch_size=FLAGS.article_batch_size,
                               count_threshold=FLAGS.count_threshold,
                               concat_mode=FLAGS.concat_mode,
                               inner_concat_mode=FLAGS.inner_concat_mode,
                               article_offset=FLAGS.article_offset
                               )
