from __future__ import print_function

import csv
import logging
import ntpath
import os
import sys
from sys import maxsize

import numpy as np
import spacy
import tensorflow as tf

import corpus
import preprocessing
import tools

tf.flags.DEFINE_string(
    'corpus_data_input_train', '/home/arne/devel/ML/data/corpora/WIKIPEDIA/documents_utf8_filtered_20pageviews.csv', #'/home/arne/devel/ML/data/corpora/WIKIPEDIA/wikipedia-23886057.csv',#'/home/arne/devel/ML/data/corpora/WIKIPEDIA/documents_utf8_filtered_20pageviews.csv', # '/home/arne/devel/ML/data/corpora/SICK/sick_train/SICK_train.txt',
    'The path to the SICK train data file.')
tf.flags.DEFINE_string(
    'corpus_data_output_dir', '/media/arne/WIN/Users/Arne/ML/data/corpora/wikipedia',#'data/corpora/wikipedia',
    'The path to the output data files (samples, embedding vectors, mappings).')
tf.flags.DEFINE_string(
    'corpus_data_output_fn', 'WIKIPEDIA',
    'Base filename of the output data files (samples, embedding vectors, mappings).')
tf.flags.DEFINE_string(
    'init_dict_filename', None, #'/media/arne/WIN/Users/Arne/ML/data/corpora/wikipedia/process_sentence7/WIKIPEDIA_articles1000_maxdepth10',#None, #'data/nlp/spacy/dict',
    'The path to embedding and mapping files (without extension) to reuse them for the new corpus.')
tf.flags.DEFINE_integer(
    'max_articles', 10000,
    'How many articles to read.')
tf.flags.DEFINE_integer(
    'article_batch_size', 1000, #250,
    'How many articles to process in one batch.')
tf.flags.DEFINE_integer(
    'max_depth', 10,
    'The maximal depth of the sequence trees.')
tf.flags.DEFINE_integer(
    'count_threshold', 3,
    'Change data types which occur less then count_threshold times to UNKNOWN')
#tf.flags.DEFINE_integer(
#    'sample_count', 14,
#    'Amount of samples per tree. This excludes the correct tree.')
tf.flags.DEFINE_string(
    'sentence_processor', 'process_sentence7', #'process_sentence8',#'process_sentence3',
    'Defines which NLP features are taken into the embedding trees.')
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


@tools.fn_timer
def convert_wikipedia(in_filename, out_filename, init_dict_filename, sentence_processor, parser, #mapping, vecs,
                      max_articles=10000, max_depth=10, batch_size=100, tree_mode=None):
    parent_dir = os.path.abspath(os.path.join(out_filename, os.pardir))
    out_base_name = ntpath.basename(out_filename)
    if not os.path.isfile(out_filename+'.data')\
            or not os.path.isfile(out_filename + '.parent')\
            or not os.path.isfile(out_filename + '.vec')\
            or not os.path.isfile(out_filename + '.depth')\
            or not os.path.isfile(out_filename + '.count'):
        if not parser:
            logging.info('load spacy ...')
            parser = spacy.load('en')
            parser.pipeline = [parser.tagger, parser.entity, parser.parser]
        # get vecs and types and save it at out_filename
        if init_dict_filename:
            vecs, types = corpus.create_or_read_dict(init_dict_filename)
            corpus.write_dict(out_filename, vecs=vecs, types=types)
        else:
            corpus.create_or_read_dict(out_filename, vocab=parser.vocab, dont_read=True)

        # parse
        parse_articles(out_filename, in_filename, parser, sentence_processor, max_articles, batch_size, tree_mode)
        # merge batches
        preprocessing.merge_numpy_batch_files(out_base_name + '.parent', parent_dir)
        preprocessing.merge_numpy_batch_files(out_base_name + '.depth', parent_dir)
        seq_data = preprocessing.merge_numpy_batch_files(out_base_name + '.data', parent_dir)
    else:
        logging.info('load data from file: ' + out_filename + '.data ...')
        seq_data = np.load(out_filename + '.data')

    if not os.path.isfile(out_filename + '.converter') or not os.path.isfile(out_filename + '.new_idx_unknown'):
        vecs, types = corpus.create_or_read_dict(out_filename, parser.vocab)
        # sort and filter vecs/mappings by counts
        converter, vecs, counts, types, new_idx_unknown = preprocessing.sort_and_cut_and_fill_dict(seq_data, vecs, types,
                                                                                 count_threshold=FLAGS.count_threshold)
        # write out vecs, mapping and tsv containing strings
        corpus.write_dict(out_filename, vecs=vecs, types=types, counts=counts)
        logging.info('dump converter to: ' + out_filename + '.converter ...')
        converter.dump(out_filename + '.converter')
        logging.info('dump new_idx_unknown to: ' + out_filename + '.new_idx_unknown ...')
        np.array(new_idx_unknown).dump(out_filename + '.new_idx_unknown')

    if os.path.isfile(out_filename + '.converter') and os.path.isfile(out_filename + '.new_idx_unknown'):
        logging.info('load converter from file: ' + out_filename + '.converter ...')
        converter = np.load(out_filename + '.converter')
        logging.info('load new_idx_unknown from file: ' + out_filename + '.new_idx_unknown ...')
        new_idx_unknown = np.load(out_filename + '.new_idx_unknown')
        logging.info('convert data ...')
        count_unknown = 0
        for i, d in enumerate(seq_data):
            if converter[d] >= 0:
                seq_data[i] = converter[d]
            # set to UNKNOWN
            else:
                seq_data[i] = new_idx_unknown  # 0 #new_idx_unknown #mapping[constants.UNKNOWN_EMBEDDING]
                count_unknown += 1
        logging.info('set ' + str(count_unknown) + ' data points to UNKNOWN')

        logging.info('dump data to: ' + out_filename + '.data ...')
        seq_data.dump(out_filename + '.data')
        logging.info('delete converter and new_idx_unknown ...')
        os.remove(out_filename + '.converter')
        os.remove(out_filename + '.new_idx_unknown')

    logging.info('load depths from file: ' + out_filename + '.depth ...')
    seq_depths = np.load(out_filename + '.depth')
    preprocessing.calc_depths_collected(out_filename, parent_dir, max_depth, seq_depths)

    logging.info('load parents from file: ' + out_filename + '.parent ...')
    seq_parents = np.load(out_filename + '.parent')
    logging.info('collect roots ...')
    roots = []
    for i, parent in enumerate(seq_parents):
        if parent == 0:
            roots.append(i)
    logging.info('dump roots to: ' + out_filename + '.root ...')
    np.array(roots, dtype=np.int32).dump(out_filename + '.root')

    #preprocessing.rearrange_children_indices(out_filename, parent_dir, max_depth, max_articles, batch_size)
    #preprocessing.concat_children_indices(out_filename, parent_dir, max_depth)

    #logging.info('load and concatenate child indices batches ...')
    #for current_depth in range(1, max_depth + 1):
    #    if not os.path.isfile(out_filename + '.children.depth' + str(current_depth)):
    #        preprocessing.merge_numpy_batch_files(out_base_name + '.children.depth' + str(current_depth), parent_dir)

    return parser


def parse_articles(out_path, in_filename, parser, sentence_processor, max_articles, batch_size, tree_mode):
    types = corpus.read_types(out_path)
    mapping = corpus.mapping_from_list(types)
    logging.info('parse articles ...')
    for offset in range(0, max_articles, batch_size):
        # all or none: otherwise the mapping lacks entries!
        if not os.path.isfile(out_path + '.data.batch' + str(offset))\
                or not os.path.isfile(out_path + '.parent.batch' + str(offset))\
                or not os.path.isfile(out_path + '.depth.batch' + str(offset)):
            logging.info('parse articles for offset=' + str(offset) + ' ...')
            current_seq_data, current_seq_parents, current_seq_depths = preprocessing.read_data(
                articles_from_csv_reader,
                sentence_processor, parser, mapping,
                args={
                    'filename': in_filename,
                    'max_articles': min(batch_size, max_articles),
                    'skip': offset
                },
                #max_depth=max_depth,
                batch_size=batch_size,
                tree_mode=tree_mode,
                calc_depths=True,
                #child_idx_offset=child_idx_offset
            )
            corpus.write_dict(out_path, types=corpus.revert_mapping_to_list(mapping))
            logging.info('dump data, parents and depths ...')
            current_seq_data.dump(out_path + '.data.batch' + str(offset))
            current_seq_parents.dump(out_path + '.parent.batch' + str(offset))
            current_seq_depths.dump(out_path + '.depth.batch' + str(offset))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    sentence_processor = getattr(preprocessing, FLAGS.sentence_processor)
    out_dir = os.path.abspath(os.path.join(FLAGS.corpus_data_output_dir, sentence_processor.func_name))
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    out_path = os.path.join(out_dir, FLAGS.corpus_data_output_fn)
    if FLAGS.tree_mode is not None:
        out_path = out_path + '_' + FLAGS.tree_mode

    out_path = out_path + '_articles' + str(FLAGS.max_articles)
    out_path = out_path + '_maxdepth' + str(FLAGS.max_depth)

    logging.info('output base file name: '+out_path)

    nlp = None
    nlp = convert_wikipedia(FLAGS.corpus_data_input_train,
                                           out_path,
                                           FLAGS.init_dict_filename,
                                           sentence_processor,
                                           nlp,
                                           max_articles=FLAGS.max_articles,
                                           max_depth=FLAGS.max_depth,
                                           batch_size=FLAGS.article_batch_size)
