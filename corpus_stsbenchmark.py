import csv
import os
import pprint
import logging
import sys

import spacy
import tensorflow as tf
import numpy as np

import corpus
import preprocessing

tf.flags.DEFINE_string(
    'corpus_data_input_train', '/home/arne/devel/ML/data/corpora/STSbenchmark/sts-train_fixed.csv',
    'The path to the STSbenchmark train data file.')
tf.flags.DEFINE_string(
    'corpus_data_input_dev', '/home/arne/devel/ML/data/corpora/STSbenchmark/sts-dev_fixed.csv',
    'The path to the STSbenchmark dev data file.')
tf.flags.DEFINE_string(
    'corpus_data_input_test', '/home/arne/devel/ML/data/corpora/STSbenchmark/sts-test_fixed.csv',
    'The path to the STSbenchmark test data file.')
tf.flags.DEFINE_string(
    'corpus_data_output_dir',
    # 'data/corpora/sick',
    '/media/arne/WIN/Users/Arne/ML/data/corpora/stsbenchmark',
    'The path to the output data files (samples, embedding vectors, mappings).')
tf.flags.DEFINE_string(
    'corpus_data_output_fn', 'STSBENCH',
    'Base filename of the output data files (samples, embedding vectors, mappings).')
tf.flags.DEFINE_string(
    'sentence_processor',
    'process_sentence3',
    'Which data types (features) are used to build the data sequence.')
tf.flags.DEFINE_string(
    'concat_mode',
    #'sequence',
    'aggregate',
    #constants.default_inner_concat_mode,
    'How to concatenate the sentence-trees with each other. '
    'A sentence-tree represents the information regarding one sentence. '
    '"sequence" -> roots point to next root, '
    '"aggregate" -> roots point to an added, artificial token (AGGREGATOR) in the end of the token sequence'
    '(NOT ALLOWED for similarity scored tuples!) None -> do not concat at all')
tf.flags.DEFINE_string(
    'inner_concat_mode',
    #'tree',
    None,
    #constants.default_inner_concat_mode,
    'How to concatenate the token-trees with each other. '
    'A token-tree represents the information regarding one token. '
    '"tree" -> use dependency parse tree'
    '"sequence" -> roots point to next root, '
    '"aggregate" -> roots point to an added, artificial token (AGGREGATOR) in the end of the token sequence'
    'None -> do not concat at all. This produces one sentence-tree per token.')
#tf.flags.DEFINE_integer(
#    'fold_count', 5,
#    'How many folds to write.')

FLAGS = tf.flags.FLAGS

pp = pprint.PrettyPrinter(indent=4)


def stsbenchmark_sentence_reader(filename):
    with open(filename, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        for row in reader:
            #if len(row) > 6:
            yield row[5].decode('utf-8')
            yield row[6].decode('utf-8')
            #else:
            #    print(row)


def stsbenchmark_score_reader(filename):
    with open(filename, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        for row in reader:
            yield float(row[4])


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
    logging.info('load spacy ...')
    nlp = spacy.load('en')
    nlp.pipeline = [nlp.tagger, nlp.entity, nlp.parser]

    vecs, types = corpus.get_dict_from_vocab(nlp.vocab)
    mapping = corpus.mapping_from_list(types)

    sentence_processor = getattr(preprocessing, FLAGS.sentence_processor)
    out_dir = os.path.abspath(os.path.join(FLAGS.corpus_data_output_dir, sentence_processor.func_name))
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    out_path = os.path.join(out_dir, FLAGS.corpus_data_output_fn)

    assert FLAGS.concat_mode is not None and FLAGS.concat_mode != 'tree', \
        "concat_mode=None or concat_mode='tree' is NOT ALLOWED for similarity scored tuples! Use 'sequence' or 'aggregate'"
    out_path = out_path + '_CM' + FLAGS.concat_mode
    if FLAGS.inner_concat_mode is not None:
        out_path = out_path + '_ICM' + FLAGS.inner_concat_mode

    def read_data(file_name):
        return corpus.parse_texts_scored(filename=file_name,
                                         reader=stsbenchmark_sentence_reader,
                                         reader_scores=stsbenchmark_score_reader,
                                         sentence_processor=sentence_processor,
                                         parser=nlp,
                                         mapping=mapping,
                                         concat_mode=FLAGS.concat_mode,
                                         inner_concat_mode=FLAGS.inner_concat_mode)

    data_train, parents_train, scores_train, _ = read_data(FLAGS.corpus_data_input_train)
    data_dev, parents_dev, scores_dev, _ = read_data(FLAGS.corpus_data_input_dev)
    data_test, parents_test, scores_test, _ = read_data(FLAGS.corpus_data_input_test)

    data = np.concatenate((data_train, data_dev, data_test))
    parents = np.concatenate((parents_train, parents_dev, parents_test))
    scores = np.concatenate((scores_train, scores_dev, scores_test))
    types = corpus.revert_mapping_to_list(mapping)
    converter, vecs, types, new_counts, new_idx_unknown = corpus.sort_and_cut_and_fill_dict(data, vecs, types, count_threshold=1)
    data = corpus.convert_data(data, converter, len(types), new_idx_unknown)
    logging.info('save data, parents, scores, vecs and types to: ' + out_path + ' ...')
    data.dump(out_path + '.data')
    parents.dump(out_path + '.parent')
    scores.dump(out_path + '.score')
    corpus.write_dict(out_path, vecs=vecs, types=types)
    logging.info('the dataset contains ' + str(len(scores)) + ' scored text tuples')
    logging.debug('calc roots ...')
    children, roots = preprocessing.children_and_roots(parents)
    logging.debug('len(roots)='+str(len(roots)))

    sim_tuples = [(i*2, i*2 + 1, (scores[i] - 1.) / 4.) for i in range(len(scores))]

    corpus.write_sim_tuple_data(out_path + '.train.0', sim_tuples[:len(scores_train)], data, children, roots)
    corpus.write_sim_tuple_data(out_path + '.train.1', sim_tuples[len(scores_train):len(scores_train)+len(scores_dev)], data, children, roots)
    corpus.write_sim_tuple_data(out_path + '.train.2', sim_tuples[len(scores_train)+len(scores_dev):], data, children, roots)

    #fold_size = len(roots) / FLAGS.fold_count
    #start_idx = 0
    #for fold in range(FLAGS.fold_count):
    #    out_fn = out_path + '.train.'+str(fold)
    #    write_data(out_fn, start_idx, fold_size, data, children, roots)


