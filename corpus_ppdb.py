import csv
import os
import pprint
import logging
import random
import sys
from itertools import izip

import spacy
import tensorflow as tf
import numpy as np

import constants
import corpus
import preprocessing
import similarity_tree_tuple_pb2  # , sequence_node_sequence_pb2

tf.flags.DEFINE_string(
    'corpus_data_input_train', '/home/arne/devel/ML/data/corpora/PPDB/ppdb-2.0-s-phrasal_1000000',
    'The path to the SICK train data file.')
tf.flags.DEFINE_string(
    'corpus_data_output_dir',
    '/media/arne/WIN/Users/Arne/ML/data/corpora/ppdb',
    'The path to the output data files (samples, embedding vectors, mappings).')
tf.flags.DEFINE_string(
    'corpus_data_output_fn', 'PPDB',
    'Base filename of the output data files (samples, embedding vectors, mappings).')
tf.flags.DEFINE_integer(
    'corpus_size', -1,
    'How many samples to write. Use a negative dummy value to set no limit.')
tf.flags.DEFINE_string(
    'sentence_processor',
    'process_sentence3',
    'Which data types (features) are used to build the data sequence.')
tf.flags.DEFINE_string(
    'concat_mode',
    #'sequence',
    'aggregate',
    #constants.default_inner_concat_mode,
    'How to concatenate the trees returned for one sentence. '
    '"tree" -> use dependency parse tree'
    '"sequence" -> roots point to next root, '
    '"aggregate" -> roots point to an added, artificial token (AGGREGATOR) in the end of the token sequence'
    '(NOT ALLOWED for similarity scored tuples!) None -> do not concat at all')
tf.flags.DEFINE_string(
    'inner_concat_mode',
    #'tree',
    None,
    #constants.default_inner_concat_mode,
    'How to concatenate the trees returned for one token. '
    '"tree" -> use dependency parse tree'
    '"sequence" -> roots point to next root, '
    '"aggregate" -> roots point to an added, artificial token (AGGREGATOR) in the end of the token sequence'
    'None -> do not concat at all')
tf.flags.DEFINE_integer(
    'fold_count', 100,
    'How many folds to write.')
tf.flags.DEFINE_integer(
    'sample_count', 1,
    'How many negative samples to add.')
tf.flags.DEFINE_integer(
    'count_threshold', 5,
    'The minimum of token occurrences to keep the token in the dictionary.')

FLAGS = tf.flags.FLAGS

pp = pprint.PrettyPrinter(indent=4)


def ppdb_sentence_reader(filename):
    with open(filename, 'rb') as file:
        for line in file:
            cols = line.split(' ||| ')
            yield cols[1].decode('utf-8')
            yield cols[2].decode('utf-8')


def ppdb_score_reader(filename):
    num_lines = sum(1 for line in open(filename))
    for _ in range(num_lines):
        yield 1.0


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
    logging.info('load spacy ...')
    nlp = spacy.load('en')
    nlp.pipeline = [nlp.tagger, nlp.entity, nlp.parser]

    # vecs, mapping = corpus.create_or_read_dict(FLAGS.dict_filename, nlp.vocab)
    # corpus.make_parent_dir(FLAGS.dict_filename)
    vecs, types = corpus.get_dict_from_vocab(nlp.vocab)  # corpus.create_or_read_dict(FLAGS.dict_filename, nlp.vocab)
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

    data_train, parents_train, scores_train, _ = corpus.parse_texts_scored(filename=FLAGS.corpus_data_input_train,
                                                                           reader=ppdb_sentence_reader,
                                                                           reader_scores=ppdb_score_reader,
                                                                           sentence_processor=sentence_processor,
                                                                           parser=nlp,
                                                                           mapping=mapping,
                                                                           concat_mode=FLAGS.concat_mode,
                                                                           inner_concat_mode=FLAGS.inner_concat_mode)


    data = data_train #np.concatenate((data_train, data_test))
    parents = parents_train #np.concatenate((parents_train, parents_test))
    scores = scores_train #np.concatenate((scores_train, scores_test))
    types = corpus.revert_mapping_to_list(mapping)
    converter, vecs, types, new_counts, new_idx_unknown = corpus.sort_and_cut_and_fill_dict(data, vecs, types,
                                                                                            count_threshold=FLAGS.count_threshold)
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
    root_pairs = list(izip(*[iter(roots)]*2))
    random.shuffle(root_pairs)

    for fold in range(FLAGS.fold_count):
        out_fn = out_path + '.train.'+str(fold)
        logging.info('write fold to: ' + out_fn + ' ...')
        with tf.python_io.TFRecordWriter(out_fn) as record_output:
            size = len(root_pairs) / FLAGS.fold_count
            for idx in range(fold * size, (fold + 1) * size):
                sim_tree_tuple = similarity_tree_tuple_pb2.SimilarityTreeTuple()
                t1 = preprocessing.build_sequence_tree(data, children, root_pairs[idx][0], sim_tree_tuple.first)
                t2 = preprocessing.build_sequence_tree(data, children, root_pairs[idx][1], sim_tree_tuple.second)
                sim_tree_tuple.similarity = scores[idx] #(scores[idx / 2] - 1.) / 4.
                record_output.write(sim_tree_tuple.SerializeToString())
                for _ in range(FLAGS.sample_count):
                    # sample just from second column
                    sample_idx = random.randint(0, len(root_pairs) - 1)

                    sim_tree_tuple = similarity_tree_tuple_pb2.SimilarityTreeTuple()
                    t1 = preprocessing.build_sequence_tree(data, children, root_pairs[idx][0], sim_tree_tuple.first)
                    t2 = preprocessing.build_sequence_tree(data, children, root_pairs[sample_idx][1], sim_tree_tuple.second)
                    sim_tree_tuple.similarity = 0.0 # scores[idx / 2]  # (scores[idx / 2] - 1.) / 4.
                    record_output.write(sim_tree_tuple.SerializeToString())

