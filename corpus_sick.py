import csv
import os
import pprint
import logging
import sys

import spacy
import tensorflow as tf
import numpy as np

import constants
import corpus
import preprocessing
import similarity_tree_tuple_pb2  # , sequence_node_sequence_pb2

tf.flags.DEFINE_string(
    'corpus_data_input_train', '/home/arne/devel/ML/data/corpora/SICK/sick_train/SICK_train.txt',
    'The path to the SICK train data file.')
tf.flags.DEFINE_string(
    'corpus_data_input_test', '/home/arne/devel/ML/data/corpora/SICK/sick_test_annotated/SICK_test_annotated.txt',
    'The path to the SICK test data file.')
tf.flags.DEFINE_string(
    'corpus_data_output_dir',
    # 'data/corpora/sick',
    '/media/arne/WIN/Users/Arne/ML/data/corpora/sick',
    'The path to the output data files (samples, embedding vectors, mappings).')
tf.flags.DEFINE_string(
    'corpus_data_output_fn', 'SICK',
    'Base filename of the output data files (samples, embedding vectors, mappings).')
# tf.flags.DEFINE_string(
#    'dict_filename', 'data/nlp/spacy/dict',
#    'The path to the output data files (samples, embedding vectors, mappings).')
tf.flags.DEFINE_integer(
    'corpus_size', -1,
    'How many samples to write. Use a negative dummy value to set no limit.')
tf.flags.DEFINE_string(
    'sentence_processor',
    'process_sentence3',
    'Which data types (features) are used to build the data sequence.')
tf.flags.DEFINE_string(
    'inner_concat_mode',
    # 'tree',
    constants.default_inner_concat_mode,
    'How to concatenate the trees returned for one token. '
    '"tree" -> use dependency parse tree'
    '"sequence" -> roots point to next root, '
    '"aggregate" -> roots point to an added, artificial token (AGGREGATOR) in the end of the token sequence'
    'None -> do not concat at all')
tf.flags.DEFINE_integer(
    'fold_count', 10,
    'How many folds to write.')

FLAGS = tf.flags.FLAGS

pp = pprint.PrettyPrinter(indent=4)


# deprecated
def sick_raw_reader_DEP(filename):
    with open(filename, 'rb') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')  # , fieldnames=['pair_ID', 'sentence_A', 'sentence_B',
        # 'relatedness_score', 'entailment_judgment'])
        for row in reader:
            yield (int(row['pair_ID']), row['sentence_A'].decode('utf-8'), row['sentence_B'].decode('utf-8'),
                   float(row['relatedness_score']))


def sick_sentence_reader(filename):
    with open(filename, 'rb') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')  # , fieldnames=['pair_ID', 'sentence_A', 'sentence_B',
        # 'relatedness_score', 'entailment_judgment'])
        for row in reader:
            # yield (int(row['pair_ID']), row['sentence_A'].decode('utf-8'), row['sentence_B'].decode('utf-8'),
            #       float(row['relatedness_score']))
            yield row['sentence_A'].decode('utf-8') + '.'
            yield row['sentence_B'].decode('utf-8') + '.'


def sick_score_reader(filename):
    with open(filename, 'rb') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')  # , fieldnames=['pair_ID', 'sentence_A', 'sentence_B',
        # 'relatedness_score', 'entailment_judgment'])
        for row in reader:
            yield float(row['relatedness_score'])


# deprecated
# build similarity_tree_tuple objects
def sick_reader(filename, sentence_processor, parser, mapping, concat_mode, inner_concat_mode):
    for i, t in enumerate(sick_raw_reader_DEP(filename)):
        _, sen1, sen2, score = t
        sim_tree_tuple = similarity_tree_tuple_pb2.SimilarityTreeTuple()
        preprocessing.build_sequence_tree_from_str(str_=sen1 + '.', sentence_processor=sentence_processor,
                                                   parser=parser,
                                                   data_maps=mapping, concat_mode=concat_mode,
                                                   inner_concat_mode=inner_concat_mode, seq_tree=sim_tree_tuple.first)
        preprocessing.build_sequence_tree_from_str(str_=sen2 + '.', sentence_processor=sentence_processor,
                                                   parser=parser,
                                                   data_maps=mapping, concat_mode=concat_mode,
                                                   inner_concat_mode=inner_concat_mode, seq_tree=sim_tree_tuple.second)
        sim_tree_tuple.similarity = (score - 1.) / 4.
        yield sim_tree_tuple


# build sequence_node_sequence objects
# def sick_reader2(filename, sentence_processor, parser, data_maps, concat_mode=None):
#    for i, t in enumerate(sick_raw_reader(filename)):
#        _, sen1, sen2, score = t
#        sequence_node_sequence = sequence_node_sequence_pb2.SequenceNodeSequence()
#        preprocessing.build_sequence_tree_from_str(sen1+'.', sentence_processor, parser, data_maps,
#                                                   sequence_node_sequence.nodes.add(), concat_mode)
#        preprocessing.build_sequence_tree_from_str(sen2+'.', sentence_processor, parser, data_maps,
#                                                   sequence_node_sequence.nodes.add(), concat_mode)
#        sequence_node_sequence.score = (score - 1.) / 4.
#        yield sequence_node_sequence


# deprecated
def convert_sick(in_filename, out_filename, sentence_processor, parser, mapping, max_tuple, inner_concat_mode):
    record_output = tf.python_io.TFRecordWriter(out_filename)
    for i, t in enumerate(sick_reader(in_filename, sentence_processor, parser, mapping, constants.default_concat_mode,
                                      inner_concat_mode)):
        if 0 < max_tuple == i:
            break
        record_output.write(t.SerializeToString())
    record_output.close()


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
    if FLAGS.inner_concat_mode is not None:
        out_path = out_path + '_' + FLAGS.inner_concat_mode

    data_train, parents_train, scores_train, _ = corpus.parse_texts_scored(filename=FLAGS.corpus_data_input_train,
                                                                           reader=sick_sentence_reader,
                                                                           reader_scores=sick_score_reader,
                                                                           sentence_processor=sentence_processor,
                                                                           parser=nlp,
                                                                           mapping=mapping,
                                                                           inner_concat_mode=FLAGS.inner_concat_mode)

    data_test, parents_test, scores_test, _ = corpus.parse_texts_scored(filename=FLAGS.corpus_data_input_test,
                                                                        reader=sick_sentence_reader,
                                                                        reader_scores=sick_score_reader,
                                                                        sentence_processor=sentence_processor,
                                                                        parser=nlp,
                                                                        mapping=mapping,
                                                                        inner_concat_mode=FLAGS.inner_concat_mode)

    data = np.concatenate((data_train, data_test))
    parents = np.concatenate((parents_train, parents_test))
    scores = np.concatenate((scores_train, scores_test))
    types = corpus.revert_mapping_to_list(mapping)
    converter, vecs, types, new_counts, new_idx_unknown = corpus.sort_and_cut_and_fill_dict(data, vecs, types, count_threshold=1)
    data = corpus.convert_data(data, converter, len(types), new_idx_unknown)
    logging.debug('calc roots ...')
    logging.info('save data, parents, scores, vecs and types to: ' + out_path + ' ...')
    data.dump(out_path + '.data')
    parents.dump(out_path + '.parent')
    scores.dump(out_path + '.score')
    corpus.write_dict(out_path, vecs=vecs, types=types)
    logging.info('the dataset contains ' + str(len(scores)) + ' scored text tuples')

    children, roots = preprocessing.children_and_roots(parents)
    for fold in range(FLAGS.fold_count):
        out_fn = out_path + '.train.'+str(fold)
        logging.info('write fold to: ' + out_fn + ' ...')
        with tf.python_io.TFRecordWriter(out_fn) as record_output:
            size = len(roots) / FLAGS.fold_count
            #offset = (fold * len(roots)) / FLAGS.fold_count
            for idx in range(fold * size, (fold + 1) * size, 2):
                sim_tree_tuple = similarity_tree_tuple_pb2.SimilarityTreeTuple()
                t1 = preprocessing.build_sequence_tree(data, children, roots[idx], sim_tree_tuple.first)
                t2 = preprocessing.build_sequence_tree(data, children, roots[idx+1], sim_tree_tuple.second)
                sim_tree_tuple.similarity = (scores[idx / 2] - 1.) / 4.
                record_output.write(sim_tree_tuple.SerializeToString())

