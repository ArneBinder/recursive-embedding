import csv
import pprint
import preprocessing
import similarity_tree_tuple_pb2#, sequence_node_sequence_pb2
import spacy
import tensorflow as tf
import pickle
import os
import numpy as np
import tools
import constants

tf.flags.DEFINE_string(
    'corpus_data_input_train', '/home/arne/devel/ML/data/corpora/SICK/sick_train/SICK_train.txt',
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

pp = pprint.PrettyPrinter(indent=4)


def sick_raw_reader(filename):
    with open(filename, 'rb') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')#, fieldnames=['pair_ID', 'sentence_A', 'sentence_B',
                                                     #'relatedness_score', 'entailment_judgment'])
        for row in reader:
            yield (int(row['pair_ID']), row['sentence_A'].decode('utf-8'), row['sentence_B'].decode('utf-8'), float(row['relatedness_score']))


# build similarity_tree_tuple objects
def sick_reader(filename, sentence_processor, parser, data_maps, tree_mode=None):
    for i, t in enumerate(sick_raw_reader(filename)):
        _, sen1, sen2, score = t
        sim_tree_tuple = similarity_tree_tuple_pb2.SimilarityTreeTuple()
        preprocessing.build_sequence_tree_from_str(sen1+'.', sentence_processor, parser, data_maps,
                                                   sim_tree_tuple.first, tree_mode)
        preprocessing.build_sequence_tree_from_str(sen2+'.', sentence_processor, parser, data_maps,
                                                   sim_tree_tuple.second, tree_mode)
        sim_tree_tuple.similarity = (score - 1.) / 4.
        yield sim_tree_tuple


# build sequence_node_sequence objects
#def sick_reader2(filename, sentence_processor, parser, data_maps, tree_mode=None):
#    for i, t in enumerate(sick_raw_reader(filename)):
#        _, sen1, sen2, score = t
#        sequence_node_sequence = sequence_node_sequence_pb2.SequenceNodeSequence()
#        preprocessing.build_sequence_tree_from_str(sen1+'.', sentence_processor, parser, data_maps,
#                                                   sequence_node_sequence.nodes.add(), tree_mode)
#        preprocessing.build_sequence_tree_from_str(sen2+'.', sentence_processor, parser, data_maps,
#                                                   sequence_node_sequence.nodes.add(), tree_mode)
#        sequence_node_sequence.score = (score - 1.) / 4.
#        yield sequence_node_sequence


def convert_sick(in_filename, out_filename, sentence_processor, parser, mapping, max_tuple=-1, tree_mode=None):
    record_output = tf.python_io.TFRecordWriter(out_filename)
    for i, t in enumerate(sick_reader(in_filename, sentence_processor, parser, mapping, tree_mode)):
        if 0 < max_tuple == i:
            break
        record_output.write(t.SerializeToString())
    record_output.close()


def write_dict(out_path, mapping, vocab_nlp, vocab_manual):
    print('dump mappings to: ' + out_path + '.mapping ...')
    with open(out_path + '.mapping', "wb") as f:
        pickle.dump(mapping, f)
    print('write tsv dict: ' + out_path + '.tsv ...')
    rev_map = tools.revert_mapping(mapping)
    with open(out_path + '.tsv', 'wb') as csvfile:
        fieldnames = ['label', 'id_orig']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter='\t', quotechar='|',
                                quoting=csv.QUOTE_MINIMAL)

        writer.writeheader()
        for i in range(len(rev_map)):
            id_orig = rev_map[i]
            if id_orig >= 0:
                label = vocab_nlp[id_orig].orth_
            else:
                label = vocab_manual[id_orig]
            writer.writerow({'label': label.encode("utf-8"), 'id_orig': str(id_orig)})


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
    convert_sick(FLAGS.corpus_data_input_train,
                 out_path + '.train',
                 sentence_processor,
                 nlp,
                 mapping,
                 FLAGS.corpus_size,
                 FLAGS.tree_mode)

    print('parse test data ...')
    convert_sick(FLAGS.corpus_data_input_test,
                 out_path + '.test',
                 sentence_processor,
                 nlp,
                 mapping,
                 FLAGS.corpus_size,
                 FLAGS.tree_mode)

    print('len(mapping): ' + str(len(mapping)))

    write_dict(out_path, mapping, nlp.vocab, constants.vocab_manual)





