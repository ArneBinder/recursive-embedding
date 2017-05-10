import csv
import pprint
import preprocessing
import similarity_tree_tuple_pb2
import spacy
import tensorflow as tf
import pickle
import os
import ntpath
import numpy as np

tf.flags.DEFINE_string(
    'corpus_data_input_file', '/home/arne/devel/ML/data/corpora/SICK/sick_train/SICK_train.txt',
    'The path to the SICK data file.')
tf.flags.DEFINE_string(
    'corpus_data_output_dir', 'data/corpora/sick',
    'The path to the output data files (samples, embedding vectors, mappings).')
tf.flags.DEFINE_string(
    'dict_filename', 'data/nlp/spacy/dict',
    'The path to the output data files (samples, embedding vectors, mappings).')
tf.flags.DEFINE_integer(
    'corpus_size', -1,
    'How many samples to write. Use a negative dummy value to set no limit.')
tf.flags.DEFINE_string(
    'sentence_processor', 'process_sentence3',
    'How long to make the expression embedding vectors.')

FLAGS = tf.flags.FLAGS

pp = pprint.PrettyPrinter(indent=4)


def sick_raw_reader(filename):
    with open(filename, 'rb') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')#, fieldnames=['pair_ID', 'sentence_A', 'sentence_B',
                                                     #'relatedness_score', 'entailment_judgment'])
        for row in reader:
            yield (int(row['pair_ID']), row['sentence_A'], row['sentence_B'], float(row['relatedness_score']))


def sick_reader(filename, sentence_processor, parser, data_maps):
    for i, t in enumerate(sick_raw_reader(filename)):
        _, sen1, sen2, score = t
        sim_tree_tuple = similarity_tree_tuple_pb2.SimilarityTreeTuple()
        preprocessing.build_sequence_tree_from_str(sen1+'.', sentence_processor, parser, data_maps, sim_tree_tuple.first)
        preprocessing.build_sequence_tree_from_str(sen2+'.', sentence_processor, parser, data_maps, sim_tree_tuple.second)
        sim_tree_tuple.similarity = (score - 1.) / 4.
        yield sim_tree_tuple


def convert_sick(in_filename, out_filename, sentence_processor, parser, mapping, max_tuple=-1):
    record_output = tf.python_io.TFRecordWriter(out_filename)
    for i, t in enumerate(sick_reader(in_filename, sentence_processor, parser, mapping)):
        if 0 < max_tuple == i:
            break
        record_output.write(t.SerializeToString())
    record_output.close()


def create_or_read_dict(fn, vocab):
    if os.path.isfile(fn+'.vecs'):
        print('load vecs from file: '+fn + '.vecs ...')
        v = np.load(fn+'.vecs')
        print('read mapping from file: ' + fn + '.mapping ...')
        m = pickle.load(open(fn+'.mapping', "rb"))
        print('vecs.shape: ' + str(v.shape))
        print('len(mapping): ' + str(len(m)))
    else:
        out_dir = os.path.abspath(os.path.join(fn, os.pardir))
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        print('extract word embeddings from spaCy ...')
        v, m = preprocessing.get_word_embeddings(vocab)
        print('vecs.shape: ' + str(v.shape))
        print('len(mapping): ' + str(len(m)))
        print('dump vecs to: ' + fn + '.vecs ...')
        v.dump(fn + '.vecs')
        print('dump mappings to: ' + fn + '.mapping ...')
        with open(fn + '.mapping', "wb") as f:
            pickle.dump(m, f)
    return v, m

if __name__ == '__main__':
    print('load spacy ...')
    nlp = spacy.load('en')
    nlp.pipeline = [nlp.tagger, nlp.parser]

    # in_dir = '/home/arne/devel/ML/data/corpora/SICK/sick_test_annotated/'
    # fn = 'SICK_test_annotated'
    #in_path = '/home/arne/devel/ML/data/corpora/SICK/sick_train/SICK_train.txt'

    vecs, mapping = create_or_read_dict(FLAGS.dict_filename, nlp.vocab)

    # use filename from input file
    out_fn = os.path.splitext(ntpath.basename(FLAGS.corpus_data_input_file))[0]
    sentence_processor = getattr(preprocessing, FLAGS.sentence_processor)
    out_dir = os.path.abspath(os.path.join(FLAGS.corpus_data_output_dir, sentence_processor.func_name))
    #'data/corpora/sick/' + sentence_processor.func_name + '/' + out_fn
   #out_dir = os.path.abspath(os.path.join(out_path, os.pardir))
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    out_path = os.path.join(out_dir, out_fn)

    #fn_mapping = out_path + '.mapping'
    #if os.path.isfile(fn_mapping):
    #    print('read mapping from file: '+fn_mapping + ' ...')
    #    mapping = pickle.load(open(fn_mapping, "rb"))

    print('parse data ...')
    convert_sick(FLAGS.corpus_data_input_file,
                 out_path,
                 sentence_processor,
                 nlp,
                 mapping,
                 FLAGS.corpus_size)
    print('len(mapping): ' + str(len(mapping)))
    print('dump mappings to: ' + out_path + '.mapping ...')
    with open(out_path + '.mapping', "wb") as f:
        pickle.dump(mapping, f)



