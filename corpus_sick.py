import csv
import pprint
import preprocessing
import similarity_tree_tuple_pb2
import spacy
import tensorflow as tf
import pickle
import os
import ntpath

tf.flags.DEFINE_string(
    'corpus_data_input_file', '/home/arne/devel/ML/data/corpora/SICK/sick_train/SICK_train.txt',
    'The path to the SICK data file.')
tf.flags.DEFINE_string(
    'corpus_data_output_dir', 'data/corpora/sick',
    'The path to the output data files (samples, embedding vectors, mappings).')
tf.flags.DEFINE_integer(
    'corpus_size', 1,
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


if __name__ == '__main__':
    print('load spacy ...')
    nlp = spacy.load('en')
    nlp.pipeline = [nlp.tagger, nlp.parser]

    # in_dir = '/home/arne/devel/ML/data/corpora/SICK/sick_test_annotated/'
    # fn = 'SICK_test_annotated'
    #in_path = '/home/arne/devel/ML/data/corpora/SICK/sick_train/SICK_train.txt'

    # use filename from input file
    out_fn = os.path.splitext(ntpath.basename(FLAGS.corpus_data_input_file))[0]
    sentence_processor = getattr(preprocessing, FLAGS.sentence_processor)
    out_dir = os.path.abspath(os.path.join(FLAGS.corpus_data_output_dir, sentence_processor.func_name))
    #'data/corpora/sick/' + sentence_processor.func_name + '/' + out_fn
   #out_dir = os.path.abspath(os.path.join(out_path, os.pardir))
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    out_path = os.path.join(out_dir, out_fn)
    fn_mapping = out_path + '.mapping'

    if os.path.isfile(fn_mapping):
        print('read mapping from file: '+fn_mapping + ' ...')
        mapping = pickle.load(open(fn_mapping, "rb"))
    else:
        print('extract word embeddings from spaCy ...')
        vecs, mapping = preprocessing.get_word_embeddings(nlp.vocab)
        print('vecs.shape: '+str(vecs.shape))
        print('len(mapping): '+str(len(mapping)))
        print('dump vecs to: ' + out_path + '.vecs ...')
        vecs.dump(out_path + '.vecs')
        #pickle.dump(vecs, open(out_dir + fn + '.vecs', "wb"))

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



