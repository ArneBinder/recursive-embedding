
import preprocessing
import test_model

import sequence_node_pb2
import sequence_trees
import train_fold
import similarity_tree_tuple_pb2, sequence_node_sequence_pb2
import pprint
import tensorflow_fold as td
import spacy
import pickle

#seq_data = [291, 1297613, 48, 1297614, 234, 1297613, 1297615, 1297614, 5132, 1297613, 1297616, 1297614, 6313, 1297613, 1297617, 1297614, 117, 1297613, 44, 1297614, 5593, 1297613, 53146, 1297614, 7686, 1297613, 1297616, 1297614, 5353, 1297613, 1297618, 1297614, 70, 1297613, 62, 1297614, 110, 1297613, 1297617, 1297614, 145, 1297613, 53146, 1297614, 139, 1297613, 1297618, 1297614, 70, 1297613, 62, 1297614]
#seq_parents = [12, -1, -2, -1, 4, -1, -2, -1, 4, -1, -2, -1, 8, -1, -2, -1, 4, -1, -2, -1, 20, -1, -2, -1, 4, -1, -2, -1, -8, -1, -2, -1, -12, -1, -2, -1, 4, -1, -2, -1, 0, -1, -2, -1, -4, -1, -2, -1, -8, -1, -2, -1]
seq_data = [291, 234, 5132, 6313, 117, 5593, 7686, 5353, 70, 110, 145, 139, 70]
seq_parents = [3, 1, 1, 2, 1, 5, 1, -2, -3, 1, 0, -1, -2]


#def test_build_sequence_tree():
#    children, roots = preprocessing.children_and_roots(seq_parents)
#    out = preprocessing.build_sequence_tree(seq_data, seq_parents)
#    print(out.SerializeToString())

def test_read_sim_tree_tuple(fn):
    pp = pprint.PrettyPrinter(indent=2)
    print('\n')
    print('\n')
    print('\n')
    print('\n')
    for i, t in enumerate(train_fold.iterate_over_tf_record_protos(fn, similarity_tree_tuple_pb2.SimilarityTreeTuple)):
        if i > 1:
            break
        pp.pprint(t)


def test_read_seq_node_seq(fn):
    pp = pprint.PrettyPrinter(indent=2)
    print('\n')
    print('\n')
    print('\n')
    print('\n')
    for i, t in enumerate(train_fold.iterate_over_tf_record_protos(fn, sequence_node_sequence_pb2.SequenceNodeSequence)):
        if i > 0:
            break
        pp.pprint(t)


def test_build_sequence_tree():
    pp = pprint.PrettyPrinter(indent=2)
    seq_parents = [0, -1, -2, -1]
    seq_data = [1, 2, 3, 4]
    children, roots = sequence_trees.children_and_roots(seq_parents)
    seq_tree = sequence_node_pb2.SequenceNode()
    sequence_trees.build_sequence_tree(seq_data, children, roots[0], seq_tree)
    s = td.proto_tools.serialized_message_to_tree('recursive_dependency_embedding.SequenceNode', seq_tree.SerializeToString())
    pp.pprint(s)


def test_sequence_node_sequence():
    print('load spacy ...')
    nlp = spacy.load('en')
    nlp.pipeline = [nlp.tagger, nlp.parser]
    print('load data_mapping from: ' + 'data/nlp/spacy/dict.mapping' + ' ...')
    data_maps = pickle.load(open('data/nlp/spacy/dict.mapping', "rb"))
    l = list(test_model.parse_iterator([['Hallo.', 'Hallo!', 'Hallo?', 'Hallo']], nlp, preprocessing.process_sentence3, data_maps))


if __name__ == '__main__':
    td.proto_tools.import_proto_file('similarity_tree_tuple.proto')
    td.proto_tools.import_proto_file('sequence_node_sequence.proto')
    #test_build_sequence_tree()
    #test_read_sim_tree_tuple('data/corpora/sick/process_sentence3/SICK_train')
    #corpus_sick.get_embeddings_from_tf_checkpoint('data/log/spacy/dict.sess')
    test_sequence_node_sequence()