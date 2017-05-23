import spacy

import corpus_wikipedia
import preprocessing

import corpus_wikipedia
import numpy as np
import pprint
import tensorflow_fold as td
import os


seq_data = [3646, 117, 112, 385, 853, 120, 109, 2014, 7453, 70, 110, 145, 139, 70]
seq_parents = [1, 10, 2, 1, -3, -1, 2, 1, -3, -8, 1, 0, -1, -2]
print('calc children and roots ...')
children, roots = preprocessing.children_and_roots(seq_parents)
print(children)
print(roots)


def test_depth():
    print('calc depths ...')
    depth = -np.ones(len(seq_data), dtype=np.int32)
    for root in roots:
        #print(idx)
        #if depth[idx] < 0:
        preprocessing.calc_depth(children, seq_parents, depth, root)
    print(depth)


def test_build_build_sequence_tree_with_candidates():
    pp = pprint.PrettyPrinter(indent=2)
    insert_idx = 5
    candidate_indices = [2, 8]
    seq_tree_c = preprocessing.build_sequence_tree_with_candidates(seq_data, seq_parents, children, roots[0], insert_idx, candidate_indices)
    #pp.pprint(seq_tree_c)
    pp.pprint(td.proto_tools.serialized_message_to_tree('recursive_dependency_embedding.SequenceNodeCandidates', seq_tree_c.SerializeToString()))


def test_build_build_sequence_tree_with_candidate():
    pp = pprint.PrettyPrinter(indent=2)
    insert_idx = 8
    candidate_idx = 8
    max_depth = 10
    max_candidate_depth = 3
    seq_tree_c = preprocessing.build_sequence_tree_with_candidate(seq_data, children, roots[0], insert_idx, candidate_idx, max_depth, max_candidate_depth)
    pp.pprint(seq_tree_c)
    #pp.pprint(td.proto_tools.serialized_message_to_tree('recursive_dependency_embedding.SequenceNodeCandidates', seq_tree_c.SerializeToString()))


def test_sequence_tree_to_arrays():
    pp = pprint.PrettyPrinter(indent=2)
    insert_idx = 8
    candidate_idx = 8
    max_depth = 10
    max_candidate_depth = 3
    seq_tree_c = preprocessing.build_sequence_tree_with_candidate(seq_data, children, roots[0], insert_idx,
                                                                  candidate_idx, max_depth, max_candidate_depth)
    pp.pprint(seq_tree_c)
    tree_ = td.proto_tools.serialized_message_to_tree('recursive_dependency_embedding.SequenceNode', seq_tree_c.SerializeToString())

    new_data, new_parents = preprocessing.sequence_node_to_arrays(tree_)
    print(new_data)
    print(new_parents)


def test_get_all_children():
    start = 11
    for max_depth in range(1, 5):
        print(preprocessing.get_all_children_rec(start, children, max_depth, max_depth_only=True))


def test_read_data_2():
    nlp = spacy.load('en')
    nlp.pipeline = [nlp.tagger, nlp.entity, nlp.parser]
    print('extract word embeddings from spaCy...')
    vecs, mapping = preprocessing.get_word_embeddings(nlp.vocab)

    sentence = 'London is a big city in the United Kingdom. I like this.'
    res = preprocessing.read_data_2(preprocessing.string_reader, preprocessing.process_sentence2, nlp, mapping,
                              args={'content': sentence})  # , tree_mode='sequence')
    print(res)


def test_collected_shuffled_child_indices():
    x = preprocessing.collected_shuffled_child_indices('/media/arne/WIN/Users/Arne/ML/data/corpora/wikipedia/process_sentence7/WIKIPEDIA_articles10000_maxdepth10', 1)
    print(x.shape)

if __name__ == '__main__':
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    td.proto_tools.map_proto_source_tree_path('', ROOT_DIR)
    td.proto_tools.import_proto_file('sequence_node.proto')
    #td.proto_tools.import_proto_file('sequence_node_sequence.proto')
    #td.proto_tools.import_proto_file('sequence_node_candidates.proto')
    #test_depth()
    #test_build_build_sequence_tree_with_candidate()
    #test_get_all_children()
    #test_read_data_2()
    #test_collected_shuffled_child_indices()
    test_sequence_tree_to_arrays()








