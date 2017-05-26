import pickle
import spacy

import constants
import corpus
import preprocessing

import numpy as np
import pprint
import tensorflow_fold as td
import os

import sequence_node_pb2
import tools
import train_fold_nce
import visualize

seq_data = [3646, 1297620, 117, 53146, 112, 48, 385, 1297621, 853, 43, 120, 60, 109, 48, 2014, 9123, 7453, 1297622, 70, 62, 110, 1297620, 145, 53146, 139, 1297623, 70, 62]#[3646, 117, 112, 385, 853, 120, 109, 2014, 7453, 70, 110, 145, 139, 70]
seq_parents = [2, -1, 0, -1, 4, -1, 2, -1, -6, -1, -2, -1, 4, -1, 2, -1, -6, -1, -16, -1, 2, -1, 0, -1, -2, -1, -4, -1]#[1, 10, 2, 1, -3, -1, 2, 1, -3, -8, 1, 0, -1, -2]
print('calc children and roots ...')
children, roots = preprocessing.children_offsets_and_roots(seq_parents)
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


def test_iterator_sequence_trees():
    pp = pprint.PrettyPrinter(indent=2)

    train_path = '/media/arne/WIN/Users/Arne/ML/data/corpora/wikipedia/process_sentence3/WIKIPEDIA_articles1000_maxdepth10'
    sample_count = 5
    max_depth = 1

    print('load mapping from file: ' + train_path + '.mapping ...')
    m = pickle.load(open(train_path + '.mapping', "rb"))
    print('len(mapping): ' + str(len(m)))
    rev_m = corpus.revert_mapping(m)
    print('load spacy ...')
    parser = spacy.load('en')

    # load corpus data
    print('load corpus data from: ' + train_path + '.data ...')
    seq_data = np.load(train_path + '.data')
    print('load corpus parents from: ' + train_path + '.parent ...')
    seq_parents = np.load(train_path + '.parent')
    print('calc children ...')
    children, roots = preprocessing.children_and_roots(seq_parents)

    #visualize.visualize('orig.png', (seq_data, seq_parents), rev_m, parser.vocab, constants.vocab_manual)

    for i, seq_tree_seq in enumerate(train_fold_nce.iterator_sequence_trees(train_path, max_depth, seq_data, children,
                                                                            sample_count)):
        if i == 1000000:
            break

        #pp.pprint(seq_tree_seq)
        t_all = [str(seq_tree_seq['trees'][j]) for j in range(sample_count + 1)]
        if len(set(t_all)) <= 1:
            print('IDX: ' + str(i))
            print(t_all[0])
            #pp.pprint(seq_tree_seq)
            visualize.visualize_seq_node_seq(seq_tree_seq, rev_m, parser.vocab, constants.vocab_manual,
                                         'forest_out_' + str(i) + '.png')


def test_iterator_sequence_trees_cbot():
    pp = pprint.PrettyPrinter(indent=2)

    train_path = '/media/arne/WIN/Users/Arne/ML/data/corpora/wikipedia/process_sentence7/WIKIPEDIA_articles10000_maxdepth10'
    sample_count = 5
    max_depth = 3

    # load corpus data
    print('load corpus data from: ' + train_path + '.data ...')
    seq_data = np.load(train_path + '.data')
    print('load corpus parents from: ' + train_path + '.parent ...')
    seq_parents = np.load(train_path + '.parent')
    print('calc children ...')
    children, roots = preprocessing.children_and_roots(seq_parents)
    print(children[3106218])

    print('load depths from: ' + train_path + '.depth1.collected')
    depth1_collected = np.load(train_path + '.depth1.collected')

    idx = depth1_collected[9663491] # idx = 3106218
    print('idx: '+str(idx))
    print('load depths ...')
    seq_depths = np.load(train_path + '.depth')
    print('depth of idx: '+str(seq_depths[idx]))

    print('children of idx:')
    print(children[idx])
    seq_tree = sequence_node_pb2.SequenceNode()
    preprocessing.build_sequence_tree(seq_data, children, idx, seq_tree, max_depth)
    pp.pprint(seq_tree)

    print('load mapping from file: ' + train_path + '.mapping ...')
    m = pickle.load(open(train_path + '.mapping', "rb"))
    print('len(mapping): ' + str(len(m)))
    rev_m = corpus.revert_mapping(m)
    print('load spacy ...')
    parser = spacy.load('en')

    #visualize.visualize('orig.png', (seq_data, seq_parents), rev_m, parser.vocab, constants.vocab_manual)

    for i, seq_tree_seq in enumerate(train_fold_nce.iterator_sequence_trees_cbot(train_path, max_depth, seq_data, children,
                                                                            sample_count, loaded_global_step=0)):
        if i == 1:
            break

        #pp.pprint(seq_tree_seq)
        t_all = [str(seq_tree_seq['trees'][j]) for j in range(sample_count + 1)]
        #if len(set(t_all)) <= 1:
        print('IDX: ' + str(i))
        #print(t_all[0])
        pp.pprint(seq_tree_seq)
        visualize.visualize_seq_node_seq(seq_tree_seq, rev_m, parser.vocab, constants.vocab_manual,
                                     'forest_out_' + str(i) + '.png')


if __name__ == '__main__':
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    td.proto_tools.map_proto_source_tree_path('', ROOT_DIR)
    td.proto_tools.import_proto_file('sequence_node.proto')
    td.proto_tools.import_proto_file('sequence_node_sequence.proto')
    td.proto_tools.import_proto_file('sequence_node_candidates.proto')
    #test_depth()
    #test_build_build_sequence_tree_with_candidate()
    #test_get_all_children()
    #test_read_data_2()
    #test_collected_shuffled_child_indices()
    #test_sequence_tree_to_arrays()
    #test_iterator_sequence_trees()
    #test_iterator_sequence_trees_cbot()
    test_depth()







