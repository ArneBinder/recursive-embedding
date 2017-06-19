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
import unittest


class Tester(unittest.TestCase):
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    td.proto_tools.map_proto_source_tree_path('', ROOT_DIR)
    td.proto_tools.import_proto_file('sequence_node.proto')
    td.proto_tools.import_proto_file('sequence_node_sequence.proto')
    td.proto_tools.import_proto_file('sequence_node_candidates.proto')

    seq_data = [3646, 1297620, 117, 53146, 112, 48, 385, 1297621, 853, 43, 120, 60, 109, 48, 2014, 9123, 7453, 1297622,
                70, 62, 110, 1297620, 145, 53146, 139, 1297623, 70,
                62]  # [3646, 117, 112, 385, 853, 120, 109, 2014, 7453, 70, 110, 145, 139, 70]
    seq_parents = [2, -1, 0, -1, 4, -1, 2, -1, -6, -1, -2, -1, 4, -1, 2, -1, -6, -1, -16, -1, 2, -1, 0, -1, -2, -1, -4,
                   -1]  # [1, 10, 2, 1, -3, -1, 2, 1, -3, -8, 1, 0, -1, -2]
    print('calc children and roots ...')
    children, roots = preprocessing.children_and_roots(seq_parents)
    print(children)
    print(roots)

    @unittest.skip("skip")
    def test_depth(self):
        print('calc depths ...')
        depth = -np.ones(len(self.seq_data), dtype=np.int32)
        for root in self.roots:
            #print(idx)
            #if depth[idx] < 0:
            preprocessing.calc_depth(self.children, self.seq_parents, depth, root)
        print(depth)

    @unittest.skip("skip")
    def test_build_build_sequence_tree_with_candidates(self):
        pp = pprint.PrettyPrinter(indent=2)
        insert_idx = 5
        candidate_indices = [2, 8]
        seq_tree_c = preprocessing.build_sequence_tree_with_candidates(self.seq_data, self.seq_parents, self.children, self.roots[0], insert_idx, candidate_indices)
        #pp.pprint(seq_tree_c)
        pp.pprint(td.proto_tools.serialized_message_to_tree('recursive_dependency_embedding.SequenceNodeCandidates', seq_tree_c.SerializeToString()))

    @unittest.skip("skip")
    def test_build_build_sequence_tree_with_candidate(self):
        pp = pprint.PrettyPrinter(indent=2)
        insert_idx = 8
        candidate_idx = 8
        max_depth = 10
        max_candidate_depth = 3
        seq_tree_c = preprocessing.build_sequence_tree_with_candidate(self.seq_data, self.children, self.roots[0], insert_idx, candidate_idx, max_depth, max_candidate_depth)
        pp.pprint(seq_tree_c)
        #pp.pprint(td.proto_tools.serialized_message_to_tree('recursive_dependency_embedding.SequenceNodeCandidates', seq_tree_c.SerializeToString()))

    @unittest.skip("skip")
    def test_sequence_tree_to_arrays(self):
        pp = pprint.PrettyPrinter(indent=2)
        insert_idx = 8
        candidate_idx = 8
        max_depth = 10
        max_candidate_depth = 3
        seq_tree_c = preprocessing.build_sequence_tree_with_candidate(self.seq_data, self.children, self.roots[0], insert_idx,
                                                                      candidate_idx, max_depth, max_candidate_depth)
        pp.pprint(seq_tree_c)
        tree_ = td.proto_tools.serialized_message_to_tree('recursive_dependency_embedding.SequenceNode', seq_tree_c.SerializeToString())

        new_data, new_parents = preprocessing.sequence_node_to_arrays(tree_)
        print(new_data)
        print(new_parents)

    @unittest.skip("skip")
    def test_get_all_children(self):
        start = 11
        for max_depth in range(1, 5):
            print(preprocessing.get_all_children_rec(start, self.children, max_depth, max_depth_only=True))

    @unittest.skip("skip")
    def test_read_data_2(self):
        nlp = spacy.load('en')
        nlp.pipeline = [nlp.tagger, nlp.entity, nlp.parser]
        print('extract word embeddings from spaCy...')
        vecs, types = corpus.get_dict_from_vocab(nlp.vocab)
        mapping = corpus.mapping_from_list(types)
        sentence = 'London is a big city in the United Kingdom. I like this.'
        res = preprocessing.read_data_2(preprocessing.string_reader, preprocessing.process_sentence2, nlp, mapping,
                                  args={'content': sentence})  # , tree_mode='sequence')
        print(res)

    @unittest.skip("skip")
    def test_collected_shuffled_child_indices(self):
        x = preprocessing.collected_shuffled_child_indices('/media/arne/WIN/Users/Arne/ML/data/corpora/wikipedia/process_sentence7/WIKIPEDIA_articles10000_maxdepth10', 1)
        print(x.shape)

    @unittest.skip("skip")
    def test_iterator_sequence_trees(self):
        pp = pprint.PrettyPrinter(indent=2)

        train_path = '/media/arne/WIN/Users/Arne/ML/data/corpora/wikipedia/process_sentence7/WIKIPEDIA_articles100_maxdepth10'
        sample_count = 5
        max_depth = 1


        types = corpus.read_types(train_path)
        #print('load mapping from file: ' + train_path + '.mapping ...')
        #m = pickle.load(open(train_path + '.mapping', "rb"))
        #print('len(mapping): ' + str(len(m)))
        #rev_m = corpus.revert_mapping(m)
        #print('load spacy ...')
        #parser = spacy.load('en')

        # load corpus data
        print('load corpus data from: ' + train_path + '.data ...')
        seq_data = np.load(train_path + '.data')
        print('load corpus parents from: ' + train_path + '.parent ...')
        seq_parents = np.load(train_path + '.parent')
        print('calc children ...')
        children, roots = preprocessing.children_and_roots(seq_parents)

        #token_list = list(corpus.create_or_read_dict_types_string(train_path, mapping=m, spacy_vocab=parser.vocab))

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
                visualize.visualize_seq_node_list(seq_tree_seq['trees'], types, 'forest_out_' + str(i) + '.png')

    @unittest.skip("skip")
    def test_iterator_sequence_trees_cbot(self):
        pp = pprint.PrettyPrinter(indent=2)

        train_path = '/media/arne/WIN/Users/Arne/ML/data/corpora/wikipedia/process_sentence7/WIKIPEDIA_articles100_offset0'
        sample_count = 5
        max_depth = 3

        types = corpus.read_types(train_path)

        # load corpus data
        print('load corpus data from: ' + train_path + '.data ...')
        seq_data = np.load(train_path + '.data')
        print('load corpus parents from: ' + train_path + '.parent ...')
        seq_parents = np.load(train_path + '.parent')
        print('calc children ...')
        children, roots = preprocessing.children_and_roots(seq_parents)
        #print(children[3106218])

        #print('load depths from: ' + train_path + '.depth1.collected')
        #depth1_collected = np.load(train_path + '.depth1.collected')

        #idx = depth1_collected[9663491] # idx = 3106218
        #print('idx: '+str(idx))
        #print('load depths ...')
        #seq_depths = np.load(train_path + '.depth')
        #print('depth of idx: '+str(seq_depths[idx]))

        #print('children of idx:')
        #print(children[idx])
        #seq_tree = sequence_node_pb2.SequenceNode()
        #preprocessing.build_sequence_tree(seq_data, children, idx, seq_tree, max_depth)
        #pp.pprint(seq_tree)

        #print('load mapping from file: ' + train_path + '.mapping ...')
        #m = pickle.load(open(train_path + '.mapping', "rb"))
        #print(4 in m.values)
        #print('len(mapping): ' + str(len(m)))
        #rev_m = corpus.revert_mapping(m)
        #print('load spacy ...')
        #parser = spacy.load('en')

        #token_list = list(corpus.create_or_read_dict_types_string(train_path))

        #visualize.visualize('orig.png', (seq_data, seq_parents), rev_m, parser.vocab, constants.vocab_manual)

        for i, seq_tree_seq in enumerate(train_fold_nce.iterator_sequence_trees_cbot(train_path, max_depth, seq_data, children,
                                                                                sample_count, loaded_global_step=0)):
            if i == 5:
                break

            #pp.pprint(seq_tree_seq)
            t_all = [str(seq_tree_seq['trees'][j]) for j in range(sample_count + 1)]
            #if len(set(t_all)) <= 1:
            print('IDX: ' + str(i))
            #print(t_all[0])
            pp.pprint(seq_tree_seq)
            visualize.visualize_seq_node_list(seq_tree_seq['trees'], types, 'forest_out_' + str(i) + '.png')

    @unittest.skip("skip")
    def test_check_depth_collected(self):
        train_path = '/media/arne/WIN/Users/Arne/ML/data/corpora/wikipedia/process_sentence7/WIKIPEDIA_articles100_maxdepth10'
        print('load depths ...')
        seq_depths = np.load(train_path + '.depth')
        print(len(seq_depths))

        max_depth = 10

        deph_collected = []
        for i in range(max_depth):
            print('load depths from: ' + train_path + '.depth'+str(i)+'.collected')
            dc = np.load(train_path + '.depth'+str(i)+'.collected')
            print(len(dc))
            deph_collected.append(dc)

        for i, current_depth in enumerate(seq_depths):
            for d in range(min(current_depth+1, max_depth)):
                if i not in deph_collected[d]:
                    print(str(i) +' not in '+str(d))

    def test_merge_dicts(self):
        vecs1 = np.array([[0.11], [0.21], [0.31], [0.51], [0.61], [0.71], [0.91]])          # missing: 0.01, 0.41, 0.81
        #vecs1 = np.ndarray(shape=(0, 1))
        vecs2 = np.array([[0.52], [0.12], [0.32], [0.82], [0.02], [0.62], [0.72], [0.42]])  # missing: 0.22, 0.92
        #vecs2 = np.ndarray(shape=(0, 1))

        types1 = ['1', '2', '3', '5', '6', '7', '9']
        #types1 = []
        types2 = ['5', '1', '3', '8', '0', '6', '7', '4']
        #types2 = []

        new_vecs, new_types = corpus.merge_dicts(vecs1, types1, vecs2, types2, add=True, remove=True)
        print(new_vecs)
        print(new_types)


#def test_create_or_read_dict_plain_token():
#    for type in corpus.create_or_read_dict_plain_token(fn='temp', mapping=mapping, spacy_vocab=nlp.vocab)




#if __name__ == '__main__':

    #test_depth()
    #test_build_build_sequence_tree_with_candidate()
    #test_get_all_children()
    #test_read_data_2()
    #test_collected_shuffled_child_indices()
    #test_sequence_tree_to_arrays()
    #test_iterator_sequence_trees()
    #test_iterator_sequence_trees_cbot()
    #test_depth()
    #test_check_depth_collected()







