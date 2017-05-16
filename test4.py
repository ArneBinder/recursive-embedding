import corpus_wikipedia
import preprocessing

import corpus_wikipedia
import numpy as np
import pprint


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
    seq_tree_c = preprocessing.build_sequence_tree_with_candidates(seq_data, children, roots[0], insert_idx, candidate_indices)
    pp.pprint(seq_tree_c)

if __name__ == '__main__':
    #test_depth()
    test_build_build_sequence_tree_with_candidates()







