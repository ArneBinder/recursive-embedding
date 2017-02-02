import numpy as np
from tools import list_powerset


# creates a list of possible graphs by trying to link the last data point indicated by ind to the graph of previous ones
# result[0] contains the correct graph
#   len(result) = (ind + 2) * r_c, r_c: count of roots (before ind)
def forest_candidates(parents, ind):
    # assert ind < len(parents), 'ind = ' + str(ind) + ' exceeds data length = ' + str(len(parents))

    parent_correct = parents[ind]
    parents[ind] = -(ind + 1)   # point to outside, should not go into roots
    roots_orig = get_roots(parents)
    roots_cutout = cutout_leaf(parents, ind)

    current_roots = roots_orig + roots_cutout

    correct_forrest_ind = -1

    forests = []
    i = 0
    for parent_candidate in range(-ind, len(parents)-ind):
        parents[ind] = parent_candidate
        # find root of parent_candidate
        parent_candidate_root = ind
        while parents[parent_candidate_root] != 0:
            parent_candidate_root = parent_candidate_root + parents[parent_candidate_root]
        # temporarily remove the root whose tree contains the newly added element
        removed_root_pos = -1
        if parent_candidate_root != ind:
            removed_root_pos = current_roots.index(parent_candidate_root)
            del current_roots[removed_root_pos]
        for roots_subset in list_powerset(current_roots):
            if np.array_equal(roots_subset, roots_cutout) and parent_candidate == parent_correct:
                correct_forrest_ind = i
            forests.append((roots_subset, parent_candidate))
            i += 1

        # re-add root
        if parent_candidate_root != ind:
            current_roots.insert(removed_root_pos, parent_candidate_root)

    return forests, correct_forrest_ind, roots_orig, roots_cutout


# creates a subgraph of the forest represented by parents
# parents outside the new graph are linked to itself
def cut_subgraph(parents):
    # assert start < len(parents), 'start_ind = ' + str(start) + ' exceeds list size = ' + str(len(parents))
    new_roots = []
    for i in range(len(parents)):
        if parents[i] < -i or parents[i] >= len(parents) - i:
            new_roots.append((i, parents[i]))
            parents[i] = 0
    return new_roots


def get_roots(parents):
    return [i[0] for i, parent in np.ndenumerate(parents) if parent == 0]


def get_children(parents):
    result = {}
    for i, parent in np.ndenumerate(parents):
        if parent == 0:
            continue
        i = i[0]
        parent_pos = i + parent
        if parent_pos not in result:
            result[parent_pos] = [i]
        else:
            result[parent_pos] += [i]
    return result


# modifies parents
# leaf parent points outside
def cutout_leaf(parents, pos):
    # assert pos < len(parents), 'pos = ' + str(pos) + ' exceeds list size = ' + str(len(parents))
    new_roots = []
    parents[pos] = -(pos + 1)
    for i, parent in np.ndenumerate(parents):
        i = i[0]
        if i+parent == pos:
            parents[i] = 0
            new_roots.append(i)

    return new_roots
