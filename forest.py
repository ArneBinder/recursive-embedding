import numpy as np
from tools import list_powerset


# creates a list of possible graphs by trying to link the last data point indicated by ind to the graph of previous ones
# result[0] contains the correct graph
#   len(result) = (ind + 2) * r_c, r_c: count of roots (before ind)
def graph_candidates(parents, ind):
    assert ind < len(parents), 'ind = ' + str(ind) + ' exceeds data length = ' + str(len(parents))
    sliced_parents = np.array(subgraph(parents, 0, ind + 1))
    sliced_parents_predict = np.array(subgraph(parents, 0, ind))
    roots = get_roots(sliced_parents_predict)

    sliced_parents_candidates = [sliced_parents]
    for parent_candidate in range(ind+1):
        candidate_temp = np.append(sliced_parents_predict, [parent_candidate - ind])
        # find root of parent_candidate
        parent_candidate_root = parent_candidate
        while candidate_temp[parent_candidate_root] != 0:
            parent_candidate_root = parent_candidate_root + candidate_temp[parent_candidate_root]
        # temporarily remove the root whose tree contains the newly added element
        if parent_candidate_root != ind:
            roots.remove(parent_candidate_root)
        for roots_subset in list_powerset(roots):
            candidate = candidate_temp.copy()
            for root in roots_subset:
                candidate[root] = ind - root
            if not np.array_equal(sliced_parents, candidate):
                sliced_parents_candidates.append(candidate)

        # re-add root
        if parent_candidate_root != ind:
            roots.append(parent_candidate_root)

    return sliced_parents_candidates


# creates a subgraph of the forest represented by parents
# parents outside the new graph are linked to itself
def subgraph(parents, start, end):
    assert start < len(parents), 'start_ind = ' + str(start) + ' exceeds list size = ' + str(len(parents))
    new_parents = parents[start:end]
    for i in range(len(new_parents)):
        if new_parents[i] < -i or new_parents[i] >= len(new_parents) - i:
            new_parents[i] = 0
    return new_parents


def get_roots(parents):
    return [i[0] for i, parent in np.ndenumerate(parents) if parent == 0]


def get_children(parents):
    result = {}
    for i, parent in np.ndenumerate(parents):
        i = i[0]
        parent_pos = i + parent
        if parent_pos not in result:
            result[parent_pos] = [i]
        else:
            result[parent_pos] += [i]
    return result

# modifies parents, but not children
def cutout_leaf(parents, children, pos):
    assert pos < len(parents), 'pos = ' + str(pos) + ' exceeds list size = ' + str(len(parents))
    parents[pos] = 0
    if pos in children:
        for child in children[pos]:
            parents[child] = 0
