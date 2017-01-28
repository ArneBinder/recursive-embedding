from __future__ import print_function
from preprocessing import read_data, articles_from_csv_reader, dummy_str_reader, get_word_embeddings
from tools import list_powerset
from visualize import visualize
import numpy as np
import spacy
import constants

dim = 300
edge_count = 60
seq_length = 10

slice_size = 75
max_forest_count = 10

nlp = spacy.load('en')
nlp.pipeline = [nlp.tagger, nlp.parser]

vecs, mapping, human_mapping = get_word_embeddings(nlp.vocab)
# for processing parser output
data_embedding_maps = {constants.WORD_EMBEDDING: mapping}
# for displaying human readable tokens etc.
data_embedding_maps_human = {constants.WORD_EMBEDDING: human_mapping}
# data vectors
data_vecs = {constants.WORD_EMBEDDING: vecs}

data_dir = '/media/arne/DATA/DEVELOPING/ML/data/'
# create data arrays
(seq_data, seq_types, seq_heads, seq_edges), edge_map = \
    read_data(articles_from_csv_reader, nlp, data_embedding_maps, max_forest_count=max_forest_count, max_sen_length=slice_size,
              args={'max_articles': 1, 'filename': data_dir + 'corpora/documents_utf8_filtered_20pageviews.csv'})


def slice_heads(heads, start, end):
    assert start < len(seq_data), 'start_ind = ' + str(start) + ' exceeds list size = ' + str(len(seq_data))
    new_heads = heads[start:end]
    for i in range(len(new_heads)):
        if new_heads[i] < -i or new_heads[i] >= len(new_heads) - i:
            new_heads[i] = 0
    return new_heads

# take first 50 token and visualize the dependency graph
start = 0
end = 50
sliced_heads = slice_heads(seq_heads, start, end)
sliced_data = seq_data[start:end]
sliced_types = seq_types[start:end]
sliced_edges = seq_edges[start:end]
visualize('forest.png', sliced_data, sliced_types, sliced_heads, sliced_edges, data_embedding_maps_human, edge_map)


# creates possible graphs by trying to link the last data point indicated by ind to the previous
#   len(result) = (ind + 2) * r_c, r_c: count of roots (before ind)
def create_graph_candidates(heads, ind):
    assert ind < len(heads), 'ind = '+str(ind)+' exceeds data length = '+str(len(heads))
    sliced_heads = np.array(slice_heads(heads, 0, ind+1))
    sliced_heads_predict = np.array(slice_heads(heads, 0, ind))
    roots = [i for i, head in enumerate(sliced_heads_predict) if head == 0]

    sliced_heads_candidates = [sliced_heads]
    for head_target in range(ind+1):
        candidate_temp = np.append(sliced_heads_predict, [head_target - ind])
        # find root of head_target
        head_target_root = head_target
        while candidate_temp[head_target_root] != 0:
            head_target_root = head_target_root + candidate_temp[head_target_root]
        # temporarily remove the root whose tree contains the newly added element
        if head_target_root != ind:
            roots.remove(head_target_root)
        for roots_subset in list_powerset(roots):
            candidate = candidate_temp.copy()
            for root in roots_subset:
                candidate[root] = ind - root
            if not np.array_equal(sliced_heads, candidate):
                sliced_heads_candidates.append(candidate)

        # re-add root
        if head_target_root != ind:
            roots.append(head_target_root)

    return sliced_heads_candidates


heads_candidates = create_graph_candidates(sliced_heads, 9)
for i, c in enumerate(heads_candidates):
    visualize('forest_'+str(i)+'.png', sliced_data[:10], sliced_types[:10], c, sliced_edges[:9]+[0], data_embedding_maps_human, edge_map)






