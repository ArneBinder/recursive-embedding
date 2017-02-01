from __future__ import print_function
from preprocessing import read_data, articles_from_csv_reader, dummy_str_reader, get_word_embeddings
import spacy
import constants
from visualize import visualize
from forest import subgraph, forest_candidates
import numpy as np

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
(seq_data, seq_types, seq_parents, seq_edges), edge_map_human = \
    read_data(articles_from_csv_reader, nlp, data_embedding_maps, max_forest_count=max_forest_count, max_sen_length=slice_size,
              args={'max_articles': 1, 'filename': '/home/arne/devel/ML/data/corpora/documents_utf8_filtered_20pageviews.csv'})

# take first 50 token and visualize the dependency graph
start = 0
end = 10
sliced_parents = subgraph(seq_parents, start, end)
sliced_data = seq_data[start:end]
sliced_types = seq_types[start:end]
sliced_edges = seq_edges[start:end]
visualize('forest.png', (sliced_data, sliced_types, sliced_parents, sliced_edges), data_embedding_maps_human, edge_map_human)


# create possible graphs for "new" data point with index = 9
graphs, correct_forrest_ind = forest_candidates(np.array(sliced_parents), 9)
for i, g in enumerate(graphs):
    visualize('forest_' + str(i) +'.png', (sliced_data, sliced_types, g, sliced_edges[:(end-1)]+[0]),
              data_embedding_maps_human, edge_map_human)


exit()

def calc_embedding(data, types, parents, edges):
    # connect roots
    roots = [i for i, parent in enumerate(parents) if parent == 0]
    for i in range(len(roots) - 1):
        parents[roots[i]] = roots[i + 1]

    root = roots[-1]

    # calc child pointer
    children = {}
    for i, parent in enumerate(parents):
        parent_pos = i + parent
        # skip circle at root pos
        if parent_pos == i:
            continue
        if parent_pos not in children:
            children[parent_pos] = [i]
        else:
            children[parent_pos] += [i]

    return calc_embedding_rec(data, types, children, edges, root)


def calc_embedding_rec(data, types, children, edges, idx):
    # embedding = data_vecs[types[idx]][data[idx]] * data_weights[types[idx]] + data_biases[types[idx]]
    embedding = data_embedding_maps_human[types[idx]][data[idx]]

    # leaf
    if idx not in children:
        return embedding

    embedding += '['

    for child in children[idx]:
        # embedding += calc_embedding_rec(data, types, children, edges, child) * edge_weights[edges[child]] + edge_biases[edges[child]]
        embedding += ' ' + edge_map_human[edges[child]] + '(' + calc_embedding_rec(data, types, children, edges, child) + ')'

    embedding += ']'

    return embedding

print(calc_embedding(sliced_data, sliced_types, sliced_parents, sliced_edges))