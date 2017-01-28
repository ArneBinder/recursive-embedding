from __future__ import print_function
from preprocessing import read_data, articles_from_csv_reader, dummy_str_reader, get_word_embeddings, subgraph, \
    graph_candidates
from visualize import visualize
import numpy as np
import spacy
import constants
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

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
(seq_data, seq_types, seq_parents, seq_edges), edge_map = \
    read_data(articles_from_csv_reader, nlp, data_embedding_maps, max_forest_count=max_forest_count, max_sen_length=slice_size,
              args={'max_articles': 1, 'filename': data_dir + 'corpora/documents_utf8_filtered_20pageviews.csv'})

# take first 50 token and visualize the dependency graph
start = 0
end = 50
sliced_parents = subgraph(seq_parents, start, end)
sliced_data = seq_data[start:end]
sliced_types = seq_types[start:end]
sliced_edges = seq_edges[start:end]
visualize('forest.png', (sliced_data, sliced_types, sliced_parents, sliced_edges), data_embedding_maps_human, edge_map)


# create possible graphs for "new" data point with index = 9
graphs = graph_candidates(sliced_parents, 9)
for i, g in enumerate(graphs):
    visualize('forest_'+str(i)+'.png', (sliced_data[:10], sliced_types[:10], g, sliced_edges[:9]+[0]),
              data_embedding_maps_human, edge_map)


class Net(nn.Module):
    def __init__(self, data_vecs, edge_count, dim):
        super(Net, self).__init__()

        # dimension of embeddings
        self.dim = dim

        self.data_vecs = data_vecs
        self.data_weights = {}
        self.data_biases = {}
        for data_type in self.data_vecs.keys():
            vecs = data_vecs[data_type]
            _, vec_dim = vecs.shape
            self.data_weights[data_type] = Variable(torch.zeros(vec_dim, dim), requires_grad=True)
            self.data_biases[data_type] = Variable(torch.zeros(dim), requires_grad=True)

        self.edge_weights = {}
        self.edge_biases = {}
        for i in range(edge_count):
            self.edge_weights[data_type] = Variable(torch.zeros(dim, dim), requires_grad=True)
            self.edge_biases[data_type] = Variable(torch.zeros(dim), requires_grad=True)

        self.score_embedding_weights = Variable(torch.zeros(1, dim), requires_grad=True)
        self.score_embedding_biases = Variable(torch.zeros(dim), requires_grad=True)
        self.score_data_weights = Variable(torch.zeros(1, dim), requires_grad=True)
        self.score_data_biases = Variable(torch.zeros(dim), requires_grad=True)

    def calc_embedding(self, data, types, parents, edges):

        # connect roots
        roots = [i for i, parent in enumerate(parents) if parent == 0]
        for i in range(len(roots) - 1):
            parents[roots[i]] = roots[i + 1]

        # TODO: implement!

        # dummy result
        return Variable(torch.zeros(self.dim), requires_grad=True)

    def calc_score(self, embedding, data_embedding):
        return (self.score_embedding_biases + embedding) * self.score_embedding_weights \
               + (self.score_data_biases + data_embedding) * self.score_data_weights

    def forward(self, data, types, edges, graphs, pos):
        data_vec = torch.from_numpy(data[pos])
        data_embedding = data_vec * self.data_weights[types[pos]] + self.data_biases[types[pos]]
        correct_edge = edges[pos]

        embeddings = []
        scores = []
        for parents in graphs:
            # calc embedding for correct graph at first
            embedding = self.calc_embedding(data, types, parents, edges)
            embeddings.append(embedding)
            scores.append(self.calc_score(embedding, data_embedding))
            for edge in range(edge_count):
                if edge == correct_edge:    # already calculated
                    continue
                edges[pos] = edge
                embedding = self.calc_embedding(data, types, parents, edges)
                embeddings.append(embedding)
                scores.append(self.calc_score(embedding, data_embedding))

        # TODO: implement selection of best-scored embedding!



