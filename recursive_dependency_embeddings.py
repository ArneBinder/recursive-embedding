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
import torch.optim as optim

dim = 300
# edge_count = 60
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
(seq_data, seq_types, seq_parents, seq_edges), edge_map_human = \
    read_data(articles_from_csv_reader, nlp, data_embedding_maps, max_forest_count=max_forest_count, max_sen_length=slice_size,
              args={'max_articles': 1, 'filename': data_dir + 'corpora/documents_utf8_filtered_20pageviews.csv'})


class Net(nn.Module):
    def __init__(self, data_vecs, edge_count, dim, slice_size, max_forest_count):
        super(Net, self).__init__()

        # dimension of embeddings
        self.dim = dim

        self.slice_size = slice_size
        self.max_forest_count = max_forest_count
        self.max_graph_count = (slice_size + 1) * 2 ** (max_forest_count - 1)

        self.data_vecs = {}
        self.data_weights = {}
        self.data_biases = {}
        for data_type in self.data_vecs.keys():
            vecs = data_vecs[data_type]
            _, vec_dim = vecs.shape
            self.data_vecs[data_type] = torch.from_numpy(vecs)
            self.data_weights[data_type] = Variable(torch.zeros(vec_dim, dim), requires_grad=True)
            self.data_biases[data_type] = Variable(torch.zeros(dim), requires_grad=True)

        self.edge_weights = Variable(torch.zeros(edge_count, dim, dim), requires_grad=True)
        self.edge_biases = Variable(torch.zeros(edge_count, dim), requires_grad=True)
        #for i in range(edge_count):
        #    self.edge_weights[i] = torch.zeros(dim, dim) # Variable(torch.zeros(dim, dim), requires_grad=True)
        #    self.edge_biases[i] = Variable(torch.zeros(dim), requires_grad=True)

        self.score_embedding_weights = Variable(torch.zeros(1, dim), requires_grad=True)
        self.score_embedding_biases = Variable(torch.zeros(dim), requires_grad=True)
        self.score_data_weights = Variable(torch.zeros(1, dim), requires_grad=True)
        self.score_data_biases = Variable(torch.zeros(dim), requires_grad=True)

    def calc_embedding(self, data, types, parents, edges):

        # connect roots
        roots = [i for i, parent in enumerate(parents) if parent == 0]
        for i in range(len(roots) - 1):
            parents[roots[i]] = roots[i + 1]

        root = roots[-1]

        # calc child pointer
        children = {}
        for i, parent in enumerate(parents):
            if i + parent not in children:
                children[i + parent] = [i]
            else:
                children[i + parent] += [i]

        return self.calc_embedding_rec(data, types, children, edges, root)

    def calc_embedding_rec(self, data, types, children, edges, idx):
        embedding = self.data_vecs[types[idx]][data[idx]] * self.data_weights[types[idx]] + self.data_biases[types[idx]]
        if idx not in children:     # leaf
            return embedding
        for child in children[idx]:
            embedding += self.calc_embedding_rec(data, types, children, edges, child) * self.edge_weights[edges[child]] \
                         + self.edge_biases[edges[child]]
        return embedding

    def calc_score(self, embedding, data_embedding):
        return (self.score_embedding_biases + embedding) * self.score_embedding_weights \
               + (self.score_data_biases + data_embedding) * self.score_data_weights

    def forward(self, data, types, edges, graphs, pos):
        data_vec = torch.from_numpy(data[pos])
        data_embedding = data_vec * self.data_weights[types[pos]] + self.data_biases[types[pos]]
        correct_edge = edges[pos]

        scores = Variable(torch.zeros(self.max_graph_count))
        i = 0
        for parents in graphs:
            # calc embedding for correct graph at first
            embedding = self.calc_embedding(data, types, parents, edges)
            scores[i] = self.calc_score(embedding, data_embedding)
            i += 1
            for edge in range(self.edge_count):
                if edge == correct_edge:    # already calculated
                    continue
                edges[pos] = edge
                embedding = self.calc_embedding(data, types, parents, edges)
                scores[i] = self.calc_score(embedding, data_embedding)
                i += 1

        scores = nn.LogSoftmax().forward(scores)
        # expected = torch.zeros(len(scores))
        # expected[0] = 1
        # criterion = nn.CrossEntropyLoss()
        # loss = criterion(scores, expected)

        return scores


net = Net(data_vecs, len(edge_map_human), dim, slice_size, max_forest_count)

params = list(net.parameters())
print(len(params))

criterion = nn.CrossEntropyLoss() # use a Classification Cross-Entropy loss
optimizer = optim.Adagrad(net.parameters(), lr=0.01, lr_decay=0, weight_decay=0)    # default meta parameters

expected = Variable(torch.zeros(slice_size))
expected[0] = 1

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    slice_start = 0
    #for i, data in enumerate(trainloader, 0):
    while slice_start < len(seq_data):
        for i in range(slice_start, min(len(seq_data), slice_start + slice_size)):
            # get the inputs
            data = seq_data[slice_start:i]
            types = seq_types[slice_start:i]
            parents = subgraph(seq_parents, slice_start, i)
            edges = seq_edges[slice_start:i]

            #inputs, labels = data

            # wrap them in Variable
            #inputs, labels = Variable(inputs), Variable(labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(Variable(data), Variable(types), Variable(parents), Variable(edges))
            loss = criterion(outputs, expected)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data[0]
            if i % 100 == 99:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

        slice_start += slice_size
print('Finished Training')