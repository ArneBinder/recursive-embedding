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
from net import Net
import math
from sys import exit


dim = 300
# edge_count = 60
#seq_length = 10

slice_size = 10
max_forest_count = 5

nlp = spacy.load('en')
nlp.pipeline = [nlp.tagger, nlp.parser]

vecs, mapping, human_mapping = get_word_embeddings(nlp.vocab)
# for processing parser output
data_embedding_maps = {constants.WORD_EMBEDDING: mapping}
# for displaying human readable tokens etc.
data_embedding_maps_human = {constants.WORD_EMBEDDING: human_mapping}
# data vectors
data_vecs = {constants.WORD_EMBEDDING: vecs}

data_dir = '/home/arne/devel/ML/data/'
# create data arrays
(seq_data, seq_types, seq_parents, seq_edges), edge_map_human = \
    read_data(articles_from_csv_reader, nlp, data_embedding_maps, max_forest_count=max_forest_count, max_sen_length=slice_size,
              args={'max_articles': 1, 'filename': data_dir + 'corpora/documents_utf8_filtered_20pageviews.csv'})

net = Net(data_vecs, len(edge_map_human), dim, slice_size, max_forest_count)
print('output size:', net.max_graph_count)

optimizer = optim.Adagrad(net.get_parameters(), lr=0.01, lr_decay=0, weight_decay=0)    # default meta parameters

ind = slice_size
data = np.array(seq_data[0:(ind+1)])
types = np.array(seq_types[0:(ind+1)])
parents = subgraph(seq_parents, 0, (ind+1))
edges = np.array(seq_edges[0:(ind+1)])

graphs = np.array(graph_candidates(parents, ind))


for epoch in range(3):
    outputs = net(data, types, graphs, edges)

    optimizer.zero_grad()

    loss_euclidean = net.loss_euclidean(outputs)
    print('loss_euclidean:', loss_euclidean.squeeze().data[0])

    #loss_cross_entropy = loss_cross_entropy(outputs)
    #print('loss_cross_entropy:', loss_cross_entropy)

    #loss_cross_entropy.backward()
    loss_euclidean.backward()
    optimizer.step()

exit()

outputs_soft = F.softmax(outputs.unsqueeze(0))
print('outputs_soft:', outputs_soft)

expected = Variable(torch.zeros(net.max_graph_count * net.edge_count).type(torch.FloatTensor), requires_grad=False)
expected[0] = 1.
print('expected:', expected)

loss = (outputs_soft - expected).pow(2).sum()
print(loss.data[0])

#optimizer.zero_grad()
#loss.backward()

#loss = F.cross_entropy(outputs_soft, expected)
#print('loss:', loss)