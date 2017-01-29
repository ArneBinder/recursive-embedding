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