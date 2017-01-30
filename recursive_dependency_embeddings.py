from __future__ import print_function
from preprocessing import read_data, articles_from_csv_reader, dummy_str_reader, get_word_embeddings, subgraph, \
    graph_candidates
from visualize import visualize
import numpy as np
import spacy
import constants
import torch.optim as optim
from net import Net

dim = 300
# edge_count = 60
# seq_length = 10

slice_size = 20         # 75
max_forest_count = 5    # 10

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
              args={'max_articles': 10, 'filename': data_dir + 'corpora/documents_utf8_filtered_20pageviews.csv'})

print('data length:', len(seq_data))

net = Net(data_vecs, len(edge_map_human), dim, slice_size, max_forest_count)

params = list(net.get_parameters())
print('variables to train:', len(params))

#criterion = nn.CrossEntropyLoss() # use a Classification Cross-Entropy loss
optimizer = optim.Adagrad(net.get_parameters(), lr=0.01, lr_decay=0, weight_decay=0)    # default meta parameters

print('max_graph_count: ', net.max_graph_count)
print('edge_count: ', net.edge_count)

#expected = Variable(torch.zeros(net.max_graph_count * net.edge_count))
#expected[0] = 1

interval_avg = 50
max_steps = 100 #len(seq_data)

#for epoch in range(2):  # loop over the dataset multiple times

running_loss = 0.0
slice_start = 0
for epoch in range(1):
    while slice_start < max_steps:
        for i in range(slice_start + 1, min(max_steps, len(seq_data) + 1, slice_start + slice_size + 1)):
            # get the inputs
            data = np.array(seq_data[slice_start:i])
            types = np.array(seq_types[slice_start:i])
            parents = subgraph(seq_parents, slice_start, i)
            edges = np.array(seq_edges[slice_start:i])
            if len([True for parent in parents if parent == 0]) > net.max_forest_count:
                continue
            graphs = np.array(graph_candidates(parents, i - slice_start - 1))

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(data, types, graphs, edges, i - slice_start - 1)
            #loss = criterion(outputs, expected)
            loss = net.loss_euclidean(outputs)
            #loss = net.loss_cross_entropy(outputs)

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.squeeze().data[0]
            #if ((i * 100) % (len(seq_data)-slice_start)*slice_size == 0):
            #if ((i * interval_avg) % num_steps) == 0 or i == 1:
                #if i > 1:
                #    average_loss = average_loss * interval_avg / num_steps
            # if i % step_size == step_size*10 -1:  # print every 2000 mini-batches
            #print('[%5d] loss: %.3f' % (i + 1, running_loss * interval_avg / num_steps))
            print('[%d, %5d] loss: %.3f size: %2d' % (epoch+1, i, running_loss, i - slice_start))
            running_loss = 0.0

        slice_start += slice_size

print('Finished Training')