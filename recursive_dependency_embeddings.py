from __future__ import print_function
from preprocessing import read_data, articles_from_csv_reader, dummy_str_reader, get_word_embeddings, subgraph, \
    graph_candidates
from visualize import visualize
import numpy as np
import spacy
import constants
import torch
import torch.optim as optim
import datetime
from torch.autograd import Variable
from net import Net
from tools import mkdir_p
from tensorboard_logger import configure, log_value
import random

dim = 300
# edge_count = 60
# seq_length = 10

max_slice_size = 20  # 75
max_forest_count = 5  # 10

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
log_dir = data_dir + 'summaries/train_{:%Y-%m-%d_%H:%M:%S}/'.format(datetime.datetime.now())
mkdir_p(log_dir)
# configure tensorboard logging
configure(log_dir, flush_secs=1)

# create data arrays
(seq_data, seq_types, seq_parents, seq_edges), edge_map_human = \
    read_data(articles_from_csv_reader, nlp, data_embedding_maps, max_forest_count=max_forest_count,
              max_sen_length=max_slice_size,
              args={'max_articles': 10, 'filename': data_dir + 'corpora/documents_utf8_filtered_20pageviews.csv'})

print('data length:', len(seq_data))

net = Net(data_vecs, len(edge_map_human), dim, max_slice_size, max_forest_count)
#loss_fn = torch.nn.L1Loss(size_average=True)
loss_fn = torch.nn.CrossEntropyLoss(size_average=True)

params = list(net.get_parameters())
print('variables to train:', len(params))

# criterion = nn.CrossEntropyLoss() # use a Classification Cross-Entropy loss
optimizer = optim.Adagrad(net.get_parameters(), lr=0.01, lr_decay=0, weight_decay=0)  # default meta parameters

epochs = 10
max_steps = 100  # per slice_size

print('max_slice_size:', max_slice_size)
print('max_forest_count:', max_forest_count)
print('max_graph_count:', net.max_graph_count)
print('edge_count:', net.edge_count)
print('epochs:', epochs)
print('max_steps:', max_steps)

interval_avg = 50


for slice_size in range(1, max_slice_size):
    for epoch in range(epochs):
        running_loss = 0.0
        slice_start = 0
        # TODO: check loop end!
        i = 0
        slice_starts = range(0, min(max_steps*slice_size, len(seq_data)-slice_size), slice_size)
        random.shuffle(slice_starts)
        for slice_start in slice_starts:
        #while slice_start < min(max_steps*slice_size, len(seq_data)-slice_size):
            #for i in range(slice_start + 1, min(max_steps, len(seq_data) + 1, slice_start + max_slice_size + 1)):
                # get the inputs
            slice_end = slice_start + slice_size
            data = np.array(seq_data[slice_start:slice_end])
            types = np.array(seq_types[slice_start:slice_end])
            parents = subgraph(seq_parents, slice_start, slice_end)
            edges = np.array(seq_edges[slice_start:slice_end])
            if len([True for parent in parents if parent == 0]) > net.max_forest_count:
                continue
            graphs = np.array(graph_candidates(parents, len(parents)-1))
            correct_edge = edges[-1]
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(data, types, graphs, edges, len(parents)-1)
            outputs_cat = torch.cat(outputs).squeeze()
            expected = Variable(torch.cat(
                (torch.zeros(correct_edge), torch.ones(1), torch.zeros(len(outputs_cat) - correct_edge - 1)))).type(
                torch.FloatTensor).squeeze()

            #loss = loss_fn(outputs_cat, expected)
            loss = loss_fn(outputs_cat, Variable(torch.ones(1)*correct_edge).type(torch.LongTensor))

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.squeeze().data[0]
            # if ((i * 100) % (len(seq_data)-slice_start)*slice_size == 0):
            # if ((i * interval_avg) % num_steps) == 0 or i == 1:
            # if i > 1:
            #    average_loss = average_loss * interval_avg / num_steps
            # if i % step_size == step_size*10 -1:  # print every 2000 mini-batches
            # print('[%5d] loss: %.3f' % (i + 1, running_loss * interval_avg / num_steps))

            #print('[%d, %5d] loss: %15.3f   size: %2d' % (epoch + 1, i, running_loss, slice_size))
            #log_value('loss', running_loss, i)


            #slice_start += slice_size
            i += 1

        print('[%2d %4d] loss: %15.3f' % (slice_size, epoch + 1, running_loss / len(slice_starts)))

        model_fn = log_dir + 'model-' + '{:03d}'.format(epoch)
        #print('write model to ' + model_fn)
        #with open(model_fn, 'w') as f:
        #    torch.save(net, f)

print('Finished Training')
