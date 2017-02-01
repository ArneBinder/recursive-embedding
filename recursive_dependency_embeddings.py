from __future__ import print_function
from preprocessing import read_data, articles_from_csv_reader, dummy_str_reader, get_word_embeddings, subgraph, \
    graph_candidates
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
import scipy.stats.stats as st
import argparse


def main():
    arg_parser = argparse.ArgumentParser()

    # data locations
    arg_parser.add_argument('-ld', '--log-dir', default='/home/arne/devel/ML/data/summaries')
    arg_parser.add_argument('-cf', '--corpus-file', default='/home/arne/devel/ML/data/corpora/documents_utf8_filtered_20pageviews.csv')
    # parsing
    arg_parser.add_argument('-a', '--max-article-count', type=int, default=10)
    # max-forest-count = 10 captures 0,9998 % of tokens in wikipedia corpus
    arg_parser.add_argument('-f', '--max-forest-count', type=int, default=10)
    # max-slice-size = 75 (sentence length while parsing) captures 0,9929 % of tokens in wikipedia corpus
    arg_parser.add_argument('-s', '--max-slice-size', type=int, default=75)
    # model
    arg_parser.add_argument('-d', '--dimensions', type=int, default=300)
    # training
    arg_parser.add_argument('-e', '--max-epochs-per-size', type=int, default=50)
    arg_parser.add_argument('-st', '--max-steps-per-epoch', type=int, default=1000)
    # increase slice_size if the skew over the last loss-history-size losses is smaller than loss-skew-threshold
    arg_parser.add_argument('-lh', '--loss-history-size', type=int, default=10)
    arg_parser.add_argument('-sk', '--loss-skew-threshold', type=float, default=0.01)
    args = arg_parser.parse_args()

    print('ARGUMENTS:')
    for v in vars(args):
        print('\t'+v+':\t', vars(args)[v])

    print('\n')

    corpus_file_name = args.corpus_file
    log_dir = args.log_dir + '/train_{:%Y-%m-%d_%H:%M:%S}/'.format(datetime.datetime.now())
    mkdir_p(log_dir)
    # configure tensorboard logging
    configure(log_dir, flush_secs=2)

    dim = args.dimensions # 300

    max_article_count = args.max_article_count #10
    max_slice_size = args.max_slice_size #75
    max_forest_count = args.max_forest_count #10

    nlp = spacy.load('en')
    nlp.pipeline = [nlp.tagger, nlp.parser]

    print('extract word embeddings from spaCy...')
    vecs, mapping, human_mapping = get_word_embeddings(nlp.vocab)
    # for processing parser output
    data_embedding_maps = {constants.WORD_EMBEDDING: mapping}
    # for displaying human readable tokens etc.
    data_embedding_maps_human = {constants.WORD_EMBEDDING: human_mapping}
    # data vectors
    data_vecs = {constants.WORD_EMBEDDING: vecs}

    # create data arrays
    (seq_data, seq_types, seq_parents, seq_edges), edge_map_human = \
        read_data(articles_from_csv_reader, nlp, data_embedding_maps, max_forest_count=max_forest_count,
                  max_sen_length=max_slice_size,
                  args={'max_articles': max_article_count, 'filename': corpus_file_name})

    print('data length (token):', len(seq_data))

    net = Net(data_vecs, len(edge_map_human), dim, max_slice_size, max_forest_count)
    print('edge_count:', net.edge_count)
    params = list(net.get_parameters())
    print('tensors to train:', len(params))
    print('total parameter_count:', net.parameter_count())
    print('max_graph_count (depends on max_slice_size and max_forest_count):', net.max_graph_count)
    print('max_class_count (max_graph_count * edge_count):', net.max_class_count())

    #loss_fn = torch.nn.L1Loss(size_average=True)
    loss_fn = torch.nn.CrossEntropyLoss(size_average=True)
    optimizer = optim.Adagrad(net.get_parameters(), lr=0.01, lr_decay=0, weight_decay=0)  # default meta parameters

    max_epochs = args.max_epochs_per_size #50
    max_steps = args.max_steps_per_epoch # 1000  # per slice_size
    loss_hist_size = args.loss_history_size # 10
    loss_skew_threshold = args.loss_skew_threshold # 0.1

    # interval_avg = 50

    print('\n')
    time_train_start = datetime.datetime.now()
    print(str(time_train_start), 'START TRAINING')
    for slice_size in range(1, max_slice_size):
        print('max_class_count (slice_size='+str(slice_size)+'):', net.max_class_count(slice_size))
        losses = []
        loss_skew = loss_skew_threshold + 1
        epoch = 0
        while epoch < max_epochs and (loss_skew > loss_skew_threshold or len(losses) < loss_hist_size):
            running_loss = 0.0
            slice_step = 0
            # get slices of full size (slice_size)
            slice_starts = range(0, min(max_steps*slice_size, len(seq_data) - slice_size + 1), slice_size)
            random.shuffle(slice_starts)
            for slice_start in slice_starts:
                # get the inputs
                slice_end = slice_start + slice_size
                data = np.array(seq_data[slice_start:slice_end])
                types = np.array(seq_types[slice_start:slice_end])
                parents = subgraph(seq_parents, slice_start, slice_end)
                edges = np.array(seq_edges[slice_start:slice_end])
                if len([True for parent in parents if parent == 0]) > net.max_forest_count:
                    continue

                # predict last token
                graphs = np.array(graph_candidates(parents, len(parents)-1))
                correct_edge = edges[-1]
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(data, types, graphs, edges, len(parents)-1)
                outputs_cat = torch.cat(outputs).squeeze()

                loss = loss_fn(outputs_cat, Variable(torch.ones(1)*correct_edge).type(torch.LongTensor))

                loss.backward()
                optimizer.step()


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
                slice_step += 1

            running_loss /= len(slice_starts)
            losses.append(running_loss)
            losses = losses[-loss_hist_size:]
            loss_skew = st.skew(losses)
            # print statistics
            print(str(datetime.datetime.now() - time_train_start)+' [%2d %4d] loss: %15.3f loss_skew: %5.2f' % (slice_size, epoch + 1, running_loss, loss_skew))
            log_value('loss', running_loss / len(slice_starts), (slice_size - 1) * max_slice_size + epoch)
            epoch += 1

        model_fn = log_dir + 'model-' + '{:03d}'.format(slice_size)
        #print('write model to ' + model_fn)
        #with open(model_fn, 'w') as f:
        #    torch.save(net, f)

    print('Finished Training')

if __name__ == "__main__":
    main()
