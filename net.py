from __future__ import print_function
import torch
import torch.nn as nn
from torch.autograd import Variable



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

    def forward(self, data, types, edges, graphs):
        pos = len(data) -1
        data_vec = torch.from_numpy(self.data_vecs[types[pos]][data[pos]])
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