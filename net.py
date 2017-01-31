from __future__ import print_function
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np


class Net(nn.Module):
    def __init__(self, data_vecs, edge_count, dim, slice_size, max_forest_count):
        super(Net, self).__init__()

        # dimension of embeddings
        self.dim = dim

        self.edge_count = edge_count

        self.slice_size = slice_size
        self.max_forest_count = max_forest_count
        self.max_graph_count = (slice_size + 1) * (2 ** (max_forest_count - 1))

        self.data_vecs = {}
        self.data_weights = {}
        self.data_biases = {}
        for data_type in data_vecs.keys():
            vecs = data_vecs[data_type]
            _, vec_dim = vecs.shape
            self.data_vecs[data_type] = Variable(torch.from_numpy(vecs), requires_grad=False)
            self.data_weights[data_type] = Variable(torch.rand(vec_dim, dim), requires_grad=True)
            self.data_biases[data_type] = Variable(torch.rand(dim), requires_grad=True)

        self.edge_weights = Variable(torch.rand(edge_count, dim, dim), requires_grad=True)
        self.edge_biases = Variable(torch.rand(edge_count, dim), requires_grad=True)
        #for i in range(edge_count):
        #    self.edge_weights[i] = torch.zeros(dim, dim) # Variable(torch.zeros(dim, dim), requires_grad=True)
        #    self.edge_biases[i] = Variable(torch.zeros(dim), requires_grad=True)

        self.score_embedding_weights = Variable(torch.rand(dim, 1), requires_grad=True)
        #self.score_embedding_biases = Variable(torch.rand(1, dim), requires_grad=True)
        self.score_data_weights = Variable(torch.rand(dim, 1), requires_grad=True)
        # self.score_data_biases = Variable(torch.rand(1, dim), requires_grad=True)
        self.score_bias = Variable(torch.rand(1,1), requires_grad=True)

    def calc_embedding(self, data, types, parents, edges):

        # connect roots
        roots = [i[0] for i, parent in np.ndenumerate(parents) if parent == 0]
        for i in range(len(roots) - 1):
            parents[roots[i]] = roots[i + 1]

        root = roots[-1]

        # let root point to outside (remove circle)
        parents[root] = -len(data)

        # calc child pointer
        children = {}
        for i, parent in np.ndenumerate(parents):
            i = i[0]
            parent_pos = i + parent
            if parent_pos not in children:
                children[parent_pos] = [i]
            else:
                children[parent_pos] += [i]

        result = self.calc_embedding_rec(data, types, children, edges, root)
        # incorporate root edge
        result = self.add_child_embedding(Variable(torch.zeros(result.size())), result, edges[root])

        # reset root for next iteration
        parents[root] = 0
        return result

    def add_child_embedding(self, embedding, child_embedding, edge_id):
        single = True
        if len(embedding.size()) == 3 or len(child_embedding.size()) == 3:
            single = False
        if edge_id < 0:  # no edge specified, take all!
            single = False
            w = self.edge_weights
            b = self.edge_biases.unsqueeze(1)
        else:
            if single:
                w = self.edge_weights[edge_id]
                b = self.edge_biases[edge_id].unsqueeze(0)
            else:
                w = torch.cat([self.edge_weights[edge_id].unsqueeze(0)] * self.edge_count)
                b = torch.cat([self.edge_biases[edge_id].unsqueeze(0)] * self.edge_count).unsqueeze(1)

        if not single and len(embedding.size()) == 2:
            embedding = torch.cat([embedding.unsqueeze(0)] * self.edge_count)
        if not single and len(child_embedding.size()) == 2:
            child_embedding = torch.cat([child_embedding.unsqueeze(0)] * self.edge_count)

        if single:

            return embedding + torch.addmm(1, b, 1, child_embedding, w)  # child_embedding.mm(w) + b
        else:
            s1 = b.size()
            s2 = child_embedding.size()
            s3 = w.size()
            return embedding + torch.baddbmm(1, b, 1, child_embedding, w)

    def calc_embedding_rec(self, data, types, children, edges, idx):
        embedding = torch.addmm(1, self.data_biases[types[idx]].unsqueeze(0), 1, self.data_vecs[types[idx]][data[idx]].unsqueeze(0), self.data_weights[types[idx]]) # self.data_vecs[types[idx]][data[idx]].unsqueeze(0).mm() +
        if idx not in children:     # leaf
            return embedding
        for child in children[idx]:
            child_embedding = self.calc_embedding_rec(data, types, children, edges, child)
            embedding = self.add_child_embedding(embedding, child_embedding, edges[child])
        return embedding.clamp(min=0)

    def calc_score(self, embedding, data_embedding_s):
        s1 = data_embedding_s.size()
        s2 = embedding.size()
        s3 = self.score_embedding_weights.size()
        if len(embedding.size()) == 3:
            s = embedding.size()[0]
            return torch.baddbmm(1, torch.cat([data_embedding_s.unsqueeze(0)]*s), 1, embedding, torch.cat([self.score_embedding_weights.unsqueeze(0)]*s))
        else:
            return torch.addmm(1, data_embedding_s, 1, embedding, self.score_embedding_weights) #embedding.mm(self.score_embedding_weights) #torch.addmm(1, data_embedding_s, 1, embedding, self.score_embedding_weights) #embedding.mm(self.score_embedding_weights) + data_embedding_s

    #def calc_score_batch(self, embedding, data_embedding_s):
    #    s1 = embedding.size()
    #    return (self.score_biases + embedding).mm(self.score_embedding_weights) + data_embedding_s

    def forward(self, data, types, graphs, edges, pos):
        #pos = len(data) - 1
        t = types[pos]
        d = data[pos]
        # data_vec = self.data_vecs[t][d].unsqueeze(0)
        #s1 = self.data_biases[t].unsqueeze(0).size()
        #s2 = self.data_vecs[t][d].unsqueeze(0).size()
        #s3 = self.data_weights[t].size()

        data_embedding = torch.addmm(1, self.data_biases[t].unsqueeze(0), 1, self.data_vecs[t][d].unsqueeze(0), self.data_weights[t])
        #data_embedding += self.data_biases[t]
        data_embedding_s = torch.addmm(1, self.score_bias, 1, data_embedding, self.score_data_weights) #  data_embedding.mm(self.score_data_weights)  #torch.mm(1, self.score_biases, 1, data_embedding, self.score_data_weights) #self.score_biases + data_embedding.mm(self.score_data_weights)
        correct_edge = edges[pos]
        edges[pos] = -1

        scores = [] # Variable(torch.zeros(self.max_graph_count * self.edge_count))

        #i = 0
        graph_count, _ = graphs.shape
        for j in range(graph_count):
        #for s, parents in np.ndenumerate(graphs):
            # calc embedding for correct graph at first
            embedding = self.calc_embedding(data, types, graphs[j], edges)
            #s1 = embedding.size()
            # print(s1)
            #scores[i] = self.calc_score(embedding, data_embedding)
            scores.append(self.calc_score(embedding, data_embedding_s).squeeze())
            #i += 1
            #for edge in range(self.edge_count):
            #    if edge == correct_edge:    # already calculated
            #        continue
            #    edges[pos] = edge
            #    embedding = self.calc_embedding(data, types, graphs[j], edges)
            #    #scores[i] = self.calc_score(embedding, data_embedding)
            #    scores.append(self.calc_score(embedding, data_embedding_s))
            #    i += 1

        return scores

    def get_parameters(self):
        return self.data_weights.values() + self.data_biases.values() \
                 + [self.edge_weights, self.edge_biases, self.score_embedding_weights,
                    self.score_data_weights, self.score_bias]
