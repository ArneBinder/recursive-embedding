from __future__ import print_function
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from forest import get_roots, get_children
import constants


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

        self.score_weights = Variable(torch.rand(dim, 1), requires_grad=True)

    def calc_embedding_single(self, data, types, children, edges, embeddings, idx):
        embedding = torch.addmm(1, self.data_biases[types[idx]].unsqueeze(0), 1,
                                self.data_vecs[types[idx]][data[idx]].unsqueeze(0), self.data_weights[types[idx]])
        if idx in children:  # leaf
            for child in children[idx]:
                child_embedding = self.calc_embedding_single(data, types, children, edges, embeddings, child)
                embedding += torch.addmm(1, self.edge_biases[edges[child]].unsqueeze(0), 1, child_embedding, self.edge_weights[edges[child]]) #self.add_child_embedding(embedding, child_embedding, edges[child])

            embedding /= len(children[idx]) + 1
        embeddings[idx] = embedding
        return embedding.clamp(min=0)

    def calc_embedding_path_up(self, parents, children, edges, embeddings, start, end):
        embedding = None
        current_pos = start
        while current_pos != end:
            cc = 1
            children_embedding = embeddings[current_pos]
            if current_pos in children:
                cc += len(children[current_pos])
                children_embedding *= cc

            if current_pos == start:
                embedding = children_embedding
                cc = 1
            else:
                embedding += children_embedding
                cc += 1
            # follow the edge
            embedding = torch.addmm(1, self.edge_biases[edges[current_pos]].unsqueeze(0), 1, (embedding / cc).clamp(min=0), self.edge_weights[edges[current_pos]])
            current_pos += parents[current_pos]
        return embedding

    def forward(self, data, types, parents, edges, pos, forests, correct_roots, new_roots):
        edges[pos] = -1
        parents[pos] = -(pos + 1)    # prevent circle when calculating embeddings

        embeddings = {}
        # calc child pointer
        children = get_children(parents)
        # calc forest embeddings (top down from roots)
        for root in correct_roots + new_roots + [pos]:
            # calc embedding and save
            self.calc_embedding_single(data, types, children, edges, embeddings, root)

        scores = []
        for (parent_children, parent_candidate) in forests:
            parents[pos] = parent_candidate
            # link parent_children and roots
            for child in parent_children:
                parents[child] = pos - child
            roots_set = set(correct_roots + new_roots).difference(parent_children)
            if parents[pos] == 0:
                roots_set.add(pos)
            roots = list(roots_set)
            roots.sort()
            rem_edges = []
            for i in range(len(roots) - 1):
                rem_edges.append((roots[i], parents[roots[i]], edges[roots[i]]))
                parents[roots[i]] = roots[i+1] - roots[i]
                edges[roots[i]] = 0

            embedding = embeddings[pos]
            cc = 1
            # re-scale, if pos has children
            if pos in children:
                cc += len(children[pos])
                embedding *= cc

            for child in parent_children:
                embedding += torch.addmm(1, self.edge_biases[edges[child]].unsqueeze(0), 1, embeddings[child].clamp(min=0), self.edge_weights[edges[child]])
            cc += len(parent_children)

            # check, if INTERTREE points to pos (pos has to be a root, but not the first)
            if pos in roots_set and roots[0] != pos:
                embedding += self.calc_embedding_path_up(parents, children, edges, embeddings, roots[0], pos)
                cc += 1

            # blow up
            embedding = torch.cat([embedding.unsqueeze(0)] * self.edge_count)
            # calc with all edges
            embedding = torch.baddbmm(1, self.edge_biases.unsqueeze(1), 1, (embedding / cc).clamp(min=0), self.edge_weights)

            parent = parents[pos]
            current_pos = pos + parent
            while parent != 0:
                cc = 1  # itself
                if current_pos in children:
                    cc += len(children[current_pos])
                current_embedding = embeddings[current_pos] * cc

                # check, if INTERTREE points to current_pos (current_pos has to be a root, but not the first)
                if current_pos in roots_set and roots[0] != current_pos:
                    embedding += self.calc_embedding_path_up(parents, children, edges, embeddings, roots[0], current_pos)
                    cc += 1

                embedding += torch.cat([current_embedding] * self.edge_count) # embedding * edge_w[edge[current_pos]] + edge_b[edge[current_pos]] + children_embedding / cc
                cc += 1
                m1 = torch.cat([self.edge_biases[edges[current_pos]].unsqueeze(0)] * self.edge_count).unsqueeze(1)
                s1 = m1.size()
                m2 = (embedding / cc).clamp(min=0)
                s2 = m2.size()
                m3 = torch.cat([self.edge_weights[edges[current_pos]].unsqueeze(0)] * self.edge_count)
                s3 = m3.size()
                embedding += torch.baddbmm(1, m1, 1,
                                           m2, m3)
                parent = parents[current_pos]
                current_pos = current_pos + parent

            # calc score
            s4 = torch.cat([self.score_weights.unsqueeze(0)] * self.edge_count).size()
            score = torch.bmm(embedding, torch.cat([self.score_weights.unsqueeze(0)] * self.edge_count))
            scores.append(score)

            # reset parent_children
            for child in parent_children:
                parents[child] = 0
            for (i, parent, edge) in rem_edges:
                parents[i] = parent
                edges[i] = edge

        return scores

    def get_parameters(self):
        return self.data_weights.values() + self.data_biases.values() \
               + [self.edge_weights, self.edge_biases, self.score_weights]

    # stats
    def max_class_count(self, slice_size=None, max_forest_count=None, edge_count=None):
        if slice_size is None:
            slice_size = self.slice_size
        if max_forest_count is None:
            max_forest_count = min(self.max_forest_count, slice_size - 1)
        if edge_count is None:
            edge_count = self.edge_count
        return int((slice_size + 1) * (2 ** (max_forest_count - 1)) * edge_count)

    def parameter_count(self):
        return sum([np.prod(np.array(v.size())) for v in self.get_parameters()])
