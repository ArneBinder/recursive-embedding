from __future__ import print_function
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from forest import get_children
import constants


class Net(nn.Module):
    def __init__(self, data_vecs, edge_count, dim, slice_size, max_forest_count, data_maps):
        super(Net, self).__init__()

        # dimension of embeddings
        self.dim = dim

        self.edge_count = edge_count

        self.slice_size = slice_size
        self.max_forest_count = max_forest_count
        self.max_graph_count = (slice_size + 1) * (2 ** (max_forest_count - 1))

        self.data_maps = data_maps

        # activation function
        self.activation_fn = nn.ReLU()

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

    # Calculates the embeddings in the subgraph rooted by idx.
    # The result is put into embeddings[idx].
    # No activation function is applied to the result.
    def calc_embedding_single(self, data, types, children, edges, idx, embeddings):
        b_c = self.data_biases[types[idx]].unsqueeze(0)
        data_vec = self.data_vecs[types[idx]][data[idx]].unsqueeze(0)
        w = self.data_weights[types[idx]]
        embedding = torch.addmm(1, b_c, 1, self.activation_fn(data_vec), w)
        if idx in children:  # leaf
            for child in children[idx]:
                self.calc_embedding_single(data, types, children, edges, child, embeddings)
                b_c = self.edge_biases[edges[child]].unsqueeze(0)
                w_c = self.edge_weights[edges[child]]
                embedding += torch.addmm(1, b_c, 1, self.activation_fn(embeddings[child]), w_c)
            embedding /= len(children[idx]) + 1
        # save embedding without activation, the edge is not yet included
        embeddings[idx] = embedding
        #return self.activation_fn(embedding)

    def calc_embedding_path_up(self, parents, children, edges, embeddings, start, end):
        embedding = None
        current_pos = start
        while current_pos != end:
            cc = 1
            children_embedding = embeddings[current_pos].clone()
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
            b = self.edge_biases[edges[current_pos]].unsqueeze(0)
            w = self.edge_weights[edges[current_pos]]
            embedding = torch.addmm(1, b, 1, self.activation_fn((embedding / cc)), w)
            current_pos += parents[current_pos]
        return embedding

    def forward(self, data, types, parents, edges, pos, forests, roots):
        #edges[pos] = -1
        parents[pos] = -(pos + 1)    # disconnect from rest to get correct children

        roots_set = set(roots)
        embeddings = {}
        # calc child pointer
        children = get_children(parents)
        # calc forest embeddings (top down from roots)
        for root in roots + [pos]:
            # calc embedding and save
            self.calc_embedding_single(data, types, children, edges, embeddings, root)

        scores = []
        for (new_children_candidate, parent_candidate) in forests:
            parents[pos] = parent_candidate
            # link new_children_candidate
            for child in new_children_candidate:
                parents[child] = pos - child
            # new_children_candidate are no roots anymore
            roots_candidate_set = roots_set.difference(new_children_candidate)
            if parents[pos] == 0:
                roots_candidate_set.add(pos)
            roots_candidate = list(roots_candidate_set)
            roots_candidate.sort()
            rem_edges = []
            # link roots with INTERTREE edge and save for restoring
            for i in range(len(roots_candidate) - 1):
                rem_edges.append((roots_candidate[i], parents[roots_candidate[i]], edges[roots_candidate[i]]))
                parents[roots_candidate[i]] = roots_candidate[i + 1] - roots_candidate[i]
                edges[roots_candidate[i]] = constants.INTER_TREE

            embedding = embeddings[pos].clone()
            cc = 1
            # re-scale, if pos has children
            if pos in children:
                cc += len(children[pos])
                embedding *= cc

            for child in new_children_candidate:
                b = self.edge_biases[edges[child]].unsqueeze(0)
                w = self.edge_weights[edges[child]]
                embedding += torch.addmm(1, b, 1, self.activation_fn(embeddings[child]), w)
            cc += len(new_children_candidate)

            # check, if INTERTREE points to pos (pos has to be a root, but not the first)
            if pos in roots_candidate_set and roots_candidate[0] != pos:
                # add embedding for root chain
                embedding += self.calc_embedding_path_up(parents, children, edges, embeddings, roots_candidate[0], pos)
                cc += 1

            # blow up
            embedding = torch.cat([embedding.unsqueeze(0)] * self.edge_count)
            # calc with all edges
            embedding = torch.baddbmm(1, self.edge_biases.unsqueeze(1), 1, self.activation_fn(embedding / cc), self.edge_weights)

            parent = parents[pos]
            last_pos = pos
            current_pos = pos + parent
            while parent != 0:
                # initialize children count
                cc = 1  # one for data vector
                # get embedding for the current node
                current_embedding = embeddings[current_pos].clone()
                if current_pos in children:
                    # If current embedding was averaged over more than one children,
                    # it has to be rescaled to incorporate the new embedding data
                    cc += len(children[current_pos])
                    current_embedding *= cc
                    # if we come from the same branch, remove previously integrated embedding of this.
                    # The embedding of this branch is newly calculated and contained in 'embedding'.
                    if last_pos in children[current_pos]:
                        current_embedding -= embeddings[last_pos]
                        cc -= 1

                # Check, if INTERTREE points to current_pos (current_pos has to be a root, but not the first)
                # to get the embedding data from the root chain.
                if current_pos in roots_candidate_set and roots_candidate[0] != current_pos:
                    # add embedding from root chain
                    current_embedding += self.calc_embedding_path_up(parents, children, edges, embeddings,
                                                                     roots_candidate[0], current_pos)
                    cc += 1

                # blow up current embedding and add it
                embedding += torch.cat([current_embedding] * self.edge_count).unsqueeze(1)
                # inc for previous embedding (from the branch we came)
                cc += 1
                b = torch.cat([self.edge_biases[edges[current_pos]].unsqueeze(0)] * self.edge_count).unsqueeze(1)
                w = torch.cat([self.edge_weights[edges[current_pos]].unsqueeze(0)] * self.edge_count)
                embedding = torch.baddbmm(1, b, 1, self.activation_fn(embedding / cc), w)
                parent = parents[current_pos]
                last_pos = current_pos
                current_pos = current_pos + parent

            # calc score
            score = torch.bmm(embedding, torch.cat([self.score_weights.unsqueeze(0)] * self.edge_count))
            scores.append(score)

            # reset new_children_candidate
            for child in new_children_candidate:
                parents[child] = 0
            # unlink roots
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

    def set_counts(self, slice_size, max_forest_count):
        self.slice_size = slice_size
        self.max_forest_count = max_forest_count
        self.max_graph_count = (slice_size + 1) * (2 ** (max_forest_count - 1))
