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

    def calc_embedding(self, data, types, parents, edges):
        # connect roots
        roots = get_roots(parents)
        for i in range(len(roots) - 1):
            parents[roots[i]] = roots[i + 1]

        root = roots[-1]

        # let root point to outside (remove circle)
        parents[root] = -len(data)

        # calc child pointer
        children = get_children(parents)

        result = self.calc_embedding_rec(data, types, children, edges, root)
        # incorporate root edge
        result = self.add_child_embedding(Variable(torch.zeros(result.size())), result, edges[root])

        # reset root for next iteration
        parents[root] = 0
        return result

    def add_child_embedding(self, embedding, child_embedding, edge_id):
        single = True  # batch disabled
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
            return embedding + torch.baddbmm(1, b, 1, child_embedding, w)

    def calc_embedding_rec(self, data, types, children, edges, idx):
        embedding = torch.addmm(1, self.data_biases[types[idx]].unsqueeze(0), 1,
                                self.data_vecs[types[idx]][data[idx]].unsqueeze(0), self.data_weights[types[idx]])
        if idx not in children:  # leaf
            return embedding
        for child in children[idx]:
            child_embedding = self.calc_embedding_rec(data, types, children, edges, child)
            embedding = self.add_child_embedding(embedding, child_embedding, edges[child])
        return (embedding/(len(children)+1)).clamp(min=0)

    def calc_score(self, embedding):
        if len(embedding.size()) == 3:  # batch edges
            s = embedding.size()[0]
            return torch.bmm(embedding, torch.cat([self.score_weights.unsqueeze(0)] * s))
        else:
            return torch.mm(embedding, self.score_weights)

    def calc_embedding_single(self, data, types, children, edges, embeddings, idx):
        embedding = torch.addmm(1, self.data_biases[types[idx]].unsqueeze(0), 1,
                                self.data_vecs[types[idx]][data[idx]].unsqueeze(0), self.data_weights[types[idx]])
        if idx not in children:  # leaf
            return embedding
        for child in children[idx]:
            child_embedding = self.calc_embedding_single(data, types, children, edges, embeddings, child)
            embedding += torch.addmm(1, self.edge_biases[edges[child]].unsqueeze(0), 1, child_embedding, self.edge_weights[edges[child]]) #self.add_child_embedding(embedding, child_embedding, edges[child])

        embedding = (embedding / (len(children) + 1))
        embeddings[idx] = embedding
        return embedding.clamp(min=0)

    #def calc_embedding_batch(self, data, types, children, edges, embeddings, idx):

    def forward(self, data, types, parents, edges, pos, forests, correct_roots, new_roots):
        edges[pos] = -1
        parents[pos] = -pos

        embeddings = {}
        # calc child pointer
        children = get_children(parents)
        # calc forest embeddings (top down from roots)
        for root in correct_roots + new_roots:
            # calc embedding and save
            self.calc_embedding_single(data, types, children, edges, embeddings, root)

        # link other roots (not in parent_children: correct_roots)
        correct_edges = []
        root_prev_ind = -1
        root_next_ind = 0
        for i in range(len(correct_roots) - 1):
            # add INTERTREE children
            if correct_roots[i + 1] not in children:
                children[correct_roots[i + 1]] = [correct_roots[i]]
            else:
                children[correct_roots[i + 1]] += [correct_roots[i]]
            if correct_roots[i] < pos:
                root_prev_ind = i
                root_next_ind = i + 1
            parents[correct_roots[i]] = correct_roots[i + 1] - correct_roots[i]
            correct_edges.append((correct_roots[i], edges[correct_roots[i]]))
            edges[correct_roots[i]] = 0     # set INTERTREE

        if correct_roots[-1] < pos:
            root_prev_ind = len(correct_roots) - 1
            root_next_ind = -1


        scores = []
        # graph_count, _ = graphs.shape
        #for j in range(graph_count):
        #    embedding = self.calc_embedding(data, types, graphs[j], edges)
        #    scores.append(self.calc_score(embedding).squeeze())
        for (parent_children, parent_candidate) in forests:
            parents[pos] = parent_candidate
            if parent_candidate == 0:  # root: insert into root chain
                if root_prev_ind >= 0:
                    parents[correct_roots[root_prev_ind]] = pos - correct_roots[root_prev_ind]
                    # remember edge and set INTERTREE
                    re_edge = edges[correct_roots[root_prev_ind]]
                    edges[correct_roots[root_prev_ind]] = 0
                if root_next_ind >= 0:
                    parents[pos] = correct_roots[root_next_ind] - pos
            # calc pos embedding (bottom up)
            parent = parent_candidate
            current_pos = pos
            embedding = embeddings[pos]
            current_children = parent_children
            if pos in children:
                current_children += children[pos]
            for child in current_children:
                if edges[child] != constants.INTER_TREE:
                    embedding += torch.addmm(1, self.edge_biases[edges[child]].unsqueeze(0), 1, embeddings[child].clamp(min=0), self.edge_weights[edges[child]])
                else:
                    # calc INTERTREE embeddings top down
                    print('implement')
            embedding = torch.cat([(embedding * len(children[pos])).unsqueeze(0)] * self.edge_count)
            while parent != 0:
                cc = len(children[current_pos])
                children_embedding = embedding[current_pos] * cc

                cc += 1 # itself
                # TODO: IMPLEMENT
                # is 0 (INTERTREE) in edges of children?
                #   cc += 1
                #   add this child embedding (calc recursive(?))
                # embedding = embedding * edge_w[edge[current_pos]] + edge_b[edge[current_pos]] + children_embedding / cc

                current_pos = current_pos + parent
                parent = parents[current_pos]

            # calc score ()
            if parent_candidate == 0:  # root: remove from root chain
                if root_prev_ind >= 0:
                    parents[correct_roots[root_prev_ind]] = 0
                    if root_prev_ind < len(correct_roots) - 1:
                        parents[correct_roots[root_prev_ind]] = correct_roots[root_prev_ind + 1] - correct_roots[root_prev_ind]
                        # reset edge
                        parents[correct_roots[root_prev_ind]] = re_edge
                if root_next_ind >= 0:
                    parents[pos] = 0

        # reset edges
        for (i, edge) in correct_edges:
            edge[i] = edge

        # reset roots:
        for root in correct_roots:
            parents[root] = 0

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
