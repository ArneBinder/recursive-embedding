from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# import google3
import tensorflow as tf
import tensorflow_fold.public.blocks as td


def sequence_tree_block(state_size, embeddings, name_):
    # state_size = embeddings.shape[1]
    expr_decl = td.ForwardDeclaration(td.PyObjectType(), state_size)

    # get the head embedding from id
    def head(name_):
        return td.Pipe(td.Scalar(dtype='int32'),
                       td.Function(embeddings),
                       # td.Embedding(lex_size, state_size, initializer = embeddings, name='head_embed')
                       name=name_)

    # get the weighted sum of all children
    def children_aggr(name_):
        return td.Pipe(td.Map(expr_decl()),
                       td.Map(td.Function(lambda x: tf.norm(x) * x)),
                       td.Reduce(td.Function(tf.add)),
                       name=name_)

    # gru_cell = td.ScopedLayer(tf.contrib.rnn.GRUCell(num_units=state_size), 'mygru')

    # TODO: use GRU cell
    #def aggr_op():
        # return (td.AllOf(head, children_aggr) >> td.RNN(gru_cell, initial_state_from_input=True))
    #    return (children_aggr)

    def cas(seq_tree):
        # process and aggregate
        if 'children' in seq_tree and 'head' in seq_tree:
            return 0
        # don't process children
        if 'children' not in seq_tree:
            return 1
        # process children only
        return 2

    cases = td.OneOf(lambda x: cas(x),
                     {0: td.Record([('head', head('head')),
                                    ('children', children_aggr('children_aggr'))]) >> td.Concat() >> td.FC(
                         state_size),
                      1: td.GetItem('head') >> td.Optional(head('just_head')),
                      2: td.GetItem('children') >> children_aggr('just_children')},
                     name=name_)

    expr_decl.resolve_to(cases)
    return cases


class SequenceTupleModel(object):
    """A Fold model for calculator examples."""

    # TODO: check (and use) scopes
    def __init__(self, embeddings):
        self._lex_size = embeddings.shape[0]
        self._state_size = embeddings.shape[1]
        # TODO: re-add embeddings
        # self._embeddings = td.Embedding(self._lex_size, self._state_size, initializer=embeddings, name='head_embed')
        self._embeddings = td.Embedding(self._lex_size, self._state_size, name='head_embed')

        # seq_tree_block = sequence_tree_block(self._state_size, self._embeddings)
        similarity = td.GetItem('similarity') >> td.Scalar(dtype='float', name='gold_similarity')

        # The AllOf block will run each of its children on the same input.
        model = td.AllOf(td.GetItem('first') >> sequence_tree_block(self._state_size, self._embeddings, name_='first'),
                         td.GetItem('second') >> sequence_tree_block(self._state_size, self._embeddings,
                                                                     name_='second'),
                         similarity)
        self._compiler = td.Compiler.create(model)

        # Get the tensorflow tensors that correspond to the outputs of model.
        (embeddings_1, embeddings_2, gold_similarities) = self._compiler.output_tensors

        normed_embeddings_1 = tf.nn.l2_normalize(embeddings_1, dim=1)
        normed_embeddings_2 = tf.nn.l2_normalize(embeddings_2, dim=1)
        cosine_similarities = tf.matmul(normed_embeddings_1, tf.transpose(normed_embeddings_2, [1, 0]))

        # self._loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        #    logits=logits, labels=labels))
        self._loss = tf.reduce_sum(tf.abs(cosine_similarities - gold_similarities))

        # self._accuracy = tf.reduce_mean(
        #    tf.cast(tf.equal(tf.argmax(labels, 1),
        #                     tf.argmax(logits, 1)),
        #            dtype=tf.float32))

        self._global_step = tf.Variable(0, name='global_step', trainable=False)
        optr = tf.train.GradientDescentOptimizer(0.01)
        self._train_op = optr.minimize(self._loss, global_step=self._global_step)

    @property
    def loss(self):
        return self._loss

    # @property
    # def accuracy(self):
    #    return self._accuracy

    @property
    def train_op(self):
        return self._train_op

    @property
    def global_step(self):
        return self._global_step

    def build_feed_dict(self, sim_trees):
        return self._compiler.build_feed_dict(sim_trees)
