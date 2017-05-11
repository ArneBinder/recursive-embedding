from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# import google3
import tensorflow as tf
import tensorflow_fold.public.blocks as td


def sequence_tree_block(state_size, embeddings, aggregator):
    # state_size = embeddings.shape[1]
    expr_decl = td.ForwardDeclaration(td.PyObjectType(), state_size)

    def embed(x):
        return tf.gather(embeddings, x)

    # get the head embedding from id
    def head(name_):
        return td.Pipe(td.Scalar(dtype='int32'),
                       td.Function(embeddings),
                       # td.Embedding(lex_size, state_size, initializer = embeddings, name='head_embed')
                       name=name_)

    # get the weighted sum of all children
    def children_aggr(name_):
        return td.Pipe(td.Map(expr_decl()),
                       #td.Map(td.Function(lambda x: tf.norm(x) * x)),
                       td.Reduce(td.Function(tf.add)),
                       name=name_)

    # gru_cell = td.ScopedLayer(tf.contrib.rnn.GRUCell(num_units=state_size), 'mygru')

    # TODO: use GRU cell
    def aggr_op(inX, scope):
        #print(scope.name)
        with tf.variable_scope(scope, reuse=None) as sc:
            #sc.reuse_variables()
            weights = tf.get_variable("fc_weights", shape=(state_size, state_size))
            biases = tf.get_variable("fc_biases", shape=(state_size,))

            pre_activation = tf.add(tf.matmul(inX, weights), biases)
            my_fc = tf.nn.relu(pre_activation, name=sc.name)

        # return (td.AllOf(head, children_aggr) >> td.RNN(gru_cell, initial_state_from_input=True))
        return my_fc

    def cas(seq_tree):
        # process and aggregate
        if len(seq_tree['children']) > 0 and 'head' in seq_tree:
            return 0
        # don't process children
        if len(seq_tree['children']) == 0: #'children' not in seq_tree:
            return 1
        # process children only
        return 2

    # DEBUG
    def p_both(x):
        return tf.Print(x, [tf.shape(x), x], message='both: ', summarize=3000)
    def p_head(x):
        return tf.Print(x, [tf.shape(x), x], message='head: ', summarize=3000)
    def p_children(x):
        return tf.Print(x, [tf.shape(x), x], message='children: ', summarize=3000)

    cases = td.OneOf(lambda x: cas(x),
                     {0: td.Record([('head', td.Scalar(dtype='int32') >> td.Function(embed)),
                                    ('children', td.Map(expr_decl()) >> td.Reduce(td.Function(tf.add)))])
                         >> td.Function(aggregator),
                         #>> td.Function(p_both),
                      1: td.GetItem('head')
                         >> td.Optional(td.Scalar(dtype='int32')
                         >> td.Function(embed)),
                         #>> td.Function(p_head),
                      2: td.GetItem('children')
                         >> td.Map(expr_decl())
                         >> td.Reduce(td.Function(tf.add)),
                         #>> td.Function(p_children)
                      })

    #cases = td.GetItem('head') >> td.Optional(head('just_head')) >> td.ScopedLayer(aggr_op, scope)

    expr_decl.resolve_to(cases)
    return cases


class SequenceTupleModel(object):
    """A Fold model for calculator examples."""

    # TODO: check (and use) scopes
    def __init__(self, lex_size, embedding_dim, embeddings):

        self._lex_size = lex_size
        self._state_size = embedding_dim
        self._embeddings = embeddings

        similarity = td.GetItem('similarity') >> td.Scalar(dtype='float', name='gold_similarity')

        #grucell = tf.contrib.rnn.GRUCell(num_units=embedding_dim)#, 'gru_cell') # tf.contrib.rnn.core_rnn_cell.GRUCell(num_units=embedding_dim)

        fc = td.FC(embedding_dim)

        def dummy_aggr(x, y):
            #h1, h2 = grucell(x, y)

            r = fc(tf.concat([x, y], 1))
            #r = x + y
            return r
        #td.Concat
        #gc = td.ScopedLayer(dummy_aggr, 'gru_cell')

        # The AllOf block will run each of its children on the same input.
        model = td.AllOf(td.GetItem('first') >> sequence_tree_block(self._state_size, self._embeddings, dummy_aggr),
                         td.GetItem('second') >> sequence_tree_block(self._state_size, self._embeddings, dummy_aggr),
                         similarity)
        self._compiler = td.Compiler.create(model)

        # Get the tensorflow tensors that correspond to the outputs of model.
        (embeddings_1, embeddings_2, gold_similarities) = self._compiler.output_tensors

        normed_embeddings_1 = tf.nn.l2_normalize(embeddings_1, dim=1)
        normed_embeddings_2 = tf.nn.l2_normalize(embeddings_2, dim=1)

        self._embeddings_1 = embeddings_1 #, [normed_embeddings_1], summarize=1000)
        self._embeddings_2 = embeddings_2 #, [normed_embeddings_2], summarize=1000)

        cosine_similarities = tf.reduce_sum(normed_embeddings_1 * normed_embeddings_2, axis=1)#tf.matmul(normed_embeddings_1, tf.transpose(normed_embeddings_2, [1, 0]))
        self._cosine_similarities = cosine_similarities #, [cosine_similarities], summarize=300)


        # self._loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        #    logits=logits, labels=labels))

        # use MSE
        self._loss = tf.reduce_sum(tf.pow(cosine_similarities - gold_similarities, 2))  #/(cosine_similarities.shape.as_list()[0]) #tf.reduce_sum(tf.metrics.mean_squared_error(labels=gold_similarities, predictions=cosine_similarities))

        # self._accuracy = tf.reduce_mean(
        #    tf.cast(tf.equal(tf.argmax(labels, 1),
        #                     tf.argmax(logits, 1)),
        #            dtype=tf.float32))

        self._global_step = tf.Variable(0, name='global_step', trainable=False)
        optr = tf.train.GradientDescentOptimizer(0.01)
        self._train_op = optr.minimize(self._loss, global_step=self._global_step)

    @property
    def embeddings_1(self):
        return self._embeddings_1

    @property
    def embeddings_2(self):
        return self._embeddings_2

    @property
    def cosine_similarities(self):
        return self._cosine_similarities

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
