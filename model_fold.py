from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# import google3
import tensorflow as tf
import tensorflow_fold.public.blocks as td


def sequence_tree_block(state_size, embeddings, aggregator):
    # state_size = embeddings.shape[1]
    expr_decl = td.ForwardDeclaration(td.PyObjectType(), state_size)

    # get the head embedding from id
    def embed(x):
        return tf.gather(embeddings, x)

    def case(seq_tree):
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

    cases = td.OneOf(lambda x: case(x),
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

        grucell = td.ScopedLayer(tf.contrib.rnn.GRUCell(num_units=embedding_dim))
        #fc = td.FC(embedding_dim)

        def aggregator_ordered(head, children):
            # inputs=head, state=children
            r, h2 = grucell(head, children)
            #r = fc(tf.concat([x, y], 1))
            #r = x + y
            return r

        # The AllOf block will run each of its children on the same input.
        model = td.AllOf(td.GetItem('first')
                         >> sequence_tree_block(self._state_size, self._embeddings, aggregator_ordered),
                         td.GetItem('second')
                         >> sequence_tree_block(self._state_size, self._embeddings, aggregator_ordered),
                         similarity)
        self._compiler = td.Compiler.create(model)

        # Get the tensorflow tensors that correspond to the outputs of model.
        (embeddings_1, embeddings_2, gold_similarities) = self._compiler.output_tensors

        normed_embeddings_1 = tf.nn.l2_normalize(embeddings_1, dim=1)
        normed_embeddings_2 = tf.nn.l2_normalize(embeddings_2, dim=1)

        self._embeddings_1 = embeddings_1
        self._embeddings_2 = embeddings_2

        cosine_similarities = tf.reduce_sum(normed_embeddings_1 * normed_embeddings_2, axis=1)
        self._cosine_similarities = cosine_similarities


        # self._loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        #    logits=logits, labels=labels))

        # use MSE
        self._loss = tf.reduce_sum(tf.pow(cosine_similarities - gold_similarities, 2))

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
