from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# import google3
import tensorflow as tf
import tensorflow_fold.public.blocks as td

DEFAULT_AGGR_ORDERED_SCOPE = 'aggregator_ordered'
DEFAULT_SCORING_SCOPE = 'scoring'


def dprint(x):
    r = tf.Print(x, [tf.shape(x)])
    return r


def SeqToTuple(T, N):
    return (td.InputTransform(lambda x: tuple(x))
            .set_input_type(td.SequenceType(T))
            .set_output_type(td.TupleType(*([T] * N))))


def sequence_tree_block(embeddings, scope):
    """Calculates an embedding over a (recursive) SequenceNode.

    Args:
      embeddings: a tensor of shape=(lex_size, state_size) containing the (pre-trained) embeddings
      scope: A scope to share variables over instances of sequence_tree_block
    """
    state_size = embeddings.shape.as_list()[1]
    expr_decl = td.ForwardDeclaration(td.PyObjectType(), state_size)
    grucell = td.ScopedLayer(tf.contrib.rnn.GRUCell(num_units=state_size), name_or_scope=scope)

    # an aggregation function which takes the order of the inputs into account
    def aggregator_order_aware(head, children):
        # inputs=head, state=children
        r, h2 = grucell(head, children)
        return r

    # an aggregation function which doesn't take the order of the inputs into account
    def aggregator_order_unaware(x, y):
        return tf.add(x, y)

    # get the head embedding from id
    def embed(x):
        return tf.gather(embeddings, x)

    # normalize (batched version -> dim=1)
    def norm(x):
        return tf.nn.l2_normalize(x, dim=1)

    def case(seq_tree):
        # children and head exist: process and aggregate
        if len(seq_tree['children']) > 0 and 'head' in seq_tree:
            return 0
        # children do not exist (but maybe a head): process (optional) head only
        if len(seq_tree['children']) == 0:
            return 1
        # otherwise (head does not exist): process children only
        return 2

    cases = td.OneOf(lambda x: case(x),
                     {0: td.Record([('head', td.Scalar(dtype='int32') >> td.Function(embed)),
                                    ('children', td.Map(expr_decl()) >> td.Reduce(td.Function(aggregator_order_unaware)))])
                         >> td.Function(aggregator_order_aware),
                      1: td.GetItem('head')
                         >> td.Optional(td.Scalar(dtype='int32')
                         >> td.Function(embed)),
                      2: td.GetItem('children')
                         >> td.Map(expr_decl())
                         >> td.Reduce(td.Function(aggregator_order_unaware)),
                      })

    expr_decl.resolve_to(cases)

    return cases >> td.Function(norm) #>> td.Function(dprint)


class SimilaritySequenceTreeTupleModel(object):
    """A Fold model for similarity scored sequence tree (SequenceNode) tuple."""

    def __init__(self, embeddings, aggregator_ordered_scope=DEFAULT_AGGR_ORDERED_SCOPE):
        self._aggregator_ordered_scope = aggregator_ordered_scope

        similarity = td.GetItem('similarity') >> td.Scalar(dtype='float', name='gold_similarity')

        with tf.variable_scope(aggregator_ordered_scope) as sc:
            # The AllOf block will run each of its children on the same input.
            model = td.AllOf(td.GetItem('first')
                             >> sequence_tree_block(embeddings, sc),
                             td.GetItem('second')
                             >> sequence_tree_block(embeddings, sc),
                             similarity)
        self._compiler = td.Compiler.create(model)

        # Get the tensorflow tensors that correspond to the outputs of model.
        (self._tree_embeddings_1, self._tree_embeddings_2, gold_similarities) = self._compiler.output_tensors

        self._cosine_similarities = tf.reduce_sum(self._tree_embeddings_1 * self._tree_embeddings_2, axis=1)

        # self._loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        #    logits=logits, labels=labels))

        # use MSE
        self._loss = tf.reduce_sum(tf.pow(self._cosine_similarities - gold_similarities, 2))

        # self._accuracy = tf.reduce_mean(
        #    tf.cast(tf.equal(tf.argmax(labels, 1),
        #                     tf.argmax(logits, 1)),
        #            dtype=tf.float32))

        self._global_step = tf.Variable(0, name='global_step', trainable=False)
        optr = tf.train.GradientDescentOptimizer(0.01)
        self._train_op = optr.minimize(self._loss, global_step=self._global_step)

    @property
    def tree_embeddings_1(self):
        return self._tree_embeddings_1

    @property
    def tree_embeddings_2(self):
        return self._tree_embeddings_2

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

    @property
    def aggregator_ordered_scope(self):
        return self._aggregator_ordered_scope

    def build_feed_dict(self, sim_trees):
        return self._compiler.build_feed_dict(sim_trees)


class SequenceTreeEmbedding(object):

    def __init__(self, embeddings, aggregator_ordered_scope=DEFAULT_AGGR_ORDERED_SCOPE):

        with tf.variable_scope(aggregator_ordered_scope) as sc:
            model = td.SerializedMessageToTree('recursive_dependency_embedding.SequenceNode') >> sequence_tree_block(embeddings, sc)
        self._compiler = td.Compiler.create(model)
        self._tree_embeddings = self._compiler.output_tensors

    @property
    def tree_embeddings(self):
        return self._tree_embeddings

    def build_feed_dict(self, sim_trees):
        return self._compiler.build_feed_dict(sim_trees)


class SequenceTreeEmbeddingSequence(object):
    """ A Fold model for training sequence tree embeddings using NCE.
        The model expects a converted (see td.proto_tools.serialized_message_to_tree) SequenceNodeSequence object 
        containing a sequence of sequence trees (see SequenceNode) and an index of the correct tree.
        It calculates all sequence tree embeddings, maps them to an 'integrity' score and calculates the maximum 
        entropy loss with regard to the correct tree.
    """

    def __init__(self, embeddings, aggregator_ordered_scope=DEFAULT_AGGR_ORDERED_SCOPE, scoring_scope=DEFAULT_SCORING_SCOPE):

        # This layer maps a sequence tree embedding to an 'integrity' score
        with tf.variable_scope(scoring_scope) as scoring_sc:
            scoring_fc = td.FC(1, name=scoring_sc)

        def squz(x):
            return tf.squeeze(x, [1])

        # takes a sequence of scalars and an index as input
        # and returns the n-th scalar normalized by the sum
        # of all the scalars
        norm = td.Composition()
        with norm.scope():
            sum_ = td.Reduce(td.Function(tf.add)).reads(norm.input[0])
            nth_ = td.Nth().reads(norm.input[0], norm.input[1])
            normed_nth = td.Function(tf.div).reads(nth_, sum_)
            norm.output.reads(normed_nth)

        with tf.variable_scope(aggregator_ordered_scope) as sc:
            tree_logits = td.Map(sequence_tree_block(embeddings, sc)
                                 >> scoring_fc >> td.Function(squz) >> td.Function(tf.exp))

        softmax_correct = td.AllOf(td.GetItem('trees') >> tree_logits, td.GetItem('idx_correct')) >> norm

        self._compiler = td.Compiler.create(softmax_correct)

        # Get the tensorflow tensors that correspond to the outputs of model.
        self._softmax_correct = self._compiler.output_tensors

        self._loss = tf.reduce_mean(-tf.log(self._softmax_correct))

        self._global_step = tf.Variable(0, name='global_step', trainable=False)
        optr = tf.train.GradientDescentOptimizer(0.01)
        self._train_op = optr.minimize(self._loss, global_step=self._global_step)

    @property
    def softmax_correct(self):
        return self._softmax_correct

    @property
    def loss(self):
        return self._loss

    @property
    def train_op(self):
        return self._train_op

    @property
    def global_step(self):
        return self._global_step

    def build_feed_dict(self, sim_trees):
        return self._compiler.build_feed_dict(sim_trees)


def sequence_tree_block_with_candidates(embeddings, scope, candidate_count=3):
    """Calculates an embedding over a (recursive) SequenceNode.

    Args:
      embeddings: a tensor of shape=(lex_size, state_size) containing the (pre-trained) embeddings
      scope: A scope to share variables over instances of sequence_tree_block
    """
    state_size = embeddings.shape.as_list()[1]
    expr_decl = td.ForwardDeclaration(td.PyObjectType(), state_size)
    grucell = td.ScopedLayer(tf.contrib.rnn.GRUCell(num_units=state_size), name_or_scope=scope)

    # an aggregation function which takes the order of the inputs into account
    def aggregator_order_aware(head, children):
        # inputs=head, state=children
        r, h2 = grucell(head, children)
        return r

    # an aggregation function which doesn't take the order of the inputs into account
    def aggregator_order_unaware(x, y):
        return tf.add(x, y)

    # get the head embedding from id
    def embed(x):
        return tf.gather(embeddings, x)

    # normalize (batched version -> dim=1)
    def norm(x):
        return tf.nn.l2_normalize(x, dim=1)

    def case(seq_tree):
        if len(seq_tree['candidates']) > 0:
            return 3
        # children and head exist: process and aggregate
        if len(seq_tree['children_outside']) > 0 and 'head_outside' in seq_tree:
            return 4
        # head exists
        if len(seq_tree['children_outside']) == 0 and 'head_outside' in seq_tree:
            return 5

        # children and head exist: process and aggregate
        if len(seq_tree['children']) > 0 and 'head' in seq_tree:
            return 0
        # children do not exist (but maybe a head): process (optional) head only
        if len(seq_tree['children']) == 0:
            return 1
        # otherwise (head does not exist): process children only
        return 2

    cases = td.OneOf(lambda x: case(x),
                     {0: td.Record([('head', td.Scalar(dtype='int32') >> td.Function(embed)),
                                    ('children', td.Map(expr_decl()) >> td.Reduce(td.Function(aggregator_order_unaware)))])
                         >> td.Function(aggregator_order_aware),
                      1: td.GetItem('head')
                         >> td.Optional(td.Scalar(dtype='int32')
                         >> td.Function(embed)),
                      2: td.GetItem('children')
                         >> td.Map(expr_decl())
                         >> td.Reduce(td.Function(aggregator_order_unaware)),
                      3: td.GetItem('candidates')
                         >> td.Map(expr_decl()),
                      4: td.GetItem('head_outside') >> td.Scalar(dtype='int32') >> td.Function(embed) >> td.Broadcast(),
                      #4: td.Record([('head_outside', td.Scalar(dtype='int32') >> td.Function(embed) >> td.Broadcast()),
                      #              ('children_outside', td.Map(expr_decl()) #>> td.NGrams(candidate_count) >> td.GetItem(0)
                      #               >> td.Reduce(td.Function(aggregator_order_unaware)))])
                      #   >> td.ZipWith(td.Function(aggregator_order_aware))


                         #0: td.Record([('head', td.Scalar(dtype='int32') >> td.Function(embed)) >> td.Broadcast(),
                      #              ('children', td.Map(expr_decl()) >> td.Reduce(td.Function(aggregator_order_unaware)))])
                      #   >> td.ZipWith(td.Function(aggregator_order_aware)),
                      #0: td.GetItem('children')
                      #   >> td.Map(expr_decl())
                      #   >> td.Reduce(td.Function(aggregator_order_unaware)),
                         #>> td.Broadcast(),
                      #1: td.GetItem('head')
                      #   >> td.Optional(td.Scalar(dtype='int32')
                      #   >> td.Function(embed))
                      #   >> td.Broadcast(),
                     #2: td.GetItem('children')
                      #   >> td.Map(expr_decl())
                      #   >> td.Reduce(td.Function(aggregator_order_unaware)),

                      })

    expr_decl.resolve_to(cases)

    return cases >> td.Map(td.Function(norm)) #>> td.Function(dprint)


class SequenceTreeEmbeddingWithCandidates(object):
    """ A Fold model for training sequence tree embeddings using NCE.
        The model expects a converted (see td.proto_tools.serialized_message_to_tree) SequenceNodeSequence object 
        containing a sequence of sequence trees (see SequenceNode) and an index of the correct tree.
        It calculates all sequence tree embeddings, maps them to an 'integrity' score and calculates the maximum 
        entropy loss with regard to the correct tree.
    """

    def __init__(self, embeddings, aggregator_ordered_scope=DEFAULT_AGGR_ORDERED_SCOPE, scoring_scope=DEFAULT_SCORING_SCOPE):

        # This layer maps a sequence tree embedding to an 'integrity' score
        with tf.variable_scope(scoring_scope) as scoring_sc:
            scoring_fc = td.FC(1, name=scoring_sc)

        def squz(x):
            return tf.squeeze(x, [1])

        first = td.Composition()
        with first.scope():
            def z(a):
                return 0
            x = td.InputTransform(z)
            nth_ = td.Nth().reads(first.input, x)
            first.output.reads(nth_)

        # takes a sequence of scalars as input
        # and returns the first scalar normalized by the sum
        # of all the scalars
        norm = td.Composition()
        with norm.scope():
            sum_ = td.Reduce(td.Function(tf.add)).reads(norm.input)
            nth_ = first.reads(norm.input)
            normed_nth = td.Function(tf.div).reads(nth_, sum_)
            norm.output.reads(normed_nth)

        with tf.variable_scope(aggregator_ordered_scope) as sc:
            tree_logits = sequence_tree_block_with_candidates(embeddings, sc) \
                          >> td.Map(scoring_fc >> td.Function(squz) >> td.Function(tf.exp))

        softmax_correct = tree_logits >> norm

        self._compiler = td.Compiler.create(softmax_correct)

        # Get the tensorflow tensors that correspond to the outputs of model.
        self._softmax_correct = self._compiler.output_tensors

        self._loss = tf.reduce_mean(-tf.log(self._softmax_correct))

        self._global_step = tf.Variable(0, name='global_step', trainable=False)
        optr = tf.train.GradientDescentOptimizer(0.01)
        self._train_op = optr.minimize(self._loss, global_step=self._global_step)

    @property
    def softmax_correct(self):
        return self._softmax_correct

    @property
    def loss(self):
        return self._loss

    @property
    def train_op(self):
        return self._train_op

    @property
    def global_step(self):
        return self._global_step

    def build_feed_dict(self, sim_trees):
        return self._compiler.build_feed_dict(sim_trees)