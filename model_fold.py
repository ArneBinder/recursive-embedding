from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# import google3
import tensorflow as tf
import tensorflow_fold.public.blocks as td

# DEFAULT_SCOPE_TREE_EMBEDDER = 'tree_embedder'   # DEPRECATED
DEFAULT_SCOPE_SCORING = 'scoring'
DIMENSION_EMBEDDINGS = 300
DIMENSION_SIM_MEASURE = 300
VAR_NAME_EMBEDDING = 'embeddings'
VAR_NAME_GLOBAL_STEP = 'global_step'
VAR_PREFIX_FC_EMBEDDING = 'FC_embedding'
VAR_PREFIX_FC_OUTPUT = 'FC_output'
VAR_PREFIX_TREE_EMBEDDING = 'TreeEmbedding'
VAR_PREFIX_SIM_MEASURE = 'sim_measure'


def dprint(x):
    r = tf.Print(x, [tf.shape(x)])
    return r


def block_info(block):
    print("%s: %s -> %s" % (block, block.input_type, block.output_type))


def SeqToTuple(T, N):
    return (td.InputTransform(lambda x: tuple(x))
            .set_input_type(td.SequenceType(T))
            .set_output_type(td.TupleType(*([T] * N))))


def fc_scoped(num_units, scope, name=None, activation_fn=tf.nn.relu):
    def fc_(inputs, scope):
        return tf.contrib.layers.fully_connected(inputs, num_units, activation_fn=activation_fn, scope=scope)

    if not name:
        name = 'FC_scoped_%d' % num_units

    with tf.variable_scope(scope):
        with tf.variable_scope(name) as sc:
            result = td.ScopedLayer(fc_, name_or_scope=sc)

    return result


def nth(n):
    c = td.Composition()
    with c.scope():
        def z(_):
            return n

        x = td.InputTransform(z)
        nth_ = td.Nth().reads(c.input, x)
        c.output.reads(nth_)
    return


def nth_py(n):
    return td.InputTransform(lambda s: s[n])


def treeLSTM(xh_linear_layer, fc_f_layer, name='treelstm', forget_bias=1.0, activation=tf.tanh):
    comp = td.Composition(name=name)
    with comp.scope():
        x = comp.input[0]
        c_k = td.Map(td.GetItem(0)).reads(comp.input[1])
        h_k = td.Map(td.GetItem(1)).reads(comp.input[1])
        h_k_sum = td.Reduce(td.Function(tf.add)).reads(h_k)
        xh_concat = td.Concat().reads(x, h_k_sum)

        xh_linear = td.Function(xh_linear_layer).reads(xh_concat)

        def split_3(v):
            return tf.split(value=v, num_or_size_splits=3, axis=1)

        iou = td.Function(split_3).reads(xh_linear)
        i_sigm = td.Function(tf.sigmoid).reads(iou[0])
        o_sigm = td.Function(tf.sigmoid).reads(iou[1])
        u_sigm = td.Function(tf.sigmoid).reads(iou[2])

        c_new1 = td.Function(tf.multiply).reads(i_sigm, u_sigm)

        x_bc = td.Broadcast().reads(x)
        xh_k_concat = (td.Zip() >> td.Map(td.Concat())).reads(x_bc, h_k)

        def add_forget_bias(x):
            return tf.add(x, forget_bias)

        f_k = td.Map(td.Function(fc_f_layer) >> td.Function(add_forget_bias) >> td.Function(tf.sigmoid)).reads(
            xh_k_concat)

        fc_k = td.Zip().reads(f_k, c_k)
        fc_k_mul = td.Map(td.Function(tf.multiply)).reads(fc_k)
        c_new2 = td.Reduce(td.Function(tf.add)).reads(fc_k_mul)  # (c_jk_mul)

        c_new = td.Function(tf.add).reads(c_new1, c_new2)

        c_new_activ = td.Function(activation).reads(c_new)
        h_new = td.Function(tf.multiply).reads(o_sigm, c_new_activ)

        comp.output.reads(c_new, h_new)
    return comp


# normalize (batched version -> dim=1)
def norm(x):
    return tf.nn.l2_normalize(x, dim=1)


class TreeEmbedding(object):
    def __init__(self, embeddings, name, embedding_fc_size_multiple=None, embedding_fc_activation=tf.nn.tanh,
                 output_fc_activation=tf.nn.tanh):  # , apply_output_fc=False):
        self._state_size = embeddings.get_shape().as_list()[1]  # state_size
        self._embeddings = embeddings
        self._apply_embedding_fc = embedding_fc_size_multiple != 0
        # self._apply_output_fc = apply_output_fc
        self._name = VAR_PREFIX_TREE_EMBEDDING + '_' + name  # + '_%d' % self._state_size

        with tf.variable_scope(self.name) as scope:
            self._scope = scope
            if self._apply_embedding_fc:
                self._embedding_fc = fc_scoped(num_units=embedding_fc_size_multiple * self.state_size,
                                               activation_fn=embedding_fc_activation, scope=scope,
                                               name=VAR_PREFIX_FC_EMBEDDING + '_%d' % (
                                                   embedding_fc_size_multiple * self.state_size))
            else:
                self._embedding_fc = td.Identity()
            if output_fc_activation:
                self._output_fc = fc_scoped(num_units=self.state_size, activation_fn=output_fc_activation, scope=scope,
                                            name=VAR_PREFIX_FC_OUTPUT + '_%d' % self.state_size)
            else:
                self._output_fc = td.Identity()

                # get the head embedding from id

    def embed(self):
        return td.Function(lambda x: tf.gather(self._embeddings, x))

    def head(self, name='head_embed'):
        return td.Pipe(td.GetItem('head'), td.Scalar(dtype='int32'), self.embed(), name=name)

    @property
    def state_size(self):
        return self._state_size

    @property
    def embeddings(self):
        return self._embeddings

    @property
    def apply_embedding_fc(self):
        return self._apply_embedding_fc

    @property
    def name(self):
        return self._name

    @property
    def scope(self):
        return self._scope

    @property
    def embedding_fc(self):
        return self._embedding_fc

    @property
    def output_fc(self):
        return self._output_fc


class TreeEmbedding_TREE_LSTM(TreeEmbedding):
    """Calculates an embedding over a (recursive) SequenceNode.

    Args:
        embeddings: a tensor of shape=(lex_size, state_size) containing the (pre-trained) embeddings
        name_or_scope: A scope to share variables over instances of sequence_tree_block
    """

    def __init__(self, embeddings, embedding_fc_activation=None, output_fc_activation=None):
        super(TreeEmbedding_TREE_LSTM, self).__init__(embeddings, name='TREE_LSTM',
                                                      embedding_fc_size_multiple=(
                                                          1 * (embedding_fc_activation is not None)),
                                                      embedding_fc_activation=embedding_fc_activation,
                                                      output_fc_activation=output_fc_activation)
        with tf.variable_scope(self.scope) as scope:
            self._xh_linear = fc_scoped(num_units=3 * self.state_size, scope=scope,
                                        name='FC_xh_linear_%d' % (3 * self.state_size), activation_fn=None)
            self._fc_f = fc_scoped(num_units=self._state_size, scope=scope,
                                   name='FC_f_linear_%d' % self.state_size, activation_fn=None)

    def __call__(self):
        zero_state = td.Zeros((self.state_size, self.state_size))
        embed_tree = td.ForwardDeclaration(input_type=td.PyObjectType(), output_type=zero_state.output_type)
        treelstm = treeLSTM(self._xh_linear, self._fc_f)

        children = td.GetItem('children') >> td.Optional(some_case=td.Map(embed_tree()),
                                                         none_case=td.Zeros(td.SequenceType(zero_state.output_type)))
        cases = td.AllOf(self.head(), children) >> treelstm
        embed_tree.resolve_to(cases)

        return cases >> td.Concat() >> self.output_fc


class TreeEmbedding_HTU_GRU_simplified(TreeEmbedding):
    """Calculates an embedding over a (recursive) SequenceNode.

    Args:
        embeddings: a tensor of shape=(lex_size, state_size) containing the (pre-trained) embeddings
        name_or_scope: A scope to share variables over instances of sequence_tree_block
    """

    def __init__(self, embeddings, embedding_fc_activation=None, output_fc_activation=None):
        super(TreeEmbedding_HTU_GRU_simplified, self).__init__(embeddings, name='HTU_GRU',
                                                               embedding_fc_size_multiple=(
                                                                   1 * (embedding_fc_activation is not None)),
                                                               embedding_fc_activation=embedding_fc_activation,
                                                               output_fc_activation=output_fc_activation)
        with tf.variable_scope(self.scope):
            self._grucell = td.ScopedLayer(tf.contrib.rnn.GRUCell(num_units=self._state_size), 'gru_cell')

    def __call__(self):
        zero_state = td.Zeros(self._state_size)
        # zero_state = td.Zeros((state_size, state_size))
        embed_tree = td.ForwardDeclaration(input_type=td.PyObjectType(), output_type=zero_state.output_type)

        # an aggregation function which takes the order of the inputs into account
        def aggregator_order_aware(head, children):
            # inputs=head, state=children
            r, h2 = self._grucell(head, children)
            return r

        # an aggregation function which doesn't take the order of the inputs into account
        def aggregator_order_unaware(x, y):
            return tf.add(x, y)

        # simplified naive version (minor modification: apply order_aware also to single head with zeros as input state)
        children = td.GetItem('children') >> td.Optional(
            some_case=(td.Map(embed_tree()) >> td.Reduce(td.Function(aggregator_order_unaware))),
            none_case=zero_state)
        cases = td.AllOf(self.head(), children) >> td.Function(aggregator_order_aware)

        embed_tree.resolve_to(cases)
        model = cases >> self.output_fc

        return model


class TreeEmbedding_HTU_GRU(TreeEmbedding):
    """Calculates an embedding over a (recursive) SequenceNode.

    Args:
        embeddings: a tensor of shape=(lex_size, state_size) containing the (pre-trained) embeddings
        name_or_scope: A scope to share variables over instances of sequence_tree_block
    """

    def __init__(self, embeddings, embedding_fc_activation=None, output_fc_activation=None):

        super(TreeEmbedding_HTU_GRU, self).__init__(embeddings, name='HTU_GRU',
                                                    embedding_fc_size_multiple=(
                                                        1 * (embedding_fc_activation is not None)),
                                                    embedding_fc_activation=embedding_fc_activation,
                                                    output_fc_activation=output_fc_activation)
        with tf.variable_scope(self.scope):
            self._grucell = td.ScopedLayer(tf.contrib.rnn.GRUCell(num_units=self.state_size), 'gru_cell')

    def __call__(self):
        zero_state = td.Zeros(self.state_size)
        # zero_state = td.Zeros((state_size, state_size))
        embed_tree = td.ForwardDeclaration(input_type=td.PyObjectType(), output_type=zero_state.output_type)

        # an aggregation function which takes the order of the inputs into account
        def aggregator_order_aware(head, children):
            # inputs=head, state=children
            r, h2 = self._grucell(head, children)
            return r

        # an aggregation function which doesn't take the order of the inputs into account
        def aggregator_order_unaware(x, y):
            return tf.add(x, y)

        # naive version
        def case(seq_tree):
            # children and head exist: process and aggregate
            if len(seq_tree['children']) > 0 and 'head' in seq_tree:
                return 0
            # children do not exist (but maybe a head): process (optional) head only
            if len(seq_tree['children']) == 0:
                return 1
            # otherwise (head does not exist): process children only
            return 2

        def children(name='children'):
            return td.Pipe(td.GetItem('children'), td.Map(embed_tree()),
                           td.Reduce(td.Function(aggregator_order_unaware)), name=name)

        cases = td.OneOf(lambda x: case(x),
                         {0: td.AllOf(self.head(), children()) >> td.Function(aggregator_order_aware),
                          1: self.head('head_only'),
                          2: children('children_only'),
                          })

        embed_tree.resolve_to(cases)

        model = cases >> self.output_fc
        return model


class TreeEmbedding_FLAT(TreeEmbedding):
    def __init__(self, embeddings, name, embedding_fc_activation=None, output_fc_activation=None):
        super(TreeEmbedding_FLAT, self).__init__(embeddings, name='FLAT_' + name,
                                                 embedding_fc_size_multiple=(
                                                     self.embedding_fc_size_multiple * (
                                                     embedding_fc_activation is not None)),
                                                 embedding_fc_activation=embedding_fc_activation,
                                                 output_fc_activation=output_fc_activation)

    def element(self, name='element'):
        return self.head()

    def sequence(self, name='sequence'):
        return td.Pipe(td.GetItem('children'), td.Map(self.element()), name=name)

    def aggregate(self, name='aggregate'):
        # an aggregation function which doesn't take the order of the inputs into account
        return td.Mean(name)

    def __call__(self):
        model = self.sequence() >> self.aggregate() >> self.output_fc
        return model

    @property
    def embedding_fc_size_multiple(self):
        return 1


class TreeEmbedding_FLAT_AVG(TreeEmbedding_FLAT):
    def __init__(self, embeddings, embedding_fc_activation=None, output_fc_activation=None):
        super(TreeEmbedding_FLAT_AVG, self).__init__(embeddings, 'AVG', embedding_fc_activation, output_fc_activation)


class TreeEmbedding_FLAT_AVG_2levels(TreeEmbedding_FLAT):
    def __init__(self, embeddings, embedding_fc_activation=None, output_fc_activation=None):
        super(TreeEmbedding_FLAT_AVG_2levels, self).__init__(embeddings, 'AVG_2levels', embedding_fc_activation,
                                                             output_fc_activation)

    def element(self, name='element'):
        # use word embedding and first child embedding
        return td.Pipe(td.AllOf(self.head(name='head_level1'),
                                td.GetItem('children') >> td.InputTransform(lambda s: s[0]) >> self.head(
                                    name='head_level2')),
                       td.Concat(), name=name)

    @property
    def embedding_fc_size_multiple(self):
        # use word embedding and first child embedding
        return 2


class TreeEmbedding_FLAT_LSTM(TreeEmbedding_FLAT):
    def __init__(self, embeddings, embedding_fc_activation=None, output_fc_activation=None):
        super(TreeEmbedding_FLAT_LSTM, self).__init__(embeddings, 'LSTM', embedding_fc_activation,
                                                         output_fc_activation)
        with tf.variable_scope(self.scope):
            self._lstm_cell = td.ScopedLayer(tf.contrib.rnn.BasicLSTMCell(num_units=self.state_size), 'lstm_cell')

    def aggregate(self, name='aggregate'):
        # apply LSTM >> take the LSTM output state(s) >> take the h state (discard the c state)
        return td.Pipe(td.RNN(self._lstm_cell), td.GetItem(1), td.GetItem(0), name=name)


class TreeEmbedding_FLAT_LSTM50(TreeEmbedding_FLAT):
    def __init__(self, embeddings, embedding_fc_activation=None, output_fc_activation=None):
        super(TreeEmbedding_FLAT_LSTM50, self).__init__(embeddings, 'LSTM50', embedding_fc_activation,
                                                         output_fc_activation)
        with tf.variable_scope(self.scope):
            self._lstm_cell = td.ScopedLayer(tf.contrib.rnn.BasicLSTMCell(num_units=50), 'lstm_cell')

    def aggregate(self, name='aggregate'):
        # apply LSTM >> take the LSTM output state(s) >> take the h state (discard the c state)
        return td.Pipe(td.RNN(self._lstm_cell), td.GetItem(1), td.GetItem(0), name=name)


class TreeEmbedding_FLAT_LSTM_2levels(TreeEmbedding_FLAT):
    def __init__(self, embeddings, embedding_fc_activation=None, output_fc_activation=None):
        super(TreeEmbedding_FLAT_LSTM_2levels, self).__init__(embeddings, 'LSTM_2levels', embedding_fc_activation,
                                                         output_fc_activation)
        with tf.variable_scope(self.scope):
            self._lstm_cell = td.ScopedLayer(tf.contrib.rnn.BasicLSTMCell(num_units=self.state_size), 'lstm_cell')

    def element(self, name='element'):
        # use word embedding and first child embedding
        return td.Pipe(td.AllOf(self.head(name='head_level1'),
                                td.GetItem('children') >> td.InputTransform(lambda s: s[0]) >> self.head(
                                    name='head_level2')),
                       td.Concat(), name=name)

    def aggregate(self, name='aggregate'):
        # apply LSTM >> take the LSTM output state(s) >> take the h state (discard the c state)
        return td.Pipe(td.RNN(self._lstm_cell), td.GetItem(1), td.GetItem(0), name=name)

    @property
    def embedding_fc_size_multiple(self):
        # use word embedding and first child embedding
        return 2


def sim_cosine(e1, e2):
    e1 = tf.nn.l2_normalize(e1, dim=1)
    e2 = tf.nn.l2_normalize(e2, dim=1)
    return tf.reduce_sum(e1 * e2, axis=1)


def sim_manhattan(e1, e2):
    abs_ = tf.abs(e1 - e2)
    sum_ = tf.reduce_sum(abs_, axis=1)
    return tf.exp(-sum_)


def sim_layer(e1, e2, hidden_size=DIMENSION_SIM_MEASURE):
    with tf.variable_scope(VAR_PREFIX_SIM_MEASURE + '_layer'):
        embeddings_dif = tf.abs(e1 - e2)
        embeddings_product = e1 * e2
        concat = tf.concat([embeddings_dif, embeddings_product], axis=1)
        h_s = tf.contrib.layers.fully_connected(concat, hidden_size, activation_fn=tf.nn.sigmoid)
        s = tf.contrib.layers.fully_connected(h_s, 1, activation_fn=tf.nn.sigmoid)
    return tf.squeeze(s, axis=[1])


class SimilaritySequenceTreeTupleModel(object):
    """A Fold model for similarity scored sequence tree (SequenceNode) tuple."""

    def __init__(self, lex_size, tree_embedder=TreeEmbedding_TREE_LSTM, learning_rate=0.01, optimizer=tf.train.GradientDescentOptimizer, sim_measure=sim_layer,
                 embeddings_trainable=True, embedding_fc_activation=None, output_fc_activation=None):
        self._embeddings = tf.Variable(tf.constant(0.0, shape=[lex_size, DIMENSION_EMBEDDINGS]),
                                 trainable=embeddings_trainable, name=VAR_NAME_EMBEDDING)
        self._embedding_placeholder = tf.placeholder(tf.float32, [lex_size, DIMENSION_EMBEDDINGS])
        self._embedding_init = self._embeddings.assign(self._embedding_placeholder)

        gold_similarity = td.GetItem('similarity') >> td.Scalar(dtype='float', name='gold_similarity')

        tree_embed = tree_embedder(self._embeddings, embedding_fc_activation, output_fc_activation)
        model = td.AllOf(td.GetItem('first') >> tree_embed(),
                         td.GetItem('second') >> tree_embed(),
                         gold_similarity)
        self._compiler = td.Compiler.create(model)

        # Get the tensorflow tensors that correspond to the outputs of model.
        (self._tree_embeddings_1, self._tree_embeddings_2, self._gold_similarities) = self._compiler.output_tensors

        self._sim = sim_measure(e1=self._tree_embeddings_1, e2=self._tree_embeddings_2)

        # self._loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        #    logits=logits, labels=labels))

        # use MSE
        self._mse = tf.pow(self._sim - self._gold_similarities, 2)

        #self._mse_comp = tf.pow((self._sim * 4.0) - (self._gold_similarities * 4.0), 2)
        self._mse_comp = self._mse * 16.0

        # self._mse = tf.square(self._sim - self._gold_similarities)
        self._loss = tf.reduce_mean(self._mse)

        # self._accuracy = tf.reduce_mean(
        #    tf.cast(tf.equal(tf.argmax(labels, 1),
        #                     tf.argmax(logits, 1)),
        #            dtype=tf.float32))

        self._global_step = tf.Variable(0, name=VAR_NAME_GLOBAL_STEP, trainable=False)
        #optr = tf.train.GradientDescentOptimizer(0.01)
        optr = optimizer(learning_rate=learning_rate)
        # self._train_op = optr.minimize(self._loss, global_step=self._global_step)

        # gradient clipping
        gradients, variables = zip(*optr.compute_gradients(self._loss))
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        self._train_op = optr.apply_gradients(grads_and_vars=zip(gradients, variables),
                                              global_step=self._global_step)

    @property
    def tree_embeddings_1(self):
        return self._tree_embeddings_1

    @property
    def tree_embeddings_2(self):
        return self._tree_embeddings_2

    # @property
    # def cosine_similarities(self):
    #    return self._cosine_similarities

    @property
    def gold_similarities(self):
        return self._gold_similarities

    @property
    def loss(self):
        return self._loss

    @property
    def mse(self):
        return self._mse

    @property
    def mse_compare(self):
        return self._mse_comp

    @property
    def sim(self):
        return self._sim

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
    def embedding_var(self):
        return self._embeddings

    @property
    def embedding_init(self):
        return self._embedding_init

    @property
    def embedding_placeholder(self):
        return self._embedding_placeholder

    @property
    def compiler(self):
        return self._compiler

    def build_feed_dict(self, sim_trees):
        return self._compiler.build_feed_dict(sim_trees)


class SequenceTreeEmbedding(object):
    def __init__(self, embeddings, tree_embedder=TreeEmbedding_TREE_LSTM, apply_embedding_fc=False,
                 scoring_enabled=True,
                 scoring_scope=DEFAULT_SCOPE_SCORING):
        def squz(x):
            return tf.squeeze(x, [1])

        self._scoring_enabled = scoring_enabled

        tree_embed = tree_embedder(embeddings, apply_embedding_fc=apply_embedding_fc)

        if scoring_enabled:
            # This layer maps a sequence tree embedding to an 'integrity' score
            with tf.variable_scope(scoring_scope) as scoring_sc:
                scoring_fc = td.FC(1, name=scoring_sc) >> td.Function(squz)
            model = td.SerializedMessageToTree(
                'recursive_dependency_embedding.SequenceNode') >> tree_embed() >> td.AllOf(td.Identity(), scoring_fc)

            self._compiler = td.Compiler.create(model)
            self._tree_embeddings, self._scores = self._compiler.output_tensors
        else:
            model = td.SerializedMessageToTree(
                'recursive_dependency_embedding.SequenceNode') >> tree_embed()
            self._compiler = td.Compiler.create(model)
            self._tree_embeddings, = self._compiler.output_tensors

    @property
    def tree_embeddings(self):
        return self._tree_embeddings

    @property
    def scores(self):
        return self._scores

    @property
    def scoring_enabled(self):
        return self._scoring_enabled

    def build_feed_dict(self, sim_trees):
        return self._compiler.build_feed_dict(sim_trees)


class SequenceTreeEmbeddingSequence(object):
    """ A Fold model for training sequence tree embeddings using negative sampling.
        The model expects a converted (see td.proto_tools.serialized_message_to_tree) SequenceNodeSequence object 
        containing a sequence of sequence trees (see SequenceNode) assuming the first is the correct one.
        It calculates all sequence tree embeddings, maps them to an 'integrity' score and calculates the maximum 
        entropy loss with regard to the correct tree.
    """

    def __init__(self, embeddings, scoring_scope=DEFAULT_SCOPE_SCORING):
        def squz(x):
            return tf.squeeze(x, [1])

        # This layer maps a sequence tree embedding to an 'integrity' score
        with tf.variable_scope(scoring_scope) as scoring_sc:
            scoring_fc = td.FC(1, name=scoring_sc) >> td.Function(squz)

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
        norm_first = td.Composition()
        with norm_first.scope():
            sum_ = td.Reduce(td.Function(tf.add)).reads(norm_first.input)
            nth_ = first.reads(norm_first.input)
            normed_nth = td.Function(tf.div).reads(nth_, sum_)
            norm_first.output.reads(normed_nth)

        ## takes a sequence of scalars and an index as input
        ## and returns the n-th scalar normalized by the sum
        ## of all the scalars
        # norm = td.Composition()
        # with norm.scope():
        #    sum_ = td.Reduce(td.Function(tf.add)).reads(norm.input[0])
        #    nth_ = td.Nth().reads(norm.input[0], norm.input[1])
        #    normed_nth = td.Function(tf.div).reads(nth_, sum_)
        #    norm.output.reads(normed_nth)

        tree_embed = TreeEmbedding_HTU_GRU(embeddings)

        # with tf.variable_scope(aggregator_ordered_scope) as sc:
        #    tree_logits = td.Map(sequence_tree_block(embeddings, sc)
        #                         >> scoring_fc >> td.Function(tf.exp))
        tree_logits = td.Map(tree_embed() >> scoring_fc >> td.Function(tf.exp))

        # softmax_correct = td.AllOf(td.GetItem('trees') >> tree_logits, td.GetItem('idx_correct')) >> norm
        softmax_correct = td.GetItem('trees') >> tree_logits >> norm_first

        self._compiler = td.Compiler.create(softmax_correct)

        # Get the tensorflow tensors that correspond to the outputs of model.
        self._softmax_correct, = self._compiler.output_tensors

        self._loss = tf.reduce_mean(-tf.log(self._softmax_correct))

        self._global_step = tf.Variable(0, name=VAR_NAME_GLOBAL_STEP, trainable=False)
        optr = tf.train.GradientDescentOptimizer(0.01)
        self._train_op = optr.minimize(self._loss, global_step=self._global_step)
        tf.summary.scalar('loss', self._loss)
        self._acc = tf.reduce_mean(self._softmax_correct)
        tf.summary.scalar('acc', self._acc)

    @property
    def softmax_correct(self):
        return self._softmax_correct

    @property
    def loss(self):
        return self._loss

    @property
    def accuracy(self):
        return self._acc

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
    expr_decl = td.ForwardDeclaration(td.PyObjectType(), td.SequenceType(td.convert_to_type(state_size)))
    expr_decl2 = td.ForwardDeclaration(td.PyObjectType(), state_size)
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

    def case_dep(seq_tree):
        if 'children_candidate' in seq_tree and len(seq_tree['children']) > 0 and 'head' in seq_tree:
            return 0
        if 'children_candidate' in seq_tree and len(seq_tree['children']) == 0 and 'head' in seq_tree:
            return 1
        if 'children_candidate' in seq_tree and len(seq_tree['children']) > 0 and 'head' not in seq_tree:
            return 2
        if 'children_candidate' in seq_tree and len(seq_tree['children']) == 0 and 'head' not in seq_tree:
            return 3
        if 'children_candidate' not in seq_tree and len(seq_tree['children']) > 0 and 'head' in seq_tree:
            return 4
        if 'children_candidate' not in seq_tree and len(seq_tree['children']) == 0 and 'head' in seq_tree:
            return 5
        if 'children_candidate' not in seq_tree and len(seq_tree['children']) > 0 and 'head' not in seq_tree:
            return 6
        # if len(seq_tree['children_candidates']) == 0 and len(seq_tree['children']) == 0 and 'head' not in seq_tree:
        #    return 7
        if len(seq_tree['candidates']) > 0:
            return 8

    cases = td.OneOf(lambda x: case_dep(x),
                     {0: td.GetItem('children_candidate') >> expr_decl(),
                      1: td.GetItem('children_candidate') >> expr_decl(),
                      2: td.GetItem('children_candidate') >> expr_decl(),
                      3: td.GetItem('children_candidate') >> expr_decl(),
                      4: td.GetItem('head')
                         >> td.Optional(td.Scalar(dtype='int32')
                                        >> td.Function(embed))
                         >> td.Broadcast(),
                      5: td.GetItem('head')
                         >> td.Optional(td.Scalar(dtype='int32')
                                        >> td.Function(embed))
                         >> td.Broadcast(),
                      6: td.GetItem('children')
                         >> td.Map(expr_decl()),
                      # >> td.Reduce(td.Function(aggregator_order_unaware)),
                      8: td.GetItem('candidates')
                         >> td.Map(expr_decl()),
                      })

    def cases_single(seq_tree):
        # children and head exist: process and aggregate
        if len(seq_tree['children']) > 0 and 'head' in seq_tree:
            return 0
        # children do not exist (but maybe a head): process (optional) head only
        if len(seq_tree['children']) == 0:
            return 1
        # otherwise (head does not exist): process children only
        return 2

    def cases_candidates(seq_tree):
        if len(seq_tree['candidates']) > 0:
            return 3
        # children and head exist: process and aggregate
        if len(seq_tree['children']) > 0 and 'head' in seq_tree:
            return 0
        # children do not exist (but maybe a head): process (optional) head only
        if len(seq_tree['children']) == 0:
            return 1
        # otherwise (head does not exist): process children only
        return 2

    expr_decl.resolve_to(cases)

    return cases >> td.Map(td.Function(norm))  # >> td.Function(dprint)


class SequenceTreeEmbeddingWithCandidates(object):
    """ A Fold model for training sequence tree embeddings using NCE.
        The model expects a converted (see td.proto_tools.serialized_message_to_tree) SequenceNodeSequence object 
        containing a sequence of sequence trees (see SequenceNode) and an index of the correct tree.
        It calculates all sequence tree embeddings, maps them to an 'integrity' score and calculates the maximum 
        entropy loss with regard to the correct tree.
    """

    def __init__(self, embeddings, scoring_scope=DEFAULT_SCOPE_SCORING):
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

        # TODO: fix
        with tf.variable_scope(aggregator_ordered_scope) as sc:
            tree_logits = sequence_tree_block_with_candidates(embeddings, sc) \
                          >> td.Map(scoring_fc >> td.Function(squz) >> td.Function(tf.exp))

        softmax_correct = tree_logits >> norm

        self._compiler = td.Compiler.create(softmax_correct)

        # Get the tensorflow tensors that correspond to the outputs of model.
        self._softmax_correct = self._compiler.output_tensors

        self._loss = tf.reduce_mean(-tf.log(self._softmax_correct))

        self._global_step = tf.Variable(0, name=VAR_NAME_GLOBAL_STEP, trainable=False)
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
