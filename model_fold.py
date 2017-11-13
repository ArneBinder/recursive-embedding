from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# import google3
import tensorflow as tf
import tensorflow_fold.public.blocks as td
from tensorflow_fold.blocks import result_types as tdt
import numpy as np

# DEFAULT_SCOPE_TREE_EMBEDDER = 'tree_embedder'   # DEPRECATED
DEFAULT_SCOPE_SCORING = 'scoring'
DIMENSION_EMBEDDINGS = 300
DIMENSION_SIM_MEASURE = 300
VAR_NAME_LEXICON = 'embeddings'
VAR_NAME_GLOBAL_STEP = 'global_step'
VAR_PREFIX_FC_LEAF = 'FC_embedding'
VAR_PREFIX_FC_ROOT = 'FC_output'
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


class SequenceToTuple(td.Block):
    """A Python function, lifted to a block."""

    def __init__(self, T, N, name=None):
        py_fn = (lambda x: tuple(x))
        if not callable(py_fn):
            raise TypeError('py_fn is not callable: %s' % str(py_fn))
        self._py_fn = py_fn
        super(SequenceToTuple, self).__init__(
            [], input_type=td.SequenceType(T), output_type=td.TupleType(*([T] * N)),
            name=name)

    def _repr_kwargs(self):
        return dict(py_fn=self.py_fn)

    @property
    def py_fn(self):
        return self._py_fn

    def _evaluate(self, _, x):
        return self._py_fn(x)


def fc_scoped(num_units, scope, name=None, activation_fn=tf.nn.relu, keep_prob=None):
    def fc_(inputs, scope):
        if keep_prob is None:
            return tf.contrib.layers.fully_connected(inputs, num_units, activation_fn=activation_fn, scope=scope)
        else:
            return tf.nn.dropout(tf.contrib.layers.fully_connected(inputs, num_units, activation_fn=activation_fn, scope=scope), keep_prob)

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


def create_lexicon(lex_size, trainable=True):
    lexicon = tf.Variable(tf.constant(0.0, shape=[lex_size, DIMENSION_EMBEDDINGS]),
                          trainable=trainable, name=VAR_NAME_LEXICON)
    lexicon_placeholder = tf.placeholder(tf.float32, [lex_size, DIMENSION_EMBEDDINGS])
    lexicon_init = lexicon.assign(lexicon_placeholder)

    return lexicon, lexicon_placeholder, lexicon_init


class TreeEmbedding(object):
    def __init__(self, name, lexicon_size, keep_prob_placeholder=None, lexicon_trainable=True, state_size=None,
                 leaf_fc_size=0, root_fc_size=0, keep_prob_fixed=1.0):
        self._lex_size = lexicon_size
        self._lexicon, self._lexicon_placeholder, self._lexicon_init = create_lexicon(lex_size=lexicon_size,
                                                                                      trainable=lexicon_trainable)
        self._keep_prob_fixed = keep_prob_fixed
        if keep_prob_placeholder is not None:
            self._keep_prob = keep_prob_placeholder
        else:
            self._keep_prob = 1.0
        if state_size:
            self._state_size = state_size
        else:
            self._state_size = self._lexicon.get_shape().as_list()[1]  # state_size
        #self._apply_leaf_fc = (leaf_fc_size > 0)
        self._name = VAR_PREFIX_TREE_EMBEDDING + '_' + name  # + '_%d' % self._state_size

        self._leaf_fc_size = leaf_fc_size
        self._root_fc_size = root_fc_size
        with tf.variable_scope(self.name) as scope:
            self._scope = scope
            if self._leaf_fc_size:
                self._leaf_fc = fc_scoped(num_units=leaf_fc_size,
                                          activation_fn=tf.nn.tanh, scope=scope, keep_prob=self.keep_prob,
                                          name=VAR_PREFIX_FC_LEAF + '_%d' % leaf_fc_size)
            else:
                self._leaf_fc = td.Identity()
            if root_fc_size:
                self._root_fc = fc_scoped(num_units=root_fc_size, activation_fn=tf.nn.tanh, scope=scope,
                                          keep_prob=self.keep_prob,
                                          name=VAR_PREFIX_FC_ROOT + '_%d' % self.state_size)
            else:
                self._root_fc = td.Identity()

    def embed(self):
        # get the head embedding from id
        #return td.Function(lambda x: tf.gather(self._lexicon, x))
        return td.OneOf(key_fn=(lambda x: x >= 0),
                        case_blocks={True: td.Scalar(dtype='int32') >> td.Function(lambda x: tf.gather(self._lexicon, x)),
                                     False: td.Void() >> td.Zeros(DIMENSION_EMBEDDINGS)})

    def head(self, name='head_embed'):
        def helper(x, keep_prob):
            if x < 0:
                return x + self.lex_size
            if np.random.random() < keep_prob:
                return x
            return -1

        #return td.Pipe(td.GetItem('head'), td.Scalar(dtype='int32'), self.embed(), name=name)
        return td.Pipe(td.GetItem('head'), td.InputTransform(lambda x: helper(x, self.keep_prob_fixed)),
                       self.embed(), name=name)

    def children(self, name='children'):
        return td.InputTransform(lambda x: x.get('children', []), name=name)

    @property
    def state_size(self):
        return self._state_size

    @property
    def lexicon(self):
        return self._lexicon

    @property
    def name(self):
        return self._name

    @property
    def scope(self):
        return self._scope

    @property
    def leaf_fc(self):
        return self._leaf_fc

    @property
    def leaf_fc_size(self):
        return self._leaf_fc_size or 0

    @property
    def root_fc_size(self):
        return self._root_fc_size or 0

    @property
    def root_fc(self):
        return self._root_fc

    #@property
    #def embedding_fc_size_multiple(self):
    #    return 1

    @property
    def lexicon_var(self):
        return self._lexicon

    @property
    def lexicon_init(self):
        return self._lexicon_init

    @property
    def lexicon_placeholder(self):
        return self._lexicon_placeholder

    @property
    def keep_prob(self):
        return self._keep_prob

    @property
    def keep_prob_fixed(self):
        return self._keep_prob_fixed

    @property
    def lex_size(self):
        return self._lex_size


class TreeEmbedding_TREE_LSTM(TreeEmbedding):
    """Calculates an embedding over a (recursive) SequenceNode.

    Args:
        embeddings: a tensor of shape=(lex_size, state_size) containing the (pre-trained) embeddings
        name_or_scope: A scope to share variables over instances of sequence_tree_block
    """

    def __init__(self, **kwargs):
        super(TreeEmbedding_TREE_LSTM, self).__init__(name='TREE_LSTM', **kwargs)
        with tf.variable_scope(self.scope) as scope:
            self._xh_linear = fc_scoped(num_units=3 * self.state_size, scope=scope,
                                        name='FC_xh_linear_%d' % (3 * self.state_size), activation_fn=None)
            self._fc_f = fc_scoped(num_units=self._state_size, scope=scope,
                                   name='FC_f_linear_%d' % self.state_size, activation_fn=None)

    def __call__(self):
        zero_state = td.Zeros((self.state_size, self.state_size))
        embed_tree = td.ForwardDeclaration(input_type=td.PyObjectType(), output_type=zero_state.output_type)
        treelstm = treeLSTM(self._xh_linear, self._fc_f, forget_bias=2.5)

        children = self.children() >> td.Map(embed_tree())
        cases = td.AllOf(self.head(), children) >> treelstm
        embed_tree.resolve_to(cases)

        # TODO: use only h state. DONE, but needs testing!
        #return cases >> td.Concat() >> self.output_fc
        return cases >> td.GetItem(1) >> self.root_fc


class TreeEmbedding_HTU_GRU(TreeEmbedding):
    """Calculates an embedding over a (recursive) SequenceNode.

    Args:
        embeddings: a tensor of shape=(lex_size, state_size) containing the (pre-trained) embeddings
        name_or_scope: A scope to share variables over instances of sequence_tree_block
    """

    def __init__(self, **kwargs):
        super(TreeEmbedding_HTU_GRU, self).__init__(name='HTU_GRU', **kwargs)
        with tf.variable_scope(self.scope):
            self._grucell = td.ScopedLayer(tf.contrib.rnn.DropoutWrapper(
                tf.contrib.rnn.GRUCell(num_units=self._state_size),
                input_keep_prob=self.keep_prob,
                output_keep_prob=self.keep_prob,
                variational_recurrent=True,
                dtype=tf.float32,
                input_size=self.leaf_fc_size or DIMENSION_EMBEDDINGS),
                'gru_cell')

    def __call__(self):
        #zero_state = td.Zeros(self._state_size)
        # zero_state = td.Zeros((state_size, state_size))
        embed_tree = td.ForwardDeclaration(input_type=td.PyObjectType(), output_type=self.state_size)

        # an aggregation function which takes the order of the inputs into account
        def aggregator_order_aware(head, children):
            # inputs=head, state=children
            r, h2 = self._grucell(head, children)
            return r

        # an aggregation function which doesn't take the order of the inputs into account
        # TODO: try td.Mean()
        #def aggregator_order_unaware(x, y):
        #    return tf.add(x, y)

        # simplified naive version (minor modification: apply order_aware also to single head with zeros as input state)
        children = self.children() >> td.Map(embed_tree()) >> td.Sum()

        cases = td.AllOf(self.head() >> self.leaf_fc, children) >> td.Function(aggregator_order_aware)

        embed_tree.resolve_to(cases)
        model = cases >> self.root_fc

        return model


class TreeEmbedding_HTU_GRU_dep(TreeEmbedding):
    """Calculates an embedding over a (recursive) SequenceNode.

    Args:
        embeddings: a tensor of shape=(lex_size, state_size) containing the (pre-trained) embeddings
        name_or_scope: A scope to share variables over instances of sequence_tree_block
    """

    def __init__(self, **kwargs):

        super(TreeEmbedding_HTU_GRU_dep, self).__init__(name='HTU_GRU_dep', **kwargs)
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
        # TODO: try td.Mean()
        #def aggregator_order_unaware(x, y):
        #    return tf.add(x, y)

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
            return td.Pipe(td.GetItem('children'), td.Map(embed_tree()), td.Sum(), name=name)

        cases = td.OneOf(lambda x: case(x),
                         {0: td.AllOf(self.head(), children()) >> td.Function(aggregator_order_aware),
                          1: self.head('head_only'),
                          2: children('children_only'),
                          })

        embed_tree.resolve_to(cases)

        model = cases >> self.root_fc
        return model


class TreeEmbedding_FLAT(TreeEmbedding):
    def __init__(self, name, **kwargs):
        super(TreeEmbedding_FLAT, self).__init__(name='FLAT_' + name, **kwargs)

    def element(self, name='element'):
        return td.Pipe(self.head(), self.leaf_fc, name=name)

    def sequence(self, name='sequence'):
        return td.Pipe(self.children(), td.Map(self.element()), name=name)

    def aggregate(self, name='aggregate'):
        raise NotImplementedError("Please Implement this method")

    def __call__(self):
        model = self.sequence() >> self.aggregate() >> self.root_fc
        if model.output_type is None:
            model.set_output_type(tdt.TensorType(shape=(self.output_size,), dtype='float32'))
        return model

    @property
    def element_size(self):
        return self.leaf_fc_size or DIMENSION_EMBEDDINGS

    @property
    def output_size(self):
        raise NotImplementedError("Please Implement this method")


class TreeEmbedding_FLAT_2levels(TreeEmbedding_FLAT):
    def __init__(self, name, **kwargs):
        super(TreeEmbedding_FLAT_2levels, self).__init__(name=name, **kwargs)

    def element(self, name='element'):
        # use word embedding and first child embedding
        return td.Pipe(td.AllOf(self.head(name='head_level1'),
                                td.GetItem('children') >> td.InputTransform(lambda s: s[0]) >> self.head(
                                    name='head_level2')),
                       td.Concat(), self.leaf_fc, name=name)

    @property
    def element_size(self):
        return self.leaf_fc_size or (DIMENSION_EMBEDDINGS * 2)


class TreeEmbedding_FLAT_AVG(TreeEmbedding_FLAT):
    def __init__(self, name=None, **kwargs):
        super(TreeEmbedding_FLAT_AVG, self).__init__(name=name or 'AVG', **kwargs)

    def aggregate(self, name='aggregate'):
        # an aggregation function which doesn't take the order of the inputs into account
        return td.Mean(name)

    @property
    def output_size(self):
        return self.root_fc_size or self.element_size


class TreeEmbedding_FLAT_AVG_2levels(TreeEmbedding_FLAT_AVG, TreeEmbedding_FLAT_2levels):
    def __init__(self, name=None, **kwargs):
        super(TreeEmbedding_FLAT_AVG_2levels, self).__init__(name=name or 'AVG_2levels', **kwargs)


class TreeEmbedding_FLAT_SUM(TreeEmbedding_FLAT):
    def __init__(self, name=None, **kwargs):
        super(TreeEmbedding_FLAT_SUM, self).__init__(name=name or 'SUM', **kwargs)

    def aggregate(self, name='aggregate'):
        # an aggregation function which doesn't take the order of the inputs into account
        return td.Sum(name)

    @property
    def output_size(self):
        return self.root_fc_size or self.element_size


class TreeEmbedding_FLAT_SUM_2levels(TreeEmbedding_FLAT_SUM, TreeEmbedding_FLAT_2levels):
    def __init__(self, name=None, **kwargs):
        super(TreeEmbedding_FLAT_SUM_2levels, self).__init__(name=name or 'SUM_2levels', **kwargs)


class TreeEmbedding_FLAT_LSTM(TreeEmbedding_FLAT):
    def __init__(self, name=None, **kwargs):
        super(TreeEmbedding_FLAT_LSTM, self).__init__(name=name or 'LSTM', **kwargs)
        with tf.variable_scope(self.scope):
            self._lstm_cell = td.ScopedLayer(
                tf.contrib.rnn.DropoutWrapper(
                    tf.contrib.rnn.BasicLSTMCell(num_units=self.state_size, forget_bias=2.5),
                    input_keep_prob=self.keep_prob,
                    output_keep_prob=self.keep_prob,
                    variational_recurrent=True,
                    dtype=tf.float32,
                    input_size=self.element_size),
                'lstm_cell')

    def aggregate(self, name='aggregate'):
        # apply LSTM >> take the LSTM output state(s) >> take the h state (discard the c state)
        return td.Pipe(td.RNN(self._lstm_cell), td.GetItem(1), td.GetItem(0), name=name)

    @property
    def output_size(self):
        return self.root_fc_size or self.state_size


# compatibility
class TreeEmbedding_FLAT_LSTM50(TreeEmbedding_FLAT_LSTM):
    def __init__(self, **kwargs):
        super(TreeEmbedding_FLAT_LSTM50, self).__init__(name='LSTM50', **kwargs)


class TreeEmbedding_FLAT_LSTM_2levels(TreeEmbedding_FLAT_LSTM, TreeEmbedding_FLAT_2levels):
    def __init__(self, name=None, **kwargs):
        super(TreeEmbedding_FLAT_LSTM_2levels, self).__init__(name=name or 'LSTM_2levels', **kwargs)


# compatibility
class TreeEmbedding_FLAT_LSTM50_2levels(TreeEmbedding_FLAT_LSTM_2levels):
    def __init__(self, **kwargs):
        super(TreeEmbedding_FLAT_LSTM50_2levels, self).__init__(name='LSTM50_2levels', **kwargs)


class TreeEmbedding_FLAT_GRU(TreeEmbedding_FLAT):
    def __init__(self, name=None, **kwargs):
        super(TreeEmbedding_FLAT_GRU, self).__init__(name=name or 'GRU', **kwargs)
        with tf.variable_scope(self.scope):
            self._gru_cell = td.ScopedLayer(
                tf.contrib.rnn.DropoutWrapper(
                    tf.contrib.rnn.GRUCell(num_units=self.state_size),
                    input_keep_prob=self.keep_prob,
                    output_keep_prob=self.keep_prob,
                    variational_recurrent=True,
                    dtype=tf.float32,
                    input_size=self.element_size),
                'gru_cell')

    def aggregate(self, name='aggregate'):
        # apply GRU >> take the GRU output state(s)
        return td.Pipe(td.RNN(self._gru_cell), td.GetItem(1), name=name)


class TreeEmbedding_FLAT_GRU_2levels(TreeEmbedding_FLAT_GRU, TreeEmbedding_FLAT_2levels):
    def __init__(self, name=None, **kwargs):
        super(TreeEmbedding_FLAT_GRU_2levels, self).__init__(name=name or 'GRU_2levels', **kwargs)


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


def get_all_heads(tree):
    current_heads = [tree['head']]
    for child in tree['children']:
        current_heads.extend(get_all_heads(child))
    return current_heads


def get_jaccard_sim(tree_tuple):
    heads1 = set(get_all_heads(tree_tuple['first']))
    heads2 = set(get_all_heads(tree_tuple['second']))
    return len(heads1 & heads2) / float(len(heads1 | heads2))


class SequenceTreeModel(object):
    def __init__(self, lex_size, tree_embedder=TreeEmbedding_TREE_LSTM, lexicon_trainable=True, keep_prob=1.0,
                 tree_count=2, prob_count=None, **kwargs):
        if prob_count is None:
            self._prob_count = tree_count
        else:
            self._prob_count = prob_count

        self._tree_count = tree_count
        self._keep_prob = tf.placeholder_with_default(keep_prob, shape=())

        self._tree_embed = tree_embedder(lexicon_size=lex_size, lexicon_trainable=lexicon_trainable,
                                         keep_prob_placeholder=self._keep_prob, **kwargs)

        embed_tree = self._tree_embed()
        model = td.AllOf(td.GetItem(0) >> td.Map(embed_tree) >> SequenceToTuple(embed_tree.output_type, self._tree_count) >> td.Concat(),
                         td.GetItem(1) >> td.Vector(self._prob_count))

        # fold model output
        self._compiler = td.Compiler.create(model)
        (self._tree_embeddings_all, self._probs_gold) = self._compiler.output_tensors

        ## TODO: doesn't work as expected!
        #self._probs_gold_flattened = tf.concat(self._probs_gold, axis=0)
        #self._tree_embeddings_all_flattened = tf.concat(self._tree_embeddings_all, axis=0)

    def build_feed_dict(self, data):
        return self._compiler.build_feed_dict(data)

    @property
    def keep_prob(self):
        return self._keep_prob

    @property
    def embedder(self):
        return self._tree_embed

    @property
    def embeddings_all(self):
        return self._tree_embeddings_all

    ## TODO: doesn't work as expected!
    #@property
    #def embeddings_all_flattened(self):
    #    return self._tree_embeddings_all_flattened

    @property
    def probs_gold(self):
        return self._probs_gold

    ## TODO: doesn't work as expected!
    #@property
    #def probs_gold_flattened(self):
    #    return self._probs_gold_flattened

    @property
    def compiler(self):
        return self._compiler

    @property
    def tree_count(self):
        return self._tree_count

    @property
    def prob_count(self):
        return self._prob_count

    @property
    def tree_output_size(self):
        return int(self._tree_embeddings_all.get_shape().as_list()[1] / self.tree_count)


class BaseTrainModel(object):
    def __init__(self, tree_model, loss, optimizer=None, learning_rate=0.1):

        self._loss = loss
        self._tree_model = tree_model

        self._global_step = tf.Variable(0, name=VAR_NAME_GLOBAL_STEP, trainable=False)

        if optimizer is not None:
            self._optimizer = optimizer(learning_rate=learning_rate)

            # gradient manipulation
            gradients, variables = zip(*self._optimizer.compute_gradients(self._loss))
            # reversal gradient integration
            # gradients_rev, _ = zip(*self._optimizer.compute_gradients(self._loss_rev))
            # gradients = list(gradients)
            # for grads_idx, grads in enumerate(gradients):
            #    if isinstance(grads, tf.Tensor):
            #        gradients[grads_idx] = gradients[grads_idx] - gradients_rev[grads_idx]
            # gradient clipping
            gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
            self._train_op = self._optimizer.apply_gradients(grads_and_vars=zip(gradients, variables),
                                                             global_step=self._global_step)
        else:
            self._train_op = None

    def optimizer_vars(self):
        slot_names = self._optimizer.get_slot_names()
        all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        opt_vars = []
        for slot_name in slot_names:
            opt_vars.extend([self._optimizer.get_slot(var=v, name=slot_name) for v in all_vars if
                             self._optimizer.get_slot(var=v, name=slot_name)])
        return opt_vars

    @property
    def train_op(self):
        return self._train_op

    @property
    def loss(self):
        return self._loss

    @property
    def global_step(self):
        return self._global_step

    @property
    def tree_model(self):
        return self._tree_model


class SimilaritySequenceTreeTupleModel(BaseTrainModel):
    """A Fold model for similarity scored sequence tree (SequenceNode) tuple."""

    def __init__(self, tree_model, learning_rate=0.01, optimizer=None, sim_measure=sim_cosine):

        # unpack scores_gold. Every prob tuple has the format: [1.0, score_gold, ...]
        self._scores_gold = tree_model.probs_gold[:, 1]

        # get first two tree embeddings
        tree_size = tree_model.tree_output_size
        self._tree_embeddings_1 = tree_model.embeddings_all[:, :tree_size]
        self._tree_embeddings_2 = tree_model.embeddings_all[:, tree_size:tree_size * 2]
        self._scores = sim_measure(e1=self._tree_embeddings_1, e2=self._tree_embeddings_2)

        BaseTrainModel.__init__(self, optimizer=optimizer, learning_rate=learning_rate,
                                tree_model=tree_model, loss=tf.reduce_mean(tf.square(self._scores - self._scores_gold)))

    @property
    def tree_embeddings_1(self):
        return self._tree_embeddings_1

    @property
    def tree_embeddings_2(self):
        return self._tree_embeddings_2

    @property
    def scores_gold(self):
        return self._scores_gold

    @property
    def scores(self):
        return self._scores


class ScoredSequenceTreeTupleModel(BaseTrainModel):
    """A Fold model for similarity scored sequence tree (SequenceNode) tuple."""

    def __init__(self, tree_model, learning_rate=0.01, optimizer=tf.train.GradientDescentOptimizer, probs_count=2):

        self._prediction_logits = tf.contrib.layers.fully_connected(tree_model.embeddings_all, probs_count,
                                                                    activation_fn=None, scope=DEFAULT_SCOPE_SCORING)
        loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tree_model.probs_gold, logits=self._prediction_logits))

        BaseTrainModel.__init__(self, tree_model=tree_model, loss=loss, optimizer=optimizer, learning_rate=learning_rate)


# TODO: not implemented yet
class ScoredSequenceTreeTupleModel_independent(BaseTrainModel):
    """A Fold model for similarity scored sequence tree (SequenceNode) tuple."""

    def __init__(self, tree_model, learning_rate=0.01, optimizer=tf.train.GradientDescentOptimizer, probs_count=2):



        self._prediction_logits = tf.contrib.layers.fully_connected(tree_model.embeddings_all, probs_count,
                                                                    activation_fn=None, scope=DEFAULT_SCOPE_SCORING)
        loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tree_model.probs_gold, logits=self._prediction_logits))

        BaseTrainModel.__init__(self, tree_model=tree_model, loss=loss, optimizer=optimizer,
                                learning_rate=learning_rate)


# DEPRECATED
class SequenceTreeEmbedding(object):
    def __init__(self, tree_model, sim_measure=sim_cosine, scoring_enabled=True, scoring_scope=DEFAULT_SCOPE_SCORING):

        self._scoring_enabled = scoring_enabled

        self._tree_model = tree_model #tree_embedder(lexicon_size=lex_size, lexicon_trainable=lexicon_trainable, **kwargs)

        model = self._tree_model.embedder()
        self._compiler = td.Compiler.create(model)
        self._tree_embeddings, = self._compiler.output_tensors

        if scoring_enabled:
            # This layer maps a sequence tree embedding to an 'integrity' score
            with tf.variable_scope(scoring_scope) as scoring_sc:
                scoring_fc = td.FC(1, name=scoring_sc) >> td.Function(lambda x: tf.squeeze(x, [1]))

            self._compiler_scoring = td.Compiler.create(td.Tensor(shape=model.output_type.shape) >> scoring_fc)
            self._scores, = self._compiler_scoring.output_tensors

        self._e1_placeholder = tf.placeholder(tf.float32, shape=[None, model.output_type.shape[0]])
        self._e2_placeholder = tf.placeholder(tf.float32, shape=[None, model.output_type.shape[0]])
        self._sim = sim_measure(e1=self._e1_placeholder, e2=self._e2_placeholder)

    @property
    def tree_embeddings(self):
        return self._tree_embeddings

    @property
    def scores(self):
        if self._scoring_enabled:
            return self._scores
        else:
            return None

    @property
    def scoring_enabled(self):
        return self._scoring_enabled

    def build_feed_dict(self, sim_trees):
        return self._compiler.build_feed_dict(sim_trees)

    def build_scoring_feed_dict(self, embeddings):
        return self._compiler_scoring.build_feed_dict(embeddings)

    @property
    def tree_model(self):
        return self._tree_model

    @property
    def sim(self):
        return self._sim

    @property
    def e1_placeholder(self):
        return self._e1_placeholder

    @property
    def e2_placeholder(self):
        return self._e2_placeholder


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

        tree_embed = TreeEmbedding_HTU_GRU_dep(embeddings)

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
