#from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# import google3
import tensorflow as tf
import tensorflow_fold.public.blocks as td
from tensorflow_fold.blocks import result_types as tdt
from tensorflow_fold.blocks import blocks as tdb
import numpy as np
import math

from constants import KEY_HEAD, KEY_CHILDREN

# DEFAULT_SCOPE_TREE_EMBEDDER = 'tree_embedder'   # DEPRECATED
DEFAULT_SCOPE_SCORING = 'scoring'
# DIMENSION_EMBEDDINGS = 300     use lexicon.dimension_embeddings
DIMENSION_SIM_MEASURE = 300
VAR_NAME_LEXICON_VAR = 'embeddings'
VAR_NAME_LEXICON_FIX = 'embeddings_fix'
VAR_NAME_GLOBAL_STEP = 'global_step'
VAR_PREFIX_FC_LEAF = 'FC_embedding'
VAR_PREFIX_FC_ROOT = 'FC_output'
VAR_PREFIX_FC_REVERSE = 'FC_reverse'
VAR_PREFIX_TREE_EMBEDDING = 'TreeEmbedding'
VAR_PREFIX_SIM_MEASURE = 'sim_measure'

MODEL_TYPE_DISCRETE = 'mt_discrete'
MODEL_TYPE_REGRESSION = 'mt_regression'


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
    result.set_output_type(td.TensorType(shape=(num_units,)))
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


def create_lexicon(lex_size, dimension_embeddings, trainable=True):
    lexicon = tf.Variable(tf.constant(0.0, shape=[lex_size, dimension_embeddings]),
                          trainable=trainable, name=VAR_NAME_LEXICON_VAR if trainable else VAR_NAME_LEXICON_FIX)
    lexicon_placeholder = tf.placeholder(tf.float32, [lex_size, dimension_embeddings])
    lexicon_init = lexicon.assign(lexicon_placeholder)

    return lexicon, lexicon_placeholder, lexicon_init


def Softmax(name=None):  # pylint: disable=invalid-name
    """
    This block calculates softmax over a sequence.
    """
    c = td.Composition(name=name)
    with c.scope():
        # _exps = td.Map(td.Function(tf.exp)).reads(c.input)
        # numerical stable version (subtract max)
        _max = td.Max().reads(c.input)
        _exps = td.Map(td.Function(lambda x, m: tf.exp(x - m))).reads(td.Zip().reads(c.input, td.Broadcast().reads(_max)))
        _exps_with_sum = td.Zip().reads(_exps, td.Broadcast().reads(td.Sum().reads(_exps)))
        _res = td.Map(td.Function(tdb._tf_batch_safe_scalar_division)).reads(_exps_with_sum)
        c.output.reads(_res)
    return c.set_constructor_name('td.Softmax')


def AttentionReduce(name=None):  # pylint: disable=invalid-name
    """
    This block calculates attention over a sequence and sums the elements weighted by this attention.
    It requires a sequence of 1-d vectors and a 1-d vector as input and produces a sequence of 1-d vectors.
    """
    c = td.Composition(name=name)
    with c.scope():
        att_scalars = td.Map(td.Function(lambda x, y: tf.reduce_sum(x * y, axis=1))).reads(td.Zip().reads(c.input[0], td.Broadcast().reads(c.input[1])))
        att_weights = Softmax().reads(att_scalars)
        wheighted = td.Map(td.Function(tdb._tf_batch_scalar_mul)).reads(td.Zip().reads(att_weights, c.input[0]))
        res = td.Sum().reads(wheighted)
        c.output.reads(res)
    return c.set_constructor_name('td.Attention')


###################################
#  Abstract TreeEmbedding models  #
###################################


class TreeEmbedding(object):
    def __init__(self, name, lex_size_fix, lex_size_var, dimension_embeddings, keep_prob_placeholder=None, state_size=None,
                 leaf_fc_size=0, root_fc_sizes=0, keep_prob_fixed=1.0, **unused):
        self._lex_size_fix = lex_size_fix
        self._lex_size_var = lex_size_var
        self._dim_embeddings = dimension_embeddings

        # compatibility
        #if self._lex_size_fix > 0:
        self._lexicon_fix, self._lexicon_fix_placeholder, self._lexicon_fix_init = create_lexicon(lex_size=self._lex_size_fix,
                                                                                                  dimension_embeddings=self._dim_embeddings,
                                                                                                  trainable=False)
        #if self._lex_size_var > 0:
        self._lexicon_var, self._lexicon_var_placeholder, self._lexicon_var_init = create_lexicon(lex_size=self._lex_size_var,
                                                                                                  dimension_embeddings=self._dim_embeddings,
                                                                                                  trainable=True)
        self._keep_prob_fixed = keep_prob_fixed
        if keep_prob_placeholder is not None:
            self._keep_prob = keep_prob_placeholder
        else:
            self._keep_prob = 1.0
        if state_size:
            self._state_size = state_size
        else:
            self._state_size = self._dim_embeddings  # state_size
        #self._apply_leaf_fc = (leaf_fc_size > 0)
        self._name = VAR_PREFIX_TREE_EMBEDDING + '_' + name  # + '_%d' % self._state_size

        self._leaf_fc_size = leaf_fc_size
        if isinstance(root_fc_sizes, (list, tuple)):
            self._root_fc_sizes = root_fc_sizes
        else:
            self._root_fc_sizes = [root_fc_sizes]
        with tf.variable_scope(self.name) as scope:
            self._scope = scope
            if self._leaf_fc_size:
                self._leaf_fc = fc_scoped(num_units=leaf_fc_size,
                                          activation_fn=tf.nn.tanh, scope=scope, keep_prob=self.keep_prob,
                                          name=VAR_PREFIX_FC_LEAF + '_%d' % leaf_fc_size)
            else:
                self._leaf_fc = td.Identity()
            #if len(root_fc_size) > 0:
            self._root_fcs = []
            for i, s in enumerate(root_fc_sizes):
                if s > 0:
                    self._root_fcs.append(fc_scoped(num_units=s, activation_fn=tf.nn.tanh, scope=scope,
                                                    keep_prob=self.keep_prob,
                                                    name=VAR_PREFIX_FC_ROOT + '%d_%d' % (i, self.state_size)))
            #else:
            #    self._root_fc = td.Identity()

            self._reverse_fc = fc_scoped(num_units=self._dim_embeddings,
                                         activation_fn=tf.nn.tanh, scope=scope, keep_prob=self.keep_prob,
                                         name=VAR_PREFIX_FC_REVERSE + '_%d' % self._dim_embeddings)

    def embed_dep(self):
        # negative values indicate enabled head dropout
        def helper(x, keep_prob):
            if x >= 0:
                return x
            if np.random.random() < keep_prob:
                return x + self.lex_size
            return -1

        # get the head embedding from id
        #return td.Function(lambda x: tf.gather(self._lexicon, x))
        return td.InputTransform(lambda x: helper(x, self.keep_prob_fixed)) \
               >> td.OneOf(key_fn=(lambda x: x >= 0),
                           case_blocks={True: td.Scalar(dtype='int32') >> td.Function(lambda x: tf.gather(self._lexicon, x)),
                                        False: td.Void() >> td.Zeros(self._dim_embeddings)})

    def embed(self):
        # get the head embedding from id
        #if self._lex_size_fix > 0 and self._lex_size_var > 0:
        return td.OneOf(key_fn=(lambda x: (x >= 0, abs(x) // (self._lex_size_fix + self._lex_size_var))),
                        case_blocks={
                            # trainable embedding
                            (True, 0):  td.Scalar(dtype='int32')
                                        >> td.Function(lambda x: tf.gather(self._lexicon_var, tf.mod(x, self.lexicon_size))),
                            # fixed embedding
                            (False, 0): td.Scalar(dtype='int32')
                                        >> td.Function(lambda x: tf.gather(self._lexicon_fix, tf.mod(tf.abs(x), self.lexicon_size))),
                            # trainable embedding for "reverted" edge
                            (True, 1):  td.Scalar(dtype='int32')
                                        >> td.Function(lambda x: tf.gather(self._lexicon_var, tf.mod(x, self.lexicon_size)))
                                        >> self._reverse_fc,
                            # fixed embedding for "reverted" edge
                            (False, 1): td.Scalar(dtype='int32')
                                        >> td.Function(lambda x: tf.gather(self._lexicon_fix, tf.mod(tf.abs(x), self.lexicon_size)))
                                        >> self._reverse_fc,
                        })
        # compatibility
        #if self._lex_size_fix > 0:
        #    return td.Scalar(dtype='int32') >> td.Function(lambda x: tf.gather(self._lexicon_fix, tf.abs(x)))
        #if self._lex_size_var > 0:
        #    return td.Scalar(dtype='int32') >> td.Function(lambda x: tf.gather(self._lexicon_var, x))

    def head(self, name='head_embed'):
        #return td.Pipe(td.GetItem(KEY_HEAD), td.Scalar(dtype='int32'), self.embed(), name=name)
        return td.Pipe(td.GetItem(KEY_HEAD), self.embed(), self.leaf_fc, name=name)

    def children(self, name=KEY_CHILDREN):
        return td.InputTransform(lambda x: x.get(KEY_CHILDREN, []), name=name)

    @property
    def state_size(self):
        return self._state_size

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
        if len(self._root_fc_sizes) > 0:
            return self._root_fc_sizes[-1]
        return 0

    @property
    def root_fc(self):
        # return self._root_fcs
        if len(self._root_fcs) == 0:
            return td.Identity()
        elif len(self._root_fcs) == 1:
            return self._root_fcs[0]
        return td.Pipe(*self._root_fcs)

    @property
    def lexicon_var(self):
        return self._lexicon_var

    @property
    def lexicon_var_init(self):
        return self._lexicon_var_init

    @property
    def lexicon_var_placeholder(self):
        return self._lexicon_var_placeholder

    @property
    def lexicon_fix(self):
        return self._lexicon_fix

    @property
    def lexicon_fix_init(self):
        return self._lexicon_fix_init

    @property
    def lexicon_fix_placeholder(self):
        return self._lexicon_fix_placeholder

    @property
    def lexicon_size(self):
        return self._lex_size_fix + self._lex_size_var

    @property
    def keep_prob(self):
        return self._keep_prob

    @property
    def keep_prob_fixed(self):
        return self._keep_prob_fixed

    @property
    def dimension_embeddings(self):
        return self._dim_embeddings

    @property
    def head_size(self):
        return self.leaf_fc_size or self._dim_embeddings


class TreeEmbedding_TREE_LSTM(TreeEmbedding):
    """Calculates an embedding over a (recursive) SequenceNode."""

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


class TreeEmbedding_map(TreeEmbedding):
    @property
    def map(self):
        """
        Applies a mapping function.
        :return: A tensorflow fold block that consumes a tuple (input, state_in) and outputs a new state with same shape
        as state_in.
        """
        raise NotImplementedError


class TreeEmbedding_reduce(TreeEmbedding):
    @property
    def reduce(self):
        """
        gets a tuple (head_in, children_sequence), where head_in is a 1-d vector and children a sequence of 1-d vectors
        (size does not have to fit head_in size) and produces a tuple (children_reduced, head_out) to feed into
        self.map.
        :return: a tuple (head_out, children_reduced) of two 1-d vectors, of different size, eventually.
        """
        raise NotImplementedError


# TODO: does not work like this! TreeEmbedding_map.map requires: (input, state_in) -> state_out
#class TreeEmbedding_mapIDENTITY(TreeEmbedding_map):
#    def __init__(self, name, **kwargs):
#        super(TreeEmbedding_mapIDENTITY, self).__init__(name='mapIDENTITY_' + name, **kwargs)
#
#    @property
#    def map(self):
#        return td.Identity()


def gru_cell(scope, state_size, keep_prob, input_size):
    with tf.variable_scope(scope):
        return td.ScopedLayer(tf.contrib.rnn.DropoutWrapper(
            tf.contrib.rnn.GRUCell(num_units=state_size),
            input_keep_prob=keep_prob,
            output_keep_prob=keep_prob,
            variational_recurrent=True,
            dtype=tf.float32,
            input_size=input_size),
            'gru_cell')


class TreeEmbedding_mapGRU(TreeEmbedding_map):
    def __init__(self, name, **kwargs):
        super(TreeEmbedding_mapGRU, self).__init__(name='mapGRU_' + name, **kwargs)
        self._rnn_cell = gru_cell(self.scope, self.state_size, self.keep_prob, self.head_size)

    @property
    def map(self):
        return self._rnn_cell >> td.GetItem(1)


class TreeEmbedding_reduceGRU(TreeEmbedding_reduce):
    def __init__(self, name, **kwargs):
        super(TreeEmbedding_reduceGRU, self).__init__(name='reduceGRU_' + name, **kwargs)
        self._rnn_cell = gru_cell(self.scope, self.state_size, self.keep_prob, self.head_size)

    @property
    def reduce(self):
        return td.AllOf(td.GetItem(0), td.GetItem(1) >> td.Pipe(td.RNN(self._rnn_cell), td.GetItem(1)))

    @property
    def output_size(self):
        return self.root_fc_size or self.state_size


def lstm_cell(scope, state_size, keep_prob, input_size, state_is_tuple=True):
    with tf.variable_scope(scope):
        return td.ScopedLayer(
            tf.contrib.rnn.DropoutWrapper(
                tf.contrib.rnn.BasicLSTMCell(num_units=state_size, forget_bias=2.5, state_is_tuple=state_is_tuple),  # , state_is_tuple=False),
                input_keep_prob=keep_prob,
                output_keep_prob=keep_prob,
                variational_recurrent=True,
                dtype=tf.float32,
                input_size=input_size),
            'lstm_cell')


class TreeEmbedding_mapLSTM(TreeEmbedding_map):
    def __init__(self, name, **kwargs):
        super(TreeEmbedding_mapLSTM, self).__init__(name='mapLSTM_' + name, **kwargs)
        self._rnn_cell = lstm_cell(self.scope, self.state_size, self.keep_prob, self.head_size, state_is_tuple=False)

    @property
    def map(self):
        #return td.AllOf(td.GetItem(0), td.GetItem(1) # requires: state_is_tuple=True
        #                >> td.Function(lambda v: tf.split(value=v, num_or_size_splits=2, axis=1))) \
        #       >> self._rnn_cell >> td.GetItem(1) >> td.Concat()
        return td.AllOf(td.GetItem(0), td.GetItem(1)) >> self._rnn_cell >> td.GetItem(1)  # requires: state_is_tuple=False


class TreeEmbedding_reduceLSTM(TreeEmbedding_reduce):
    def __init__(self, name, **kwargs):
        super(TreeEmbedding_reduceLSTM, self).__init__(name='reduceLSTM_' + name, **kwargs)
        self._rnn_cell = lstm_cell(self.scope, self.state_size, self.keep_prob, self.head_size)

    # TODO: use both states as output: return td.Pipe(td.RNN(self._lstm_cell), td.GetItem(1), td.Concat)
    @property
    def reduce(self):
        #return td.Pipe(td.RNN(self._lstm_cell), td.GetItem(1))
        _reduce = td.Pipe(td.RNN(self._rnn_cell), td.GetItem(1), td.GetItem(0))
        return td.AllOf(td.GetItem(0), td.GetItem(1) >> _reduce)

    @property
    def output_size(self):
        return self.root_fc_size or self.state_size


class TreeEmbedding_mapFC(TreeEmbedding_map):
    def __init__(self, name, **kwargs):
        super(TreeEmbedding_mapFC, self).__init__(name='mapFC_' + name, **kwargs)
        self._fc = td.FC(self.state_size, activation=tf.nn.tanh, input_keep_prob=self.keep_prob, name='fc_cell')

    @property
    def map(self):
        return td.Concat() >> self._fc


class TreeEmbedding_reduceSUM(TreeEmbedding_reduce):
    """Calculates an embedding over a (recursive) SequenceNode."""

    def __init__(self, name, **kwargs):
        super(TreeEmbedding_reduceSUM, self).__init__(name='reduceSUM_' + name, **kwargs)

    @property
    def reduce(self):
        return td.AllOf(td.GetItem(0), td.GetItem(1) >> td.Sum())


class TreeEmbedding_reduceAVG(TreeEmbedding_reduce):
    """Calculates an embedding over a (recursive) SequenceNode."""

    def __init__(self, name, **kwargs):
        super(TreeEmbedding_reduceAVG, self).__init__(name='reduceAVG_' + name, **kwargs)

    @property
    def reduce(self):
        return td.AllOf(td.GetItem(0), td.GetItem(1) >> td.Mean())


class TreeEmbedding_reduceMAX(TreeEmbedding_reduce):
    """Calculates an embedding over a (recursive) SequenceNode."""

    def __init__(self, name, **kwargs):
        super(TreeEmbedding_reduceMAX, self).__init__(name='reduceMAX_' + name, **kwargs)

    @property
    def reduce(self):
        return td.AllOf(td.GetItem(0), td.GetItem(1) >> td.Max())


class TreeEmbedding_reduceATT(TreeEmbedding_reduce):
    """Calculates an embedding over a (recursive) SequenceNode."""

    def __init__(self, name, **kwargs):
        super(TreeEmbedding_reduceATT, self).__init__(name='ATT_' + name, **kwargs)
        #self._fc_map = td.FC(self.head_size, activation=tf.nn.tanh, input_keep_prob=self.keep_prob, name='fc_rnn')
        self._fc_att = td.FC(self.state_size, activation=tf.nn.tanh, input_keep_prob=self.keep_prob, name='fc_att')

    @property
    def reduce(self):
        # head_in, children_sequence --> head_out, children_reduced
        #_fc_rnn = td.FC(self.head_size, activation=tf.nn.tanh, input_keep_prob=self.keep_prob, name='fc_rnn')
        #_fc_att = td.FC(self.state_size, activation=tf.nn.tanh, input_keep_prob=self.keep_prob, name='fc_att')
        #return td.AllOf(td.GetItem(0) >> self._fc_map, td.AllOf(td.GetItem(1), td.GetItem(0) >> self._fc_att) >> AttentionReduce())
        return td.AllOf(td.GetItem(0), td.AllOf(td.GetItem(1), td.GetItem(0) >> self._fc_att) >> AttentionReduce())


class TreeEmbedding_reduceATTsplit(TreeEmbedding_reduce):
    """Calculates an embedding over a (recursive) SequenceNode."""

    def __init__(self, name, **kwargs):
        super(TreeEmbedding_reduceATTsplit, self).__init__(name='ATT_split_' + name, **kwargs)
        # head_in, children_sequence --> head_out, children_reduced
        self._c = td.Composition()
        with self._c.scope():
            head_split = td.Function(lambda v: tf.split(value=v, num_or_size_splits=2, axis=1)).reads(self._c.input[0])
            head_att = td.Pipe(td.FC(self.state_size, activation=tf.nn.tanh, input_keep_prob=self.keep_prob,
                                     name='fc_attention'),
                               name='att_pipe').reads(head_split[0])
            head_rnn = td.Pipe(td.FC(self.head_size, activation=tf.nn.tanh,
                                     input_keep_prob=self.keep_prob, name='fc_rnn'),
                               name='rnn_pipe').reads(head_split[1])
            children_attention = AttentionReduce().reads(self._c.input[1], head_att)
            self._c.output.reads(head_rnn, children_attention)

    @property
    def reduce(self):
        return self._c


class TreeEmbedding_reduceATTsingle(TreeEmbedding_reduce):
    """Calculates an embedding over a (recursive) SequenceNode."""

    def __init__(self, name, **kwargs):
        super(TreeEmbedding_reduceATTsingle, self).__init__(name='ATT_single_' + name, **kwargs)
        self._att_weights = tf.Variable(tf.truncated_normal([self.state_size], stddev=1.0 / math.sqrt(float(self.state_size))),
                                        name='att_weights')
        self._att = AttentionReduce()

    @property
    def reduce(self):
        # head_in, children_sequence --> head_out, children_reduced
        return td.AllOf(td.GetItem(0), td.AllOf(td.GetItem(1), td.Void() >> td.FromTensor(self._att_weights)) >> self._att)


class TreeEmbedding_HTU(TreeEmbedding_reduce, TreeEmbedding_map):
    """Calculates an embedding over a (recursive) SequenceNode."""

    def __init__(self, name, **kwargs):
        super(TreeEmbedding_HTU, self).__init__(name='HTU_' + name, **kwargs)

    def new_state(self, head, children):
        return td.AllOf(head, children) >> self.reduce >> self.map

    def __call__(self):
        embed_tree = td.ForwardDeclaration(input_type=td.PyObjectType(), output_type=self.map.output_type)
        children = self.children() >> td.Map(embed_tree())
        #state = td.AllOf(self.head(), children) >> self.reduce >> self.map
        state = self.new_state(self.head(), children)
        embed_tree.resolve_to(state)
        model = state >> self.root_fc
        return model


class TreeEmbedding_HTUrev(TreeEmbedding_HTU):
    """Calculates an embedding over a (recursive) SequenceNode."""

    def __init__(self, name, **kwargs):
        super(TreeEmbedding_HTUrev, self).__init__(name='rev_' + name, **kwargs)

    def new_state(self, head, children):
        children_mapped = td.AllOf(head >> td.Broadcast(), children) >> td.Zip() >> td.Map(self.map)
        return td.AllOf(self.head(), children_mapped) >> self.reduce >> td.GetItem(1)


class TreeEmbedding_HTUdep(TreeEmbedding_reduce, TreeEmbedding_map):
    """Calculates an embedding over a (recursive) SequenceNode."""

    def __init__(self, name,  state_size, dimension_embeddings, leaf_fc_size=0, **kwargs):
        assert leaf_fc_size > 0 or state_size == dimension_embeddings, \
            'state_size==%i has to equal dimension_embeddings==%i or if leaf_fc_size==0' \
            % (state_size, dimension_embeddings)
        assert leaf_fc_size == 0 or state_size == leaf_fc_size, \
            'state_size==%i has to equal leaf_fc_size==%i if leaf_fc_size > 0' \
            % (state_size, leaf_fc_size)
        super(TreeEmbedding_HTUdep, self).__init__(name='HTUdep_' + name,
                                                   state_size=state_size,
                                                   dimension_embeddings=dimension_embeddings,
                                                   leaf_fc_size=leaf_fc_size,
                                                   **kwargs)

    def __call__(self):
        embed_tree = td.ForwardDeclaration(input_type=td.PyObjectType(), output_type=self.map.output_type)

        # naive version
        def case(seq_tree):
            # children and head exist: process and aggregate
            if len(seq_tree[KEY_CHILDREN]) > 0 and KEY_HEAD in seq_tree:
                return 0
            # children do not exist (but maybe a head): process (optional) head only
            if len(seq_tree[KEY_CHILDREN]) == 0:
                return 1
            # otherwise (head does not exist): process children only
            return 2

        def children(name='children'):
            return td.Pipe(td.GetItem(KEY_CHILDREN), td.Map(embed_tree()), td.Sum(), name=name)

        cases = td.OneOf(lambda x: case(x),
                         {0: td.AllOf(self.head(), children()) >> self.map,
                          1: self.head('head_only'),
                          2: children('children_only'),
                          })

        embed_tree.resolve_to(cases)

        model = cases >> self.root_fc
        return model


class TreeEmbedding_HTUBatchedHead(TreeEmbedding_HTU):
    """ Calculates batch_size embeddings given a sequence of children and batch_size heads """

    def __init__(self, name, tree_count, root_fc_sizes=0, **kwargs):
        assert root_fc_sizes == 0, 'no root_fc allowed for HTUBatchedHead'
        super(TreeEmbedding_HTUBatchedHead, self).__init__(name='BatchedHead_' + name, root_fc_size=root_fc_sizes,
                                                           **kwargs)
        self._batch_size = tree_count

    def __call__(self):
        _htu_model = super(TreeEmbedding_HTUBatchedHead, self).__call__()
        #print('_htu_model: %s' % str(_htu_model.output_type))
        trees_children = td.GetItem(0) >> td.Map(_htu_model)
        #print('trees_children: %s' % str(trees_children.output_type))
        dummy_head = td.Zeros(output_type=_htu_model.output_type)
        reduced_children = td.AllOf(dummy_head, trees_children) >> self.reduce >> td.GetItem(1)
        #print('reduced_children: %s' % str(reduced_children.output_type))
        heads_embedded = td.GetItem(1) >> td.Map(self.embed() >> self.leaf_fc)
        #print('heads_embedded: %s' % str(heads_embedded.output_type))
        batched = td.AllOf(heads_embedded, reduced_children >> td.Broadcast()) >> td.Zip() >> td.Map(self.map)
        #print('batched: %s' % str(batched.output_type))
        return batched


class TreeEmbedding_FLAT(TreeEmbedding_reduce):
    """
        FLAT TreeEmbedding models take all first level children of the root as input and reduce them.
    """
    def __init__(self, name, **kwargs):
        super(TreeEmbedding_FLAT, self).__init__(name='FLAT_' + name, **kwargs)

    def __call__(self):
        model = self.children() >> td.Map(self.head()) >> td.AllOf(td.Void(), td.Identity()) >> self.reduce >> td.GetItem(1) >> self.root_fc
        if model.output_type is None:
            model.set_output_type(tdt.TensorType(shape=(self.output_size,), dtype='float32'))
        return model

    @property
    def output_size(self):
        raise NotImplementedError("Please Implement this method")


class TreeEmbedding_FLAT2levels(TreeEmbedding_FLAT):
    """
        FLAT_2levels TreeEmbedding models take all first level children of the root as input and reduce them.
    """
    def __init__(self, name, **kwargs):
        super(TreeEmbedding_FLAT2levels, self).__init__(name=name, **kwargs)

    def children(self, name='children'):
        # return only children that have at least one child themselves
        def get_children(x):
            if KEY_CHILDREN not in x:
                return []
            res = [c for c in x[KEY_CHILDREN] if KEY_CHILDREN in c and len(c[KEY_CHILDREN]) > 0]
            #if len(res) != len(x[KEY_CHILDREN]):
                # warn, if children have been removed
                #logging.warning('removed children: %i' % (len(x[KEY_CHILDREN]) - len(res)))
            return res
        return td.InputTransform(get_children, name=name)

    def head(self, name='head_embed'):
        # use word embedding and first child embedding
        return td.Pipe(td.AllOf(td.Pipe(td.GetItem(KEY_HEAD), self.embed(), name='head_level1'),
                                td.GetItem(KEY_CHILDREN) >> td.InputTransform(lambda s: s[0])
                                >> td.Pipe(td.GetItem(KEY_HEAD), self.embed(), name='head_level2')),
                       td.Concat(), self.leaf_fc, name=name)

    @property
    def head_size(self):
        return self.leaf_fc_size or (self.dimension_embeddings * 2)


#######################################
#  Instantiable TreeEmbedding models  #
#######################################


class TreeEmbedding_HTU_reduceSUM_mapGRU(TreeEmbedding_reduceSUM, TreeEmbedding_mapGRU, TreeEmbedding_HTU):
    def __init__(self, name='', **kwargs):
        super(TreeEmbedding_HTU_reduceSUM_mapGRU, self).__init__(name=name, **kwargs)


class TreeEmbedding_HTU_reduceSUM_mapLSTM(TreeEmbedding_reduceSUM, TreeEmbedding_mapLSTM, TreeEmbedding_HTU):
    def __init__(self, name='', **kwargs):
        super(TreeEmbedding_HTU_reduceSUM_mapLSTM, self).__init__(name=name, **kwargs)


class TreeEmbedding_HTU_reduceSUM_mapFC(TreeEmbedding_reduceSUM, TreeEmbedding_mapFC, TreeEmbedding_HTU):
    def __init__(self, name='', **kwargs):
        super(TreeEmbedding_HTU_reduceSUM_mapFC, self).__init__(name=name, **kwargs)


class TreeEmbedding_HTUrev_reduceSUM_mapGRU(TreeEmbedding_mapGRU, TreeEmbedding_reduceSUM, TreeEmbedding_HTUrev):
    def __init__(self, name='', **kwargs):
        super(TreeEmbedding_HTUrev_reduceSUM_mapGRU, self).__init__(name=name, **kwargs)


class TreeEmbedding_HTUdep_mapGRU(TreeEmbedding_HTUdep, TreeEmbedding_mapGRU):
    def __init__(self, name='', **kwargs):
        super(TreeEmbedding_HTUdep_mapGRU, self).__init__(name=name, **kwargs)


class TreeEmbedding_HTU_reduceATT_mapGRU(TreeEmbedding_reduceATT, TreeEmbedding_mapGRU, TreeEmbedding_HTU):
    def __init__(self, name='', **kwargs):
        super(TreeEmbedding_HTU_reduceATT_mapGRU, self).__init__(name=name, **kwargs)


class TreeEmbedding_HTU_reduceATTsplit_mapGRU(TreeEmbedding_reduceATTsplit, TreeEmbedding_mapGRU, TreeEmbedding_HTU):
    def __init__(self, name='', **kwargs):
        super(TreeEmbedding_HTU_reduceATTsplit_mapGRU, self).__init__(name=name, **kwargs)


class TreeEmbedding_HTU_reduceATTsingle_mapGRU(TreeEmbedding_reduceATTsingle, TreeEmbedding_mapGRU, TreeEmbedding_HTU):
    def __init__(self, name='', **kwargs):
        super(TreeEmbedding_HTU_reduceATTsingle_mapGRU, self).__init__(name=name, **kwargs)


#TODO: check this!
#class TreeEmbedding_HTU_reduceGRU_mapIDENTITY(TreeEmbedding_reduceGRU, TreeEmbedding_mapIDENTITY, TreeEmbedding_HTU):
#    def __init__(self, name='', **kwargs):
#        super(TreeEmbedding_HTU_reduceGRU_mapIDENTITY, self).__init__(name=name, **kwargs)


class TreeEmbedding_FLAT_AVG(TreeEmbedding_reduceAVG, TreeEmbedding_FLAT):
    def __init__(self, name='', **kwargs):
        super(TreeEmbedding_FLAT_AVG, self).__init__(name=name, **kwargs)

    @property
    def output_size(self):
        return self.root_fc_size or self.head_size


class TreeEmbedding_FLAT_AVG_2levels(TreeEmbedding_FLAT_AVG, TreeEmbedding_FLAT2levels):
    def __init__(self, name='', **kwargs):
        super(TreeEmbedding_FLAT_AVG_2levels, self).__init__(name=name, **kwargs)


class TreeEmbedding_FLAT_SUM(TreeEmbedding_reduceSUM, TreeEmbedding_FLAT):
    def __init__(self, name='', **kwargs):
        super(TreeEmbedding_FLAT_SUM, self).__init__(name=name, **kwargs)

    @property
    def output_size(self):
        return self.root_fc_size or self.head_size


class TreeEmbedding_FLAT_SUM_2levels(TreeEmbedding_FLAT_SUM, TreeEmbedding_FLAT2levels):
    def __init__(self, name='', **kwargs):
        super(TreeEmbedding_FLAT_SUM_2levels, self).__init__(name=name, **kwargs)


class TreeEmbedding_FLAT_LSTM(TreeEmbedding_reduceLSTM, TreeEmbedding_FLAT):
    def __init__(self, name='', **kwargs):
        super(TreeEmbedding_FLAT_LSTM, self).__init__(name=name, **kwargs)


class TreeEmbedding_FLAT_LSTM_2levels(TreeEmbedding_FLAT_LSTM, TreeEmbedding_FLAT2levels):
    def __init__(self, name='', **kwargs):
        super(TreeEmbedding_FLAT_LSTM, self).__init__(name=name, **kwargs)


class TreeEmbedding_FLAT_reduceGRU(TreeEmbedding_reduceGRU, TreeEmbedding_FLAT):
    def __init__(self, name='', **kwargs):
        super(TreeEmbedding_FLAT_reduceGRU, self).__init__(name=name, **kwargs)


class TreeEmbedding_FLAT2levels_reduceGRU(TreeEmbedding_FLAT_reduceGRU, TreeEmbedding_FLAT2levels):
    def __init__(self, name='', **kwargs):
        super(TreeEmbedding_FLAT2levels_reduceGRU, self).__init__(name=name, **kwargs)


class TreeEmbedding_HTUBatchedHead_reduceSUM_mapGRU(TreeEmbedding_reduceSUM, TreeEmbedding_mapGRU, TreeEmbedding_HTUBatchedHead):
    def __init__(self, name='', **kwargs):
        super(TreeEmbedding_HTUBatchedHead_reduceSUM_mapGRU, self).__init__(name=name, **kwargs)

########################################################################################################################


def sim_cosine_DEP(e1, e2):
    e1 = tf.nn.l2_normalize(e1, dim=1)
    e2 = tf.nn.l2_normalize(e2, dim=1)
    return tf.reduce_sum(e1 * e2, axis=1)


def sim_cosine(embeddings):
    """
    :param embeddings: paired embeddings with shape = [batch_count, 2, embedding_dimension]
    :return: clipped similarity scores with shape = [batch_count]
    """
    es_norm = tf.nn.l2_normalize(embeddings, dim=-1)
    cos = tf.reduce_sum(es_norm[:, 0, :] * es_norm[:, 1, :], axis=-1)
    return tf.clip_by_value(cos, 0.0, 1.0)


def sim_manhattan_DEP(e1, e2):
    abs_ = tf.abs(e1 - e2)
    sum_ = tf.reduce_sum(abs_, axis=1)
    return tf.exp(-sum_)


def sim_layer_DEP(e1, e2, hidden_size=DIMENSION_SIM_MEASURE):
    with tf.variable_scope(VAR_PREFIX_SIM_MEASURE + '_layer'):
        embeddings_dif = tf.abs(e1 - e2)
        embeddings_product = e1 * e2
        concat = tf.concat([embeddings_dif, embeddings_product], axis=1)
        h_s = tf.contrib.layers.fully_connected(concat, hidden_size, activation_fn=tf.nn.sigmoid)
        s = tf.contrib.layers.fully_connected(h_s, 1, activation_fn=tf.nn.sigmoid)
    return tf.squeeze(s, axis=[1])


def get_all_heads(tree):
    current_heads = [tree[KEY_HEAD]]
    for child in tree[KEY_CHILDREN]:
        current_heads.extend(get_all_heads(child))
    return current_heads


def get_jaccard_sim(tree_tuple):
    heads1 = set(get_all_heads(tree_tuple['first']))
    heads2 = set(get_all_heads(tree_tuple['second']))
    return len(heads1 & heads2) / float(len(heads1 | heads2))


class SequenceTreeModel(object):
    def __init__(self, tree_embedder=TreeEmbedding_TREE_LSTM, keep_prob=1.0, tree_count=2, discrete_values_gold=False,
                 **kwargs):
        if discrete_values_gold:
            self._values_gold_dtype = 'int32'
        else:
            self._values_gold_dtype = 'float'

        self._tree_count = tree_count
        self._keep_prob = tf.placeholder_with_default(keep_prob, shape=())

        self._tree_embed = tree_embedder(keep_prob_placeholder=self._keep_prob, tree_count=tree_count, **kwargs)

        embed_tree = self._tree_embed()
        # do not map tree embedding model to input, if it produces a sequence itself
        if isinstance(embed_tree.output_type, tdt.SequenceType):
            model = td.AllOf(td.GetItem(0) >> embed_tree
                             >> SequenceToTuple(embed_tree.output_type.element_type, self._tree_count) >> td.Concat(),
                             td.GetItem(1) >> td.Vector(self._tree_count, dtype=self._values_gold_dtype))
        else:
            model = td.AllOf(td.GetItem(0) >> td.Map(embed_tree)
                             >> SequenceToTuple(embed_tree.output_type, self._tree_count) >> td.Concat(),
                             td.GetItem(1) >> td.Vector(self._tree_count, dtype=self._values_gold_dtype))

        # fold model output
        self._compiler = td.Compiler.create(model)
        (self._tree_embeddings_all, self._probs_gold) = self._compiler.output_tensors

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

    @property
    def embeddings_shaped(self):
        return tf.reshape(self._tree_embeddings_all, shape=[-1, self.tree_output_size])

    @property
    def values_gold(self):
        return self._probs_gold

    @property
    def values_gold_shaped(self):
        return tf.reshape(self._probs_gold, shape=[-1])

    @property
    def values_gold_dtype(self):
        return self._values_gold_dtype

    @property
    def compiler(self):
        return self._compiler

    @property
    def tree_count(self):
        return self._tree_count

    #@property
    #def prob_count(self):
    #    return self._prob_count

    @property
    def tree_output_size(self):
        return self._tree_embeddings_all.get_shape().as_list()[1] // self.tree_count


class BaseTrainModel(object):
    def __init__(self, tree_model, loss, optimizer=None, learning_rate=0.1, clipping_threshold=5.0):

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
            gradients, _ = tf.clip_by_global_norm(gradients, clipping_threshold)
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

    def __init__(self, tree_model, sim_measure=sim_cosine, **kwargs):

        # unpack scores_gold. Every prob tuple has the format: [1.0, score_gold, ...]
        self._scores_gold = tf.reshape(tree_model.values_gold_shaped, shape=[-1, 2])[:, 1]
        # pack tree embeddings in pairs of two
        self._tree_embeddings_reshaped = tf.reshape(tree_model.embeddings_shaped, shape=[-1, 2, tree_model.tree_output_size])
        # apply sim measure
        self._scores = sim_measure(self._tree_embeddings_reshaped)

        BaseTrainModel.__init__(self, tree_model=tree_model,
                                loss=tf.reduce_mean(tf.square(self._scores - self._scores_gold)), **kwargs)

    @property
    def values_gold(self):
        return self._scores_gold

    @property
    def values_predicted(self):
        return self._scores

    @property
    def model_type(self):
        return MODEL_TYPE_REGRESSION


class SimilaritySequenceTreeTupleModel_sample(BaseTrainModel):
    """A Fold model for similarity scored sequence tree (SequenceNode) tuple."""

    def __init__(self, tree_model, **kwargs):

        # unpack scores_gold. Every prob tuple has the format: [1.0, ...]
        #self._scores_gold = tf.reshape(tree_model.probs_gold_shaped, shape=[-1, 2])[:, 0]

        # pack tree embeddings in pairs of two
        tree_embeddings = tf.reshape(tree_model.embeddings_shaped, shape=[-1, 2, tree_model.tree_output_size])

        batch_size = tf.shape(tree_embeddings)[0]

        self._labels_gold = tf.reshape(tf.eye(batch_size, dtype=np.int32), [batch_size ** 2])
        embeddings_0_tiled = tf.tile(tree_embeddings[:, 0, :], multiples=[batch_size, 1])
        embeddings_1_tiled = tf.tile(tree_embeddings[:, 1, :], multiples=[batch_size, 1])

        embeddings_0_m = tf.reshape(embeddings_0_tiled, shape=[batch_size, batch_size, tree_model.tree_output_size])
        embeddings_1_m = tf.reshape(embeddings_1_tiled, shape=[batch_size, batch_size, tree_model.tree_output_size])
        embeddings_1_m_trans = tf.transpose(embeddings_1_m, perm=[1, 0, 2])

        embeddings_0_list = tf.reshape(embeddings_0_m, shape=[batch_size**2, tree_model.tree_output_size])
        embeddings_1_list_trans = tf.reshape(embeddings_1_m_trans, shape=[batch_size ** 2, tree_model.tree_output_size])
        #tree_embeddings_tiled_stacked = tf.reshape(tf.stack([embeddings_0_list, embeddings_1_list_trans]),
        #                                           shape=[batch_size ** 2, 2, tree_model.tree_output_size])
        stacked = tf.stack([embeddings_0_list, embeddings_1_list_trans], axis=1)
        tree_embeddings_tiled_stacked = tf.reshape(stacked, shape=[batch_size ** 2, tree_model.tree_output_size * 2])

        # apply sim measure
        #self._scores = sim_measure(tree_embeddings_tiled_stacked)
        fc = tf.contrib.layers.fully_connected(inputs=tree_embeddings_tiled_stacked, num_outputs=1000)
        logits = tf.contrib.layers.fully_connected(inputs=fc, num_outputs=2, activation_fn=None)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self._labels_gold, logits=logits)

        #BaseTrainModel.__init__(self, tree_model=tree_model,
        #                       loss=tf.reduce_mean(tf.square(self._scores - self._scores_gold)), **kwargs)
        BaseTrainModel.__init__(self, tree_model=tree_model,
                                loss=tf.reduce_mean(cross_entropy), **kwargs)

        softmax = tf.nn.softmax(logits)
        self._probs = softmax[:, 1]

    @property
    def values_gold(self):
        return self._labels_gold

    @property
    def values_predicted(self):
        return self._probs

    @property
    def model_type(self):
        return MODEL_TYPE_DISCRETE


class ScoredSequenceTreeTupleModel(BaseTrainModel):
    """A Fold model for similarity scored sequence tree (SequenceNode) tuple."""

    def __init__(self, tree_model, probs_count=2, **kwargs):

        self._prediction_logits = tf.contrib.layers.fully_connected(tree_model.embeddings_all, probs_count,
                                                                    activation_fn=None, scope=DEFAULT_SCOPE_SCORING)
        loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tree_model.values_gold, logits=self._prediction_logits))

        BaseTrainModel.__init__(self, tree_model=tree_model, loss=loss, **kwargs)


# DEPRECATED
class ScoredSequenceTreeTupleModel_independent(BaseTrainModel):
    """A Fold model for similarity scored sequence tree (SequenceNode) tuple."""

    def __init__(self, tree_model, count=None, **kwargs):
        if count is None:
            count = tree_model.tree_count
        #assert tree_model.prob_count >= count, 'tree_model produces %i prob values per batch entry, but count=%i ' \
        #                                        'requested' % (tree_model.prob_count, count)
        assert tree_model.tree_count >= count, 'tree_model produces %i tree embeddings per batch entry, but count=%i ' \
                                               'requested' % (tree_model.tree_count, count)
        # cut inputs to 'count'
        #probs = tree_model.probs_gold[:, :count]
        #trees = tree_model.embeddings_all[:, :count * tree_model.tree_output_size]
        #input_layer = tf.reshape(trees, [-1, count, tree_model.tree_output_size, 1])

        #conv = tf.layers.conv2d(inputs=input_layer, filters=1,
        #                        kernel_size=[1, tree_model.tree_output_size], activation=None,
        #                        name=DEFAULT_SCOPE_SCORING)
        #self._prediction_logits = tf.reshape(conv, shape=[-1, count])


        _weights = tf.Variable(tf.truncated_normal([tree_model.tree_output_size, 1],
                                                   stddev=1.0 / math.sqrt(float(tree_model.tree_output_size))),
                               name='scoring_weights')
        _bias = tf.Variable(tf.truncated_normal([1]), name='scoring_bias')
        _prediction_logits = tf.reshape(tf.matmul(tree_model.embeddings_shaped, _weights) + _bias, shape=[-1])
        #loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=probs, logits=self._prediction_logits))
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tree_model.values_gold_shaped, logits=_prediction_logits))

        BaseTrainModel.__init__(self, tree_model=tree_model, loss=loss, **kwargs)


class SequenceTreeRerootModel(BaseTrainModel):

    def __init__(self, tree_model, **kwargs):

        #batch_size = tf.shape(tree_model.embeddings_shaped)[0]

        # unpack tree embeddings
        tree_embeddings = tf.reshape(tree_model.embeddings_shaped, shape=[-1, tree_model.tree_output_size])

        # unpack (flatten) labels_gold
        self._labels_gold = tf.reshape(tree_model.values_gold, shape=[tf.shape(tree_embeddings)[0]])

        fc = tf.contrib.layers.fully_connected(inputs=tree_embeddings, num_outputs=1000)
        logits = tf.contrib.layers.fully_connected(inputs=fc, num_outputs=2, activation_fn=None)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self._labels_gold, logits=logits)

        BaseTrainModel.__init__(self, tree_model=tree_model, loss=tf.reduce_mean(cross_entropy), **kwargs)

        softmax = tf.nn.softmax(logits)
        self._probs = softmax[:, 1]

    @property
    def values_gold(self):
        return self._labels_gold

    @property
    def values_predicted(self):
        return self._probs

    @property
    def model_type(self):
        return MODEL_TYPE_DISCRETE


class HighestSimsModel:

    def __init__(self, embedding_size, number_of_embeddings):
        #self._normed_reference_embedding = tf.placeholder(tf.float32, [embedding_size])
        self._reference_idx = tf.placeholder(tf.int32, shape=[])
        self._number_of_embeddings = tf.placeholder(tf.int32, shape=[])
        #self._normed_embeddings = tf.placeholder(tf.float32, [None, embedding_size])
        self._normed_embeddings = tf.Variable(tf.constant(0.0, shape=[number_of_embeddings, embedding_size]),
                                              trainable=False, name='EMBEDDED_DOCS')

        self._normed_embeddings_placeholder = tf.placeholder(tf.float32, shape=[number_of_embeddings, embedding_size])
        self._normed_embeddings_init = self._normed_embeddings.assign(self._normed_embeddings_placeholder)

        _normed_reference_embedding = tf.gather(self._normed_embeddings, indices=self._reference_idx)

        _batch_size = tf.shape(self._normed_embeddings)[0]
        _reference_embedding_tiled = tf.tile(tf.reshape(_normed_reference_embedding, shape=(1, embedding_size)),
                                             multiples=[_batch_size, 1])

        self._sims = tf.reduce_sum(_reference_embedding_tiled * self._normed_embeddings, axis=-1)

    #@property
    #def normed_reference_embedding(self):
    #    return self._normed_reference_embedding

    @property
    def reference_idx(self):
        return self._reference_idx

    @property
    def normed_embeddings(self):
        return self._normed_embeddings

    @property
    def normed_embeddings_placeholder(self):
        return self._normed_embeddings_placeholder

    @property
    def normed_embeddings_init(self):
        return self._normed_embeddings_init

    #@property
    #def number_of_embeddings(self):
    #    return self._number_of_embeddings

    @property
    def sims(self):
        return self._sims
