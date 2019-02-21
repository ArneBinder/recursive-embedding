#from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
# import google3
import tensorflow as tf
import tensorflow_fold.public.blocks as td
from tensorflow_fold.blocks import result_types as tdt
from tensorflow_fold.blocks import blocks as tdb
import numpy as np
import math
from scipy.sparse import csr_matrix

from constants import KEY_HEAD, KEY_CHILDREN, KEY_CANDIDATES, LOGGING_FORMAT, KEY_HEAD_CONCAT

# DEFAULT_SCOPE_TREE_EMBEDDER = 'tree_embedder'   # DEPRECATED
DEFAULT_SCOPE_SCORING = 'scoring'
# DIMENSION_EMBEDDINGS = 300     use lexicon.dimension_embeddings
DIMENSION_SIM_MEASURE = 300
VAR_NAME_LEXICON_VAR = 'embeddings'
VAR_NAME_LEXICON_FIX = 'embeddings_fix'
VAR_NAME_GLOBAL_STEP = 'global_step'
VAR_PREFIX_FC_LEAF = 'FC_leaf'
VAR_PREFIX_FC_PLAIN_LEAF = 'FC_plain_leaf'
VAR_PREFIX_FC_ROOT = 'FC_root'
VAR_PREFIX_FC_REVERSE = 'FC_reverse'
VAR_PREFIX_TREE_EMBEDDING = 'TreeEmbedding'
VAR_PREFIX_SIM_MEASURE = 'sim_measure'

MODEL_TYPE_DISCRETE = 'mt_discrete'
MODEL_TYPE_REGRESSION = 'mt_regression'

logger = logging.getLogger('model_fold')
logger.setLevel(logging.DEBUG)
logger_streamhandler = logging.StreamHandler()
logger_streamhandler.setLevel(logging.DEBUG)
logger_streamhandler.setFormatter(logging.Formatter(LOGGING_FORMAT))


def dprint(x, message=None):
    r = tf.Print(x, [tf.shape(x)], message=message)
    return r


def block_info(block):
    print("%s: %s -> %s" % (block, block.input_type, block.output_type))


def convert_sparse_matrix_to_sparse_tensor(X):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensorValue(indices, coo.data, coo.shape)


def convert_sparse_tensor_to_sparse_matrix(X):
    indices = X.indices.T
    return csr_matrix((X.values, (indices[0], indices[1])), shape=X.dense_shape)


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
        fc = tf.contrib.layers.fully_connected(inputs, num_units, activation_fn=activation_fn, scope=scope)
        if keep_prob is not None:
            fc = tf.nn.dropout(fc, keep_prob)
        return fc

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
    return c


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
        bias = tf.Variable(np.ones(shape=(), dtype='float32'))
        att_scalars = td.Map(td.Function(lambda x, y: tf.reduce_sum(x * y, axis=1) + bias)).reads(td.Zip().reads(c.input[0], td.Broadcast().reads(c.input[1])))
        att_weights = Softmax().reads(att_scalars)
        wheighted = td.Map(td.Function(tdb._tf_batch_scalar_mul)).reads(td.Zip().reads(att_weights, c.input[0]))
        res = td.Sum().reads(wheighted)
        c.output.reads(res)
    return c.set_constructor_name('td.Attention')


###################################
#  Abstract TreeEmbedding models  #
###################################


class TreeEmbedding(object):
    def __init__(self, name, lex_size_fix, lex_size_var, dimension_embeddings, keep_prob_placeholder=None,
                 state_size=None, leaf_fc_size=0,
                 keep_prob_fixed=1.0, additional_heads_dims=(), **unused):
        self._lex_size_fix = lex_size_fix
        self._lex_size_var = lex_size_var
        self._dim_embeddings = dimension_embeddings
        #self._additional_heads = additional_heads
        self._additional_heads_dims = additional_heads_dims

        self._lexicon_var, self._lexicon_var_placeholder, self._lexicon_var_init = create_lexicon(lex_size=self._lex_size_var,
                                                                                                  dimension_embeddings=self._dim_embeddings,
                                                                                                  trainable=True)
        self._lexicon_fix, self._lexicon_fix_placeholder, self._lexicon_fix_init = create_lexicon(lex_size=self._lex_size_fix,
                                                                                                  dimension_embeddings=self._dim_embeddings,
                                                                                                  trainable=False)
        self._lexicon = tf.concat((self._lexicon_var, self._lexicon_fix), axis=0)

        self._keep_prob_fixed = keep_prob_fixed
        if keep_prob_placeholder is not None:
            self._keep_prob = keep_prob_placeholder
        else:
            self._keep_prob = 1.0
        if state_size:
            if type(state_size) == unicode or type(state_size) == str:
                _parts = state_size.split(',')
                self._state_sizes = [int(s.strip()) for s in _parts if int(s.strip()) != 0]
            else:
                self._state_sizes = [state_size]
        else:
            self._state_sizes = [self._dim_embeddings]

        self._name = VAR_PREFIX_TREE_EMBEDDING + '/' + name  # + '_%d' % self._state_size

        self._leaf_fc_size = leaf_fc_size
        # handled in TreeEmbedding_HTU_plain_leaf directly (and added only for leafs, not for inner nodes!)
        #if isinstance(self, TreeEmbedding_HTU_plain_leaf):
        #    self._leaf_fc_size = self.state_size
        #    logger.debug('set leaf_fc_size to state_size=%i because model is instance of HTU_plain_leaf'
        #                 % self._leaf_fc_size)
        with tf.variable_scope(self.name) as scope:
            self._scope = scope
            if self._leaf_fc_size:
                self._leaf_fc = fc_scoped(num_units=leaf_fc_size,
                                          activation_fn=tf.nn.tanh, scope=scope, keep_prob=self.keep_prob,
                                          name=VAR_PREFIX_FC_LEAF + '_%d' % leaf_fc_size)
            else:
                self._leaf_fc = td.Identity()
            # NOT USED ANYMORE, kept for compatibility (loading old models)
            self._reverse_fc = fc_scoped(num_units=self._dim_embeddings,
                                         activation_fn=tf.nn.tanh, scope=scope, keep_prob=self.keep_prob,
                                         name=VAR_PREFIX_FC_REVERSE + '_%d' % self._dim_embeddings)

        # implemented for batches
        self._reference_indices = tf.placeholder(dtype=tf.int32)
        self._candidate_indices = tf.placeholder(dtype=tf.int32)
        self._reference_vs_candidate = self.calc_reference_vs_candidate(reference_indices=self.reference_indices,
                                                                        candidate_indices=self.candidate_indices)

    # used only in HTUBatchedHead and FLAT2levels
    def embed(self, max_dims=None):
        # get the head embedding from id
        #res = td.OneOf(key_fn=(lambda x: x // self.lexicon_size),
        #                case_blocks={
        #                    # normal embedding
        #                    0: td.Scalar(dtype='int32')
        #                       >> td.Function(lambda x: tf.gather(self._lexicon, tf.mod(x, self.lexicon_size))),
        #                    # "reverted" edge embedding
        #                    1: td.Scalar(dtype='int32')
        #                       >> td.Function(lambda x: tf.gather(self._lexicon, tf.mod(x, self.lexicon_size)))
        #                       >> self._reverse_fc,
        #                })
        res = td.Scalar(dtype='int32') >> td.Function(lambda x: tf.gather(self._lexicon, x))
        if max_dims is not None:
            res = td.Pipe(res, td.Function(lambda x: x[:, :max_dims]))
            res.set_output_type(tdt.TensorType(shape=[max_dims], dtype='float32'))
        return res

    def embed_w_direction(self):
        # translate id into head embedding and direction (0 / 1)
        return td.AllOf(td.Scalar(dtype='int32')
                               >> td.Function(lambda x: tf.gather(self._lexicon, tf.mod(x, self.lexicon_size))),
                        td.InputTransform(lambda x: x // self.lexicon_size))

    def head(self, name='head_embed'):
        raise NotImplementedError('head() is deprecated, because it does not consider additional_heads. use head_w_direction() instead')
        return td.Pipe(td.GetItem(KEY_HEAD), self.embed(), self.leaf_fc, name=name)

    def head_w_direction(self, name='head_embed_w_direction'):
        #return td.Pipe(td.GetItem(KEY_HEAD), self.embed_w_direction(), name=name)
        comp = td.Composition(name=name)
        with comp.scope():
            head = td.GetItem(KEY_HEAD).reads(comp.input)
            head_embedded_w_direction = self.embed_w_direction().reads(head)
            head_embedded = td.GetItem(0).reads(head_embedded_w_direction)
            direction = td.GetItem(1).reads(head_embedded_w_direction)
            nbr_additional_heads = len(self.additional_heads_dims)
            if nbr_additional_heads > 0:
                heads_other = td.InputTransform(lambda x: x[KEY_HEAD_CONCAT]).reads(comp.input)
                heads_other_embedded = [self.embed(max_dims=dims).reads(td.GetItem(i).reads(heads_other)) for i, dims in enumerate(self.additional_heads_dims)]
                head_embedded = td.Concat().reads(head_embedded, *heads_other_embedded)
            comp.output.reads(head_embedded, direction)
        if self.leaf_fc_size > 0:
            return comp >> td.AllOf(td.GetItem(0) >> self.leaf_fc, td.GetItem(1))
        return comp

    def children(self, name=KEY_CHILDREN):
        return td.InputTransform(lambda x: x.get(KEY_CHILDREN, []), name=name)

    def has_children(self):
        return td.InputTransform(lambda x: len(x.get(KEY_CHILDREN, [])) != 0)

    def calc_reference_vs_candidate(self, reference_indices, candidate_indices):
        reference_embedding = tf.nn.l2_normalize(tf.gather(self.lexicon, reference_indices), dim=-1)
        candidate_embedding = tf.nn.l2_normalize(tf.gather(self.lexicon, candidate_indices), dim=-1)
        _shape = [tf.shape(reference_embedding)[0], tf.shape(candidate_embedding)[0], self.dimension_embeddings]
        reference_embedding_tiled = tf.tile(reference_embedding, multiples=[1, tf.shape(candidate_embedding)[0]])
        candidate_embedding_tiled = tf.tile(candidate_embedding, multiples=[tf.shape(reference_embedding)[0], 1])
        reference_embedding_tiled_reshaped = tf.reshape(reference_embedding_tiled, _shape)
        candidate_embedding_tiled_reshaped = tf.reshape(candidate_embedding_tiled, _shape)

        _mul = candidate_embedding_tiled_reshaped * reference_embedding_tiled_reshaped
        return tf.reduce_sum(_mul, axis=-1)

    @property
    def reference_vs_candidate(self):
        return self._reference_vs_candidate

    @property
    def state_size(self):
        return sum(self._state_sizes)

    @property
    def state_sizes(self):
        return self._state_sizes

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
        return self._lex_size_var + self._lex_size_fix

    @property
    def lexicon(self):
        return self._lexicon

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
        return self.leaf_fc_size or (self.dimension_embeddings + sum(self.additional_heads_dims))

    @property
    def output_size(self):
        raise NotImplementedError

    @property
    def reference_indices(self):
        return self._reference_indices

    @property
    def candidate_indices(self):
        return self._candidate_indices

    @property
    def additional_heads_dims(self):
        return self._additional_heads_dims


def get_lexicon_vars():
    lexicon_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=VAR_NAME_LEXICON_VAR) \
                   + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=VAR_NAME_LEXICON_FIX)
    return lexicon_vars


def get_tree_embedder_vars():
    tree_embedder_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=VAR_PREFIX_TREE_EMBEDDING)
    return tree_embedder_vars


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
        return cases >> td.GetItem(1) #>> self.root_fc


class TreeEmbedding_map(TreeEmbedding):
    @property
    def map(self):
        """
        Applies a mapping function.
        :return: A tensorflow fold block that consumes a tuple ((input, direction), state_in) and outputs a new state
                 with same shape as state_in
        as state_in.
        """
        raise NotImplementedError

    @property
    def input_wo_direction(self):
        return td.AllOf(td.GetItem(0) >> td.GetItem(0), td.GetItem(1))


class TreeEmbedding_reduce(TreeEmbedding):
    @property
    def reduce(self):
        """
        gets a tuple (head_in, children_sequence), where head_in is a 1-d vector and children a sequence of 1-d vectors
        (size does not have to fit head_in size) and produces a tuple (children_reduced, head_out) to feed into
        self.map.
        NOTE: head_in and head_out have to be tuples: (head_vector, head_direction), with head_direction being 0 or 1
        :return: a tuple (head_out, children_reduced) of two 1-d vectors, of different size, eventually.
        """
        raise NotImplementedError

    @property
    def reduce_output_size_mapping(self):
        """
        Returns a function mapping from input size of reduce to its output size. It is required for HTUrev models.
        Defaults to identity.
        :return: the mapping function
        """
        return lambda x: x


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


class TreeEmbedding_RNN(TreeEmbedding):
    def __init__(self, name, **kwargs):
        super(TreeEmbedding_RNN, self).__init__(name=name, **kwargs)

    def rnn(self, rnn_cells, name='RNN'):
        if len(rnn_cells) == 1:
            rnn = rnn_cells[0]
        else:
            logger.debug('create RNN with state sizes: %s' % str(self.state_sizes))
            rnn = td.Composition(name=name)
            with rnn.scope():
                head = rnn.input[0]
                states_concat = rnn.input[1]
                _start = 0
                start_dims = [0]
                for i, size in enumerate(self.state_sizes):
                    _start += size
                    start_dims.append(_start)
                states_split = td.Function(lambda s: [s[:, start_dims[i]:start_dims[i+1]] for i in range(len(self.state_sizes))]).reads(states_concat)
                current_head = head
                new_states = []
                for i, cell in enumerate(rnn_cells):
                    # wrap cell into pipe to enable .reads function
                    _cell = td.Pipe(cell).reads(current_head, states_split[i])
                    current_head = td.GetItem(0).reads(_cell)
                    new_states.append(td.GetItem(1).reads(_cell))
                new_state = td.Concat().reads(*new_states)
                rnn.output.reads(current_head, new_state)
        return rnn


class TreeEmbedding_mapGRU(TreeEmbedding_map, TreeEmbedding_RNN):
    def __init__(self, name, **kwargs):
        super(TreeEmbedding_mapGRU, self).__init__(name='mapGRU_' + name, **kwargs)
        input_sizes = [self.head_size] + self.state_sizes[:-1]
        self._rnn_cells = [gru_cell(self.scope, state_size, self.keep_prob, input_sizes[i]) for i, state_size in enumerate(self.state_sizes)]

    @property
    def map(self):
        return self.input_wo_direction >> self.rnn(self._rnn_cells) >> td.GetItem(1)


class TreeEmbedding_mapGRU_w_direction(TreeEmbedding_map, TreeEmbedding_RNN):
    def __init__(self, name, **kwargs):
        super(TreeEmbedding_mapGRU_w_direction, self).__init__(name='mapGRU_w_direction' + name, **kwargs)
        input_sizes = [self.head_size] + self.state_sizes[:-1]
        with tf.variable_scope(self.scope):
            with tf.variable_scope('map/forward') as scope:
                self._rnn_cells_fw = [gru_cell(scope, state_size, self.keep_prob, input_sizes[i]) for i, state_size in enumerate(self.state_sizes)]
            with tf.variable_scope('map/backward') as scope:
                self._rnn_cells_bw = [gru_cell(scope, state_size, self.keep_prob, input_sizes[i]) for i, state_size in enumerate(self.state_sizes)]

    @property
    def map(self):
        # check direction (0 or 1) of head
        return td.OneOf(key_fn=td.GetItem(0) >> td.GetItem(1),
                        case_blocks={
                            0: self.input_wo_direction >> self.rnn(self._rnn_cells_fw, name='GRU_fw'),
                            1: self.input_wo_direction >> self.rnn(self._rnn_cells_bw, name='GRU_bw')
                        }) >> td.GetItem(1)


class TreeEmbedding_mapGRU2(TreeEmbedding_map):
    def __init__(self, name, **kwargs):
        super(TreeEmbedding_mapGRU2, self).__init__(name='mapGRU2_' + name, **kwargs)
        # self._rnn_cell reads (head, state) and returns (output, state)
        self._rnn_cell = gru_cell(self.scope, self.state_size, self.keep_prob, self.head_size)

    @property
    def map(self):
        # temp:  (head, [previous_summed_outputs, previous_state]) -> ((current_output, new_state), summed_outputs)
        temp = td.AllOf(td.Function(lambda head, state: (head, state[:, self.state_size:])) >> self._rnn_cell,
                        td.Function(lambda head, state: state[:, :self.state_size]))
        res = self.input_wo_direction >> temp \
              >> td.AllOf(self.input_wo_direction
                          # sum current_output with previous_summed_outputs
                          >> td.Function(lambda x, y: x + y),
                          # concatenate (summed_outputs, new_state)
                          td.GetItem(0) >> td.GetItem(1)) >> td.Concat()
        res.set_output_type(tdt.TensorType(shape=[self.state_size * 2], dtype='float32'))
        return res


class TreeEmbedding_reduceGRU(TreeEmbedding_reduce, TreeEmbedding_RNN):
    def __init__(self, name, **kwargs):
        super(TreeEmbedding_reduceGRU, self).__init__(name='reduceGRU_' + name, **kwargs)
        input_sizes = [self.head_size] + self.state_sizes[:-1]
        self._rnn_cells = [gru_cell(self.scope, state_size, self.keep_prob, input_sizes[i]) for i, state_size in enumerate(self.state_sizes)]

    @property
    def reduce(self):
        return td.AllOf(td.GetItem(0), td.GetItem(1) >> td.RNN(self.rnn(self._rnn_cells)) >> td.GetItem(1))

    @property
    def output_size(self):
        return self.state_size


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
        #return td.AllOf(td.GetItem(0), td.GetItem(1)) >> self._rnn_cell >> td.GetItem(1)  # requires: state_is_tuple=False
        return self.input_wo_direction >> self._rnn_cell >> td.GetItem(1)  # requires: state_is_tuple=False


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

    # TODO: replace by reduce_output_size??
    #@property
    #def output_size(self):
    #    #return self.root_fc_size or self.state_size
    #    return self.state_size

    # TODO: test this! eventually has to be: 2 * self.state_size
    @property
    def reduce_output_size_mapping(self):
        return lambda _: self.state_size


class TreeEmbedding_mapFC(TreeEmbedding_map):
    def __init__(self, name, **kwargs):
        super(TreeEmbedding_mapFC, self).__init__(name='mapFC_' + name, **kwargs)
        self._fc = td.FC(self.state_size, activation=tf.nn.tanh, input_keep_prob=self.keep_prob, name='fc_cell')

    @property
    def map(self):
        return self.input_wo_direction >> td.Concat() >> self._fc


class TreeEmbedding_mapCCFC(TreeEmbedding_map):
    def __init__(self, name, **kwargs):
        super(TreeEmbedding_mapCCFC, self).__init__(name='mapCCFC_' + name, **kwargs)
        self._fc = td.FC(self.head_size, activation=tf.nn.tanh, input_keep_prob=self.keep_prob, name='fc_cell')

    @property
    def map(self):
        return self.input_wo_direction >> td.Function(circular_correlation) >> self._fc


class TreeEmbedding_mapAVG(TreeEmbedding_map):
    def __init__(self, name, **kwargs):
        super(TreeEmbedding_mapAVG, self).__init__(name='mapAVG_' + name, **kwargs)

    @property
    def map(self):
        _mapped = td.Mean()
        _mapped.set_output_type(tdt.TensorType(shape=[self.head_size], dtype='float32'))
        return self.input_wo_direction >> _mapped


class TreeEmbedding_mapSUM(TreeEmbedding_map):
    def __init__(self, name, **kwargs):
        super(TreeEmbedding_mapSUM, self).__init__(name='mapSUM_' + name, **kwargs)

    @property
    def map(self):
        _mapped = td.Sum()
        _mapped.set_output_type(tdt.TensorType(shape=[self.head_size], dtype='float32'))
        return self.input_wo_direction >> _mapped


class TreeEmbedding_reduceSUM(TreeEmbedding_reduce):
    """Calculates an embedding over a (recursive) SequenceNode."""

    def __init__(self, name, **kwargs):
        super(TreeEmbedding_reduceSUM, self).__init__(name='reduceSUM_' + name, **kwargs)

    @property
    def reduce(self):
        #_reduced = td.AllOf(td.GetItem(0), td.GetItem(1) >> td.Sum())
        #_reduced.set_output_type(tdt.TupleType(td.GetItem(0).output_type, td.GetItem(1).output_type))
        #return _reduced
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
        # TODO: use correct FC size (can be state_size OR head_size)
        #self._fc_att = td.FC(self.state_size, activation=tf.nn.tanh, input_keep_prob=self.keep_prob, name='fc_att')

        # initialize with zeros to start with naive averaging (bias in AttentionReduce is set to 1.0)
        self._fc_att = td.FC(self.head_size, initializer=tf.zeros_initializer(), activation=tf.nn.tanh,
                             input_keep_prob=self.keep_prob, name='fc_att')
        self._att = AttentionReduce()
        #self._bias = tf.get_variable("attention_bias", [self.head_size])

    @property
    def reduce(self):
        # head_in, children_sequence --> head_out, children_reduced
        #return td.AllOf(td.GetItem(0) >> self._fc_map, td.AllOf(td.GetItem(1), td.GetItem(0) >> self._fc_att) >> AttentionReduce())
        return td.AllOf(td.GetItem(0), td.AllOf(td.GetItem(1), td.GetItem(0) >> self._fc_att) >> self._att)

    #@property
    #def reduce_output_size_mapping(self):
    #    return lambda _: self.state_size


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
        #self._att_weights = tf.Variable(tf.truncated_normal([self.head_size], stddev=1.0 / math.sqrt(float(self.head_size))),
        #                                name='att_weights')

        # initialize with zeros to start with naive averaging (bias in AttentionReduce is set to 1.0)
        self._att_weights = tf.Variable(np.zeros(shape=[self.head_size], dtype='float32'))
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
        # discard direction
        return td.AllOf(head(), children) >> self.reduce >> self.map

    def __call__(self):
        embed_tree = td.ForwardDeclaration(input_type=td.PyObjectType(), output_type=self.map.output_type)
        state = self.new_state(head=self.head_w_direction,
                               children=self.children() >> td.Map(embed_tree()))
        embed_tree.resolve_to(state)
        return state

    @property
    def output_size(self):
        # depends on self.new_state
        ot = self.map.output_type
        return ot.shape[-1]


class TreeEmbedding_HTU_plain_leaf(TreeEmbedding_HTU):
    """Calculates an embedding over a (recursive) SequenceNode."""

    def __init__(self, name, **kwargs):
        super(TreeEmbedding_HTU_plain_leaf, self).__init__(name='HTU_' + name, **kwargs)
        #assert self.head_size == self.state_size, 'head_size [%i] has to equal sum of state_sizes [%i] for HTU_plain_leaf' \
        #                                          % (self.head_size, self.state_size)
        with tf.variable_scope(self.name) as scope:
            self._plain_leaf_fc = fc_scoped(num_units=self.state_size,
                                            activation_fn=tf.nn.tanh, scope=scope, keep_prob=self.keep_prob,
                                            name=VAR_PREFIX_FC_PLAIN_LEAF + '_%d' % self.state_size)

    def __call__(self):
        embed_tree = td.ForwardDeclaration(input_type=td.PyObjectType(), output_type=self.map.output_type)
        state = td.OneOf(key_fn=self.has_children(),
                         case_blocks={
                             True: self.new_state(head=self.head_w_direction,
                                                  children=self.children() >> td.Map(embed_tree())),
                             False: self.head_w_direction() >> td.GetItem(0) >> self._plain_leaf_fc
                         })
        embed_tree.resolve_to(state)
        return state


class TreeEmbedding_HTU_init_state(TreeEmbedding_HTU):
    """Calculates an embedding over a (recursive) SequenceNode."""

    def __init__(self, name, **kwargs):
        super(TreeEmbedding_HTU_init_state, self).__init__(name='HTU_init_state_' + name, **kwargs)
        with tf.variable_scope(self.name):
            self._init_state = tf.Variable(tf.constant(0.0, shape=[self.state_size]), trainable=True,
                                           name='initial_state')

    def __call__(self):
        embed_tree = td.ForwardDeclaration(input_type=td.PyObjectType(), output_type=self.map.output_type)
        state = td.OneOf(key_fn=self.has_children(),
                         case_blocks={
                             True: td.AllOf(self.head_w_direction(), self.children() >> td.Map(embed_tree())) >> self.reduce,
                             False: td.AllOf(self.head_w_direction(), td.Void() >> self._init_state)
                         }) >> self.map
        embed_tree.resolve_to(state)
        return state


class TreeEmbedding_HTU_w_direction(TreeEmbedding_HTU):
    def __init__(self, name, **kwargs):
        super(TreeEmbedding_HTU_w_direction, self).__init__(name=name, **kwargs)

    def new_state(self, head, children):
        # discard direction
        return td.AllOf(head(), children) >> self.reduce >> self.map


class TreeEmbedding_HTU_mapIDENTITY(TreeEmbedding_reduce):
    """Calculates an embedding over a (recursive) SequenceNode."""

    def __init__(self, name, **kwargs):
        super(TreeEmbedding_HTU_mapIDENTITY, self).__init__(name='HTU_' + name, **kwargs)

    def new_state(self, head, children):
        _ns = td.OneOf(key_fn=td.InputTransform(lambda x: KEY_CHILDREN in x and len(x[KEY_CHILDREN]) > 0),
                       case_blocks={
                           True: td.AllOf(head(), children) >> self.reduce >> td.GetItem(1),
                           False: head()
                       })
        return _ns

    def __call__(self):
        embed_tree = td.ForwardDeclaration(input_type=td.PyObjectType(), output_type=tdt.TensorType(shape=[self.output_size]))
        children = self.children() >> td.Map(embed_tree())
        state = self.new_state(self.head, children)
        embed_tree.resolve_to(state)
        return state

    @property
    def output_size(self):
        # depends on self.new_state
        #ot = self.map.output_type
        #return ot.shape[-1]
        os = self.reduce_output_size_mapping(self.head_size)
        return os


class TreeEmbedding_HTUrev(TreeEmbedding_HTU):
    """Calculates an embedding over a (recursive) SequenceNode."""

    def __init__(self, name, **kwargs):
        super(TreeEmbedding_HTUrev, self).__init__(name='rev_' + name, **kwargs)

    def new_state(self, head, children):
        children_mapped = td.AllOf(head() >> td.Broadcast(), children) >> td.Zip() >> td.Map(self.map)
        return td.AllOf(td.Void(), children_mapped) >> self.reduce >> td.GetItem(1)

    @property
    def output_size(self):
        # depends on self.new_state
        _map_ot = self.map.output_type
        _map_os = _map_ot.shape[-1]
        _reduced_os = self.reduce_output_size_mapping(_map_os)
        return _reduced_os


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
        return cases


class TreeEmbedding_HTUBatchedHead(TreeEmbedding_HTU):
    """ Calculates batch_size embeddings given a sequence of children and batch_size heads """

    def __init__(self, name, **kwargs):
        super(TreeEmbedding_HTUBatchedHead, self).__init__(name=name, **kwargs)

    def __call__(self):
        _htu_model = super(TreeEmbedding_HTUBatchedHead, self).__call__()
        trees_children = td.GetItem(KEY_CHILDREN) >> td.Map(_htu_model)
        # dummy_head is just passed through reduce, shouldn't be touched
        reduced_children = td.AllOf(td.Void(), trees_children) >> self.reduce >> td.GetItem(1)
        # add id to candidate embeddings
        # do not use leaf_fc, directionality and add_heads for candidates!
        heads_embedded = td.GetItem(KEY_CANDIDATES) \
                         >> td.Map(td.AllOf(self.embed(), td.InputTransform(lambda x: [x]) >> td.Vector(size=1))
                                   >> td.Concat())

        model = td.AllOf(reduced_children >> td.Broadcast(), heads_embedded) >> td.Zip() \
                  >> td.Map(td.Function(lambda x, y: tf.concat((x, y), axis=-1)))

        if model.output_type is None:
            model.set_output_type(tdt.SequenceType(tdt.TensorType(shape=(self.output_size,), dtype='float32')))

        return model

    @property
    def output_size(self):
        return self.dimension_embeddings + self.state_size + 1


class TreeEmbedding_HTUBatchedHead_init_state(TreeEmbedding_HTU_init_state):
    """ Calculates batch_size embeddings given a sequence of children and batch_size heads """

    def __init__(self, name, **kwargs):
        super(TreeEmbedding_HTUBatchedHead_init_state, self).__init__(name=name, **kwargs)

    def __call__(self):
        _htu_model = super(TreeEmbedding_HTUBatchedHead_init_state, self).__call__()
        ## dummy_head is just passed through reduce, shouldn't be touched
        reduced_children = td.OneOf(key_fn=self.has_children(),
                                    case_blocks={
                                        True: td.AllOf(td.Void(), td.GetItem(KEY_CHILDREN) >> td.Map(_htu_model))
                                              >> self.reduce >> td.GetItem(1),
                                        False: td.Void() >> self._init_state
                                    }
                                    )

        # add id to candidate embeddings
        # do not use leaf_fc, directionality and add_heads for candidates!
        heads_embedded = td.GetItem(KEY_CANDIDATES) \
                         >> td.Map(td.AllOf(self.embed(), td.InputTransform(lambda x: [x]) >> td.Vector(size=1))
                                   >> td.Concat())

        model = td.AllOf(reduced_children >> td.Broadcast(), heads_embedded) >> td.Zip() \
                  >> td.Map(td.Function(lambda x, y: tf.concat((x, y), axis=-1)))

        if model.output_type is None:
            model.set_output_type(tdt.SequenceType(tdt.TensorType(shape=(self.output_size,), dtype='float32')))

        return model

    @property
    def output_size(self):
        return self.dimension_embeddings + self.state_size + 1


class TreeEmbedding_HTUBatchedHeadX_init_state(TreeEmbedding_HTU_init_state):
    """ Calculates batch_size embeddings given a sequence of children and batch_size heads """

    def __init__(self, name, **kwargs):
        super(TreeEmbedding_HTUBatchedHeadX_init_state, self).__init__(name=name, **kwargs)

    def __call__(self):
        _htu_model = super(TreeEmbedding_HTUBatchedHeadX_init_state, self).__call__()
        #trees_children = td.GetItem(KEY_CHILDREN) >> td.Map(_htu_model)
        # dummy_head is just passed through reduce, shouldn't be touched
        #reduced_children = td.AllOf(td.Void(), trees_children) >> self.reduce >> td.GetItem(1)
        reduced_children = td.OneOf(key_fn=self.has_children(),
                                    case_blocks={
                                        True: td.AllOf(td.Void(), td.GetItem(KEY_CHILDREN) >> td.Map(_htu_model))
                                              >> self.reduce >> td.GetItem(1),
                                        False: td.Void() >> self._init_state
                                    }
                                    )
        # add id to candidate embeddings
        # do not use leaf_fc for candidates!
        #heads_embedded = td.GetItem(KEY_CANDIDATES) \
        #                 >> td.Map(td.AllOf(self.embed(), td.InputTransform(lambda x: [x]) >> td.Vector(size=1))
        #                           >> td.Concat())
        #heads_embedded = td.GetItem(KEY_CANDIDATES) >> td.Map(self.embed_w_direction())
        heads_embedded = td.InputTransform(lambda x: [{KEY_HEAD: c, KEY_HEAD_CONCAT: x.get(KEY_HEAD_CONCAT, [])} for c in x[KEY_CANDIDATES]]) \
                         >> td.Map(self.head_w_direction())
        model = td.AllOf(
            td.AllOf(heads_embedded, reduced_children >> td.Broadcast()) >> td.Zip() >> td.Map(self.map),
            td.GetItem(KEY_CANDIDATES) >> td.Map(td.InputTransform(lambda x: [x]) >> td.Vector(size=1))
        ) >> td.Zip() >> td.Map(td.Concat())

        if model.output_type is None:
            model.set_output_type(tdt.SequenceType(tdt.TensorType(shape=(self.output_size,), dtype='float32')))

        return model

    @property
    def output_size(self):
        #return self.dimension_embeddings + self.state_size + 1
        return self.state_size + 1


class TreeEmbedding_HTUBatchedHeadX(TreeEmbedding_HTU):
    """ Calculates batch_size embeddings given a sequence of children and batch_size heads """

    def __init__(self, name, **kwargs):
        super(TreeEmbedding_HTUBatchedHeadX, self).__init__(name=name, **kwargs)

    def __call__(self):
        _htu_model = super(TreeEmbedding_HTUBatchedHeadX, self).__call__()
        # dummy_head is just passed through reduce, shouldn't be touched
        reduced_children = td.AllOf(td.Void(), td.GetItem(KEY_CHILDREN) >> td.Map(_htu_model)) >> self.reduce >> td.GetItem(1)

        # add id to candidate embeddings
        # do not use leaf_fc for candidates!
        #heads_embedded = td.GetItem(KEY_CANDIDATES) \
        #                 >> td.Map(td.AllOf(self.embed(), td.InputTransform(lambda x: [x]) >> td.Vector(size=1))
        #                           >> td.Concat())
        #heads_embedded = td.GetItem(KEY_CANDIDATES) >> td.Map(self.embed_w_direction())
        heads_embedded = td.InputTransform(lambda x: [{KEY_HEAD: c, KEY_HEAD_CONCAT: x.get(KEY_HEAD_CONCAT, [])} for c in x[KEY_CANDIDATES]]) \
                         >> td.Map(self.head_w_direction())
        model = td.AllOf(
            td.AllOf(heads_embedded, reduced_children >> td.Broadcast()) >> td.Zip() >> td.Map(self.map),
            td.GetItem(KEY_CANDIDATES) >> td.Map(td.InputTransform(lambda x: [x]) >> td.Vector(size=1))
        ) >> td.Zip() >> td.Map(td.Concat())

        if model.output_type is None:
            model.set_output_type(tdt.SequenceType(tdt.TensorType(shape=(self.output_size,), dtype='float32')))

        return model

    @property
    def output_size(self):
        #return self.dimension_embeddings + self.state_size + 1
        return self.state_size + 1


class TreeEmbedding_FLAT(TreeEmbedding_reduce):
    """
        FLAT TreeEmbedding models take all first level children of the root as input and reduce them.
    """
    def __init__(self, name, **kwargs):
        super(TreeEmbedding_FLAT, self).__init__(name='FLAT_' + name, **kwargs)

    def __call__(self):
        model = self.children() >> td.Map(self.head_w_direction() >> td.GetItem(0)) >> td.AllOf(td.Void(), td.Identity()) >> self.reduce >> td.GetItem(1)
        if model.output_type is None:
            model.set_output_type(tdt.TensorType(shape=(self.output_size,), dtype='float32'))
        return model

    @property
    def output_size(self):
        raise NotImplementedError("Please implement this method")


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


class TreeEmbedding_FLATconcat(TreeEmbedding):
    """
        FLAT TreeEmbedding models take all first level children of the root as input and reduce them.
        The FLATconcat model pads the list of token ids (direct children) to sequence_length, embeds them and
        returns the concatenation of the embeddings and the amount of real token ids
        i.e. the result shape is (batch_size, sequence_length * embedding_size + 1).
    """
    def __init__(self, name, sequence_length, padding_id, merge_factor=1, **kwargs):
        assert merge_factor == 1, 'merge_factor is deprecated, but it is set to %i' % merge_factor
        #self._merge_factor = merge_factor
        self._sequence_length = sequence_length
        self._padding_id = padding_id
        super(TreeEmbedding_FLATconcat, self).__init__(name='FLATconcat_' + name, **kwargs)

    def __call__(self):
        padding_element = {KEY_HEAD: self._padding_id, KEY_CHILDREN: [],
                           KEY_HEAD_CONCAT: [self._padding_id] * len(self.additional_heads_dims)}

        def adjust_length(l):
            if len(l) >= self.sequence_length:
                new_l = l[:self.sequence_length]
            else:
                new_l = l + [padding_element] * (self.sequence_length - len(l))
            # return adjusted list and the length of valid entries
            return new_l, float(min(len(l), self.sequence_length))

        model = self.children() >> td.InputTransform(adjust_length) >> td.AllOf(
            td.GetItem(0) >> td.Map(self.head_w_direction() >> td.GetItem(0))
            >> SequenceToTuple(tdt.TensorType(shape=[self.head_size], dtype='float32'), self.sequence_length)
            >> td.Concat(),
            td.GetItem(1) >> td.Scalar(dtype='float32')
        ) >> td.Concat()

        if model.output_type is None:
            model.set_output_type(tdt.TensorType(shape=(self.head_size * self.sequence_length + 1,), dtype='float32'))
        return model

    def reduce_concatenated(self, concatenated_embeddings_with_length):
        concatenated_embeddings = concatenated_embeddings_with_length[:, :-1]
        length = concatenated_embeddings_with_length[:, -1]
        embeddings_sequence = tf.reshape(concatenated_embeddings, shape=[-1, int(self.sequence_length), self.head_size])
        return self.reduce_flat(embeddings_sequence, length)

    def reduce_flat(self, embeddings, actual_length):
        raise NotImplementedError('Implement this')

    # This is not the actual output size, but the output is adapted to this
    # in SequenceTreeModel.__init__ via TreeEmbedding_FLAT_CONCAT.reduce_concatenated
    @property
    def output_size(self):
        raise NotImplementedError('Implement this')

    @property
    def sequence_length(self):
        return self._sequence_length

    #@property
    #def merge_factor(self):
    #    return self._merge_factor


#######################################
#  Instantiable TreeEmbedding models  #
#######################################


class TreeEmbedding_HTU_reduceSUM_mapGRU(TreeEmbedding_reduceSUM, TreeEmbedding_mapGRU, TreeEmbedding_HTU):
    def __init__(self, name='', **kwargs):
        super(TreeEmbedding_HTU_reduceSUM_mapGRU, self).__init__(name=name, **kwargs)


class TreeEmbedding_HTU_reduceMAX_mapGRU(TreeEmbedding_reduceMAX, TreeEmbedding_mapGRU, TreeEmbedding_HTU):
    def __init__(self, name='', **kwargs):
        super(TreeEmbedding_HTU_reduceMAX_mapGRU, self).__init__(name=name, **kwargs)

class TreeEmbedding_HTU_reduceAVG_mapGRU(TreeEmbedding_reduceAVG, TreeEmbedding_mapGRU, TreeEmbedding_HTU):
    def __init__(self, name='', **kwargs):
        super(TreeEmbedding_HTU_reduceAVG_mapGRU, self).__init__(name=name, **kwargs)

class TreeEmbedding_HTU_reduceSUM_mapGRU2(TreeEmbedding_reduceSUM, TreeEmbedding_mapGRU2, TreeEmbedding_HTU):
    def __init__(self, name='', **kwargs):
        super(TreeEmbedding_HTU_reduceSUM_mapGRU2, self).__init__(name=name, **kwargs)


class TreeEmbedding_HTU_reduceSUM_mapGRU_wd(TreeEmbedding_reduceSUM, TreeEmbedding_mapGRU_w_direction, TreeEmbedding_HTU_w_direction):
    def __init__(self, name='', **kwargs):
        super(TreeEmbedding_HTU_reduceSUM_mapGRU_wd, self).__init__(name=name, **kwargs)


class TreeEmbedding_HTU_reduceMAX_mapGRU_wd(TreeEmbedding_reduceMAX, TreeEmbedding_mapGRU_w_direction, TreeEmbedding_HTU_w_direction):
    def __init__(self, name='', **kwargs):
        super(TreeEmbedding_HTU_reduceMAX_mapGRU_wd, self).__init__(name=name, **kwargs)


class TreeEmbedding_HTU_reduceSUM_mapGRU_pl(TreeEmbedding_reduceSUM, TreeEmbedding_mapGRU, TreeEmbedding_HTU_plain_leaf):
    def __init__(self, name='', **kwargs):
        super(TreeEmbedding_HTU_reduceSUM_mapGRU_pl, self).__init__(name=name, **kwargs)


class TreeEmbedding_HTU_reduceMAX_mapGRU_pl(TreeEmbedding_reduceMAX, TreeEmbedding_mapGRU, TreeEmbedding_HTU_plain_leaf):
    def __init__(self, name='', **kwargs):
        super(TreeEmbedding_HTU_reduceMAX_mapGRU_pl, self).__init__(name=name, **kwargs)


class TreeEmbedding_HTU_reduceSUM_mapGRU_is(TreeEmbedding_reduceSUM, TreeEmbedding_mapGRU, TreeEmbedding_HTU_init_state):
    def __init__(self, name='', **kwargs):
        super(TreeEmbedding_HTU_reduceSUM_mapGRU_is, self).__init__(name=name, **kwargs)


class TreeEmbedding_HTU_reduceMAX_mapGRU_is(TreeEmbedding_reduceMAX, TreeEmbedding_mapGRU, TreeEmbedding_HTU_init_state):
    def __init__(self, name='', **kwargs):
        super(TreeEmbedding_HTU_reduceMAX_mapGRU_is, self).__init__(name=name, **kwargs)


class TreeEmbedding_HTU_reduceSUM_mapLSTM(TreeEmbedding_reduceSUM, TreeEmbedding_mapLSTM, TreeEmbedding_HTU):
    def __init__(self, name='', **kwargs):
        super(TreeEmbedding_HTU_reduceSUM_mapLSTM, self).__init__(name=name, **kwargs)


class TreeEmbedding_HTU_reduceSUM_mapFC(TreeEmbedding_reduceSUM, TreeEmbedding_mapFC, TreeEmbedding_HTU):
    def __init__(self, name='', **kwargs):
        super(TreeEmbedding_HTU_reduceSUM_mapFC, self).__init__(name=name, **kwargs)


class TreeEmbedding_HTU_reduceSUM_mapCCFC(TreeEmbedding_reduceSUM, TreeEmbedding_mapCCFC, TreeEmbedding_HTU):
    def __init__(self, name='', **kwargs):
        super(TreeEmbedding_HTU_reduceSUM_mapCCFC, self).__init__(name=name, **kwargs)


class TreeEmbedding_HTUrev_reduceSUM_mapGRU(TreeEmbedding_mapGRU, TreeEmbedding_reduceSUM, TreeEmbedding_HTUrev):
    def __init__(self, name='', **kwargs):
        super(TreeEmbedding_HTUrev_reduceSUM_mapGRU, self).__init__(name=name, **kwargs)


class TreeEmbedding_HTUrev_reduceAVG_mapGRU(TreeEmbedding_mapGRU, TreeEmbedding_reduceAVG, TreeEmbedding_HTUrev):
    def __init__(self, name='', **kwargs):
        super(TreeEmbedding_HTUrev_reduceAVG_mapGRU, self).__init__(name=name, **kwargs)


class TreeEmbedding_HTUrev_reduceMAX_mapGRU(TreeEmbedding_mapGRU, TreeEmbedding_reduceMAX, TreeEmbedding_HTUrev):
    def __init__(self, name='', **kwargs):
        super(TreeEmbedding_HTUrev_reduceMAX_mapGRU, self).__init__(name=name, **kwargs)


class TreeEmbedding_HTUdep_mapGRU(TreeEmbedding_HTUdep, TreeEmbedding_mapGRU):
    def __init__(self, name='', **kwargs):
        super(TreeEmbedding_HTUdep_mapGRU, self).__init__(name=name, **kwargs)


class TreeEmbedding_HTU_reduceATT_mapGRU(TreeEmbedding_reduceATT, TreeEmbedding_mapGRU, TreeEmbedding_HTU):
    def __init__(self, name='', **kwargs):
        super(TreeEmbedding_HTU_reduceATT_mapGRU, self).__init__(name=name, **kwargs)


class TreeEmbedding_HTU_reduceATT_mapAVG(TreeEmbedding_reduceATT, TreeEmbedding_mapAVG, TreeEmbedding_HTU):
    def __init__(self, name='', **kwargs):
        super(TreeEmbedding_HTU_reduceATT_mapAVG, self).__init__(name=name, **kwargs)


class TreeEmbedding_HTU_reduceATT_mapSUM(TreeEmbedding_reduceATT, TreeEmbedding_mapSUM, TreeEmbedding_HTU):
    def __init__(self, name='', **kwargs):
        super(TreeEmbedding_HTU_reduceATT_mapSUM, self).__init__(name=name, **kwargs)


class TE_HTU_reduceATT_mapIDENTITY(TreeEmbedding_reduceATT, TreeEmbedding_HTU_mapIDENTITY):
    def __init__(self, name='', **kwargs):
        super(TE_HTU_reduceATT_mapIDENTITY, self).__init__(name=name, **kwargs)


class TE_HTU_reduceATTSINGLE_mapIDENTITY(TreeEmbedding_reduceATTsingle, TreeEmbedding_HTU_mapIDENTITY):
    def __init__(self, name='', **kwargs):
        super(TE_HTU_reduceATTSINGLE_mapIDENTITY, self).__init__(name=name, **kwargs)


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
        return self.head_size


class TreeEmbedding_FLAT2levels_AVG(TreeEmbedding_FLAT_AVG, TreeEmbedding_FLAT2levels):
    def __init__(self, name='', **kwargs):
        super(TreeEmbedding_FLAT2levels_AVG, self).__init__(name=name, **kwargs)


class TreeEmbedding_FLAT_SUM(TreeEmbedding_reduceSUM, TreeEmbedding_FLAT):
    def __init__(self, name='', **kwargs):
        super(TreeEmbedding_FLAT_SUM, self).__init__(name=name, **kwargs)

    @property
    def output_size(self):
        return self.head_size


class TreeEmbedding_FLAT2levels_SUM(TreeEmbedding_FLAT_SUM, TreeEmbedding_FLAT2levels):
    def __init__(self, name='', **kwargs):
        super(TreeEmbedding_FLAT2levels_SUM, self).__init__(name=name, **kwargs)


class TreeEmbedding_FLAT_MAX(TreeEmbedding_reduceMAX, TreeEmbedding_FLAT):
    def __init__(self, name='', **kwargs):
        super(TreeEmbedding_FLAT_MAX, self).__init__(name=name, **kwargs)

    @property
    def output_size(self):
        return self.head_size


class TreeEmbedding_FLAT_LSTM(TreeEmbedding_reduceLSTM, TreeEmbedding_FLAT):
    def __init__(self, name='', **kwargs):
        super(TreeEmbedding_FLAT_LSTM, self).__init__(name=name, **kwargs)


class TreeEmbedding_FLAT2levels_LSTM(TreeEmbedding_FLAT_LSTM, TreeEmbedding_FLAT2levels):
    def __init__(self, name='', **kwargs):
        super(TreeEmbedding_FLAT_LSTM, self).__init__(name=name, **kwargs)


class TreeEmbedding_FLAT_GRU(TreeEmbedding_reduceGRU, TreeEmbedding_FLAT):
    def __init__(self, name='', **kwargs):
        super(TreeEmbedding_FLAT_GRU, self).__init__(name=name, **kwargs)


class TreeEmbedding_FLAT2Levels_GRU(TreeEmbedding_FLAT_GRU, TreeEmbedding_FLAT2levels):
    def __init__(self, name='', **kwargs):
        super(TreeEmbedding_FLAT2Levels_GRU, self).__init__(name=name, **kwargs)


class TreeEmbedding_HTUBatchedHead_reduceSUM_mapGRU(TreeEmbedding_reduceSUM, TreeEmbedding_mapGRU, TreeEmbedding_HTUBatchedHead):
    def __init__(self, name='', **kwargs):
        super(TreeEmbedding_HTUBatchedHead_reduceSUM_mapGRU, self).__init__(name=name, **kwargs)


class TreeEmbedding_HTUBatchedHead_reduceMAX_mapGRU(TreeEmbedding_reduceMAX, TreeEmbedding_mapGRU, TreeEmbedding_HTUBatchedHead):
    def __init__(self, name='', **kwargs):
        super(TreeEmbedding_HTUBatchedHead_reduceMAX_mapGRU, self).__init__(name=name, **kwargs)


class TreeEmbedding_HTUBatchedHead_reduceSUM_mapCCFC(TreeEmbedding_reduceSUM, TreeEmbedding_mapCCFC, TreeEmbedding_HTUBatchedHead):
    def __init__(self, name='', **kwargs):
        super(TreeEmbedding_HTUBatchedHead_reduceSUM_mapCCFC, self).__init__(name=name, **kwargs)


class TreeEmbedding_HTUBatchedHead_reduceSUM_mapGRU_wd(TreeEmbedding_reduceSUM, TreeEmbedding_mapGRU_w_direction, TreeEmbedding_HTUBatchedHead):
    def __init__(self, name='', **kwargs):
        super(TreeEmbedding_HTUBatchedHead_reduceSUM_mapGRU_wd, self).__init__(name=name, **kwargs)


class TreeEmbedding_HTUBatchedHead_reduceMAX_mapGRU_wd(TreeEmbedding_reduceMAX, TreeEmbedding_mapGRU_w_direction, TreeEmbedding_HTUBatchedHead):
    def __init__(self, name='', **kwargs):
        super(TreeEmbedding_HTUBatchedHead_reduceMAX_mapGRU_wd, self).__init__(name=name, **kwargs)


class TreeEmbedding_HTUBatchedHead_reduceMAX_mapGRU_wd_is(TreeEmbedding_reduceMAX, TreeEmbedding_mapGRU_w_direction, TreeEmbedding_HTUBatchedHead_init_state):
    def __init__(self, name='', **kwargs):
        super(TreeEmbedding_HTUBatchedHead_reduceMAX_mapGRU_wd_is, self).__init__(name=name, **kwargs)


class TreeEmbedding_HTUBatchedHeadX_reduceSUM_mapGRU_wd_is(TreeEmbedding_reduceSUM, TreeEmbedding_mapGRU_w_direction, TreeEmbedding_HTUBatchedHeadX_init_state):
    def __init__(self, name='', **kwargs):
        super(TreeEmbedding_HTUBatchedHeadX_reduceSUM_mapGRU_wd_is, self).__init__(name=name, **kwargs)


class TreeEmbedding_HTUBatchedHeadX_reduceMAX_mapGRU_wd_is(TreeEmbedding_reduceMAX, TreeEmbedding_mapGRU_w_direction, TreeEmbedding_HTUBatchedHeadX_init_state):
    def __init__(self, name='', **kwargs):
        super(TreeEmbedding_HTUBatchedHeadX_reduceMAX_mapGRU_wd_is, self).__init__(name=name, **kwargs)


class TreeEmbedding_HTUBatchedHeadX_reduceMAX_mapGRU_wd(TreeEmbedding_reduceMAX, TreeEmbedding_mapGRU_w_direction, TreeEmbedding_HTUBatchedHeadX):
    def __init__(self, name='', **kwargs):
        super(TreeEmbedding_HTUBatchedHeadX_reduceMAX_mapGRU_wd, self).__init__(name=name, **kwargs)


class TreeEmbedding_HTUBatchedHeadX_reduceMAX_mapGRU(TreeEmbedding_reduceMAX, TreeEmbedding_mapGRU, TreeEmbedding_HTUBatchedHeadX):
    def __init__(self, name='', **kwargs):
        super(TreeEmbedding_HTUBatchedHeadX_reduceMAX_mapGRU, self).__init__(name=name, **kwargs)


class TreeEmbedding_FLATconcat_GRU_DEP(TreeEmbedding_FLATconcat):
    def __init__(self, **kwargs):
        TreeEmbedding_FLATconcat.__init__(self, name='GRU', **kwargs)

    def reduce_flat(self, embeddings, actual_length):
        cell = tf.nn.rnn_cell.GRUCell(num_units=self.state_size)
        cell_dropout = tf.nn.rnn_cell.DropoutWrapper(
            cell, input_keep_prob=self.keep_prob, output_keep_prob=self.keep_prob, state_keep_prob=self.keep_prob,
            variational_recurrent=True, dtype=embeddings.dtype, input_size=embeddings.shape[-1])
        inputs = tf.unstack(embeddings, axis=1)
        outputs, state = tf.nn.static_rnn(cell_dropout, inputs, sequence_length=actual_length,
                                          dtype=embeddings.dtype)
        return state

    # This is not the actual output size, but the output is adapted to this in SequenceTreeModel.__init__
    @property
    def output_size(self):
        return self.state_size


class TreeEmbedding_FLATconcat_GRU(TreeEmbedding_FLATconcat):
    def __init__(self, name=None, use_summed_outputs=True, **kwargs):
        self._use_summed_outputs = use_summed_outputs
        TreeEmbedding_FLATconcat.__init__(self, name=name or 'GRU', **kwargs)

    def reduce_flat(self, embeddings, actual_length):

        _dtype = embeddings.dtype
        inputs = tf.unstack(embeddings, axis=1)

        states = []
        i = 0
        for size in self.state_sizes:
            if size > 0:
                with tf.variable_scope(self.name + '/layer_' + str(i)):
                    assert len(inputs) > 0, 'number of inputs (sequence_length) for (BI)GRU is zero'
                    input_size = inputs[0].shape[-1]
                    cells = []
                    for _ in range(self.nbr_cells):
                        cell = tf.nn.rnn_cell.GRUCell(num_units=self.state_size)
                        cell_dropout = tf.nn.rnn_cell.DropoutWrapper(
                            cell, input_keep_prob=self.keep_prob, output_keep_prob=self.keep_prob,
                            state_keep_prob=self.keep_prob, variational_recurrent=True, input_size=input_size,
                            dtype=_dtype)
                        cells.append(cell_dropout)
                    inputs, current_states = self.create_rnn_layer(cells=cells, inputs=inputs,
                                                                   sequence_length=actual_length, dtype=_dtype)
                    states.extend(current_states)
                i += 1

        if self._use_summed_outputs:
            summed_outputs = tf.add_n(inputs)
            states.append(summed_outputs)
        res = tf.concat(states, axis=-1)
        return res

    @property
    def nbr_cells(self):
        return 1

    @staticmethod
    def create_rnn_layer(cells, inputs, sequence_length, dtype):
        outputs, state = tf.nn.static_rnn(cells[0], inputs, sequence_length=sequence_length, dtype=dtype)
        return outputs, [state]


class TreeEmbedding_FLATconcat_BIGRU(TreeEmbedding_FLATconcat_GRU):
    def __init__(self, name=None, **kwargs):
        TreeEmbedding_FLATconcat_GRU.__init__(self, name=name or 'BIGRU', **kwargs)

    @staticmethod
    def create_rnn_layer(cells, inputs, sequence_length, dtype):
        outputs, state_fw, state_bw = tf.nn.static_bidirectional_rnn(
            cells[0], cells[1], inputs, sequence_length=sequence_length, dtype=dtype)
        return outputs, [state_fw, state_bw]

    @property
    def nbr_cells(self):
        return 2


class TreeEmbedding_FLATconcat_GRU0(TreeEmbedding_FLATconcat_GRU):
    def __init__(self, name=None, **kwargs):
        TreeEmbedding_FLATconcat_GRU.__init__(self, name=name or 'GRU0', use_summed_outputs=False, **kwargs)


class TreeEmbedding_FLATconcat_BIGRU0(TreeEmbedding_FLATconcat_BIGRU):
    def __init__(self, name=None, **kwargs):
        TreeEmbedding_FLATconcat_BIGRU.__init__(self, name=name or 'BIGRU0', use_summed_outputs=False, **kwargs)


# should be equivalent to FLAT_SUM if merge_factor == 1
class TreeEmbedding_FLATconcat_SUM(TreeEmbedding_FLATconcat):
    def __init__(self, name=None, **kwargs):
        TreeEmbedding_FLATconcat.__init__(self, name=name or 'SUM', **kwargs)

    def reduce_flat(self, embeddings, actual_length):
        return tf.reduce_sum(embeddings, axis=1)


class TreeEmbedding_FLATconcat_AVG(TreeEmbedding_FLATconcat):
    def __init__(self, name=None, **kwargs):
        TreeEmbedding_FLATconcat.__init__(self, name=name or 'AVG', **kwargs)

    def reduce_flat(self, embeddings, actual_length):
        embedded_size = int(embeddings.get_shape().as_list()[-1])
        length_tiled = tf.tile(tf.expand_dims(actual_length, 1), multiples=(1, embedded_size))
        return tf.reduce_sum(embeddings, axis=1) / length_tiled


########################################################################################################################


def sim_cosine_DEP(e1, e2):
    e1 = tf.nn.l2_normalize(e1, dim=1)
    e2 = tf.nn.l2_normalize(e2, dim=1)
    return tf.reduce_sum(e1 * e2, axis=1)


def sim_cosine(embeddings, normalize=True, clip=True):
    """
    :param embeddings: paired embeddings with shape = [batch_count, 2, embedding_dimension]
    :return: clipped similarity scores with shape = [batch_count]
    """
    if normalize:
        embeddings = tf.nn.l2_normalize(embeddings, dim=-1)
    cos = tf.reduce_sum(embeddings[:, 0, :] * embeddings[:, 1, :], axis=-1)
    if clip:
        return tf.clip_by_value(cos, 0.0, 1.0)
    else:
        return cos


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


def circular_correlation(a, b):
    a_len = a.get_shape().as_list()[-1]
    aa = tf.concat((a, a), axis=-1)
    #i_end = i + a_len
    # [0..a_len]
    indices_simple = tf.range(a_len)
    # [[0..a_len], .., [0..a_len]]    (a_len times)
    indices_multi = tf.reshape(tf.tile(indices_simple, (a_len,)), (a_len, a_len))
    # [[0..a_len], .., [i..(a_len + 1)], .. [a_len..a_len * 2]]
    indices = indices_multi + tf.expand_dims(indices_simple, 1)
    rolled = tf.gather(aa, indices, axis=-1)

    y = tf.matmul(rolled, tf.expand_dims(b, -1))
    return tf.squeeze(y, axis=-1)


class TreeModel(object):
    def __init__(self, embeddings_plain, prepared_embeddings_plain=None, keep_prob_placeholder=None,
                 keep_prob_default=1.0, root_fc_sizes=0, discard_tree_embeddings=False,
                  discard_prepared_embeddings=False, **kwargs):

        if keep_prob_placeholder is None:
            keep_prob_placeholder = tf.placeholder_with_default(keep_prob_default, shape=())
        self._keep_prob = keep_prob_placeholder

        if isinstance(root_fc_sizes, (list, tuple)):
            self._root_fc_sizes = root_fc_sizes
        else:
            self._root_fc_sizes = [root_fc_sizes]

        self._embeddings_plain = embeddings_plain
        if discard_tree_embeddings:
            self._embeddings_plain = tf.zeros_like(self._embeddings_plain)
        if prepared_embeddings_plain is not None:
            if discard_prepared_embeddings:
                prepared_embeddings_plain = tf.zeros_like(prepared_embeddings_plain)
            self._embeddings_plain = tf.concat((self._embeddings_plain, prepared_embeddings_plain), axis=-1)

        #self._output_size = int(self._embeddings_plain.shape[-1])

        i = 0
        for s in self._root_fc_sizes:
            if s > 0:
                with tf.variable_scope(VAR_PREFIX_TREE_EMBEDDING+'/'+VAR_PREFIX_FC_ROOT + '/' + str(i)) as scope:
                    fc = tf.contrib.layers.fully_connected(inputs=self._embeddings_plain, num_outputs=s,
                                                           activation_fn=tf.nn.tanh, scope=scope)
                    self._embeddings_plain = tf.nn.dropout(fc, keep_prob=self._keep_prob)
                    #self._output_size = s
                    i += 1

    @property
    def embeddings_all(self):
        return self._embeddings_plain

    @property
    def keep_prob(self):
        return self._keep_prob

    @property
    def tree_output_size(self):
        return int(self._embeddings_plain.shape[-1])
        #return self._output_size


class SequenceTreeModel(TreeModel):
    def __init__(self, nbr_trees_out, tree_embedder, keep_prob_default, state_size, prepared_embeddings_dim=-1,
                 prepared_embeddings_sparse=False, **kwargs):
        self._nbr_trees_out = nbr_trees_out or 1
        prepared_embeddings_plain = None
        if prepared_embeddings_dim > 0:
            if prepared_embeddings_sparse:
                self._prepared_embeddings_placeholder = tf.sparse_placeholder(shape=[None, prepared_embeddings_dim], dtype=tf.float32)
                prepared_embeddings_plain = tf.reshape(
                    tf.sparse_tensor_to_dense(self._prepared_embeddings_placeholder, validate_indices=False),
                    shape=[-1, prepared_embeddings_dim])
            else:
                self._prepared_embeddings_placeholder = tf.placeholder(shape=[None, prepared_embeddings_dim], dtype=tf.float32)
                prepared_embeddings_plain = tf.reshape(self._prepared_embeddings_placeholder, shape=[-1, prepared_embeddings_dim])

        keep_prob_placeholder = tf.placeholder_with_default(keep_prob_default, shape=())

        self._tree_embed = tree_embedder(keep_prob_placeholder=keep_prob_placeholder, state_size=state_size, **kwargs)

        embed_tree = self._tree_embed()
        # do not map tree embedding model to input, if it produces a sequence itself
        if isinstance(embed_tree.output_type, tdt.SequenceType):
            # TODO: move td.GetItem(0) into BatchedHead model
            # input contains only one tree (with candidate heads)
            model = td.GetItem(0) >> embed_tree >> SequenceToTuple(embed_tree.output_type.element_type, self.nbr_trees_out) >> td.Concat()
        else:
            model = td.Map(embed_tree) >> SequenceToTuple(embed_tree.output_type, self.nbr_trees_out) >> td.Concat()

        # fold model output
        self._compiler = td.Compiler.create(model)
        self._tree_embeddings_all, = self._compiler.output_tensors

        # For FLATconcat models, the tree embedder just embedded the leafs and the composition takes place in
        # reduce_concatenated. Furthermore, we take the embedding_size from that output.
        if isinstance(self._tree_embed, TreeEmbedding_FLATconcat):
            self._tree_embeddings_all = self.embedder.reduce_concatenated(self._tree_embeddings_all)
            embeddings_size = int(self._tree_embeddings_all.shape[-1])
        else:
            embeddings_size = self.embedder.output_size

        super(SequenceTreeModel, self).__init__(
            embeddings_plain=tf.reshape(self._tree_embeddings_all, shape=[-1, embeddings_size]),
            prepared_embeddings_plain=prepared_embeddings_plain,
            keep_prob_placeholder=keep_prob_placeholder,
            **kwargs)

    def build_feed_dict(self, data):
        return self._compiler.build_feed_dict(data)

    @property
    def embedder(self):
        return self._tree_embed

    @property
    def compiler(self):
        return self._compiler

    @property
    def prepared_embeddings_placeholder(self):
        return self._prepared_embeddings_placeholder

    @property
    def nbr_trees_out(self):
        return self._nbr_trees_out


class DummyTreeModel(TreeModel):
    def __init__(self, embeddings_dim, sparse=False, **kwargs):

        if sparse:
            self._embeddings_placeholder = tf.sparse_placeholder(shape=[None, embeddings_dim], dtype=tf.float32)
            embeddings_plain = tf.reshape(
                tf.sparse_tensor_to_dense(self._embeddings_placeholder, validate_indices=False),
                shape=[-1, embeddings_dim])
        else:
            self._embeddings_placeholder = tf.placeholder(shape=[None, embeddings_dim], dtype=tf.float32)
            embeddings_plain = tf.reshape(self._embeddings_placeholder, shape=[-1, embeddings_dim])

        super(DummyTreeModel, self).__init__(embeddings_plain=embeddings_plain, **kwargs)

    @property
    def prepared_embeddings_placeholder(self):
        return self._embeddings_placeholder


class BaseTrainModel(object):
    def __init__(self, tree_model, loss, nbr_embeddings_in=1, optimizer=None, learning_rate=0.1, clipping_threshold=5.0, metrics={},
                 metric_reset_op=(), **unused):
        # _nbr_embeddings_in my be already set
        try:
            self.nbr_embeddings_in
        except AttributeError:
            self._nbr_embeddings_in = nbr_embeddings_in
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

        #self._metrics = {'auc': tf.metrics.auc(labels=self.gold_eval, predictions=self.predicted_eval)}
        self._metrics = metrics
        self._reset_metrics = metric_reset_op

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

    @property
    def metrics(self):
        return {k: self._metrics[k][0] for k in self._metrics.keys()}

    @property
    def update_metrics(self):
        return {k: self._metrics[k][1] for k in self._metrics.keys()}

    @property
    def reset_metrics(self):
        return self._reset_metrics

    @property
    def nbr_embeddings_in(self):
        return self._nbr_embeddings_in


class TreeTupleModel(BaseTrainModel):
    """A Fold model for similarity scored sequence tree (SequenceNode) tuple."""

    def __init__(self, tree_model, sim_measure=sim_cosine, **kwargs):

        self._labels_gold = tf.placeholder(dtype=tf.float32)
        # pack tree embeddings in pairs of two
        self._tree_embeddings_reshaped = tf.reshape(tree_model.embeddings_all, shape=[-1, 2, tree_model.tree_output_size])
        # apply sim measure
        self._scores = sim_measure(self._tree_embeddings_reshaped)

        with tf.variable_scope("reset_metrics_scope") as scope:
            metrics = {
                'pearson_r': tf.contrib.metrics.streaming_pearson_correlation(
                    labels=self.values_gold, predictions=self.values_predicted),
                'mse': tf.contrib.metrics.streaming_mean_squared_error(
                    labels=self.values_gold, predictions=self.values_predicted),
            }
            vars = tf.contrib.framework.get_variables(scope, collection=tf.GraphKeys.LOCAL_VARIABLES)
            reset_op = tf.variables_initializer(vars)

        BaseTrainModel.__init__(self, tree_model=tree_model,
                                loss=tf.reduce_mean(tf.square(self._scores - self._labels_gold)),
                                metrics=metrics, metric_reset_op=reset_op, nbr_embeddings_in=2, **kwargs)

    @property
    def values_gold(self):
        return self._labels_gold

    @property
    def values_predicted(self):
        return self._scores

    @property
    def model_type(self):
        return MODEL_TYPE_REGRESSION


# not used
class SimilaritySequenceTreeTupleModel_sample(BaseTrainModel):
    """A Fold model for similarity scored sequence tree (SequenceNode) tuple."""

    def __init__(self, tree_model, **kwargs):

        # unpack scores_gold. Every prob tuple has the format: [1.0, ...]
        #self._scores_gold = tf.reshape(tree_model.probs_gold_shaped, shape=[-1, 2])[:, 0]

        # pack tree embeddings in pairs of two
        tree_embeddings = tf.reshape(tree_model.embeddings_all, shape=[-1, 2, tree_model.tree_output_size])

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

        softmax = tf.nn.softmax(logits)
        self._probs = softmax[:, 1]

        #BaseTrainModel.__init__(self, tree_model=tree_model,
        #                       loss=tf.reduce_mean(tf.square(self._scores - self._scores_gold)), **kwargs)
        BaseTrainModel.__init__(self, tree_model=tree_model,
                                loss=tf.reduce_mean(cross_entropy), **kwargs)

    @property
    def values_gold(self):
        return self._labels_gold

    @property
    def values_predicted(self):
        return self._probs

    @property
    def model_type(self):
        return MODEL_TYPE_DISCRETE


class TreeScoringModel_with_candidates(BaseTrainModel):
    """Predict the correct embeddings among multiple ones. The first embedding is assumed to be the correct one."""

    def __init__(self, tree_model, nbr_embeddings_in, fc_sizes=1000, use_circular_correlation=False, embedded_root=False, **kwargs):
        self._labels_gold = tf.placeholder(dtype=tf.float32)
        #self._candidate_count = tf.shape(self._labels_gold)[-1]
        self._nbr_embeddings_in = nbr_embeddings_in
        tree_embeddings = tf.reshape(tree_model.embeddings_all,
                                     shape=[-1, self.nbr_embeddings_in, tree_model.tree_output_size])
        # last entry of every embedding contains the candidate id (see TreeEmbedding_HTUBatchedHead)
        self._candidate_indices = tf.cast(tf.reshape(tree_embeddings[:,:,-1], shape=[-1, self.nbr_embeddings_in]), dtype=tf.int32)
        final_vecs = self._final_vecs(tree_embeddings[:,:,:-1], tree_model.tree_output_size - 1)

        if not isinstance(fc_sizes, (list, tuple)):
            fc_sizes = [fc_sizes]

        # add multiple fc layers
        for s in fc_sizes:
            if s > 0:
                # disabled, because it should be asymmetric
                raise NotImplementedError('fc_sizes are currently not implemented')
                fc = tf.contrib.layers.fully_connected(inputs=final_vecs, num_outputs=s)
                final_vecs = tf.nn.dropout(fc, keep_prob=tree_model.keep_prob)

        # add circular self correlation
        #if use_circular_correlation:
        #    logger.debug('add circular self correlation')
        #    circ_cor = circular_correlation(final_vecs, final_vecs)
        #    final_vecs = tf.concat((final_vecs, circ_cor), axis=-1)

        # shape: batch_size, candidate_count, concat_embeddings_dim
        _shape = final_vecs.get_shape().as_list()
        concat_embeddings_dim = _shape[-1]
        vecs_reshaped = tf.reshape(final_vecs, shape=(-1, concat_embeddings_dim))

        # assume, we get vectors that are concatenations of: (aggregated children, candidate head)
        if not embedded_root:
            ## split vecs into two
            #assert concat_embeddings_dim % 2 == 0, \
            #    'dimension of concatenated embeddings has to be multiple of two, but it is: %i' % _shape[-1]
            #final_vecs_split = tf.reshape(final_vecs, shape=(-1, 2, concat_embeddings_dim // 2))

            candidate_dims = tree_model.embedder.dimension_embeddings
            reference_dims = concat_embeddings_dim - candidate_dims
            vecs_reference = vecs_reshaped[:, :reference_dims]
            vecs_candidate = tf.expand_dims(vecs_reshaped[:, reference_dims:], axis=1)
            with tf.name_scope(name='fc_reference') as sc:
                fc_reference = tf.contrib.layers.fully_connected(inputs=vecs_reference, num_outputs=candidate_dims, scope=sc)
                vecs_reference_scaled = tf.expand_dims(tf.nn.dropout(fc_reference, keep_prob=tree_model.keep_prob), axis=1)
            final_vecs_split = tf.concat((vecs_reference_scaled, vecs_candidate), axis=1)

            # use circular correlation
            if use_circular_correlation:
                logger.debug('add circular correlation')
                circ_cor = circular_correlation(final_vecs_split[:, 0, :], final_vecs_split[:, 1, :])
                with tf.name_scope(name='logits') as sc:
                    logits_single = tf.contrib.layers.fully_connected(inputs=circ_cor, num_outputs=1, activation_fn=None, scope=sc)
                logits = tf.reshape(logits_single, shape=(-1, self.candidate_count))
            # use cosine similarity
            else:
                logits = tf.reshape(sim_cosine(final_vecs_split), shape=(-1, self.candidate_count))
        # assume, we get just root embeddings and score them
        else:
            # regression
            with tf.name_scope(name='fc_scoring') as sc:
                logits = tf.reshape(tf.contrib.layers.fully_connected(inputs=vecs_reshaped, activation_fn=None,
                                                                      num_outputs=1, scope=sc),
                                    shape=(-1, self.candidate_count))
            # just sum
            #logits = tf.reshape(tf.reduce_sum(vecs_reshaped, axis=-1), shape=(-1, self.candidate_count))
            # l2 norm
            #logits = tf.reshape(tf.sqrt(tf.reduce_sum(tf.square(vecs_reshaped), axis=-1)), shape=(-1, self.candidate_count))

        # use fully connected layer
        #batch_size = tf.shape(tree_embeddings)[0]
        #_logits = tf.contrib.layers.fully_connected(inputs=final_vecs, num_outputs=1, activation_fn=None)
        #logits = tf.reshape(_logits, shape=[batch_size, self.candidate_count])
        labels_gold_normed = self._labels_gold / tf.reduce_sum(self._labels_gold, axis=-1, keep_dims=True)
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels_gold_normed))
        self._probs = tf.nn.softmax(logits)
        self._scores = tf.sigmoid(logits)

        # m_ts = [0.1, 0.33, 0.5, 0.66, 0.9]
        # m_ts = [0.5]
        m_ts = [0.33, 0.5, 0.66]
        map_ts = lambda x: str(int(x * 100))  # format thresholds
        with tf.variable_scope("reset_metrics_scope") as scope:
            metrics = {  # 'roc': tf.metrics.auc(labels=labels_gold_dense, predictions=self.values_predicted),
                'precision:' + ','.join(map(map_ts, m_ts)): tf.metrics.precision_at_thresholds(
                    labels=labels_gold_normed, predictions=self.values_predicted, thresholds=m_ts),
                'recall:' + ','.join(map(map_ts, m_ts)): tf.metrics.recall_at_thresholds(
                    labels=labels_gold_normed, predictions=self.values_predicted, thresholds=m_ts),
            }
            for k in [1, 2, 3, 5]:
                if self.candidate_count > k:
                    metrics['recall@%i' % k] = tf.metrics.recall_at_k(
                        labels=tf.argmax(self._labels_gold, axis=1), predictions=self.values_predicted, k=k, class_id=0)

            vars = tf.contrib.framework.get_variables(scope, collection=tf.GraphKeys.LOCAL_VARIABLES)
            reset_op = tf.variables_initializer(vars)
        BaseTrainModel.__init__(
            self, tree_model=tree_model, loss=tf.reduce_mean(cross_entropy), metrics=metrics, metric_reset_op=reset_op,
            **kwargs)

    def _final_vecs(self, tree_embeddings, embedding_dim):
        raise NotImplementedError('implement this method')

    @property
    def values_gold(self):
        return self._labels_gold

    @property
    def values_predicted(self):
        return self._probs

    @property
    def scores(self):
        return self._scores

    @property
    def model_type(self):
        return MODEL_TYPE_DISCRETE

    @property
    def candidate_count(self):
        raise NotImplementedError('Implement this method')

    @property
    def candidate_indices(self):
        return self._candidate_indices


class TreeTupleModel_with_candidates(TreeScoringModel_with_candidates):
    def _final_vecs(self, tree_embeddings, embedding_dim):

        ref_tree_embedding = tree_embeddings[:, 0, :]
        candidate_tree_embeddings = tree_embeddings[:, 1:, :]
        ref_tree_embedding_tiled = tf.tile(ref_tree_embedding, multiples=[1, self.candidate_count])
        ref_tree_embedding_tiled_reshaped = tf.reshape(ref_tree_embedding_tiled,
                                                       shape=[-1, self.candidate_count, embedding_dim])
        concat = tf.concat([ref_tree_embedding_tiled_reshaped, candidate_tree_embeddings], axis=-1)
        return concat

    @property
    def candidate_count(self):
        return self.nbr_embeddings_in - 1


class TreeSingleModel_with_candidates(TreeScoringModel_with_candidates):
    def _final_vecs(self, tree_embeddings, embedding_dim):
        return tf.reshape(tree_embeddings, shape=[-1, self.candidate_count, embedding_dim])

    @property
    def candidate_count(self):
        return self.nbr_embeddings_in


class TreeMultiClassModel(BaseTrainModel):
    def __init__(self, tree_model, nbr_classes, nbr_embeddings_in=1, exclusive_classes=True, fc_sizes=1000,
                 use_circular_correlation=False, **kwargs):

        self._labels_gold = tf.sparse_placeholder(dtype=tf.float32)
        tree_embeddings = tf.reshape(tree_model.embeddings_all,
                                     shape=[-1, tree_model.tree_output_size * nbr_embeddings_in])
        final_vecs = tree_embeddings

        if not isinstance(fc_sizes, (list, tuple)):
            fc_sizes = [fc_sizes]

        # add multiple fc layers
        for s in fc_sizes:
            if s > 0:
                with tf.name_scope(name='fc') as sc:
                    fc = tf.contrib.layers.fully_connected(inputs=final_vecs, num_outputs=s, scope=sc)
                final_vecs = tf.nn.dropout(fc, keep_prob=tree_model.keep_prob)

        # add circular self correlation
        if use_circular_correlation:
            logger.debug('add circular self correlation')
            circ_cor = circular_correlation(final_vecs, final_vecs)
            final_vecs = tf.concat((final_vecs, circ_cor), axis=-1)

        with tf.name_scope(name='logits') as sc:
            logits = tf.contrib.layers.fully_connected(inputs=final_vecs, num_outputs=nbr_classes, activation_fn=None, scope=sc)
        labels_gold_dense = tf.sparse_tensor_to_dense(self._labels_gold)
        if exclusive_classes:
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                                                   labels=labels_gold_dense))
            self._probs = tf.nn.softmax(logits)
        else:
            cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                                                   labels=labels_gold_dense))
            self._probs = tf.sigmoid(logits)
        #m_ts = [0.1, 0.33, 0.5, 0.66, 0.9]
        #m_ts = [0.5]
        m_ts = [0.33, 0.5, 0.66]
        map_ts = lambda x: str(int(x*100))  # format thresholds
        with tf.variable_scope("reset_metrics_scope") as scope:
            metrics = {#'roc': tf.metrics.auc(labels=labels_gold_dense, predictions=self.values_predicted),
                       'precision:' + ','.join(map(map_ts, m_ts)): tf.metrics.precision_at_thresholds(
                           labels=labels_gold_dense, predictions=self.values_predicted, thresholds=m_ts),
                       'recall:' + ','.join(map(map_ts, m_ts)): tf.metrics.recall_at_thresholds(
                           labels=labels_gold_dense, predictions=self.values_predicted, thresholds=m_ts),
                       }
            for ts in m_ts:
                metrics['accuracy_t%i' % (ts*100)] = tf.metrics.accuracy(
                    labels=labels_gold_dense, predictions=tf.cast(self.values_predicted + 1.0 - ts, tf.int32))
            vars = tf.contrib.framework.get_variables(scope, collection=tf.GraphKeys.LOCAL_VARIABLES)
        reset_op = tf.variables_initializer(vars)
        BaseTrainModel.__init__(
            self, tree_model=tree_model, loss=tf.reduce_mean(cross_entropy), metrics=metrics, metric_reset_op=reset_op,
            nbr_embeddings_in=nbr_embeddings_in,
            **kwargs)

    @property
    def values_gold(self):
        return self._labels_gold

    @property
    def values_predicted(self):
        return self._probs

    @property
    def model_type(self):
        return MODEL_TYPE_DISCRETE


# not used
class ScoredSequenceTreeTupleModel(BaseTrainModel):
    """A Fold model for similarity scored sequence tree (SequenceNode) tuple."""

    def __init__(self, tree_model, probs_count=2, **kwargs):

        # TODO: check shapes! (is embeddings_shaped correct?)
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
        _prediction_logits = tf.reshape(tf.matmul(tree_model.embeddings_all, _weights) + _bias, shape=[-1])
        #loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=probs, logits=self._prediction_logits))
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tree_model.values_gold_shaped, logits=_prediction_logits))

        BaseTrainModel.__init__(self, tree_model=tree_model, loss=loss, **kwargs)


#not used
# test this!
class SequenceTreeRerootModel(BaseTrainModel):

    def __init__(self, tree_model, candidate_count, fc_sizes=1000, **kwargs):

        # unpack tree embeddings
        tree_embeddings = tf.reshape(tree_model.embeddings_all, shape=[-1, tree_model.tree_output_size])
        #batch_size = tf.shape(tree_model.embeddings_shaped)[0] // (tree_model.neg_samples + 1)

        # create labels_gold: first entry is the correct one
        #labels_gold_batched = tf.zeros(shape=(batch_size, tree_model.neg_samples + 1), dtype=tf.int32)
        #labels_gold_batched[:, 0] = 1
        # flatten labels_gold
        #self._labels_gold = tf.reshape(labels_gold_batched, shape=[batch_size * (tree_model.neg_samples + 1)])

        # unpack (flatten) labels_gold
        #self._labels_gold = tf.reshape(tree_model.values_gold, shape=[tf.shape(tree_embeddings)[0]])

        self._labels_gold = tf.placeholder(dtype=tf.int32, shape=[None, candidate_count])

        if not isinstance(fc_sizes, (list, tuple)):
            fc_sizes = [fc_sizes]

        # add multiple fc layers
        for s in fc_sizes:
            if s > 0:
                fc = tf.contrib.layers.fully_connected(inputs=tree_embeddings, num_outputs=s)
                tree_embeddings = tf.nn.dropout(fc, keep_prob=tree_model.keep_prob)

        #fc = tf.contrib.layers.fully_connected(inputs=tree_embeddings, num_outputs=1000)
        logits = tf.contrib.layers.fully_connected(inputs=tree_embeddings, num_outputs=2, activation_fn=None)
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

    def __init__(self, number_of_embeddings, embedding_size):
        #self._sparse = sparse
        #self._normed_reference_embedding = tf.placeholder(tf.float32, [embedding_size])
        self._reference_idx = tf.placeholder(tf.int32, shape=[])
        #self._number_of_embeddings = tf.placeholder(tf.int32, shape=[])
        #self._normed_embeddings = tf.placeholder(tf.float32, [None, embedding_size])
        self._normed_embeddings = tf.Variable(tf.constant(0.0, shape=[number_of_embeddings, embedding_size]),
                                              trainable=False, name='EMBEDDED_DOCS')
        #if sparse:
        #    self._normed_embeddings_placeholder = tf.sparse_placeholder(shape=[number_of_embeddings, embedding_size], dtype=tf.float32)
        #    #tf.sparse_tensor_to_dense(self._embeddings_placeholder, validate_indices=False)
        #    self._normed_embeddings_init = self._normed_embeddings.assign(
        #        tf.sparse_tensor_to_dense(self._normed_embeddings_placeholder, validate_indices=False))
        #else:
        self._normed_embeddings_placeholder = tf.placeholder(shape=[number_of_embeddings, embedding_size], dtype=tf.float32)
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

    #@property
    #def sparse(self):
    #    return self._sparse
