from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# import google3
import tensorflow as tf
import tensorflow_fold.public.blocks as td
from tensorflow_fold.blocks import result_types as tdt
from tensorflow_fold.blocks import blocks as tdb
import numpy as np
import logging

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


def create_lexicon(lex_size, trainable=True):
    lexicon = tf.Variable(tf.constant(0.0, shape=[lex_size, DIMENSION_EMBEDDINGS]),
                          trainable=trainable, name=VAR_NAME_LEXICON)
    lexicon_placeholder = tf.placeholder(tf.float32, [lex_size, DIMENSION_EMBEDDINGS])
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


def Attention(name=None):  # pylint: disable=invalid-name
    """
    This block calculates attention over a sequence.
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


class TreeEmbedding(object):
    def __init__(self, name, lex_size_fix, lex_size_var, keep_prob_placeholder=None, state_size=None,
                 leaf_fc_size=0, root_fc_size=0, keep_prob_fixed=1.0):
        self._lex_size_fix = lex_size_fix
        self._lex_size_var = lex_size_var

        # compatibility
        if self._lex_size_fix > 0:
            self._lexicon_fix, self._lexicon_fix_placeholder, self._lexicon_fix_init = create_lexicon(lex_size=self._lex_size_fix,
                                                                                                      trainable=False)
        if self._lex_size_var > 0:
            self._lexicon_var, self._lexicon_var_placeholder, self._lexicon_var_init = create_lexicon(lex_size=self._lex_size_var,
                                                                                                      trainable=True)
        self._keep_prob_fixed = keep_prob_fixed
        if keep_prob_placeholder is not None:
            self._keep_prob = keep_prob_placeholder
        else:
            self._keep_prob = 1.0
        if state_size:
            self._state_size = state_size
        else:
            self._state_size = DIMENSION_EMBEDDINGS  # state_size
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
                                        False: td.Void() >> td.Zeros(DIMENSION_EMBEDDINGS)})

    def embed(self):
        # get the head embedding from id
        if self._lex_size_fix > 0 and self._lex_size_var > 0:
            return td.OneOf(key_fn=(lambda x: x >= 0),
                            case_blocks={True: td.Scalar(dtype='int32') >> td.Function(lambda x: tf.gather(self._lexicon_var, x)),
                                         False: td.Scalar(dtype='int32') >> td.Function(lambda x: tf.gather(self._lexicon_fix, tf.abs(x)))})
        # compatibility
        if self._lex_size_fix > 0:
            return td.Scalar(dtype='int32') >> td.Function(lambda x: tf.gather(self._lexicon_fix, tf.abs(x)))
        if self._lex_size_var > 0:
            return td.Scalar(dtype='int32') >> td.Function(lambda x: tf.gather(self._lexicon_var, x))

    def head(self, name='head_embed'):
        #return td.Pipe(td.GetItem('head'), td.Scalar(dtype='int32'), self.embed(), name=name)
        return td.Pipe(td.GetItem('head'), self.embed(), self.leaf_fc, name=name)

    def children(self, name='children'):
        return td.InputTransform(lambda x: x.get('children', []), name=name)

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
        return self._root_fc_size or 0

    @property
    def root_fc(self):
        return self._root_fc

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
    def keep_prob(self):
        return self._keep_prob

    @property
    def keep_prob_fixed(self):
        return self._keep_prob_fixed


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


class TreeEmbedding_RNN(TreeEmbedding):
    def __init__(self, input_size=None, **kwargs):
        super(TreeEmbedding_RNN, self).__init__(**kwargs)
        if input_size is None:
            input_size = self.leaf_fc_size or DIMENSION_EMBEDDINGS
        self._rnn_input_size = input_size

    @property
    def rnn_input_size(self):
        return self._rnn_input_size

    @property
    def rnn_step(self):
        raise NotImplementedError

    @property
    def rnn_cell(self):
        raise NotImplementedError

    @property
    def rnn_reduce(self):
        raise NotImplementedError


class TreeEmbedding_GRU(TreeEmbedding_RNN):
    def __init__(self, **kwargs):
        super(TreeEmbedding_GRU, self).__init__(**kwargs)
        with tf.variable_scope(self.scope):
            self._grucell = td.ScopedLayer(tf.contrib.rnn.DropoutWrapper(
                tf.contrib.rnn.GRUCell(num_units=self._state_size),
                input_keep_prob=self.keep_prob,
                output_keep_prob=self.keep_prob,
                variational_recurrent=True,
                dtype=tf.float32,
                input_size=self.rnn_input_size),
                'gru_cell')

    @property
    def rnn_step(self):
        return self._grucell >> td.GetItem(1)

    @property
    def rnn_cell(self):
        return self._grucell

    @property
    def rnn_reduce(self):
        return td.Pipe(td.RNN(self._grucell), td.GetItem(1))


class TreeEmbedding_LSTM(TreeEmbedding_RNN):
    def __init__(self, **kwargs):
        super(TreeEmbedding_LSTM, self).__init__(**kwargs)
        with tf.variable_scope(self.scope):
            self._lstm_cell = td.ScopedLayer(
                tf.contrib.rnn.DropoutWrapper(
                    tf.contrib.rnn.BasicLSTMCell(num_units=self.state_size, forget_bias=2.5),#, state_is_tuple=False),
                    input_keep_prob=self.keep_prob,
                    output_keep_prob=self.keep_prob,
                    variational_recurrent=True,
                    dtype=tf.float32,
                    input_size=self.rnn_input_size),
                'lstm_cell')

    @property
    def rnn_step(self):
        #return self._lstm_cell >> td.GetItem(1)    # requires: state_is_tuple=False
        return td.AllOf(td.GetItem(0), td.GetItem(1)
                        >> td.Function(lambda v: tf.split(value=v, num_or_size_splits=2, axis=1))) \
               >> self._lstm_cell >> td.GetItem(1) >> td.Concat()

    @property
    def rnn_cell(self):
        return self._lstm_cell

    # TODO: use both states as output: return td.Pipe(td.RNN(self._lstm_cell), td.GetItem(1), td.Concat)
    @property
    def rnn_reduce(self):
        #return td.Pipe(td.RNN(self._lstm_cell), td.GetItem(1))
        return td.Pipe(td.RNN(self._lstm_cell), td.GetItem(1), td.GetItem(0))


class TreeEmbedding_FC(TreeEmbedding_RNN):
    def __init__(self, **kwargs):
        super(TreeEmbedding_FC, self).__init__(**kwargs)
        self._fc = td.FC(self.state_size, activation=tf.nn.tanh, input_keep_prob=self.keep_prob, name='fc_cell')

    @property
    def rnn_step(self):
        return td.Concat() >> self._fc

    #@property
    #def rnn_cell(self):
    #    return self._grucell

    #@property
    #def rnn_reduce(self):
    #    return td.Pipe(td.RNN(self._grucell), td.GetItem(1))


class TreeEmbedding_HTU(TreeEmbedding_RNN):
    """Calculates an embedding over a (recursive) SequenceNode."""

    def __init__(self, name, **kwargs):
        super(TreeEmbedding_HTU, self).__init__(name='HTU_' + name, **kwargs)

    def __call__(self):
        embed_tree = td.ForwardDeclaration(input_type=td.PyObjectType(), output_type=self.rnn_step.output_type)
        # simplified naive version (minor modification: apply order_aware also to single head with zeros as input state)
        children = self.children() >> td.Map(embed_tree()) >> td.Sum()
        #head = self.head() >> self.leaf_fc
        cases = td.AllOf(self.head(), children) >> self.rnn_step
        embed_tree.resolve_to(cases)
        model = cases >> self.root_fc
        return model


class TreeEmbedding_HTU_GRU(TreeEmbedding_HTU, TreeEmbedding_GRU):
    def __init__(self, **kwargs):
        super(TreeEmbedding_HTU_GRU, self).__init__(name='HTU', **kwargs)


class TreeEmbedding_HTU_LSTM(TreeEmbedding_HTU, TreeEmbedding_LSTM):
    def __init__(self, **kwargs):
        super(TreeEmbedding_HTU_LSTM, self).__init__(name='LSTM', **kwargs)


class TreeEmbedding_HTU_FC(TreeEmbedding_HTU, TreeEmbedding_FC):
    def __init__(self, **kwargs):
        super(TreeEmbedding_HTU_FC, self).__init__(name='FC', **kwargs)


class TreeEmbedding_HTU_rev(TreeEmbedding_RNN):
    """Calculates an embedding over a (recursive) SequenceNode."""

    def __init__(self, name, **kwargs):
        super(TreeEmbedding_HTU_rev, self).__init__(name='HTU_rev_' + name, **kwargs)

    def __call__(self):
        embed_tree = td.ForwardDeclaration(input_type=td.PyObjectType(), output_type=self.rnn_step.output_type)
        children = self.children() >> td.Map(embed_tree())
        cases = td.AllOf(self.head() >> td.Broadcast(), children) \
                >> td.Zip() >> td.Map(self.rnn_step) >> td.Sum()
        embed_tree.resolve_to(cases)
        model = cases >> self.root_fc
        return model


class TreeEmbedding_HTU_rev_GRU(TreeEmbedding_HTU_rev, TreeEmbedding_GRU):
    def __init__(self, **kwargs):
        super(TreeEmbedding_HTU_rev_GRU, self).__init__(name='GRU', **kwargs)


class TreeEmbedding_HTU_dep(TreeEmbedding_RNN):
    """Calculates an embedding over a (recursive) SequenceNode."""

    def __init__(self, name, **kwargs):
        super(TreeEmbedding_HTU_dep, self).__init__(name='HTU_dep_' + name, **kwargs)

    def __call__(self):
        embed_tree = td.ForwardDeclaration(input_type=td.PyObjectType(), output_type=self.rnn_step.output_type)

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
                         {0: td.AllOf(self.head(), children()) >> self.rnn_step,
                          1: self.head('head_only'),
                          2: children('children_only'),
                          })

        embed_tree.resolve_to(cases)

        model = cases >> self.root_fc
        return model


class TreeEmbedding_HTU_dep_GRU(TreeEmbedding_HTU_dep, TreeEmbedding_GRU):
    def __init__(self, **kwargs):
        super(TreeEmbedding_HTU_dep_GRU, self).__init__(name='GRU', **kwargs)


class TreeEmbedding_HTU_ATT(TreeEmbedding_RNN):
    """Calculates an embedding over a (recursive) SequenceNode."""

    def __init__(self, name, **kwargs):
        super(TreeEmbedding_HTU_ATT, self).__init__(name='HTU_ATT_' + name, **kwargs)

    def __call__(self):
        embed_tree = td.ForwardDeclaration(input_type=td.PyObjectType(), output_type=self.rnn_step.output_type)

        children = self.children() >> td.Map(embed_tree())
        #head = self.head() >> self.leaf_fc
        c = td.Composition()
        with c.scope():
            head_att = td.Pipe(td.FC(self.state_size, activation=tf.nn.tanh, input_keep_prob=self.keep_prob,
                                     name='fc_attention'),
                               name='att_pipe').reads(c.input[0])
            head_gru = td.Pipe(td.FC(self.leaf_fc_size or DIMENSION_EMBEDDINGS, activation=tf.nn.tanh,
                                     input_keep_prob=self.keep_prob, name='fc_gru'),
                               name='gru_pipe').reads(c.input[0])
            children_attention = Attention().reads(c.input[1], head_att)
            gru_out = self.rnn_step.reads(head_gru, children_attention)
            c.output.reads(gru_out)

        cases = td.AllOf(self.head(), children) >> c
        embed_tree.resolve_to(cases)
        model = cases >> self.root_fc
        return model


class TreeEmbedding_HTU_ATT_GRU(TreeEmbedding_HTU_ATT, TreeEmbedding_GRU):
    def __init__(self, **kwargs):
        super(TreeEmbedding_HTU_ATT_GRU, self).__init__(name='GRU', **kwargs)


class TreeEmbedding_HTU_ATT_split(TreeEmbedding_RNN):
    """Calculates an embedding over a (recursive) SequenceNode."""

    def __init__(self, name, **kwargs):
        super(TreeEmbedding_HTU_ATT_split, self).__init__(name='HTU_ATT_split_' + name, **kwargs)

    def __call__(self):
        embed_tree = td.ForwardDeclaration(input_type=td.PyObjectType(), output_type=self.state_size)

        children = self.children() >> td.Map(embed_tree())

        c = td.Composition()
        with c.scope():
            head_split = td.Function(lambda v: tf.split(value=v, num_or_size_splits=2, axis=1)).reads(c.input[0])
            head_att = td.Pipe(td.FC(self.state_size, activation=tf.nn.tanh, input_keep_prob=self.keep_prob,
                                     name='fc_attention'),
                               name='att_pipe').reads(head_split[0])
            head_gru = td.Pipe(td.FC(self.leaf_fc_size or DIMENSION_EMBEDDINGS, activation=tf.nn.tanh,
                                     input_keep_prob=self.keep_prob, name='fc_rnn'),
                               name='rnn_pipe').reads(head_split[1])
            children_attention = Attention().reads(c.input[1], head_att)
            rnn_out = self.rnn_step.reads(head_gru, children_attention)
            c.output.reads(rnn_out)

        cases = td.AllOf(self.head(), children) >> c
        embed_tree.resolve_to(cases)
        model = cases >> self.root_fc
        return model


class TreeEmbedding_HTU_ATT_split_GRU(TreeEmbedding_HTU_ATT_split, TreeEmbedding_GRU):
    def __init__(self, **kwargs):
        super(TreeEmbedding_HTU_ATT_split_GRU, self).__init__(name='GRU', **kwargs)


class TreeEmbedding_FLAT(TreeEmbedding):
    def __init__(self, name, **kwargs):
        super(TreeEmbedding_FLAT, self).__init__(name='FLAT_' + name, **kwargs)

    def element(self, name='element'):
        return td.Pipe(self.head(), name=name)

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

    def children(self, name='children'):
        # return only children that have at least one child themselves
        def get_children(x):
            if 'children' not in x:
                return []
            res = [c for c in x['children'] if 'children' in c and len(c['children']) > 0]
            #if len(res) != len(x['children']):
                # warn, if children have been removed
                #logging.warning('removed children: %i' % (len(x['children']) - len(res)))
            return res
        return td.InputTransform(get_children, name=name)

    def element(self, name='element'):
        # use word embedding and first child embedding
        return td.Pipe(td.AllOf(td.Pipe(td.GetItem('head'), self.embed(), name='head_level1'), # self.head(name='head_level1'),
                                td.GetItem('children') >> td.InputTransform(lambda s: s[0])
                                >> td.Pipe(td.GetItem('head'), self.embed(), name='head_level2')), # self.head(name='head_level2')),
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


class TreeEmbedding_FLAT_LSTM(TreeEmbedding_FLAT, TreeEmbedding_LSTM):
    def __init__(self, name=None, input_size=None, leaf_fc_size=0, **kwargs):
        super(TreeEmbedding_FLAT_LSTM, self).__init__(name=name or 'LSTM',
                                                      input_size=input_size or leaf_fc_size or DIMENSION_EMBEDDINGS,
                                                      leaf_fc_size=leaf_fc_size, **kwargs)

    def aggregate(self, name='aggregate'):
        # apply LSTM >> take the LSTM output state(s) >> take the h state (discard the c state)
        #return td.Pipe(td.RNN(self._lstm_cell), td.GetItem(1), td.GetItem(0), name=name)
        return self.rnn_reduce

    @property
    def output_size(self):
        return self.root_fc_size or self.state_size


# compatibility
class TreeEmbedding_FLAT_LSTM50(TreeEmbedding_FLAT_LSTM):
    def __init__(self, **kwargs):
        super(TreeEmbedding_FLAT_LSTM50, self).__init__(name='LSTM50', **kwargs)


class TreeEmbedding_FLAT_LSTM_2levels(TreeEmbedding_FLAT_LSTM, TreeEmbedding_FLAT_2levels):
    def __init__(self, name=None, leaf_fc_size=0, **kwargs):
        super(TreeEmbedding_FLAT_LSTM, self).__init__(name=name or 'LSTM_2levels',
                                                      input_size=leaf_fc_size or DIMENSION_EMBEDDINGS * 2,
                                                      leaf_fc_size=leaf_fc_size, **kwargs
                                                      )


# compatibility
class TreeEmbedding_FLAT_LSTM50_2levels(TreeEmbedding_FLAT_LSTM_2levels):
    def __init__(self, **kwargs):
        super(TreeEmbedding_FLAT_LSTM50_2levels, self).__init__(name='LSTM50_2levels', **kwargs)


class TreeEmbedding_FLAT_GRU(TreeEmbedding_FLAT, TreeEmbedding_GRU):
    def __init__(self, name=None, input_size=None, leaf_fc_size=0, **kwargs):
        super(TreeEmbedding_FLAT_GRU, self).__init__(name=name or 'GRU',
                                                     input_size=input_size or leaf_fc_size or DIMENSION_EMBEDDINGS,
                                                     leaf_fc_size=leaf_fc_size, **kwargs)

    def aggregate(self, name='aggregate'):
        return self.rnn_reduce


class TreeEmbedding_FLAT_GRU_2levels(TreeEmbedding_FLAT_GRU, TreeEmbedding_FLAT_2levels):
    def __init__(self, name=None, leaf_fc_size=0, **kwargs):
        super(TreeEmbedding_FLAT_GRU_2levels, self).__init__(name=name or 'GRU_2levels',
                                                             input_size=leaf_fc_size or DIMENSION_EMBEDDINGS * 2,
                                                             leaf_fc_size=leaf_fc_size, **kwargs
                                                             )


def sim_cosine_DEP(e1, e2):
    e1 = tf.nn.l2_normalize(e1, dim=1)
    e2 = tf.nn.l2_normalize(e2, dim=1)
    return tf.reduce_sum(e1 * e2, axis=1)


def sim_cosine(embeddings):
    es_norm = tf.nn.l2_normalize(embeddings, dim=-1)
    return tf.reduce_sum(es_norm[:, 0, :] * es_norm[:, 1, :], axis=-1)


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
    def __init__(self, tree_embedder=TreeEmbedding_TREE_LSTM, keep_prob=1.0, tree_count=2, prob_count=None, **kwargs):
        if prob_count is None:
            self._prob_count = tree_count
        else:
            self._prob_count = prob_count

        self._tree_count = tree_count
        self._keep_prob = tf.placeholder_with_default(keep_prob, shape=())

        self._tree_embed = tree_embedder(keep_prob_placeholder=self._keep_prob, **kwargs)

        embed_tree = self._tree_embed()
        model = td.AllOf(td.GetItem(0) >> td.Map(embed_tree) >> SequenceToTuple(embed_tree.output_type, self._tree_count) >> td.Concat(),
                         td.GetItem(1) >> td.Vector(self._prob_count))

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
    def probs_gold(self):
        return self._probs_gold

    @property
    def probs_gold_shaped(self):
        return tf.reshape(self._probs_gold, shape=[-1])

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
        #self._scores_gold = tree_model.probs_gold[:, 1]
        self._scores_gold = tf.reshape(tree_model.probs_gold_shaped, shape=[-1, 2])[:, 1]

        tree_size = tree_model.tree_output_size
        # get first two tree embeddings
        #self._tree_embeddings_1 = tree_model.embeddings_all[:, :tree_size]
        #self._tree_embeddings_2 = tree_model.embeddings_all[:, tree_size:tree_size * 2]
        #self._scores = sim_measure(e1=self._tree_embeddings_1, e2=self._tree_embeddings_2)

        self._tree_embeddings_reshaped = tf.reshape(tree_model.embeddings_shaped, shape=[-1, 2, tree_size])
        self._scores = sim_measure(self._tree_embeddings_reshaped)

        BaseTrainModel.__init__(self, tree_model=tree_model,
                                loss=tf.reduce_mean(tf.square(self._scores - self._scores_gold)), **kwargs)

    @property
    def scores_gold(self):
        return self._scores_gold

    @property
    def scores(self):
        return self._scores


class ScoredSequenceTreeTupleModel(BaseTrainModel):
    """A Fold model for similarity scored sequence tree (SequenceNode) tuple."""

    def __init__(self, tree_model, probs_count=2, **kwargs):

        self._prediction_logits = tf.contrib.layers.fully_connected(tree_model.embeddings_all, probs_count,
                                                                    activation_fn=None, scope=DEFAULT_SCOPE_SCORING)
        loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tree_model.probs_gold, logits=self._prediction_logits))

        BaseTrainModel.__init__(self, tree_model=tree_model, loss=loss, **kwargs)


class ScoredSequenceTreeTupleModel_independent(BaseTrainModel):
    """A Fold model for similarity scored sequence tree (SequenceNode) tuple."""

    def __init__(self, tree_model, count=None, **kwargs):
        if count is None:
            count = tree_model.tree_count
        assert tree_model.prob_count >= count, 'tree_model produces %i prob values per batch entry, but count=%i ' \
                                                'requested' % (tree_model.prob_count, count)
        assert tree_model.tree_count >= count, 'tree_model produces %i tree embeddings per batch entry, but count=%i ' \
                                               'requested' % (tree_model.tree_count, count)
        # cut inputs to 'count'
        probs = tree_model.probs_gold[:, :count]
        trees = tree_model.embeddings_all[:, :count * tree_model.tree_output_size]
        input_layer = tf.reshape(trees, [-1, count, tree_model.tree_output_size, 1])

        conv = tf.layers.conv2d(inputs=input_layer, filters=1,
                                kernel_size=[1, tree_model.tree_output_size], activation=None,
                                name=DEFAULT_SCOPE_SCORING)
        self._prediction_logits = tf.reshape(conv, shape=[-1, count])
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=probs, logits=self._prediction_logits))

        BaseTrainModel.__init__(self, tree_model=tree_model, loss=loss, **kwargs)


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
