
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# import google3
import tensorflow as tf
import tensorflow_fold.public.blocks as td

class SequenceModel(object):
    """A Fold model for calculator examples."""

    # TODO: pass embeddings to model (use convert_to_block?)
    def __init__(self, state_size, lex_size):
        expr_decl = td.ForwardDeclaration(td.PyObjectType(), state_size)

        # get the head embedding from id
        head = td.GetItem('head') >> td.Scalar(dtype='int32') >> td.Function(
            td.Embedding(10, state_size, name='head_embed'))
        # get the weighted sum of all children
        children_aggr = td.GetItem('children') >> td.Map(expr_decl()) >> td.Map(
            td.Function(lambda x: tf.norm(x) * x)) >> td.Reduce(td.Function(tf.add))

        # gru_cell = td.ScopedLayer(tf.contrib.rnn.GRUCell(num_units=state_size), 'mygru')

        def aggr_op():
            # return (td.AllOf(head, children_aggr) >> td.RNN(gru_cell, initial_state_from_input=True))
            return (children_aggr)

        cases = td.OneOf(lambda x: len(x['children']) == 0,
                         {True: head,
                          False: aggr_op()})

        expr_decl.resolve_to(cases)

        # Get logits from the root of the expression tree
        expression_logits = (expression >>
                             td.FC(NUM_LABELS, activation=None, name='FC_logits'))

        # The result is stored in the expression itself.
        # We ignore it in td.Record above, and pull it out here.
        expression_label = (td.GetItem('result') >>
                            td.InputTransform(result_sign) >>
                            td.OneHot(NUM_LABELS))

        # For the overall model, return a pair of (logits, labels)
        # The AllOf block will run each of its children on the same input.
        model = td.AllOf(expression_logits, expression_label)
        self._compiler = td.Compiler.create(model)

        # Get the tensorflow tensors that correspond to the outputs of model.
        # `logits` and `labels` are TF tensors, and we can use them to
        # compute losses in the usual way.
        (logits, labels) = self._compiler.output_tensors

        self._loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=labels))

        self._accuracy = tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(labels, 1),
                             tf.argmax(logits, 1)),
                    dtype=tf.float32))

        self._global_step = tf.Variable(0, name='global_step', trainable=False)
        optr = tf.train.GradientDescentOptimizer(0.01)
        self._train_op = optr.minimize(self._loss, global_step=self._global_step)

    @property
    def loss(self):
        return self._loss

    @property
    def accuracy(self):
        return self._accuracy

    @property
    def train_op(self):
        return self._train_op

    @property
    def global_step(self):
        return self._global_step

    def build_feed_dict(self, expressions):
        return self._compiler.build_feed_dict(expressions)