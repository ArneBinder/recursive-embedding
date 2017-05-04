
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# import google3
import tensorflow as tf
import tensorflow_fold.public.blocks as td

class SequenceModel(object):
    """A Fold model for calculator examples."""

    # TODO: pass embeddings to model (use convert_to_block?)
    def __init__(self, state_size, embeddings):
        expr_decl = td.ForwardDeclaration(td.PyObjectType(), state_size)

        # get the head embedding from id
        def head(name_):
            return td.Pipe(td.Scalar(dtype='int32'),
                           td.Function(embeddings), #td.Embedding(lex_size, state_size, initializer = embeddings, name='head_embed')
                           name=name_)

        # get the weighted sum of all children
        def children_aggr(name_):
            return td.Pipe(td.Map(expr_decl()),
                           td.Map(td.Function(lambda x: tf.norm(x) * x)),
                           td.Reduce(td.Function(tf.add)),
                           name=name_)

        # gru_cell = td.ScopedLayer(tf.contrib.rnn.GRUCell(num_units=state_size), 'mygru')

        # TODO: use GRU cell
        def aggr_op():
            # return (td.AllOf(head, children_aggr) >> td.RNN(gru_cell, initial_state_from_input=True))
            return (children_aggr)

        def cas(seq_tree):
            # process and aggregate
            if len(seq_tree['children']) > 0 and seq_tree['head'] is not None:
                return 0
            # dont process children
            if len(seq_tree['children']) == 0:
                return 1
            # process children only
            return 2

        cases = td.OneOf(lambda x: cas(x),
                         {1: td.Record([('head', head('head')),
                                        ('children', children_aggr('children_aggr'))]) >> td.Concat() >> td.FC(
                             state_size),
                          2: td.GetItem('head') >> td.Optional(head('just_head')),
                          3: td.GetItem('children') >> children_aggr('just_children')})


        #tree = td.InputTransform(preprocess_tree) >> cases
        #expr_decl.resolve_to(tree)
        expr_decl.resolve_to(cases)

        # Get logits from the root of the expression tree
        #expression_logits = (expression >>
        #                     td.FC(NUM_LABELS, activation=None, name='FC_logits'))
        #tree_embedding = cases

        # The result is stored in the expression itself.
        # We ignore it in td.Record above, and pull it out here.
        #expression_label = (td.GetItem('result') >>
        #                    td.InputTransform(result_sign) >>
        #                    td.OneHot(NUM_LABELS))

        # For the overall model, return a pair of (logits, labels)
        # The AllOf block will run each of its children on the same input.
        #model = td.AllOf(expression_logits, expression_label)
        self._compiler = td.Compiler.create(cases)

        # Get the tensorflow tensors that correspond to the outputs of model.
        # `logits` and `labels` are TF tensors, and we can use them to
        # compute losses in the usual way.
        #(logits, labels) = self._compiler.output_tensors
        tree_embeddings = self._compiler.output_tensors

        #self._loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        #    logits=logits, labels=labels))

        #self._accuracy = tf.reduce_mean(
        #    tf.cast(tf.equal(tf.argmax(labels, 1),
        #                     tf.argmax(logits, 1)),
        #            dtype=tf.float32))

        #self._global_step = tf.Variable(0, name='global_step', trainable=False)
        #optr = tf.train.GradientDescentOptimizer(0.01)
        #self._train_op = optr.minimize(self._loss, global_step=self._global_step)

    #@property
    #def loss(self):
    #    return self._loss

    #@property
    #def accuracy(self):
    #    return self._accuracy

    #@property
    #def train_op(self):
    #    return self._train_op

    #@property
    #def global_step(self):
    #    return self._global_step

    def build_feed_dict(self, sequence_trees):
        return self._compiler.build_feed_dict(sequence_trees)