from __future__ import print_function
import tensorflow as tf
import model_fold
from tensorflow.python.framework import graph_util

tf.flags.DEFINE_string('model_dir', '/home/arne/tmp/tf/log',
                       'model file')
FLAGS = tf.flags.FLAGS

lex_size = 1300000
embedding_dim = 300


def main(unused_argv):
    # We retrieve our checkpoint fullpath
    checkpoint = tf.train.get_checkpoint_state(FLAGS.model_dir)
    input_checkpoint = checkpoint.model_checkpoint_path

    # We precise the file fullname of our freezed graph
    absolute_model_folder = "/".join(input_checkpoint.split('/')[:-1])
    output_graph = absolute_model_folder + "/frozen_model.pb"

    with tf.Graph().as_default():
        embed_w = tf.Variable(tf.constant(0.0, shape=[lex_size, embedding_dim]),
                              trainable=True, name='embeddings')
        embedder = model_fold.SequenceTreeEmbedding(embed_w)
        tree_embeddings = embedder.tree_embeddings

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()

        # We retrieve the protobuf graph definition
        graph = tf.get_default_graph()
        input_graph_def = graph.as_graph_def()

        output_node_names = tree_embeddings[0]._op.name # 'output_gathers/float32_300'

        # Later, launch the model, use the saver to restore variables from disk, and
        # do some work with the model.
        with tf.Session() as sess:
            # Restore variables from disk.
            print('restore model ...')
            saver.restore(sess, input_checkpoint)
            # We use a built-in TF helper to export variables to constants
            output_graph_def = graph_util.convert_variables_to_constants(
                sess,  # The session is used to retrieve the weights
                input_graph_def,  # The graph_def is used to retrieve the nodes
                output_node_names.split(",")  # The output node names are used to select the usefull nodes
            )

            # Finally we serialize and dump the output graph to the filesystem
            with tf.gfile.GFile(output_graph, "wb") as f:
                f.write(output_graph_def.SerializeToString())
            print("%d ops in the final graph." % len(output_graph_def.node))


if __name__ == '__main__':
    tf.app.run()
