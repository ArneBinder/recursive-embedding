import tensorflow as tf
import os
import tensorflow_fold as td
import spacy
import pickle
import preprocessing
import pprint

tf.flags.DEFINE_string('frozen_model_filename', '/home/arne/tmp/tf/log/frozen_model.pb',
                       'Frozen model file to import')
#output_gathers/float32_300__TensorFlowFoldOutputTag_0
tf.flags.DEFINE_string('frozen_model_out_op', 'output_gathers/float32_300__TensorFlowFoldOutputTag_0',
                       'The output op of the frozen model.')
tf.flags.DEFINE_string('data_mapping_path', 'data/nlp/spacy/dict.mapping',
                       'model file')

FLAGS = tf.flags.FLAGS

PROTO_PACKAGE_NAME = 'recursive_dependency_embedding'
PROTO_CLASS = 'SequenceNode'

DEFAULT_PREFIX = 'prefix'

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we can use again a convenient built-in function to import a graph_def into the
    # current default Graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name=DEFAULT_PREFIX,
            op_dict=None,
            producer_op_list=None
        )
    return graph


def parse_iterator(sequences, parser, sentence_processor, data_maps):
    pp = pprint.PrettyPrinter(indent=2)
    for s in sequences:
        seq_tree = preprocessing.build_sequence_tree_from_str(s, sentence_processor, parser, data_maps)
        pp.pprint(seq_tree)
        yield seq_tree.SerializeToString()


if __name__ == '__main__':
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    td.proto_tools.map_proto_source_tree_path('', ROOT_DIR)
    td.proto_tools.import_proto_file('sequence_node.proto')

    print('load spacy ...')
    nlp = spacy.load('en')
    nlp.pipeline = [nlp.tagger, nlp.parser]
    print('load data_mapping from: '+FLAGS.data_mapping_path + ' ...')
    data_maps = pickle.load(open(FLAGS.data_mapping_path, "rb"))

    # We use our "load_graph" function
    graph = load_graph(FLAGS.frozen_model_filename)

    # We can verify that we can access the list of operations in the graph
    for op in graph.get_operations():
        print(op.name)
        # prefix/Placeholder/inputs_placeholder
        # ...
        # prefix/Accuracy/predictions

    # We access the input and output nodes
    x = graph.get_tensor_by_name('prefix/DeserializingWeaver:0')
    y = graph.get_tensor_by_name(DEFAULT_PREFIX+'/'+FLAGS.frozen_model_out_op+':0')

    # We launch a Session
    with tf.Session(graph=graph) as sess:
        # Note: we didn't initialize/restore anything, everything is stored in the graph_def
        batch = list(parse_iterator(['Hallo.', 'Hallo!', 'Hallo?', 'Hallo'],
                                    nlp, preprocessing.process_sentence3, data_maps))
        #fdict = embedder.build_feed_dict(batch)

        y_out = sess.run(y, feed_dict={
            x: list(parse_iterator(['Hallo!'], nlp, preprocessing.process_sentence3, data_maps))  # < 45
        })
    print(y_out)  # [[ False ]] Yay, it works!