from __future__ import print_function

import datetime
import fnmatch

import tensorflow as tf
import tensorflow_fold as td

import corpus
import model_fold
import preprocessing
import pprint
import os
import sequence_node_sequence_pb2
import sequence_node_pb2
import numpy as np
import json
import shutil
import logging
import sys
from tensorflow.contrib.tensorboard.plugins import projector


tf.flags.DEFINE_string('logdir', '/home/arne/ML_local/tf/log',  # '/home/arne/tmp/tf/log',
                       'Directory in which to write event logs and model checkpoints.')
tf.flags.DEFINE_string('train_data_path',
                       '/media/arne/WIN/Users/Arne/ML/data/corpora/wikipedia/process_sentence8/WIKIPEDIA_articles10000_offset0',
                       # '/home/arne/tmp/tf/log/model.ckpt-976',
                       'train data base path (without extension)')
# tf.flags.DEFINE_string('data_mapping_path', 'data/nlp/spacy/dict.mapping',
#                       'model file')
# tf.flags.DEFINE_string('train_dict_file', 'data/nlp/spacy/dict.vec',
#                       'A file containing a numpy array which is used to initialize the embedding vectors.')
# tf.flags.DEFINE_integer('pad_embeddings_to_size', 1310000,
#                        'The initial GloVe embedding matrix loaded from spaCy is padded to hold unknown lexical ids '
#                        '(dependency edge types, pos tag types, or any other type added by the sentence_processor to '
#                        'mark identity). This value has to be larger then the initial gloVe size ()')
tf.flags.DEFINE_integer('max_depth', 3,
                        'The maximal depth of the sequence trees.')
tf.flags.DEFINE_integer('sample_count', 15,
                        'The amount of generated samples per correct sequence tree.')
tf.flags.DEFINE_integer('batch_size', 250,  # 1000,
                        'How many samples to read per batch.')
tf.flags.DEFINE_integer('max_steps', 180000,  # 5000,
                        'The maximum number of batches to run the trainer for.')
tf.flags.DEFINE_integer('summary_step_size', 10,
                        'Emit summary values every summary_step_size steps.')
tf.flags.DEFINE_integer('save_step_size', 200,  # 200,
                        'Save the model every save_step_size steps.')
tf.flags.DEFINE_boolean('force_reload_embeddings', False,  #False, #
                        'Force initialization of embeddings from numpy array in the train directory, even if a model'
                        'checkpoint file is available.')
tf.flags.DEFINE_string('train_mode', 'cbot', #None,#
                       'The way to prepare the train data sets. '
                       'possible values: "cbot" (continuous bag of trees) or None')
tf.flags.DEFINE_string('master', '',
                       'Tensorflow master to use.')
tf.flags.DEFINE_integer('task', 0,
                        'Task ID of the replica running the training.')
tf.flags.DEFINE_integer('ps_tasks', 0,
                        'Number of PS tasks in the job.')
FLAGS = tf.flags.FLAGS

PROTO_PACKAGE_NAME = 'recursive_dependency_embedding'
PROTO_CLASS = 'SequenceNodeSequence'
PROTO_FILE_NAME = 'sequence_node_sequence.proto'

MODEL_FILENAME = 'model.ckpt'


def extract_model_embeddings(model_fn=None, out_fn=None):
    if model_fn is None:
        # We retrieve our checkpoint fullpath
        checkpoint = tf.train.get_checkpoint_state(FLAGS.logdir)
        assert checkpoint is not None, 'no checkpoint file found in logdir: ' + FLAGS.logdir
        model_fn = checkpoint.model_checkpoint_path
    if out_fn is None:
        out_fn = FLAGS.train_data_path + '.vec'

    # backup previous embeddings
    if os.path.isfile(out_fn):
        os.rename(out_fn, out_fn + '.previous')

    with tf.Graph().as_default():
        embeddings = tf.Variable(initial_value=tf.constant(0.0), validate_shape=False, name=model_fold.VAR_NAME_EMBEDDING)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            print('restore model from: ' + model_fn)
            saver.restore(sess, model_fn)
            embeddings_np = sess.run(embeddings)
            print('embeddings shape:')
            print(embeddings_np.shape)
            print('dump embeddings to: ' + out_fn + ' ...')
            embeddings_np.dump(out_fn)


def dump_flags(out_fn, global_step, add_data=None):
    print('dump flags to: ' + out_fn + ' ...')
    runs = []
    if os.path.exists(out_fn):
        with open(out_fn) as data_file:
            runs = json.load(data_file)

    current_run = {}
    current_run['time'] = datetime.datetime.now().isoformat()
    current_run['logdir'] = FLAGS.logdir
    current_run['train_data_path'] = FLAGS.train_data_path
    current_run['max_depth'] = FLAGS.max_depth
    current_run['sample_count'] = FLAGS.sample_count
    current_run['batch_size'] = FLAGS.batch_size
    current_run['max_steps'] = FLAGS.max_steps
    current_run['summary_step_size'] = FLAGS.summary_step_size
    current_run['save_step_size'] = FLAGS.save_step_size
    current_run['force_reload_embeddings'] = FLAGS.force_reload_embeddings
    current_run['train_mode'] = FLAGS.train_mode
    current_run['step_start'] = global_step
    if add_data is not None:
        for k in add_data.keys():
            current_run[k] = add_data[k]

    if len(runs) > 0:
        last_run = sorted(runs, key=lambda run: run['id'])[-1]
        current_run['id'] = last_run['id'] + 1
        last_run_steps = global_step - last_run['step_start']
        last_run_data_count = last_run_steps * last_run['batch_size']
        current_run['data_offset'] = last_run['data_offset'] + last_run_data_count
    else:
        current_run['id'] = 0
        current_run['data_offset'] = 0

    runs.append(current_run)
    with open(out_fn, 'w') as outfile:
        json.dump(runs, outfile, indent=2, sort_keys=True)
    return current_run


# DEPRECATED
def parse_iterator_candidates(sequences, parser, sentence_processor, data_maps):
    pp = pprint.PrettyPrinter(indent=2)
    for s in sequences:
        seq_data, seq_parents = preprocessing.read_data(preprocessing.identity_reader, sentence_processor, parser,
                                                        data_maps,
                                                        args={'content': s}, expand_dict=False)
        children, roots = preprocessing.children_and_roots(seq_parents)

        # dummy position
        insert_idx = 5
        candidate_indices = [2, 8]
        max_depth = 6
        max_dandidate_depth = 1
        seq_tree_seq = sequence_node_sequence_pb2.SequenceNodeSequence()
        seq_tree_seq.idx_correct = 0
        for candidate_idx in candidate_indices:
            preprocessing.build_sequence_tree_with_candidate(seq_data, children, roots[0], insert_idx, max_depth,
                                                             max_dandidate_depth, candidate_idx,
                                                             seq_tree=seq_tree_seq.trees.add())
        pp.pprint(seq_tree_seq)
        yield td.proto_tools.serialized_message_to_tree('recursive_dependency_embedding.SequenceNodeSequence',
                                                        seq_tree_seq.SerializeToString())


def iterator_sequence_trees(corpus_path, max_depth, seq_data, children, sample_count, loaded_global_step=0):
    pp = pprint.PrettyPrinter(indent=2)

    # load corpus depth_max dependent data:
    print('create collected shuffled children indices ...')
    children_indices = preprocessing.collected_shuffled_child_indices(corpus_path, max_depth)
    # print(children_indices.shape)
    size = len(children_indices)
    print('train data size: ' + str(size))
    offset = 0
    # save training info
    if 'FLAGS' in globals():
        current_run = dump_flags(os.path.join(FLAGS.logdir, 'runs.json'), loaded_global_step, add_data={'corpus_size': size})
        offset = current_run['data_offset'] % size
        print('data_offset: ' + str(current_run['data_offset']) + ' (modulo size: ' + str(offset) + ')')
    all_depths_collected = []
    for current_depth in range(max_depth):
        print('load depths from: ' + corpus_path + '.depth' + str(max_depth - 1) + '.collected')
        depths_collected = np.load(corpus_path + '.depth' + str(max_depth - 1) + '.collected')
        all_depths_collected.append(depths_collected)
    # print('current depth size: '+str(len(depths_collected)))
    # repeat infinitely
    while True:
        for child_tuple in children_indices:
            if offset > 0:
                offset -= 1
                continue

            seq_tree_seq = preprocessing.create_seq_tree_seq(child_tuple, seq_data, children, max_depth, sample_count,
                                                             all_depths_collected)
            seq_tree_seq_ = td.proto_tools.serialized_message_to_tree('recursive_dependency_embedding.SequenceNodeSequence',
                                                            seq_tree_seq.SerializeToString())
            # debug
            #pp.pprint(seq_tree_seq)
            #visualize.visualize_seq_node_seq(seq_tree_seq_, rev_m, parser.vocab, constants.vocab_manual)

            yield seq_tree_seq_


# continuous bag of trees model
def iterator_sequence_trees_cbot(corpus_path, max_depth, seq_data, children, sample_count, loaded_global_step):
    print('load depths from: ' + corpus_path + '.depth1.collected')
    depth1_collected = np.load(corpus_path + '.depth1.collected')
    size = len(depth1_collected)
    print('train data size: ' + str(size))
    offset = 0
    # save training info
    if 'FLAGS' in globals():
        current_run = dump_flags(os.path.join(FLAGS.logdir, 'runs.json'), loaded_global_step, add_data={'corpus_size': size})
        offset = current_run['data_offset'] % size
        print('data_offset: ' + str(current_run['data_offset']) + ' (modulo size: ' + str(offset) + ')')
    while True:
        # take all trees with depth > 0 as train data
        for idx in depth1_collected:
            if offset > 0:
                offset -= 1
                continue

            seq_tree = sequence_node_pb2.SequenceNode()
            preprocessing.build_sequence_tree(seq_data, children, idx, seq_tree, max_depth)
            seq_tree_ = td.proto_tools.serialized_message_to_tree('recursive_dependency_embedding.SequenceNode',
                                                            seq_tree.SerializeToString())
            seq_tree_seq = {'trees': [seq_tree_]}
            for _ in range(sample_count):
                seq_tree_new_ = seq_tree_.copy()
                new_head = seq_data[np.random.choice(depth1_collected)]
                seq_tree_new_['head'] = new_head
                seq_tree_seq['trees'].append(seq_tree_new_)

            yield seq_tree_seq


# unused
def optimistic_restore(session, save_file):
    reader = tf.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()
    var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
                        if var.name.split(':')[0] in saved_shapes])
    restore_vars = []
    name2var = dict(zip(map(lambda x: x.name.split(':')[0], tf.global_variables()), tf.global_variables()))
    with tf.variable_scope('', reuse=True):
        for var_name, saved_var_name in var_names:
            curr_var = name2var[saved_var_name]
            var_shape = curr_var.get_shape().as_list()
            if var_shape == saved_shapes[saved_var_name]:
                restore_vars.append(curr_var)
    saver = tf.train.Saver(restore_vars)
    saver.restore(session, save_file)


def save_checkpoint(saver, session, step, logdir):
    print('save checkpoint ...')
    # last checkpoint
    previous_checkpoint_fn = tf.train.get_checkpoint_state(logdir).model_checkpoint_path
    # save checkpoint
    saver.save(session, os.path.join(logdir, MODEL_FILENAME), global_step=step)
    current_checkpoint_fn = tf.train.get_checkpoint_state(logdir).model_checkpoint_path
    print('copy types ...')
    shutil.copyfile(previous_checkpoint_fn + '.type', current_checkpoint_fn + '.type')

    # clean type files
    print('clean type files ...')
    type_files = [s[:-len('.type')] for s in fnmatch.filter(os.listdir(logdir), MODEL_FILENAME + '-*.type')]
    meta_files = [s[:-len('.meta')] for s in fnmatch.filter(os.listdir(logdir), MODEL_FILENAME + '-*.meta')]
    for fn in type_files:
        if fn not in meta_files:
            os.remove(os.path.join(logdir, fn + '.type'))


def main(unused_argv):
    if not os.path.isdir(FLAGS.logdir):
        os.makedirs(FLAGS.logdir)

    loaded_global_step = 0
    embeddings_corpus = None
    # get lexicon size from saved model or numpy array
    checkpoint = tf.train.get_checkpoint_state(FLAGS.logdir)
    if checkpoint:
        input_checkpoint = checkpoint.model_checkpoint_path
        reader = tf.train.NewCheckpointReader(input_checkpoint)
        loaded_global_step = reader.get_tensor(model_fold.VAR_NAME_GLOBAL_STEP).astype(int)
        print('loaded_global_step: '+str(loaded_global_step))
        if FLAGS.force_reload_embeddings:
            print('extract embeddings from checkpoint: ' + input_checkpoint + ' ...')
            embeddings_checkpoint = reader.get_tensor(model_fold.VAR_NAME_EMBEDDING)
            embeddings_corpus = np.load(FLAGS.train_data_path + '.vec')
            types_checkpoint = corpus.read_types(input_checkpoint)
            types_corpus = corpus.read_types(FLAGS.train_data_path)
            print('merge checkpoint dict into corpus dict (add all entries from checkpoint, don\'t remove anything '
                  'from corpus dict) ...')
            embeddings_corpus, types_corpus = corpus.merge_dicts(embeddings_corpus, types_corpus, embeddings_checkpoint, types_checkpoint, add=True, remove=False)
            lex_size = embeddings_corpus.shape[0]
        else:
            print('extract lexicon size from model: ' + input_checkpoint + ' ...')
            saved_shapes = reader.get_variable_to_shape_map()
            embed_shape = saved_shapes[model_fold.VAR_NAME_EMBEDDING]
            lex_size = embed_shape[0]
    else:
        print('load embeddings from: ' + FLAGS.train_data_path + '.vec ...')
        embeddings_corpus = np.load(FLAGS.train_data_path + '.vec')
        lex_size = embeddings_corpus.shape[0]

    print('lex_size: ' + str(lex_size))

    # load corpus data
    print('load corpus data from: ' + FLAGS.train_data_path + '.data ...')
    seq_data = np.load(FLAGS.train_data_path + '.data')
    logging.info('loaded ' + str(len(seq_data)) + ' data points')
    print('load corpus parents from: ' + FLAGS.train_data_path + '.parent ...')
    seq_parents = np.load(FLAGS.train_data_path + '.parent')
    print('calc children ...')
    children, roots = preprocessing.children_and_roots(seq_parents)

    if not FLAGS.train_mode:
        train_iterator = iterator_sequence_trees(FLAGS.train_data_path, FLAGS.max_depth, seq_data, children,
                                                 FLAGS.sample_count, loaded_global_step)
    elif FLAGS.train_mode == 'cbot':
        train_iterator = iterator_sequence_trees_cbot(FLAGS.train_data_path, FLAGS.max_depth, seq_data, children,
                                                      FLAGS.sample_count, loaded_global_step)
    else:
        raise NameError('unknown train_mode: '+FLAGS.train_mode)

    with tf.Graph().as_default() as graph:
        with tf.device(tf.train.replica_device_setter(FLAGS.ps_tasks)):
            embed_w = tf.Variable(tf.constant(0.0, shape=[lex_size, model_fold.DIMENSION_EMBEDDINGS]),
                                  trainable=True, name=model_fold.VAR_NAME_EMBEDDING)
            embedding_placeholder = tf.placeholder(tf.float32, [lex_size, model_fold.DIMENSION_EMBEDDINGS])
            embedding_init = embed_w.assign(embedding_placeholder)

            trainer = model_fold.SequenceTreeEmbeddingSequence(embed_w)

            # softmax_correct = trainer.softmax_correct
            loss = trainer.loss
            acc = trainer.accuracy
            train_op = trainer.train_op
            global_step = trainer.global_step

            # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(FLAGS.logdir + '/train', graph)

            # collect important variables
            scoring_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=model_fold.DEFAULT_SCOPE_SCORING)
            aggr_ordered_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                  scope=model_fold.DEFAULT_SCOPE_AGGR_ORDERED)
            saver_small = tf.train.Saver(var_list=scoring_vars + aggr_ordered_vars + [global_step])
            #saver_embeddings = tf.train.Saver(var_list=[embed_w])

            saver = tf.train.Saver()

            config = projector.ProjectorConfig()
            # You can add multiple embeddings. Here we add only one.
            embedding = config.embeddings.add()
            embedding.tensor_name = embed_w.name
            # Link this tensor to its metadata file (e.g. labels).
            embedding.metadata_path = os.path.join(FLAGS.logdir, 'type.tsv')

            # The next line writes a projector_config.pbtxt in the LOG_DIR. TensorBoard will
            # read this file during startup.
            projector.visualize_embeddings(train_writer, config)

            with tf.Session() as sess:
                if checkpoint and not FLAGS.force_reload_embeddings:
                    input_checkpoint = checkpoint.model_checkpoint_path
                    print('restore all variables from: ' + input_checkpoint)
                    saver.restore(sess, input_checkpoint)
                else:
                    print('init embeddings with external vectors ...')
                    sess.run(embedding_init, feed_dict={embedding_placeholder: embeddings_corpus})
                    if checkpoint:
                        input_checkpoint = checkpoint.model_checkpoint_path
                        print('restore variables (except embeddings) from: ' + input_checkpoint + ' ...')
                        saver_small.restore(sess, input_checkpoint)
                    else:
                        print('initialize variables (except embeddings) ...')
                        # exclude embedding, will be initialized afterwards
                        init_vars = [v for v in tf.global_variables() if v != embed_w]
                        tf.variables_initializer(init_vars).run()

                step = 0
                if not checkpoint:
                    print('save initial checkpoint ...')
                    saver.save(sess, os.path.join(FLAGS.logdir, MODEL_FILENAME), global_step=step)
                    checkpoint_fn = tf.train.get_checkpoint_state(FLAGS.logdir).model_checkpoint_path
                    print('copy types to: ' + checkpoint_fn + '.type')
                    shutil.copyfile(FLAGS.train_data_path + '.type', checkpoint_fn + '.type')

                for _ in xrange(FLAGS.max_steps):
                    batch = [next(train_iterator) for _ in xrange(FLAGS.batch_size)]
                    fdict = trainer.build_feed_dict(batch)
                    if step % FLAGS.summary_step_size == 0:
                        summary, _, step, loss_v, accuracy = sess.run([merged, train_op, global_step, loss, acc],
                                                                      feed_dict=fdict)
                        train_writer.add_summary(summary, step)
                    else:
                        _, step, loss_v, accuracy = sess.run([train_op, global_step, loss, acc], feed_dict=fdict)
                    print('step=%d: loss=%f    accuracy=%f' % (step, loss_v, accuracy))

                    if step % FLAGS.save_step_size == 0:
                        save_checkpoint(saver=saver, session=sess, step=step, logdir=FLAGS.logdir)

                save_checkpoint(saver=saver, session=sess, step=step, logdir=FLAGS.logdir)


if __name__ == '__main__':

    ## debug
    #print('load mapping from file: ' + FLAGS.train_data_path + '.mapping ...')
    #m = pickle.load(open(FLAGS.train_data_path + '.mapping', "rb"))
    #print('len(mapping): ' + str(len(m)))
    #rev_m = tools.revert_mapping(m)
    #print('load spacy ...')
    #parser = spacy.load('en')
    ## debug_end
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    td.proto_tools.map_proto_source_tree_path('', ROOT_DIR)
    td.proto_tools.import_proto_file(PROTO_FILE_NAME)
    tf.app.run()
    # extract_model_embeddings()
