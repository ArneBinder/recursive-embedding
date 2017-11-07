import fnmatch
import logging
import ntpath
import os
import random
from collections import Counter
from functools import partial

import numpy as np
import spacy
import tensorflow as tf
import tensorflow_fold as td

import constants
import corpus
import lexicon as lex
import preprocessing
import scored_tree_pb2
import sequence_trees
import similarity_tree_tuple_pb2
import mytools

#PROTO_PACKAGE_NAME = 'recursive_dependency_embedding'
#s_root = os.path.dirname(__file__)
# Make sure serialized_message_to_tree can find the similarity_tree_tuple proto:
td.proto_tools.map_proto_source_tree_path('', os.path.dirname(__file__))
td.proto_tools.import_proto_file('similarity_tree_tuple.proto')
td.proto_tools.import_proto_file('scored_tree.proto')

tf.flags.DEFINE_string('corpora_source_root',
                       '/home/arne/devel/ML/data/corpora',
                       'location of raw corpora directories')
tf.flags.DEFINE_string('corpora_target_root',
                       '/media/arne/WIN/Users/Arne/ML/data/corpora',
                       'location of raw corpora directories')
tf.flags.DEFINE_string(
    'sentence_processor',
    'process_sentence3_marked',
    'Which data types (features) are used to build the data sequence.')
tf.flags.DEFINE_string(
    'concat_mode',
    # 'sequence',
    'aggregate',
    # constants.default_inner_concat_mode,
    'How to concatenate the sentence-trees with each other. '
    'A sentence-tree represents the information regarding one sentence. '
    '"sequence" -> roots point to next root, '
    '"aggregate" -> roots point to an added, artificial token (AGGREGATOR) in the end of the token sequence'
    '(NOT ALLOWED for similarity scored tuples!) None -> do not concat at all')
tf.flags.DEFINE_string(
    'inner_concat_mode',
    # 'tree',
    None,
    # constants.default_inner_concat_mode,
    'How to concatenate the token-trees with each other. '
    'A token-tree represents the information regarding one token. '
    '"tree" -> use dependency parse tree'
    '"sequence" -> roots point to next root, '
    '"aggregate" -> roots point to an added, artificial token (AGGREGATOR) in the end of the token sequence'
    'None -> do not concat at all. This produces one sentence-tree per token.')
tf.flags.DEFINE_integer(
    'count_threshold',
    1,
    # TODO: check if less or equal-less
    'remove token which occur less then count_threshold times in the corpus')
tf.flags.DEFINE_integer(
    'neg_samples',
    0,
    'amount of negative samples to add'
)
tf.flags.DEFINE_boolean(
    'one_hot_dep',
    True,
    'Whether to replace all dependence edge embeddings with one hot embeddings.'
)

FLAGS = tf.flags.FLAGS
mytools.logging_init()


def sim_jaccard(ids1, ids2):
    ids1_set = set(ids1)
    ids2_set = set(ids2)
    return len(ids1_set & ids2_set) * 1.0 / len(ids1_set | ids2_set)


def continuous_binning(hist_src, hist_dest):
    c_src = Counter(hist_src)
    c_dest = Counter(hist_dest)
    keys_src = sorted(c_src.keys())
    keys_dest = sorted(c_dest.keys())
    prob_map = {}
    last_dest = []
    last_src = []

    def process_probs():
        sum_dest = sum([c_dest[d] for d in last_dest])
        sum_src = sum([c_src[d] for d in last_src])
        for x in last_src:
            prob_map[x] = sum_dest / float(sum_src * len(hist_dest))
        del last_dest[:]
        del last_src[:]

    i_dest = i_src = 0
    while i_dest < len(c_dest) and i_src < len(c_src):
        if keys_dest[i_dest] <= keys_src[i_src]:
            if len(last_src) > 0 and len(last_dest) > 0:
                process_probs()
            last_dest.append(keys_dest[i_dest])
            i_dest += 1
        else:
            if len(last_src) > 0 and len(last_dest) > 0:
                process_probs()
            last_src.append(keys_src[i_src])
            i_src += 1
    # add remaining
    last_dest.extend(keys_dest[i_dest:])
    last_src.extend(keys_src[i_src:])
    process_probs()

    return prob_map


def sample_indices(idx, subtrees, sims_correct, prog_bar=None):
    #idx, subtrees, sims_correct = idx_subtrees_simscorrect
    n = len(sims_correct)
    # sample according to sims_correct probability distribution
    sims = np.zeros(shape=n)
    for j in range(n):
        sims[j] = sim_jaccard(subtrees[idx * 2][0], subtrees[j * 2 + 1][0])
    sim_original = sims[idx]
    sims_sorted = np.sort(sims)
    prob_map = continuous_binning(hist_src=sims_sorted, hist_dest=sims_correct)

    # set probabilities according to prob_map ...
    p = [prob_map[d] for d in sims]
    # ...  but set to 0.0 if this is the original pair (can occur multiple times)
    # or if the two subtrees are equal
    # c_debug = 0
    for j, d in enumerate(sims):
        if (d == sim_original and (idx == j or np.array_equal(subtrees[2 * idx + 1], subtrees[2 * j + 1]))) \
                or (d == 1.0 and np.array_equal(subtrees[2 * idx], subtrees[2 * j + 1])):
            p[j] = 0.0
            # c_debug += 1
    # if c_debug > 1:
    #    logging.debug('%i:%i' % (i, c_debug))

    # normalize probs
    p = np.array(p)
    p = p / p.sum()
    try:
        new_indices = np.random.choice(n, size=FLAGS.neg_samples, p=p, replace=False)
    except ValueError as e:
        logging.warning(
            'Error: "%s" (source tuple index: %i) Retry sampling with repeated elements allowed ...' % (
                e.message, idx))
        new_indices = np.random.choice(n, size=FLAGS.neg_samples, p=p, replace=True)
    if prog_bar:
        prog_bar.next()
    return new_indices
    #sample_indices[i * FLAGS.neg_samples:(i + 1) * FLAGS.neg_samples] = new_indices


def create_corpus(reader_sentences, reader_score, corpus_name, file_names, output_suffix=None, reader_roots=None, neg_sample_last=True):
    """
    Creates a training corpus consisting of the following files (enumerated by file extension):
        * .train.0, .train.1, ...:      training/development/... data files (for every file name in file_names)
        * .type:                        a types mapping, e.g. a list of types (strings), where its list
                                        position serves as index
        * .vec:                         embedding vectors (indexed by types mapping)
    :param reader_sentences: a file reader that yields sentences of the tuples where every two succinct sentences come
                            from the same tuple: (sentence0, sentence1), (sentence2, sentence3), ...
    :param reader_score: a file reader that yields similarity scores in the range of [0.0..1.0]
    :param corpus_name: name of the corpus. This is taken for input- and output folder names
    :param file_names: the input file names
    :param output_suffix:
    """

    logging.info('load spacy ...')
    nlp = spacy.load('en')
    nlp.pipeline = [nlp.tagger, nlp.entity, nlp.parser]

    vecs, types = lex.get_dict_from_vocab(nlp.vocab)
    mapping = lex.mapping_from_list(types)

    sentence_processor = getattr(preprocessing, FLAGS.sentence_processor)
    out_dir = os.path.abspath(os.path.join(FLAGS.corpora_target_root, corpus_name, sentence_processor.func_name))
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    out_path = os.path.join(out_dir, corpus_name + (output_suffix or ''))

    assert FLAGS.concat_mode is not None and FLAGS.concat_mode != 'tree', \
        "concat_mode=None or concat_mode='tree' is NOT ALLOWED for similarity scored tuples! Use 'sequence' or 'aggregate'"
    out_path = out_path + '_cm' + FLAGS.concat_mode.upper()
    if FLAGS.inner_concat_mode is not None:
        out_path = out_path + '_icm' + FLAGS.inner_concat_mode.upper()
    if FLAGS.neg_samples:
        out_path = out_path + '_negs' + str(FLAGS.neg_samples)

    def read_data(file_name):
        return corpus.parse_texts_scored(filename=file_name,
                                         reader=reader_sentences,
                                         reader_scores=reader_score,
                                         sentence_processor=sentence_processor,
                                         parser=nlp,
                                         mapping=mapping,
                                         concat_mode=FLAGS.concat_mode,
                                         inner_concat_mode=FLAGS.inner_concat_mode,
                                         reader_roots=reader_roots)

    file_names = [os.path.join(FLAGS.corpora_source_root, corpus_name, fn) for fn in file_names]

    _data = [None] * len(file_names)
    _parents = [None] * len(file_names)
    _scores = [None] * len(file_names)

    for i, fn in enumerate(file_names):
        _data[i], _parents[i], _scores[i], _ = read_data(fn)

    sizes = [len(s) for s in _scores]

    logging.debug('sizes: %s' % str(sizes))
    data = np.concatenate(_data)
    parents = np.concatenate(_parents)
    scores = np.concatenate(_scores)

    types = lex.revert_mapping_to_list(mapping)
    converter, vecs, types, new_counts, new_idx_unknown = lex.sort_and_cut_and_fill_dict(data, vecs, types,
                                                                                         count_threshold=FLAGS.count_threshold)
    data = corpus.convert_data(data, converter, len(types), new_idx_unknown)

    if FLAGS.one_hot_dep:
        out_path = out_path + '_onehotdep'
        # one_hot_types = []
        # one_hot_types = [u'DEP#det', u'DEP#punct', u'DEP#pobj', u'DEP#ROOT', u'DEP#prep', u'DEP#aux', u'DEP#nsubj',
        # u'DEP#dobj', u'DEP#amod', u'DEP#conj', u'DEP#cc', u'DEP#compound', u'DEP#nummod', u'DEP#advmod', u'DEP#acl',
        # u'DEP#attr', u'DEP#auxpass', u'DEP#expl', u'DEP#nsubjpass', u'DEP#poss', u'DEP#agent', u'DEP#neg', u'DEP#prt',
        # u'DEP#relcl', u'DEP#acomp', u'DEP#advcl', u'DEP#case', u'DEP#npadvmod', u'DEP#xcomp', u'DEP#ccomp', u'DEP#pcomp',
        # u'DEP#oprd', u'DEP#nmod', u'DEP#mark', u'DEP#appos', u'DEP#dep', u'DEP#dative', u'DEP#quantmod', u'DEP#csubj',
        # u'DEP#']
        one_hot_types = [t for t in types if t.startswith(preprocessing.MARKER_DEP_EDGE)]
        mapping = lex.mapping_from_list(types)
        one_hot_ids = [mapping[t] for t in one_hot_types]
        if len(one_hot_ids) > vecs.shape[1]:
            logging.warning('Setting more then vecs-size=%i lex entries to one-hot encoding.'
                            ' That overrides previously added one hot embeddings!' % vecs.shape[1])
        for i, idx in enumerate(one_hot_ids):
            vecs[idx] = np.zeros(shape=vecs.shape[1], dtype=vecs.dtype)
            vecs[idx][i % vecs.shape[1]] = 1.0

    logging.info('save data, parents, scores, vecs and types to: ' + out_path + ' ...')
    data.dump(out_path + '.data')
    parents.dump(out_path + '.parent')
    scores.dump(out_path + '.score')

    # set identity embedding to zero vector
    #if constants.vocab_manual[constants.IDENTITY_EMBEDDING] in types:
    IDENTITY_idx = types.index(constants.vocab_manual[constants.IDENTITY_EMBEDDING])
    vecs[IDENTITY_idx] = np.zeros(shape=vecs.shape[1], dtype=vecs.dtype)
    #else:
    #    types.append(constants.vocab_manual[constants.IDENTITY_EMBEDDING])
    #    vecs = np.concatenate([vecs, np.zeros(shape=(1, vecs.shape[1]), dtype=vecs.dtype)])

    lex.write_dict(out_path, vecs=vecs, types=types)
    n = len(scores)
    logging.info('the dataset contains %i scored text tuples' % n)
    logging.debug('calc roots ...')
    children, roots = sequence_trees.children_and_roots(parents)
    logging.debug('len(roots)=%i' % len(roots))

    if FLAGS.neg_samples:
        # separate into subtrees (e.g. sentences)
        logging.debug('split into subtrees ...')
        subtrees = []
        for root in roots:
            # blank roots (e.g. SOURCE/...)
            data[root] = len(types)
            descendant_indices = sequence_trees.get_descendant_indices(children, root)
            new_subtree = zip(*[(data[idx], parents[idx]) for idx in sorted(descendant_indices)])
            new_subtree = np.array(new_subtree, dtype=np.int32)
            # if new_subtree in subtrees:
            #    repl_roots.append(root)
            subtrees.append(new_subtree)
            # TODO: check for replicates? (if yes, check successive tuples!)
        assert n == len(subtrees) / 2, '(subtree_count / 2)=%i does not fit score_count=%i' % (len(subtrees) / 2, n)
        logging.debug('calc sims_correct ...')
        sims_correct = np.zeros(shape=n)
        for i in range(n):
            sims_correct[i] = sim_jaccard(subtrees[i * 2][0], subtrees[i * 2 + 1][0])

        sims_correct.sort()

        logging.debug('start sampling with neg_samples=%i ...' % FLAGS.neg_samples)

        #_sampled_indices = mytools.parallel_process(range(n), partial(sample_indices, subtrees=subtrees, sims_correct=sims_correct, prog_bar=None))
        _sampled_indices = mytools.parallel_process_simple(range(n), partial(sample_indices, subtrees=subtrees, sims_correct=sims_correct, prog_bar=None))
        sampled_indices = np.concatenate(_sampled_indices)

        # load (un-blanked) data
        data = np.load(out_path + '.data')

    # debug
    #sims_jac = np.zeros(shape=n * (1 + FLAGS.neg_samples))
    #sims_cor = np.zeros(shape=sims_jac.shape)
    #offset = 0
    # debug end

    sim_tuples = []
    start = 0
    for idx, end in enumerate(np.cumsum(sizes)):
        current_sim_tuples = [(i * 2, i * 2 + 1, scores[i]) for i in range(start, end)]
        # add negative samples, but not for last train (aka test) file
        if FLAGS.neg_samples and (neg_sample_last or idx != len(sizes) - 1):
            neg_sample_tuples = [((i / FLAGS.neg_samples) * 2, sampled_indices[i] * 2 + 1, 0.0) for i in range(start * FLAGS.neg_samples, end * FLAGS.neg_samples)]
            current_sim_tuples.extend(neg_sample_tuples)
            # shuffle
            random.shuffle(current_sim_tuples)
        sim_tuples.append(current_sim_tuples)
        start = end

        # debug
        #for i, st in enumerate(sim_tuples[-1]):
        #    sims_jac[offset + i] = sim_jaccard(subtrees[st[0]][0], subtrees[st[1]][0])
        #    sims_cor[offset + i] = st[2]

        #offset += len(sim_tuples[-1])
        # debug end

    # debug
    #sims_jac.dump('sims_jac')
    #sims_cor.dump('sims_cor')
    # debug end

    for i, _sim_tuples in enumerate(sim_tuples):
        write_sim_tuple_data('%s.train.%i' % (out_path, i), _sim_tuples, data, children, roots)
        write_sim_tuple_data_single('%s.train.%i.single' % (out_path, i), _sim_tuples, data, children, roots)
        np.array(_sim_tuples).dump('%s.idx.%i' % (out_path, i))


def iterate_sim_tuple_data(paths):
    count = 0
    for path in paths:
        for v in tf.python_io.tf_record_iterator(path):
            res = td.proto_tools.serialized_message_to_tree('recursive_dependency_embedding.' + similarity_tree_tuple_pb2.SimilarityTreeTuple.__name__, v)
            res['id'] = count
            yield res
            count += 1


def iterate_scored_tree_data(paths):
    for path in paths:
        for v in tf.python_io.tf_record_iterator(path):
            res = td.proto_tools.serialized_message_to_tree(
                'recursive_dependency_embedding.' + scored_tree_pb2.ScoredTree.__name__, v)
            yield res


def write_sim_tuple_data(out_fn, sim_tuples, data, children, roots):
    """
    Write sim_tuple(s) to file.

    :param out_fn the   file name to write the data into
    :param sim_tuples   list of tuples (root_idx1, root_idx2, similarity), where root_idx1 and root_idx2 are indices to
                        roots which thereby point to the sequence tree for the first / second sequence. Similarity is a
                        float value in [0.0, 1.0].
    :param data         the data sequence
    :param children     preprocessed child information, see preprocessing.children_and_roots
    :param roots        preprocessed root information (indices to roots in data), see preprocessing.children_and_roots
    """

    logging.info('write data to: ' + out_fn + ' ...')
    with tf.python_io.TFRecordWriter(out_fn) as record_output:
        for idx in range(len(sim_tuples)):
            sim_tree_tuple = similarity_tree_tuple_pb2.SimilarityTreeTuple()
            sequence_trees.build_sequence_tree(data, children, roots[sim_tuples[idx][0]], sim_tree_tuple.first)
            # set root of second to root of first (in case of negative samples)
            data_root2 = data[roots[sim_tuples[idx][1]]]
            data[roots[sim_tuples[idx][1]]] = data[roots[sim_tuples[idx][0]]]
            sequence_trees.build_sequence_tree(data, children, roots[sim_tuples[idx][1]], sim_tree_tuple.second)
            data[roots[sim_tuples[idx][1]]] = data_root2
            sim_tree_tuple.similarity = sim_tuples[idx][2]
            record_output.write(sim_tree_tuple.SerializeToString())


def write_sim_tuple_data_single(out_fn, sim_tuples, data, children, roots):
    """
    Write sim_tuple(s) to file.

    :param out_fn the   file name to write the data into
    :param sim_tuples   list of tuples (root_idx1, root_idx2, similarity), where root_idx1 and root_idx2 are indices to
                        roots which thereby point to the sequence tree for the first / second sequence. Similarity is a
                        float value in [0.0, 1.0].
    :param data         the data sequence
    :param children     preprocessed child information, see preprocessing.children_and_roots
    :param roots        preprocessed root information (indices to roots in data), see preprocessing.children_and_roots
    """

    # ensure every left sequence_tree occurs only once
    scored_root_ids_collected = {}
    for sim_tuple in sim_tuples:
        scored_ids = scored_root_ids_collected.get(sim_tuple[0], [])
        scored_ids.append((sim_tuple[1], sim_tuple[2]))
        scored_root_ids_collected[sim_tuple[0]] = scored_ids

    logging.info('write data to: ' + out_fn + ' ...')
    with tf.python_io.TFRecordWriter(out_fn) as record_output:
        for root_idx in scored_root_ids_collected:
            scored_tree = scored_tree_pb2.ScoredTree()
            scored_tree.score = 1.0
            sequence_trees.build_sequence_tree(data, children, roots[root_idx], scored_tree.tree)
            record_output.write(scored_tree.SerializeToString())

            scored_ids = scored_root_ids_collected[root_idx]
            for root_idx_target, score in scored_ids:
                data_target_backup = data[roots[root_idx_target]]
                data[roots[root_idx_target]] = data[roots[root_idx]]

                scored_tree = scored_tree_pb2.ScoredTree()
                scored_tree.score = score
                sequence_trees.build_sequence_tree(data, children, roots[root_idx_target], scored_tree.tree)
                record_output.write(scored_tree.SerializeToString())

                data[roots[root_idx_target]] = data_target_backup


def load_sim_tuple_indices(filename):
    _loaded = np.load(filename).T
    ids1 = _loaded[0].astype(int)
    ids2 = _loaded[1].astype(int)
    loaded = zip(ids1, ids2, _loaded[2])
    return loaded


def merge_into_corpus(corpus_fn1, corpus_fn2):
    """
    Merges corpus2 into corpus1 e.g. merges types and vecs and converts data2 according to new types dict and writes
    training files from index files (file extension: .idx.<id>)

    :param corpus_fn1: file name of source corpus
    :param corpus_fn2: file name of target corpus
    :return:
    """
    vecs, types = lex.read_dict(corpus_fn1)
    vecs2, types2 = lex.read_dict(corpus_fn2)
    vecs, types = lex.merge_dicts(vecs, types, vecs2, types2, add=True, remove=False)
    data2, parents2 = lex.load_data_and_parents(corpus_fn2)
    m = lex.mapping_from_list(types)
    mapping = {i: m[t] for i, t in enumerate(types2)}
    data2_converted = np.array([mapping[d] for d in data2], dtype=data2.dtype)
    dir2 = os.path.abspath(os.path.join(corpus_fn2, os.pardir))
    indices2_fnames = fnmatch.filter(os.listdir(dir2), ntpath.basename(corpus_fn2) + '.idx.[0-9]*')
    indices2 = [load_sim_tuple_indices(os.path.join(dir2, fn)) for fn in indices2_fnames]
    children2, roots2 = sequence_trees.children_and_roots(parents2)
    for i, sim_tuples in enumerate(indices2):
        write_sim_tuple_data('%s.merged.train.%i' % (corpus_fn1, i), sim_tuples, data2_converted, children2, roots2)
        write_sim_tuple_data_single('%s.merged.train.%i.single' % (corpus_fn1, i), sim_tuples, data2_converted, children2, roots2)
    lex.write_dict('%s.merged' % corpus_fn1, vecs=vecs, types=types)