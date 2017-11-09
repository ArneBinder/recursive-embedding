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
import sequence_trees as sequ_trees
import similarity_tree_tuple_pb2
import mytools

#PROTO_PACKAGE_NAME = 'recursive_dependency_embedding'
#s_root = os.path.dirname(__file__)
# Make sure serialized_message_to_tree can find the similarity_tree_tuple proto:
import visualize

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


# unused
def sample_all(data, parents, children, roots):
    # calc depths
    logging.debug('calc depths ...')
    depths = sequ_trees.calc_seq_depth(children, roots, parents)
    max_depth = np.max(depths)
    logging.debug('max_depth: %i' % max_depth)
    sampled_sim_tuples = []
    # sample for every depth only from trees with this depth
    for depth in range(max_depth + 1):
        data_temp = np.array(data, copy=True)
        depth_indices = np.where(depths == depth)[0]
        logging.debug('sample for depth=%i (%i indices) ...' % (depth, len(depth_indices)))
        current_new_simtuples = []
        if depth == 0:
            for i, d_i in enumerate(depth_indices):
                current_new_simtuples.append([d_i, d_i, 1.0])
            sampled_sim_tuples.extend(current_new_simtuples)
            continue

        descendants = [None] * len(depth_indices)
        for i, d_i in enumerate(depth_indices):
            descendants[i] = sorted(sequ_trees.get_descendant_indices(children, d_i))
            # blank "roots"
            data_temp[d_i] = -1

        for i, d_i in enumerate(depth_indices):
            current_data = map(lambda x: data_temp[x], descendants[i])
            probs = np.ones(len(depth_indices), dtype=np.float32)
            probs[:i] = np.zeros(i, dtype=probs.dtype)
            for _j, d_j in enumerate(depth_indices[i:]):
                j = i + _j
                if current_data == map(lambda x: data_temp[x], descendants[j]):
                    probs[j] = 0.0
            probs /= np.sum(probs)
            new_indices = np.random.choice(depth_indices, size=FLAGS.neg_samples, p=probs)
            current_new_simtuples.extend([(d_i, idx, 0.0) for idx in new_indices])

        sampled_sim_tuples.extend(current_new_simtuples)
    return sampled_sim_tuples


def create_corpus(reader_sentences, reader_scores, corpus_name, file_names, output_suffix=None, reader_roots=None,
                  overwrite=False):
    """
    Creates a training corpus consisting of the following files (enumerated by file extension):
        * .train.0, .train.1, ...:      training/development/... data files (for every file name in file_names)
        * .type:                        a types mapping, e.g. a list of types (strings), where its list
                                        position serves as index
        * .vec:                         embedding vectors (indexed by types mapping)
    :param reader_sentences: a file reader that yields sentences of the tuples where every two succinct sentences come
                            from the same tuple: (sentence0, sentence1), (sentence2, sentence3), ...
    :param reader_scores: a file reader that yields similarity scores in the range of [0.0..1.0]
    :param corpus_name: name of the corpus. This is taken for input- and output folder names
    :param file_names: the input file names
    :param output_suffix:
    """

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

    if FLAGS.one_hot_dep:
        out_path = out_path + '_onehotdep'

    if (not corpus.exist(out_path) and not os.path.isfile(out_path + '.score')) or overwrite:
        logging.info('load spacy ...')
        nlp = spacy.load('en')
        nlp.pipeline = [nlp.tagger, nlp.entity, nlp.parser]

        lexicon = lex.Lexicon(nlp_vocab=nlp.vocab)

        def read_data(file_name):
            logging.info('convert texts scored ...')
            logging.debug('len(lexicon)=%i' % len(lexicon))
            _sequence_trees = lexicon.read_data(reader=reader_sentences, sentence_processor=sentence_processor,
                                                parser=nlp, reader_args={'filename': file_name},
                                                batch_size=10000, concat_mode=FLAGS.concat_mode,
                                                inner_concat_mode=FLAGS.inner_concat_mode, expand_dict=True,
                                                #reader_roots=reader_roots,
                                                #reader_roots_args={'filename': os.path.basename(file_name)})
                                                reader_roots_args = {'root_label': constants.vocab_manual[constants.ROOT_EMBEDDING]})
            logging.debug('len(lexicon)=%i (after parsing)' % len(lexicon))
            _s = np.fromiter(reader_scores(file_name), np.float)
            logging.info('scores read: %i' % len(_s))
            assert 2 * len(_s) == len(_sequence_trees.roots), 'len(roots): %i != 2 * len(scores): %i' % (
                2 * len(_s), len(_sequence_trees.roots))
            return _sequence_trees.trees, _s

        file_names = [os.path.join(FLAGS.corpora_source_root, corpus_name, fn) for fn in file_names]

        _trees, _scores = zip(*[read_data(fn) for fn in file_names])

        sizes = [len(s) for s in _scores]

        logging.debug('sizes: %s' % str(sizes))
        np.array(sizes).dump(out_path + '.size')

        sequence_trees = sequ_trees.SequenceTrees(trees=np.concatenate(_trees, axis=1))
        scores = np.concatenate(_scores)

        converter, new_counts, new_idx_unknown = lexicon.sort_and_cut_and_fill_dict(data=sequence_trees.data,
                                                                                    count_threshold=FLAGS.count_threshold)
        sequence_trees.convert_data(converter=converter, lex_size=len(lexicon), new_idx_unknown=new_idx_unknown)

        if FLAGS.one_hot_dep:
            lexicon.set_to_onehot(prefix=constants.vocab_manual[constants.DEPENDENCY_EMBEDDING])

        sequence_trees.dump(out_path)
        scores.dump(out_path + '.score')
        lexicon.set_man_vocab_vec(man_vocab_id=constants.IDENTITY_EMBEDDING)
        lexicon.dump(out_path)
    else:
        lexicon = lex.Lexicon(filename=out_path)
        sequence_trees = sequ_trees.SequenceTrees(filename=out_path)
        scores = np.load(out_path + '.score')
        sizes = np.load(out_path + '.size')

    n = len(scores)
    logging.info('the dataset contains %i scored text tuples' % n)

    collect_unique = True
    if collect_unique:
        if not corpus.exist(out_path+'.unique'):
            # separate into root_trees (e.g. sentences)
            logging.debug('split into root_trees ...')
            root_trees = list(sequence_trees.subtrees())
            assert n == len(root_trees) / 2, '(subtree_count / 2)=%i does not fit score_count=%i' % (len(root_trees) / 2, n)
            id_unique = 0
            #unique_collected = {}
            for i, i_root in enumerate(sequence_trees.roots):
                print(i)
                if not lex.has_vocab_prefix(lexicon[sequence_trees.data[i_root]], constants.UNIQUE_EMBEDDING):
                    sequence_trees.data[i_root] = lexicon[lex.vocab_prefix(constants.UNIQUE_EMBEDDING) + str(id_unique)]
                    #unique_collected[id_unique] = unique_collected.get(id_unique, []).append(i_root)
                    for _j, j_root in enumerate(sequence_trees.roots[i:]):
                        j = i + _j
                        if not lex.has_vocab_prefix(lexicon[sequence_trees.data[j_root]], constants.UNIQUE_EMBEDDING):
                            j_root_data_backup = sequence_trees.data[j_root]
                            sequence_trees.data[j_root] = sequence_trees.data[i_root]
                            if np.array_equal(root_trees[i], root_trees[j]):
                                sequence_trees.data[j_root] = lexicon[lex.vocab_prefix(constants.UNIQUE_EMBEDDING) + str(id_unique)]
                                #unique_collected[id_unique].append(j_root)
                            else:
                                sequence_trees.data[j_root] = j_root_data_backup
                    id_unique += 1

            logging.debug('unique collection finished')

            lexicon.pad()
            lexicon.dump(out_path + '.unique')
            sequence_trees.dump(out_path + '.unique')

        else:
            sequence_trees = sequ_trees.SequenceTrees(filename=out_path + '.unique')

    if FLAGS.neg_samples:
        if not os.path.isfile(out_path + '.idx.neg'):

            logging.debug('calc sims_correct ...')
            # TODO: re-enable blanking?
            sims_correct = np.zeros(shape=n)
            for i in range(n):
                sims_correct[i] = sim_jaccard(root_trees[i * 2][0], root_trees[i * 2 + 1][0])

            sims_correct.sort()

            logging.debug('start sampling with neg_samples=%i ...' % FLAGS.neg_samples)
            _sampled_indices = mytools.parallel_process_simple(range(n), partial(sample_indices, subtrees=root_trees,
                                                                                 sims_correct=sims_correct, prog_bar=None))
            sampled_indices = np.concatenate(_sampled_indices)

            start = 0
            for idx, end in enumerate(np.cumsum(sizes)):
                neg_sample_tuples = [(sequence_trees.roots[(i / FLAGS.neg_samples) * 2], sequence_trees.roots[sampled_indices[i] * 2 + 1], 0.0) for i in range(start * FLAGS.neg_samples, end * FLAGS.neg_samples)]
                np.array(neg_sample_tuples).dump('%s.idx.%i.negs%i' % (out_path, idx, FLAGS.neg_samples))
                start = end

            # load (un-blanked) data
            # TODO: re-enable?
            #data = np.load(out_path + '.data')

        #sampled_all = sample_all(out_path, parents, children, roots)

    start = 0
    for idx, end in enumerate(np.cumsum(sizes)):
        current_sim_tuples = [(sequence_trees.roots[i * 2], sequence_trees.roots[i * 2 + 1], scores[i]) for i in range(start, end)]
        write_sim_tuple_data('%s.train.%i' % (out_path, idx), current_sim_tuples, sequence_trees.data, sequence_trees.children)
        np.array(current_sim_tuples).dump('%s.idx.%i' % (out_path, idx))
        start = end


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


def write_sim_tuple_data(out_fn, sim_tuples, data, children):
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
            sequ_trees.build_sequence_tree(data, children, sim_tuples[idx][0], sim_tree_tuple.first)
            # set root of second to root of first (in case of negative samples)
            data_root2 = data[sim_tuples[idx][1]]
            data[sim_tuples[idx][1]] = data[sim_tuples[idx][0]]
            sequ_trees.build_sequence_tree(data, children, sim_tuples[idx][1], sim_tree_tuple.second)
            data[sim_tuples[idx][1]] = data_root2
            sim_tree_tuple.similarity = sim_tuples[idx][2]
            record_output.write(sim_tree_tuple.SerializeToString())


def write_sim_tuple_data_single(out_fn, sim_tuples, data, children):
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
    # TODO: check this!
    # ensure every left sequence_tree occurs only once
    new_root_data_scored_collected = {}
    for sim_tuple in sim_tuples:
        new_root_data_scored = new_root_data_scored_collected.get(sim_tuple[0], set())
        new_root_data_scored.add((data[sim_tuple[1]], sim_tuple[2]))
        new_root_data_scored_collected[sim_tuple[0]] = new_root_data_scored

        if sim_tuple[1] != sim_tuple[2]:
            new_root_data_scored = new_root_data_scored_collected.get(sim_tuple[1], set())
            new_root_data_scored.add((data[sim_tuple[0]], sim_tuple[2]))
            new_root_data_scored_collected[sim_tuple[1]] = new_root_data_scored

    logging.info('write data to: ' + out_fn + ' ...')
    with tf.python_io.TFRecordWriter(out_fn) as record_output:
        for root in new_root_data_scored_collected:
            scored_tree = scored_tree_pb2.ScoredTree()
            scored_tree.score = 1.0
            sequ_trees.build_sequence_tree(data, children, root, scored_tree.tree)
            record_output.write(scored_tree.SerializeToString())

            data_target_backup = data[root]
            new_root_data_scored = new_root_data_scored_collected[root]
            for new_root_data, score in new_root_data_scored:
                if new_root_data != data[root]:
                    data[root] = new_root_data
                    scored_tree = scored_tree_pb2.ScoredTree()
                    scored_tree.score = score
                    sequ_trees.build_sequence_tree(data, children, root, scored_tree.tree)
                    record_output.write(scored_tree.SerializeToString())
                #else:
                #    print(root)

            data[root] = data_target_backup


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
    vecs, types = lex.load(corpus_fn1)
    vecs2, types2 = lex.load(corpus_fn2)
    vecs, types = lex.merge_dicts(vecs, types, vecs2, types2, add=True, remove=False)
    data2, parents2 = sequ_trees.load(corpus_fn2)
    m = lex.mapping_from_list(types)
    mapping = {i: m[t] for i, t in enumerate(types2)}
    data2_converted = np.array([mapping[d] for d in data2], dtype=data2.dtype)
    dir2 = os.path.abspath(os.path.join(corpus_fn2, os.pardir))
    indices2_fnames = fnmatch.filter(os.listdir(dir2), ntpath.basename(corpus_fn2) + '.idx.[0-9]*')
    indices2 = [load_sim_tuple_indices(os.path.join(dir2, fn)) for fn in indices2_fnames]
    children2, roots2 = sequ_trees.children_and_roots(parents2)
    for i, sim_tuples in enumerate(indices2):
        write_sim_tuple_data('%s.merged.train.%i' % (corpus_fn1, i), sim_tuples, data2_converted, children2)
        write_sim_tuple_data_single('%s.merged.train.%i.single' % (corpus_fn1, i), sim_tuples, data2_converted, children2)
    lex.dump('%s.merged' % corpus_fn1, vecs=vecs, types=types)