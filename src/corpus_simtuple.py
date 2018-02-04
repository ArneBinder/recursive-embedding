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
#import tensorflow_fold as td

import constants
import corpus
import lexicon as lex
import preprocessing
#import scored_tree_pb2
import sequence_trees as sequ_trees
#import similarity_tree_tuple_pb2
import mytools

#PROTO_PACKAGE_NAME = 'recursive_dependency_embedding'
#s_root = os.path.dirname(__file__)
# Make sure serialized_message_to_tree can find the similarity_tree_tuple proto:
import visualize

#td.proto_tools.map_proto_source_tree_path('', os.path.dirname(__file__))
#td.proto_tools.import_proto_file('similarity_tree_tuple.proto')
#td.proto_tools.import_proto_file('scored_tree.proto')

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
    'sample_count',
    0,
    'amount of negative samples to add'
)
tf.flags.DEFINE_boolean(
    'one_hot_dep',
    True,
    'Whether to replace all dependence edge embeddings with one hot embeddings.'
)
tf.flags.DEFINE_boolean(
    'sample_all',
    False,
    'Whether to create negative samples for every data point in the tree sequence.'
)
tf.flags.DEFINE_boolean(
    'create_unique',
    False,
    'Whether to create unique tuple roots even if negative sampling is disabled.'
)
tf.flags.DEFINE_boolean(
    'sample_adapt_distribution',
    True,
    'Adapt the distribution of the data for sampling negative examples.'
)
tf.flags.DEFINE_boolean(
    'sample_check_equality',
    True,
    'Adapt the distribution of the data for sampling negative examples.'
)
tf.flags.DEFINE_boolean(
    'random_vecs',
    False,
    'Set random embedding vectors for all lexicon entries.'
)

FLAGS = tf.flags.FLAGS
mytools.logging_init()


# Does not consider multiple mentions! Use MultiSet instead?
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


def sample_indices(idx, trees, unique_root_data=None, sims_correct=None, prog_bar=None, check_equality=True):
    unique_roots = (unique_root_data is not None)
    adapt_distribution = (sims_correct is not None)
    n = len(trees) / 2
    if adapt_distribution:
        # sample according to sims_correct probability distribution
        sims = np.zeros(shape=n)
        if unique_roots:
            sims_unique = {}
            for j in range(n):
                r1_data = unique_root_data[idx * 2]
                r2_data = unique_root_data[j * 2 + 1]
                if (r1_data, r2_data) in sims_unique:
                    sims[j] = sims_unique[(r1_data, r2_data)]
                else:
                    sims[j] = sim_jaccard(trees[idx * 2][0], trees[j * 2 + 1][0])
                    sims_unique[(r1_data, r2_data)] = sims[j]
                    sims_unique[(r2_data, r1_data)] = sims[j]
        else:
            for j in range(n):
                sims[j] = sim_jaccard(trees[idx * 2][0], trees[j * 2 + 1][0])
        sim_original = sims[idx]
        sims_sorted = np.sort(sims)
        prob_map = continuous_binning(hist_src=sims_sorted, hist_dest=sims_correct)

        # set probabilities according to prob_map ...
        p = np.array([prob_map[sim] for sim in sims])
    else:
        p = np.ones(n, dtype=np.float32)
    # ...  but set to 0.0 if this is the original pair (can occur multiple times)
    # or if the two subtrees are equal
    if check_equality:
        #c_debug = []
        if unique_roots:
            for j in range(n):
                if unique_root_data[2 * idx + 1] == unique_root_data[2 * j + 1] or unique_root_data[2 * idx] == unique_root_data[2 * j + 1]:
                    p[j] = 0.0
        else:
            for j in range(n):
                if (((not adapt_distribution) or sims[j] == sim_original) and (idx == j or np.array_equal(trees[2 * idx + 1], trees[2 * j + 1]))) \
                        or (((not adapt_distribution) or sims[j] == 1.0) and np.array_equal(trees[2 * idx], trees[2 * j + 1])):
                    p[j] = 0.0
        #            c_debug.append(j)
        #if len(c_debug) > 1:
        #   logging.debug('%i:%s' % (idx, str(c_debug)))

    # normalize probs
    p = p / p.sum()
    try:
        new_indices = np.random.choice(n, size=FLAGS.sample_count, p=p, replace=False)
    except ValueError as e:
        logging.warning(
            'Error: "%s" (source tuple index: %i) Retry sampling with repeated elements allowed ...' % (
                e.message, idx))
        new_indices = np.random.choice(n, size=FLAGS.sample_count, p=p, replace=True)
    if prog_bar:
        prog_bar.next()
    return new_indices


def write_sim_tuple_indices(path, sim_tuple_indices, sizes, path_suffix=''):
    start = 0
    for idx, end in enumerate(np.cumsum(sizes)):
        logging.info('write sim_tuple_indices to: %s.idx.%i%s ...' % (path, idx, path_suffix))
        np.array(sim_tuple_indices[start:end]).dump('%s.idx.%i%s' % (path, idx, path_suffix))
        start = end


def create_corpus(reader_sentences, reader_scores, corpus_name, file_names, output_suffix=None, overwrite=False,
                  reader_roots=None, reader_roots_args=None):
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

    if FLAGS.random_vecs:
        out_path = out_path + '_randomvecs'

    if (not corpus.exist(out_path) and not os.path.isfile(out_path + '.score')) or overwrite:
        logging.info('load spacy ...')
        nlp = spacy.load('en_core_web_md')
        #nlp.pipeline = [nlp.tagger, nlp.entity, nlp.parser]

        lexicon = lex.Lexicon(nlp_vocab=nlp.vocab)
        if reader_roots_args is None:
            reader_roots_args = {'root_labels': constants.vocab_manual[constants.ROOT_EMBEDDING]}

        def read_data(file_name):
            logging.info('convert texts scored ...')
            logging.debug('len(lexicon)=%i' % len(lexicon))
            _forest = lexicon.read_data(reader=reader_sentences, sentence_processor=sentence_processor,
                                        parser=nlp, reader_args={'filename': file_name},
                                        batch_size=10000, concat_mode=FLAGS.concat_mode,
                                        inner_concat_mode=FLAGS.inner_concat_mode, expand_dict=True,
                                        reader_roots=reader_roots,
                                        reader_roots_args=reader_roots_args)
            logging.debug('len(lexicon)=%i (after parsing)' % len(lexicon))
            _s = np.fromiter(reader_scores(file_name), np.float)
            logging.info('scores read: %i' % len(_s))
            assert 2 * len(_s) == len(_forest.roots), 'len(roots): %i != 2 * len(scores): %i' % (
                2 * len(_s), len(_forest.roots))
            return _forest.forest, _s

        file_names = [os.path.join(FLAGS.corpora_source_root, corpus_name, fn) for fn in file_names]

        _forests, _scores = zip(*[read_data(fn) for fn in file_names])

        sizes = [len(s) for s in _scores]

        logging.debug('sizes: %s' % str(sizes))
        np.array(sizes).dump(out_path + '.size')

        forest = sequ_trees.Forest(forest=np.concatenate(_forests, axis=1), lexicon=lexicon)
        scores = np.concatenate(_scores)

        converter, new_counts, new_idx_unknown = lexicon.sort_and_cut_and_fill_dict(data=forest.data,
                                                                                    count_threshold=FLAGS.count_threshold)
        forest.convert_data(converter=converter, new_idx_unknown=new_idx_unknown)

        if FLAGS.one_hot_dep:
            lexicon.set_to_onehot(prefix=constants.vocab_manual[constants.DEPENDENCY_EMBEDDING])

        forest.dump(out_path)
        scores.dump(out_path + '.score')
        lexicon.set_man_vocab_vec(man_vocab_id=constants.IDENTITY_EMBEDDING)
        if FLAGS.random_vecs:
            lexicon.set_to_random()
        lexicon.dump(out_path)
    else:
        lexicon = lex.Lexicon(filename=out_path)
        forest = sequ_trees.Forest(filename=out_path, lexicon=lexicon)
        scores = np.load(out_path + '.score')
        sizes = np.load(out_path + '.size')

    sim_tuples = [[forest.roots[tuple_idx * 2], forest.roots[tuple_idx * 2 + 1], scores[tuple_idx]]
                  for tuple_idx in range(len(scores))]
    write_sim_tuple_indices(path=out_path, sim_tuple_indices=sim_tuples, sizes=sizes)

    n = len(scores)
    logging.info('the dataset contains %i scored text tuples' % n)

    if FLAGS.create_unique:
        # use unique by now
        out_path += '.unique'
        if not corpus.exist(out_path) or overwrite:
            # separate into trees (e.g. sentences)
            logging.debug('split into trees ...')
            trees = list(forest.trees())
            assert n == len(trees) / 2, '(subtree_count / 2)=%i does not fit score_count=%i' % (len(trees) / 2, n)
            logging.info('collect unique ...')
            id_unique = 0
            #unique_collected = {}
            for i, i_root in enumerate(forest.roots):
                #logging.debug('i_root: %i' % i_root)
                if not lex.has_vocab_prefix(lexicon[forest.data[i_root]], constants.UNIQUE_EMBEDDING):
                    forest.data[i_root] = lexicon[lex.vocab_prefix(constants.UNIQUE_EMBEDDING) + str(id_unique)]
                    #unique_collected[id_unique] = unique_collected.get(id_unique, []).append(i_root)
                    for _j, j_root in enumerate(forest.roots[i:]):
                        j = i + _j
                        if not lex.has_vocab_prefix(lexicon[forest.data[j_root]], constants.UNIQUE_EMBEDDING):
                            j_root_data_backup = forest.data[j_root]
                            forest.data[j_root] = forest.data[i_root]
                            if np.array_equal(trees[i], trees[j]):
                                forest.data[j_root] = lexicon[lex.vocab_prefix(constants.UNIQUE_EMBEDDING) + str(id_unique)]
                                #unique_collected[id_unique].append(j_root)
                            else:
                                forest.data[j_root] = j_root_data_backup
                    id_unique += 1

            logging.debug('unique collection finished')

            lexicon.pad()
            lexicon.dump(out_path)
            forest.dump(out_path)
        else:
            forest = sequ_trees.Forest(filename=out_path)

        # write out unique
        sim_tuples = [[forest.roots[tuple_idx * 2], forest.roots[tuple_idx * 2 + 1], scores[tuple_idx]]
                      for tuple_idx in range(len(scores))]
        write_sim_tuple_indices(path=out_path, sim_tuple_indices=sim_tuples, sizes=sizes)

    if FLAGS.sample_count:
        path_suffix = ''
        if FLAGS.sample_check_equality:
            path_suffix += '.equal'
        if FLAGS.sample_adapt_distribution:
            path_suffix += '.adapt'
        sampled_roots_fn = '%s.idx.negs%i%s' % (out_path, FLAGS.sample_count, path_suffix)
        if not os.path.isfile(sampled_roots_fn) or overwrite:
            trees = list(forest.trees())
            if FLAGS.create_unique:
                unique_root_data = forest.indices_to_forest(forest.roots)[0]
            else:
                unique_root_data = None
            if FLAGS.sample_adapt_distribution:
                logging.info('adapt distribution from data for sampling ...')
                logging.debug('calc sims_correct ...')
                sims_correct = np.zeros(shape=n)
                if FLAGS.create_unique:
                    sims_unique = {}
                    for i in range(n):
                        r1_data = unique_root_data[i * 2]
                        r2_data = unique_root_data[i * 2 + 1]
                        if (r1_data, r2_data) in sims_unique:
                            sims_correct[i] = sims_unique[(r1_data, r2_data)]
                        else:
                            sims_correct[i] = sim_jaccard(trees[i * 2][0], trees[i * 2 + 1][0])
                            sims_unique[(r1_data, r2_data)] = sims_correct[i]
                            sims_unique[(r2_data, r1_data)] = sims_correct[i]
                else:
                    for i in range(n):
                        sims_correct[i] = sim_jaccard(trees[i * 2][0], trees[i * 2 + 1][0])
                sims_correct.sort()
            else:
                sims_correct = None

            logging.info('start sampling with sample_count=%i ...' % FLAGS.sample_count)
            _sampled_root_indices = mytools.parallel_process_simple(range(n), partial(sample_indices,
                                                                                      trees=trees,
                                                                                      unique_root_data=unique_root_data,
                                                                                      sims_correct=sims_correct,
                                                                                      prog_bar=None,
                                                                                      check_equality=FLAGS.sample_check_equality))
            sampled_indices = np.concatenate(_sampled_root_indices)
            root_indices_sampled = sampled_indices.reshape((n, FLAGS.sample_count)) * 2 + 1
            logging.debug('dump sampled root indices to: %s' % sampled_roots_fn)
            root_indices_sampled.dump(sampled_roots_fn)
        else:
            root_indices_sampled = np.load(sampled_roots_fn)

        root_indices_original = np.array(range(n)).reshape(n, 1) * 2
        root_indices_all = np.concatenate([root_indices_original, root_indices_sampled], axis=1)
        # convert root indices to roots (indices in forest)
        sampled_tuples = []
        for tuple_idx in range(len(scores)):
            sampled_tuples.append([forest.roots[i] for i in root_indices_all[tuple_idx].tolist()])
        # write out tuple samples
        write_sim_tuple_indices(path=out_path, sim_tuple_indices=sampled_tuples, sizes=sizes,
                                path_suffix='.negs%i%s' % (FLAGS.sample_count, path_suffix))

    if FLAGS.sample_all and FLAGS.sample_count:
        all_samples = forest.sample_all(sample_count=FLAGS.sample_count)
        logging.info('write sim_tuple_indices to: %s.idx.negs%i' % (out_path, FLAGS.sample_count))
        np.array(all_samples).dump('%s.idx.negs%i' % (out_path, FLAGS.sample_count))


def load_sim_tuple_indices(filename, extensions=None):
    if extensions is None:
        extensions = ['']
    probs = []
    indices = []
    for ext in extensions:
        if not os.path.isfile(filename + ext):
            raise IOError('file not found: %s' % filename + ext)
        logging.debug('load idx file: %s' % filename + ext)
        _loaded = np.load(filename + ext).T
        if _loaded.dtype.kind == 'f':
            n = (len(_loaded) - 1) / 2
            _correct = _loaded[0].astype(int)
            _indices = _loaded[1:-n].astype(int)
            _probs = _loaded[-n:]
        else:
            n = (len(_loaded) - 1)
            _correct = _loaded[0]
            _indices = _loaded[1:]
            _probs = np.zeros(shape=(n, len(_correct)), dtype=np.float32)
        if len(indices) > 0:
            if not np.array_equal(indices[0][0], _correct):
                raise ValueError
        else:
            indices.append(_correct.reshape((1, len(_correct))))
            probs.append(np.ones(shape=(1, len(_correct)), dtype=np.float32))
        probs.append(_probs)
        indices.append(_indices)

    #loaded = zip(ids1, ids2, _loaded[2])
    #return loaded
    return np.concatenate(indices).T, np.concatenate(probs).T


# TODO: check changes
def merge_into_corpus(corpus_fn1, corpus_fn2):
    """
    Merges corpus2 into corpus1 e.g. merges types and vecs and converts data2 according to new types dict and writes
    training files from index files (file extension: .idx.<id>)

    :param corpus_fn1: file name of source corpus
    :param corpus_fn2: file name of target corpus
    :return:
    """
    #vecs, types = lex.load(corpus_fn1)
    lexicon1 = lex.Lexicon(filename=corpus_fn1)
    #vecs2, types2 = lex.load(corpus_fn2)
    lexicon2 = lex.Lexicon(filename=corpus_fn2)
    #vecs, types = lex.merge_dicts(vecs, types, vecs2, types2, add=True, remove=False)
    data_converter = lexicon1.merge(lexicon2, add=True, remove=False)
    #data2, parents2 = sequ_trees.load(corpus_fn2)
    sequence_trees2 = sequ_trees.Forest(filename=corpus_fn2, lexicon=lexicon1)
    #m = lex.mapping_from_list(types)
    #mapping = {i: m[t] for i, t in enumerate(types2)}
    #converter = [m[t] for t in types2]
    #data2_converted = np.array([mapping[d] for d in data2], dtype=data2.dtype)
    sequence_trees2.convert_data(converter=data_converter,
                                 new_idx_unknown=lexicon1[constants.vocab_manual[constants.UNKNOWN_EMBEDDING]])
    #dir2 = os.path.abspath(os.path.join(corpus_fn2, os.pardir))
    #indices2_fnames = fnmatch.filter(os.listdir(dir2), ntpath.basename(corpus_fn2) + '.idx.[0-9]*')
    #indices2 = [load_sim_tuple_indices(os.path.join(dir2, fn)) for fn in indices2_fnames]
    #children2, roots2 = sequ_trees.children_and_roots(parents2)
    #for i, sim_tuples in enumerate(indices2):
    #    write_sim_tuple_data('%s.merged.train.%i' % (corpus_fn1, i), sim_tuples, data2_converted, children2)
    #    write_sim_tuple_data_single('%s.merged.train.%i.single' % (corpus_fn1, i), sim_tuples, data2_converted, children2)
    #lex.dump('%s.merged' % corpus_fn1, vecs=vecs, types=types)
    sequence_trees2.dump('%s.merged' % corpus_fn1)
    lexicon1.dump('%s.merged' % corpus_fn1)