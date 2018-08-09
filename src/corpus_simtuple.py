import logging
import os
import sys
from collections import Counter
from functools import partial

import numpy as np
import spacy
import tensorflow as tf

import constants
from lexicon import Lexicon
import preprocessing
from sequence_trees import Forest
import mytools


tf.flags.DEFINE_string('corpora_source_root',
                       #'/home/arne/devel/ML/data/corpora',
                       '/home/arne/DATA/ML/data/corpora_in',
                       'location of raw corpora directories')
tf.flags.DEFINE_string('corpora_target_root',
                       #'/media/arne/WIN/Users/Arne/ML/data/corpora',
                       '/home/arne/DATA/ML/data/corpora_out/corpora',
                       'location of raw corpora directories')
tf.flags.DEFINE_string(
    'sentence_processor',
    'process_sentence3_marked',
    'Which data types (features) are used to build the data sequence.')
#tf.flags.DEFINE_string(
#    'concat_mode',
#    # 'sequence',
#    'aggregate',
#    # constants.default_inner_concat_mode,
#    'How to concatenate the sentence-trees with each other. '
#    'A sentence-tree represents the information regarding one sentence. '
#    '"sequence" -> roots point to next root, '
#    '"aggregate" -> roots point to an added, artificial token (AGGREGATOR) in the end of the token sequence'
#    '(NOT ALLOWED for similarity scored tuples!) None -> do not concat at all')
#tf.flags.DEFINE_string(
#    'inner_concat_mode',
#    # 'tree',
#    None,
#    # constants.default_inner_concat_mode,
#    'How to concatenate the token-trees with each other. '
#    'A token-tree represents the information regarding one token. '
#    '"tree" -> use dependency parse tree'
#    '"sequence" -> roots point to next root, '
#    '"aggregate" -> roots point to an added, artificial token (AGGREGATOR) in the end of the token sequence'
#    'None -> do not concat at all. This produces one sentence-tree per token.')
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
#mytools.logging_init()
logging_format = '%(asctime)s %(levelname)s %(message)s'
#tf.logging._logger.propagate = False
#tf.logging._handler.setFormatter(logging.Formatter(logging_format))
#tf.logging._logger.format = logging_format
logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format=logging_format)


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
        mytools.numpy_dump(filename='%s.idx.%i%s' % (path, idx, path_suffix), ndarray=np.array(sim_tuple_indices[start:end]))
        start = end


def create_corpus(reader_sentences, reader_scores, corpus_name, file_names, output_suffix=None, overwrite=False,
                  reader_roots=None, reader_roots_args=None):
    """
    DEPRECATED
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

    #assert FLAGS.concat_mode is not None and FLAGS.concat_mode != 'tree', \
    #    "concat_mode=None or concat_mode='tree' is NOT ALLOWED for similarity scored tuples! Use 'sequence' or 'aggregate'"
    #out_path = out_path + '_cm' + FLAGS.concat_mode.upper()
    #if FLAGS.inner_concat_mode is not None:
    #    out_path = out_path + '_icm' + FLAGS.inner_concat_mode.upper()

    if FLAGS.one_hot_dep:
        out_path = out_path + '_onehotdep'

    if FLAGS.random_vecs:
        out_path = out_path + '_randomvecs'

    if not (Forest.exist(out_path) and Lexicon.exist(out_path) and os.path.isfile(out_path + '.score')) or overwrite:
        logging.info('load spacy ...')
        nlp = spacy.load('en')

        logging.info('extract lexicon and vecs from spacy vocab ...')
        #lexicon = Lexicon(nlp_vocab=nlp.vocab)
        lexicon = Lexicon()
        if reader_roots_args is None:
            reader_roots_args = {'root_label': constants.vocab_manual[constants.ROOT_EMBEDDING]}

        def read_data(file_name):
            logging.info('convert texts scored ...')
            logging.debug('lexicon size: %i' % len(lexicon))
            _forest = lexicon.read_data(reader=reader_sentences, sentence_processor=sentence_processor,
                                        parser=nlp, reader_args={'filename': file_name},
                                        batch_size=10000, concat_mode='sequence',
                                        inner_concat_mode='tree', expand_dict=True,
                                        reader_roots=reader_roots,
                                        reader_roots_args=reader_roots_args,
                                        return_hashes=True)
            logging.debug('lexicon size: %i (after parsing)' % len(lexicon))
            _s = np.fromiter(reader_scores(file_name), np.float)
            logging.info('scores read: %i' % len(_s))
            assert 2 * len(_s) == len(_forest.roots), 'len(roots): %i != 2 * len(scores): %i' % (
                2 * len(_s), len(_forest.roots))
            return _forest, _s

        file_names = [os.path.join(FLAGS.corpora_source_root, corpus_name, fn) for fn in file_names]

        _forests, _scores = zip(*[read_data(fn) for fn in file_names])

        sizes = [len(s) for s in _scores]

        logging.debug('sizes: %s' % str(sizes))
        np.array(sizes).dump(out_path + '.size')

        # forest = Forest(forest=np.concatenate(_forests, axis=1), lexicon=lexicon)
        forest = Forest.concatenate(_forests)
        scores = np.concatenate(_scores)

        logging.info('add vocab_manual ... ')
        lexicon.add_all(constants.vocab_manual.values())

        logging.info('sort and cut lexicon ... ')
        keep_hash_values = [lexicon.strings[s] for s in constants.vocab_manual.values()]
        lexicon.sort_and_cut_and_fill_dict(data=forest.data, keep_values=keep_hash_values, count_threshold=FLAGS.count_threshold)

        logging.info('init vecs: use nlp vocab and fill missing ...')
        lexicon.init_vecs(vocab=nlp.vocab)

        lexicon.set_to_mean(indices=lexicon.ids_fixed, indices_as_blacklist=True)

        if FLAGS.one_hot_dep:
            lexicon.set_to_onehot(prefix=constants.TYPE_DEPENDENCY_RELATION)

        # convert data: hashes to indices
        forest.hashes_to_indices()

        # convert and set children arrays
        #forest.children_dict_to_arrays()
        forest.set_children_with_parents()

        forest.dump(out_path)
        scores.dump(out_path + '.score')
        lexicon.set_man_vocab_vec(man_vocab_id=constants.IDENTITY_EMBEDDING)
        if FLAGS.random_vecs:
            lexicon.set_to_random()
        lexicon.dump(out_path)
    else:
        lexicon = Lexicon(filename=out_path)
        forest = Forest(filename=out_path, lexicon=lexicon)
        scores = np.load(out_path + '.score')
        sizes = np.load(out_path + '.size')

    sim_tuples = [[forest.roots[tuple_idx * 2], forest.roots[tuple_idx * 2 + 1], scores[tuple_idx]]
                  for tuple_idx in range(len(scores))]
    write_sim_tuple_indices(path=out_path, sim_tuple_indices=sim_tuples, sizes=sizes)

    n = len(scores)
    logging.info('the dataset contains %i scored text tuples' % n)

    # DEPRECATED
    # UNIQUE_EMBEDDING: u'UNIQUE'
    if FLAGS.create_unique:
        if forest.data_as_hashes:
            raise NotImplementedError('create_unique not implemented for data_as_hashes')
        # use unique by now
        out_path += '.unique'
        if not (Forest.exist(out_path) and Lexicon.exist(out_path)) or overwrite:
            # separate into trees (e.g. sentences)
            logging.debug('split into trees ...')
            trees = list(forest.trees())
            assert n == len(trees) / 2, '(subtree_count / 2)=%i does not fit score_count=%i' % (len(trees) / 2, n)
            logging.info('collect unique ...')
            id_unique = 0
            #unique_collected = {}
            for i, i_root in enumerate(forest.roots):
                #logging.debug('i_root: %i' % i_root)
                if not Lexicon.has_vocab_prefix(lexicon[forest.data[i_root]], constants.UNIQUE_EMBEDDING):
                    forest.data[i_root] = lexicon[Lexicon.vocab_prefix(constants.UNIQUE_EMBEDDING) + str(id_unique)]
                    #unique_collected[id_unique] = unique_collected.get(id_unique, []).append(i_root)
                    for _j, j_root in enumerate(forest.roots[i:]):
                        j = i + _j
                        if not Lexicon.has_vocab_prefix(lexicon[forest.data[j_root]], constants.UNIQUE_EMBEDDING):
                            j_root_data_backup = forest.data[j_root]
                            forest.data[j_root] = forest.data[i_root]
                            if np.array_equal(trees[i][0], trees[j][0]) and np.array_equal(trees[i][1], trees[j][1]):
                                forest.data[j_root] = lexicon[Lexicon.vocab_prefix(constants.UNIQUE_EMBEDDING) + str(id_unique)]
                                #unique_collected[id_unique].append(j_root)
                            else:
                                forest.data[j_root] = j_root_data_backup
                    id_unique += 1

            logging.debug('unique collection finished')

            lexicon.pad()
            lexicon.dump(out_path)
            forest.dump(out_path)
        else:
            forest = Forest(filename=out_path)

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



