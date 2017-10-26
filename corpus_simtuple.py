import logging
import os
import sys

import numpy as np
import spacy
import tensorflow as tf

import corpus
import preprocessing

corpora_source_root = '/home/arne/devel/ML/data/corpora'


def set_flags(corpus_name, fn_train, fn_dev=None, fn_test=None, output_suffix=None):
    tf.flags.DEFINE_string(
        'corpus_data_input_train', corpora_source_root + '/' + corpus_name + '/' + fn_train,
        'The path to the ' + corpus_name + ' train data file.')
    tf.flags.DEFINE_string(
        'corpus_data_input_dev', corpora_source_root + '/' + corpus_name + '/' + fn_dev if fn_dev else None,
        'The path to the ' + corpus_name + ' dev data file.')
    tf.flags.DEFINE_string(
        'corpus_data_input_test',
        corpora_source_root + '/' + corpus_name + '/' + fn_test if fn_test else None,
        'The path to the ' + corpus_name + ' test data file.')
    tf.flags.DEFINE_string(
        'corpus_data_output_dir',
        '/media/arne/WIN/Users/Arne/ML/data/corpora/' + corpus_name,
        'The path to the output data files (samples, embedding vectors, mappings).')
    tf.flags.DEFINE_string(
        'corpus_data_output_fn', corpus_name + (output_suffix or ''),
        'Base filename of the output data files (samples, embedding vectors, mappings).')
    tf.flags.DEFINE_string(
        'sentence_processor',
        'process_sentence3_marked',
        'Which data types (features) are used to build the data sequence.')
    tf.flags.DEFINE_string(
        'concat_mode',
        #'sequence',
        'aggregate',
        #constants.default_inner_concat_mode,
        'How to concatenate the sentence-trees with each other. '
        'A sentence-tree represents the information regarding one sentence. '
        '"sequence" -> roots point to next root, '
        '"aggregate" -> roots point to an added, artificial token (AGGREGATOR) in the end of the token sequence'
        '(NOT ALLOWED for similarity scored tuples!) None -> do not concat at all')
    tf.flags.DEFINE_string(
        'inner_concat_mode',
        #'tree',
        None,
        #constants.default_inner_concat_mode,
        'How to concatenate the token-trees with each other. '
        'A token-tree represents the information regarding one token. '
        '"tree" -> use dependency parse tree'
        '"sequence" -> roots point to next root, '
        '"aggregate" -> roots point to an added, artificial token (AGGREGATOR) in the end of the token sequence'
        'None -> do not concat at all. This produces one sentence-tree per token.')
    tf.flags.DEFINE_integer(
        'count_threshold',
        1,
        #TODO: check if less or equal-less
        'remove token which occur less then count_threshold times in the corpus')
    tf.flags.DEFINE_integer(
        'neg_samples',
        0,
        'amount of negative samples to add'
    )


def distance_jaccard(ids1, ids2):
    ids1_set = set(ids1)
    ids2_set = set(ids2)
    return len(ids1_set & ids2_set) * 1.0 / len(ids1_set | ids2_set)


def create_corpus(reader_sentences, reader_score, FLAGS):
    """
    Creates a training corpus consisting of the following files (enumerated by file extension):
        * .train.0, .train.1, (.train.2.test):  training, development and, optionally, test data
        * .type:                                a types mapping, e.g. a list of types (strings), where its list
                                                position serves as index
        * .vec:                                 embedding vectors (indexed by types mapping)
    :param reader_sentences: a file reader that yields sentences of the tuples where every two succinct sentences come
                            from the same tuple: (sentence0, sentence1), (sentence2, sentence3), ...
    :param reader_score: a file reader that yields similarity scores in the range of [0.0..1.0]
    :param FLAGS: the flags
    """

    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
    logging.info('load spacy ...')
    nlp = spacy.load('en')
    nlp.pipeline = [nlp.tagger, nlp.entity, nlp.parser]

    vecs, types = corpus.get_dict_from_vocab(nlp.vocab)
    mapping = corpus.mapping_from_list(types)

    sentence_processor = getattr(preprocessing, FLAGS.sentence_processor)
    out_dir = os.path.abspath(os.path.join(FLAGS.corpus_data_output_dir, sentence_processor.func_name))
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    out_path = os.path.join(out_dir, FLAGS.corpus_data_output_fn)

    assert FLAGS.concat_mode is not None and FLAGS.concat_mode != 'tree', \
        "concat_mode=None or concat_mode='tree' is NOT ALLOWED for similarity scored tuples! Use 'sequence' or 'aggregate'"
    out_path = out_path + '_CM' + FLAGS.concat_mode
    if FLAGS.inner_concat_mode is not None:
        out_path = out_path + '_ICM' + FLAGS.inner_concat_mode

    def read_data(file_name):
        return corpus.parse_texts_scored(filename=file_name,
                                         reader=reader_sentences,
                                         reader_scores=reader_score,
                                         sentence_processor=sentence_processor,
                                         parser=nlp,
                                         mapping=mapping,
                                         concat_mode=FLAGS.concat_mode,
                                         inner_concat_mode=FLAGS.inner_concat_mode)

    file_names = [FLAGS.corpus_data_input_train]
    if FLAGS.corpus_data_input_dev:
        file_names.append(FLAGS.corpus_data_input_dev)
    if FLAGS.corpus_data_input_test:
        file_names.append(FLAGS.corpus_data_input_test)

    _data = [None]*len(file_names)
    _parents = [None]*len(file_names)
    _scores = [None]*len(file_names)
    sizes = []
    for i, fn in enumerate(file_names):
        _data[i], _parents[i], _scores[i], _ = read_data(fn)
        sizes.append(len(_scores[i]))

    logging.debug('sizes: %s' % str(sizes))
    data = np.concatenate(_data)
    parents = np.concatenate(_parents)
    scores = np.concatenate(_scores)

    types = corpus.revert_mapping_to_list(mapping)
    converter, vecs, types, new_counts, new_idx_unknown = corpus.sort_and_cut_and_fill_dict(data, vecs, types,
                                                                                            count_threshold=FLAGS.count_threshold)
    data = corpus.convert_data(data, converter, len(types), new_idx_unknown)
    logging.info('save data, parents, scores, vecs and types to: ' + out_path + ' ...')
    data.dump(out_path + '.data')
    parents.dump(out_path + '.parent')
    scores.dump(out_path + '.score')

    #one_hot_types = []
    #one_hot_types = [u'DEP#det', u'DEP#punct', u'DEP#pobj', u'DEP#ROOT', u'DEP#prep', u'DEP#aux', u'DEP#nsubj', u'DEP#dobj', u'DEP#amod', u'DEP#conj', u'DEP#cc', u'DEP#compound', u'DEP#nummod', u'DEP#advmod', u'DEP#acl', u'DEP#attr', u'DEP#auxpass', u'DEP#expl', u'DEP#nsubjpass', u'DEP#poss', u'DEP#agent', u'DEP#neg', u'DEP#prt', u'DEP#relcl', u'DEP#acomp', u'DEP#advcl', u'DEP#case', u'DEP#npadvmod', u'DEP#xcomp', u'DEP#ccomp', u'DEP#pcomp', u'DEP#oprd', u'DEP#nmod', u'DEP#mark', u'DEP#appos', u'DEP#dep', u'DEP#dative', u'DEP#quantmod', u'DEP#csubj', u'DEP#']
    one_hot_types = [t for t in types if t.startswith('DEP#')]
    mapping = corpus.mapping_from_list(types)
    one_hot_ids = [mapping[t] for t in one_hot_types]
    if len(one_hot_ids) > vecs.shape[1]:
        logging.warning('Setting more then vecs-size=%i lex entries to one-hot encoding.'
                        ' That overrides previously added one hot embeddings!' % vecs.shape[1])
    for i, idx in enumerate(one_hot_ids):
        vecs[idx] = np.zeros(shape=vecs.shape[1], dtype=vecs.dtype)
        vecs[idx][i % vecs.shape[1]] = 1.0

    corpus.write_dict(out_path, vecs=vecs, types=types)
    logging.info('the dataset contains ' + str(len(scores)) + ' scored text tuples')
    logging.debug('calc roots ...')
    children, roots = preprocessing.children_and_roots(parents)
    logging.debug('len(roots)=' + str(len(roots)))

    if FLAGS.neg_samples:
        # separate into subtrees (e.g. sentences)
        subtrees = []
        for root in roots:
            descendant_indices = preprocessing.get_descendant_indices(children, root)
            new_subtree = zip(*[(data[idx], parents[idx]) for idx in sorted(descendant_indices)])
            #if new_subtree in subtrees:
            #    repl_roots.append(root)
            subtrees.append(new_subtree)
            #TODO: check for replicates? (if yes, check successive tuples!)
        n = len(subtrees) / 2
        distances_correct = np.zeros(shape=n)
        for i in range(n):
            distances_correct[i] = distance_jaccard(subtrees[i * 2][0], subtrees[i * 2 +1][0])
        distances_correct.sort()

        new_subtrees = []
        for i in range(n):
            distances = np.zeros(shape=n)
            for j in range(n):
                distances[j] = distance_jaccard(subtrees[i * 2][0], subtrees[j * 2 + 1][0])
            correct_dist = distances[i]
            # calc p according to distances_correct (and set p[i]=0.0)
            distances_sorted = np.sort(distances)
            prob_map = {}
            last_cu = []
            last_co = []
            i_cu = i_co = 0
            while i_cu < n and i_co < n:
                if distances_sorted[i_cu] <= distances_correct[i_co]:
                    if not len(last_co) == 0:
                        for x in set(last_cu):
                            prob_map[x] = len(last_co) / float(len(last_cu) * n)
                        last_cu = []
                        last_co = []
                    if distances_sorted[i_cu] != correct_dist:
                        last_cu.append(distances_sorted[i_cu])
                    i_cu += 1
                else:
                    #if not len(last_cu) == 0:
                    last_co.append(distances_correct[i_co])
                    i_co += 1
            # add remaining
            while i_cu < n:
                if distances_sorted[i_cu] != correct_dist:
                    last_cu.append(distances_sorted[i_cu])
                i_cu += 1
            while i_co < n:
                last_co.append(distances_correct[i_co])
                i_co += 1
            if 1.0 not in last_cu:
                last_cu.append(1.0)
            for x in set(last_cu):
                prob_map[x] = len(last_co) / float(len(last_cu) * n)
            # remove
            prob_map[1.0] = 0.0

            p = [prob_map.get(d, 0.0) for d in distances]
            # normalize probs
            p = np.array(p)
            s = p.sum()
            p = p / s

            new_indices = np.random.choice(n, size=FLAGS.neg_samples, p=p)

            # add original
            current_new_subtrees = [subtrees[j * 2 + 1] for j in new_indices]
            current_subtrees = [subtrees[i * 2]] * FLAGS.neg_samples
            _current_new_subtrees = zip(current_subtrees, current_new_subtrees)
            new_subtrees.extend(_current_new_subtrees)
        # rearrange from tuples of subtrees to flattened list of subtrees
        new_subtrees = list(sum(new_subtrees, ()))
        # get data and parents
        new_data, new_parents = zip(*new_subtrees)
        new_data = np.concatenate(new_data)
        new_parents = np.concatenate(new_parents)
        new_scores = np.zeros(shape=n * FLAGS.neg_samples, dtype=scores.dtype)
        data = np.concatenate((data, new_data))
        parents = np.concatenate((parents, new_parents))
        scores = np.concatenate((scores, new_scores))
        logging.debug('calc roots ...')
        children, roots = preprocessing.children_and_roots(parents)
        logging.debug('new root count: %i' % len(roots))
        sizes[0] += n * FLAGS.neg_samples
        # TODO: implement for multiple data files (train, dev test)

    sim_tuples = [(i * 2, i * 2 + 1, scores[i]) for i in range(len(scores))]
    #sim_tuples = zip(np.array(range(len(scores)))*2, np.array(range(len(scores)))*2+1, scores)

    corpus.write_sim_tuple_data(out_path + '.train.0', sim_tuples[:sizes[0]], data, children, roots)
    if len(sizes) > 1:
        corpus.write_sim_tuple_data(out_path + '.train.1', sim_tuples[sizes[0]:sizes[0] + sizes[1]], data, children,
                                    roots)
    if FLAGS.corpus_data_input_test:
        corpus.write_sim_tuple_data(out_path + '.train.2.test', sim_tuples[sizes[0] + sizes[1]:], data,
                                    children, roots)


