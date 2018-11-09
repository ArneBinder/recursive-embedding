import logging
import os
from datetime import datetime
from functools import partial

import numpy as np
import plac
import spacy
from spacy.strings import hash_string

from constants import DTYPE_HASH, DTYPE_COUNT, UNKNOWN_EMBEDDING, vocab_manual, LOGGING_FORMAT, OFFSET_SEEALSO_ROOT, \
    DTYPE_IDX, CLASSES_FNS
from lexicon import Lexicon, FE_STRINGS
from mytools import numpy_dump, numpy_load, numpy_exists
from sequence_trees import Forest, FE_ROOT_ID, FE_ROOT_POS

#FE_RESOURCE_HASHES = 'resource.hash'
FE_ROOT_ID_FAILED = 'root.id.failed'
FE_FAILED = 'failed'
FE_UNIQUE_HASHES = 'unique.hash'
FE_COUNTS = 'count'
FE_UNIQUE_HASHES_FILTERED = 'unique.hash.filtered'
FE_UNIQUE_HASHES_DISCARDED = 'unique.hash.discarded'
FE_UNIQUE_COUNTS_FILTERED = 'count.hash.filtered'
FE_UNIQUE_COUNTS_DISCARDED = 'count.hash.discarded'
FE_ROOT_SEEALSO_COUNT = 'root.seealso.count'
FE_ROOT_CONTEXT_SIZE = 'root.context.size'
FE_CLASS_IDS = 'classes.id'
FE_CLASS_STRINGS = 'classes.strings'

DIR_BATCHES = 'batches'
DIR_BATCHES_CONVERTED = 'batches_converted'
DIR_MERGED = 'merged'

PREFIX_FN = 'forest'


logger = logging.getLogger('corpus')
logger.setLevel(logging.DEBUG)
logger_streamhandler = logging.StreamHandler()
logger_streamhandler.setLevel(logging.DEBUG)
logger_streamhandler.setFormatter(logging.Formatter(LOGGING_FORMAT))
logger.addHandler(logger_streamhandler)
logger.propagate = False


def load_class_ids(dir_path, prefix_type):
    fn = '%s.%s.%s' % (dir_path, CLASSES_FNS.get(prefix_type, prefix_type.replace('/', '-')), FE_CLASS_IDS)
    logger.debug('load class ids for %s from: %s' % (prefix_type, fn))
    classes_ids = numpy_load(filename=fn)
    logger.info('number of classes for %s to predict: %s' % (prefix_type, len(classes_ids)))
    fn_strings = '%s.%s.%s' % (dir_path, CLASSES_FNS.get(prefix_type, prefix_type.replace('/', '-')), FE_CLASS_STRINGS)
    classes_strings = None
    if os.path.exists(fn_strings):
        logger.debug('load class strings for %s to: %s' % (prefix_type, fn_strings))
        classes_strings = list((s[:-1] for s in open(fn_strings).readlines()))
    return classes_ids, classes_strings


def save_class_ids(dir_path, prefix_type, classes_ids, classes_strings=None):
    logger.info('number of classes for %s to predict: %i.' % (prefix_type, len(classes_ids)))
    fn = '%s.%s.%s' % (dir_path, CLASSES_FNS.get(prefix_type, prefix_type.replace('/', '-')), FE_CLASS_IDS)
    logger.debug('save class ids for %s to: %s' % (prefix_type, fn))
    numpy_dump(filename=fn, ndarray=classes_ids)
    if classes_strings is not None:
        fn_strings = '%s.%s.%s' % (dir_path, CLASSES_FNS.get(prefix_type, prefix_type.replace('/', '-')), FE_CLASS_STRINGS)
        logger.debug('save class strings for %s to: %s' % (prefix_type, fn_strings))
        with open(fn_strings, 'w') as f:
            f.writelines(map(lambda s: s+'\n', classes_strings))


def process_records(records, out_base_name, record_reader, sentence_processor, parser, concat_mode='sequence',
                    batch_size=1000, n_threads=4, adjust_forest_func=None):
    if not Lexicon.exist(out_base_name, types_only=True) \
            or not Forest.exist(out_base_name) \
            or not numpy_exists('%s.%s' % (out_base_name, FE_UNIQUE_HASHES)) \
            or not numpy_exists('%s.%s' % (out_base_name, FE_COUNTS)):
        _reader = partial(record_reader, records=records)
        logger.debug('parse records ...')
        # forest, lexicon, lexicon_roots = corpus.process_records(parser=nlp, reader=_reader)

        lexicon = Lexicon()
        forest = lexicon.read_data(reader=_reader, sentence_processor=sentence_processor,
                                   parser=parser, batch_size=batch_size, concat_mode=concat_mode,
                                   inner_concat_mode='tree', expand_dict=True, as_tuples=True,
                                   return_hashes=True, n_threads=n_threads)
        if adjust_forest_func is not None:
            forest = adjust_forest_func(forest)

        forest.set_children_with_parents()
        #roots = forest.roots
        # ids are at one position after roots
        #root_ids = forest.data[roots + OFFSET_ID]
        #forest.set_root_ids(root_ids=root_ids)
        forest.set_root_data_by_offset()

        #out_path = os.path.join(out_path, os.path.basename(in_file))
        lexicon.dump(filename=out_base_name, strings_only=True)
        forest.dump(filename=out_base_name)

        unique, counts = np.unique(forest.data, return_counts=True)
        # unique.dump('%s.%s' % (filename, FE_UNIQUE_HASHES))
        # counts.dump('%s.%s' % (filename, FE_COUNTS))
        numpy_dump('%s.%s' % (out_base_name, FE_UNIQUE_HASHES), unique)
        numpy_dump('%s.%s' % (out_base_name, FE_COUNTS), counts)
    else:
        logger.debug('%s was already processed' % out_base_name)


def collect_file_names(out_path_batches):
    logger.info('collect file names ...')
    t_start = datetime.now()
    suffix = '.' + FE_STRINGS
    l = len(suffix)
    f_names = []
    for file in os.listdir(out_path_batches):
        if file.endswith(suffix) and Forest.exist(os.path.join(out_path_batches, file[:-l])):
            f_names.append(file[:-l])
    try:
        f_names = sorted(f_names, key=lambda fn: int(fn[len(PREFIX_FN)+1:]))
    except ValueError:
        logger.warning('Could not sort file names by NUMBER (expected format: "%s-NUMBER"). Sort as strings.' % PREFIX_FN)
    f_paths = [os.path.join(out_path_batches, f) for f in f_names]
    logger.info('finished. %s' % str(datetime.now()-t_start))
    return f_names, f_paths


def collect_counts_merged(f_paths):
    logger.info('collect counts ...')
    t_start = datetime.now()
    counts_merged = {}
    for fn in f_paths:
        #counts = np.load('%s.%s' % (fn, FE_COUNTS))
        #uniques = np.load('%s.%s' % (fn, FE_UNIQUE_HASHES))
        counts = numpy_load('%s.%s' % (fn, FE_COUNTS), assert_exists=True)
        uniques = numpy_load('%s.%s' % (fn, FE_UNIQUE_HASHES), assert_exists=True)
        for i, c in enumerate(counts):
            _c = counts_merged.get(uniques[i], 0)
            counts_merged[uniques[i]] = _c + c
    logger.info('finished. %s' % str(datetime.now() - t_start))
    return counts_merged


def collect_root_ids(f_paths, out_path_merged):
    logger.info('collect root_ids ...')
    fn_root_ids = '%s.%s' % (out_path_merged, FE_ROOT_ID)
    if os.path.exists(fn_root_ids):
        logger.info('found root_ids (%s). load from file.' % fn_root_ids)
        #return np.load(fn_root_ids)
        return numpy_load(fn_root_ids, assert_exists=True)

    t_start = datetime.now()
    root_ids = []
    for fn in f_paths:
        #root_ids.append(np.load('%s.%s' % (fn, FE_ROOT_ID)))
        root_ids.append(numpy_load('%s.%s' % (fn, FE_ROOT_ID), assert_exists=True))
    root_ids = np.concatenate(root_ids)

    count_root_ids_unique = len(np.unique(root_ids))
    assert len(root_ids) == count_root_ids_unique, '%i root ids are duplicated' \
                                                   % (len(root_ids) - count_root_ids_unique)
    #root_ids.dump(fn_root_ids)
    numpy_dump(fn_root_ids, root_ids)
    logger.info('finished. %s' % str(datetime.now()-t_start))
    return root_ids


def filter_uniques(f_paths, out_path_merged, min_count=None, coverage=None):

    fn_uniques_filtered = '%s.%s' % (out_path_merged, FE_UNIQUE_HASHES_FILTERED)
    fn_uniques_discarded = '%s.%s' % (out_path_merged, FE_UNIQUE_HASHES_DISCARDED)
    fn_counts_filtered = '%s.%s' % (out_path_merged, FE_UNIQUE_COUNTS_FILTERED)
    fn_counts_discarded = '%s.%s' % (out_path_merged, FE_UNIQUE_COUNTS_DISCARDED)
    if os.path.exists(fn_uniques_filtered):
        logger.info('found uniques_filtered (%s). load from file.' % fn_uniques_filtered)
        assert os.path.exists(fn_uniques_discarded), 'found uniques_filtered (%s), but misses files for ' \
                                                     'uniques_discarded (%s).' % (fn_uniques_filtered,
                                                                                  fn_uniques_discarded)
        assert os.path.exists(fn_counts_filtered), 'found uniques_filtered (%s), but misses files for ' \
                                                   'counts_filtered (%s).' % (fn_uniques_filtered, fn_counts_filtered)
        assert os.path.exists(fn_counts_discarded), 'found uniques_filtered (%s), but misses files for ' \
                                                    'counts_discarded (%s).' % (fn_uniques_filtered,
                                                                                fn_counts_discarded)
        #return np.load(fn_uniques_filtered)
        return numpy_load(fn_uniques_filtered, assert_exists=True)

    counts_merged = collect_counts_merged(f_paths)
    root_id_hashes = collect_root_ids(f_paths, out_path_merged)
    root_ids_set = set(root_id_hashes)
    assert len(root_ids_set) == len(root_id_hashes), 'root_id_hashes contains %i duplicates' % (len(root_id_hashes) - len(root_ids_set))

    # calculate coverage
    unique, counts = zip(*((v, counts_merged[v]) for v in counts_merged.keys()))
    unique = np.array(unique)
    counts = np.array(counts)
    unique_mapping = {u: i for i, u in enumerate(unique)}
    root_ids_arg = np.array([unique_mapping[root_id_hash] for root_id_hash in root_id_hashes])
    non_root_ids_indices = np.delete(np.arange(len(unique)), root_ids_arg)
    #unique_no_roots = unique[non_root_ids_indices]
    counts_no_roots = counts[non_root_ids_indices]
    if coverage is not None and coverage > 0:
        count_uniques, count_counts = np.unique(counts_no_roots, return_counts=True)
        count_counts_sorted_indices = np.argsort(count_uniques)
        count_uniques_sorted = count_uniques[count_counts_sorted_indices]
        count_counts_sorted = count_counts[count_counts_sorted_indices]
        counts_summed = np.cumsum((count_uniques_sorted * count_counts_sorted)[::-1])
        counts_coverage = counts_summed / float(counts_summed[-1])
        if counts_coverage[-1] != 1.0:
            logger.warning('Maximal calculated counts_coverage is not 1.0 (%f). Set it to 1.0.' % counts_coverage[-1])
        idx_rev = np.searchsorted(counts_coverage, coverage)
        min_count = count_uniques_sorted[-(idx_rev+1)]
        coverage = counts_coverage[idx_rev]
    else:
        assert min_count is not None and min_count > 0, 'coverage and min_count are None or invalid (less or equal zero)'
        counts_sum = np.sum(counts_no_roots)
        counts_filtered_sum = np.sum(counts_no_roots[np.where(counts_no_roots >= min_count)])
        coverage = float(counts_filtered_sum) / counts_sum
    logger.info('data coverage (without roots) for min_count=%i: %.3f' % (min_count, coverage))

    logger.info('filter uniques by count ...')
    t_start = datetime.now()
    uniques_filtered = np.zeros(shape=len(counts_merged.keys()) - len(root_id_hashes), dtype=DTYPE_HASH)
    uniques_discarded = np.zeros(shape=len(counts_merged.keys()) - len(root_id_hashes), dtype=DTYPE_HASH)
    counts_filtered = np.zeros(shape=len(counts_merged.keys()) - len(root_id_hashes), dtype=DTYPE_COUNT)
    counts_discarded = np.zeros(shape=len(counts_merged.keys()) - len(root_id_hashes), dtype=DTYPE_COUNT)
    i_filtered = 0
    i_discarded = 0
    for u in counts_merged.keys():
        #if (u not in root_ids_set and counts_merged[u] >= min_count) \
        #        or (u in root_ids_set and counts_merged[u] >= min_count_root_id >= 0):
        if u in root_ids_set:
            pass
        elif counts_merged[u] >= min_count:
            uniques_filtered[i_filtered] = u
            counts_filtered[i_filtered] = counts_merged[u]
            i_filtered += 1
        else:
            uniques_discarded[i_discarded] = u
            counts_discarded[i_discarded] = counts_merged[u]
            i_discarded += 1
    uniques_filtered = uniques_filtered[:i_filtered]
    uniques_discarded = uniques_discarded[:i_discarded]
    counts_filtered = counts_filtered[:i_filtered]
    counts_discarded = counts_discarded[:i_discarded]
    #uniques_filtered.dump(fn_uniques_filtered)
    #uniques_discarded.dump(fn_uniques_discarded)
    #counts_filtered.dump(fn_counts_filtered)
    #counts_discarded.dump(fn_counts_discarded)
    numpy_dump(fn_uniques_filtered, uniques_filtered)
    numpy_dump(fn_uniques_discarded, uniques_discarded)
    numpy_dump(fn_counts_filtered, counts_filtered)
    numpy_dump(fn_counts_discarded, counts_discarded)
    logger.info('%i entries remain in lexicon, %i were discarded.' % (i_filtered, i_discarded))

    logger.info('finished. %s' % str(datetime.now() - t_start))
    return uniques_filtered, root_id_hashes


def merge_and_filter_lexicon(uniques_filtered, root_id_hashes, f_paths, out_path_merged):
    logger.info('merge and filter lexicon ...')
    fn_lexicon_discarded = '%s.discarded' % out_path_merged
    fn_lexicon_root_id_hashes = '%s.root.id' % out_path_merged
    if Lexicon.exist(filename=out_path_merged, types_only=True):
        logger.info('found lexicon (%s). load from file.' % out_path_merged)
        assert Lexicon.exist(filename=fn_lexicon_discarded, types_only=True), \
            'found lexicon (%s), but misses lexicon_discarded (%s).' % (out_path_merged, fn_lexicon_discarded)
        assert Lexicon.exist(filename=fn_lexicon_root_id_hashes, types_only=True), \
            'found lexicon (%s), but misses lexicon_root_data (%s).' % (out_path_merged, fn_lexicon_root_id_hashes)
        # Note: Load with vecs to skip _lexicon_add_vecs, eventually.
        return Lexicon(filename=out_path_merged, load_vecs=False), Lexicon(filename=fn_lexicon_root_id_hashes, load_vecs=False)
    t_start = datetime.now()
    uniques_filtered_set = set(uniques_filtered)
    root_id_hashes_set = set(root_id_hashes)
    lexicon = Lexicon()
    lexicon.add_all(vocab_manual.values())
    lexicon_discarded = Lexicon()
    lexicon_root_data = Lexicon()
    for fn in f_paths:
        lex = Lexicon(filename=fn)
        for s in lex.strings:
            h = hash_string(s)
            if h in uniques_filtered_set:
                lexicon.add(s)
            elif h in root_id_hashes_set:
                lexicon_root_data.add(s)
            else:
                lexicon_discarded.add(s)
    lexicon.dump(filename=out_path_merged, strings_only=True)
    lexicon_discarded.dump(filename=fn_lexicon_discarded, strings_only=True)

    #for root_id in root_id_hashes:
    #    root_id_s = lexicon_discarded.strings[root_id]
    #    lexicon_root_data.strings.add(root_id_s)

    # Has to be ordered by root_id_hashes. They do not occur in the order of the roots in the data because of links
    # that can appear before the root they are pointing to.
    logger.debug('order by lexicon_roots by hashes...')
    lexicon_root_data.order_by_hashes(root_id_hashes)
    lexicon_root_data.dump(filename=fn_lexicon_root_id_hashes, strings_only=True)

    logger.info('finished. %s' % str(datetime.now() - t_start))
    return lexicon, lexicon_root_data


def filter_and_convert_data_batches(lexicon, lexicon_roots, f_names, out_dir_batches, out_dir_batches_converted):
    logger.info('filter and convert batches ...')
    t_start = datetime.now()
    assert vocab_manual[UNKNOWN_EMBEDDING] in lexicon.strings or not lexicon.frozen, 'UNKNOWN_EMBEDDING not in ' \
                                                                                     'lexicon, but it is frozen'
    lexicon.add(vocab_manual[UNKNOWN_EMBEDDING])
    assert vocab_manual[UNKNOWN_EMBEDDING] in lexicon_roots.strings or not lexicon_roots.frozen, 'UNKNOWN_EMBEDDING not in ' \
                                                                                     'lexicon_roots, but it is frozen'
    lexicon_roots.add(vocab_manual[UNKNOWN_EMBEDDING])
    count_skipped = 0
    for fn in f_names:
        fn_path_in = os.path.join(out_dir_batches, fn)
        fn_path_out = os.path.join(out_dir_batches_converted, fn)
        if Forest.exist(filename=fn_path_out):
            count_skipped += 1
            continue
        forest = Forest(filename=fn_path_in, lexicon=lexicon, lexicon_roots=lexicon_roots)
        forest.hashes_to_indices()
        forest.dump(filename=fn_path_out)
    logger.info('finished (processed: %i, skipped: %i). %s' % (len(f_names) - count_skipped, count_skipped,
                                                               str(datetime.now() - t_start)))


def lexicon_add_vecs(lexicon, out_path_merged):
    logger.info('add vecs ...')
    if lexicon.has_vecs:
        logger.info('lexicon has vecs already.')
        return lexicon
    t_start = datetime.now()
    logger.info('lexicon size: %i' % len(lexicon))
    logger.info('load spacy ...')
    nlp = spacy.load('en')
    lexicon.init_vecs(vocab=nlp.vocab)
    logger.info('lexicon fixed size: %i' % len(lexicon.ids_fixed))
    lexicon.set_to_random(indices=lexicon.ids_fixed, indices_as_blacklist=True)
    lexicon.dump(filename=out_path_merged)
    logger.info('finished. %s' % str(datetime.now() - t_start))
    return lexicon


def merge_converted_batches(f_names, out_dir_batches_converted, out_path_merged):
    logger.info('merge converted batches ...')
    if Forest.exist(out_path_merged):
        logger.info('found forest_merged (%s). load from file.' % out_path_merged)
        return Forest(filename=out_path_merged)
    t_start = datetime.now()
    forests = []
    for fn in f_names:
        fn_path_out = os.path.join(out_dir_batches_converted, fn)
        forests.append(Forest(filename=fn_path_out))
    forest_merged = Forest.concatenate(forests)
    forest_merged.dump(filename=out_path_merged)
    logger.info('finished. %s' % str(datetime.now() - t_start))
    return forest_merged


def collect_root_seealso_counts(forest_merged, out_path_merged):
    logger.info('collect root seealso counts ...')
    fn_root_seealso_counts = '%s.%s' % (out_path_merged, FE_ROOT_SEEALSO_COUNT)
    if os.path.exists(fn_root_seealso_counts):
        logger.info('found root_seealso_counts (%s). load from file.' % fn_root_seealso_counts)
        #return np.load(fn_root_seealso_counts)
        return numpy_load(fn_root_seealso_counts, assert_exists=True)
    t_start = datetime.now()
    root_seealso_counts = forest_merged.get_children_counts(forest_merged.roots + OFFSET_SEEALSO_ROOT)
    #root_seealso_counts.dump(fn_root_seealso_counts)
    numpy_dump(fn_root_seealso_counts, root_seealso_counts)
    logger.info('finished. %s' % str(datetime.now()-t_start))
    return root_seealso_counts


def collect_root_context_sizes(forest_merged, out_path_merged, root_seealso_counts=None):
    logger.info('collect root context sizes ...')
    fn_root_context_sizes = '%s.%s' % (out_path_merged, FE_ROOT_CONTEXT_SIZE)
    if os.path.exists(fn_root_context_sizes):
        logger.info('found root_context_sizes (%s). load from file.' % fn_root_context_sizes)
        #return np.load(fn_root_seealso_counts)
        return numpy_load(fn_root_context_sizes, assert_exists=True)
    t_start = datetime.now()
    #root_seealso_counts = forest_merged.get_children_counts(forest_merged.roots + OFFSET_SEEALSO_ROOT)
    #root_seealso_counts.dump(fn_root_seealso_counts)

    # get node counts of roots by root positions
    root_shifted = np.concatenate([forest_merged.roots[1:], [len(forest_merged)]])
    root_length = root_shifted - forest_merged.roots
    # use root_seealso_counts, if available
    if root_seealso_counts is not None:
        root_context_sizes = (root_length - (root_seealso_counts * 2 + 1)) - OFFSET_SEEALSO_ROOT
    else:
        # otherwise: last child of root points to first textual data node
        last_root_child_offsets = np.array([forest_merged.get_children(root)[-1] for root in forest_merged.roots])
        root_context_sizes = root_length - last_root_child_offsets

    numpy_dump(fn_root_context_sizes, root_context_sizes)
    logger.info('finished. %s' % str(datetime.now()-t_start))
    return root_context_sizes


@plac.annotations(
    out_path=('corpora out path', 'option', 'o', str),
    min_count=('minimal count a token has to occur to stay in the lexicon', 'option', 'c', int),
    coverage=('percentage of tokens that should remain in the data. Tokens that occur at least are set to UNKNOWN.',
              'option', 'v', float),
    use_see_also_counts=('use SeeAlso counts to help determining length of textual data', 'flag', 'u'),
    # shows the coverage for min_count=100:
    # 1 - (len(data) - sum(counts_sorted[-len(np.where(counts_sorted >= 100)[0]):])) / float(len(data))
    # 0.951
    # min_count_root_id=('minimal count a root_id has to occur to stay in the lexicon', 'option', 'r', int),
)
def merge_batches(out_path, min_count=1, coverage=-1, use_see_also_counts=False, *unused):  # , min_count_root_id=-1):
    logger_fh = logging.FileHandler(os.path.join(out_path, 'corpus-merge.info.log'))
    logger_fh.setLevel(logging.INFO)
    logger_fh.setFormatter(logging.Formatter(LOGGING_FORMAT))
    logger.addHandler(logger_fh)

    logger_fh = logging.FileHandler(os.path.join(out_path, 'corpus-merge.debug.log'))
    logger_fh.setLevel(logging.DEBUG)
    logger_fh.setFormatter(logging.Formatter(LOGGING_FORMAT))
    logger.addHandler(logger_fh)

    # logger.info('min_count=%i min_count_root_id=%i out_path=%s' % (min_count, min_count_root_id, out_path))
    logger.info('min_count=%i out_path=%s' % (min_count, out_path))

    out_dir_batches = os.path.join(out_path, DIR_BATCHES)
    out_dir_batches_converted = os.path.join(out_path, DIR_BATCHES_CONVERTED)
    if not os.path.exists(out_dir_batches_converted):
        os.mkdir(out_dir_batches_converted)
    out_path_merged = os.path.join(out_path, DIR_MERGED)
    if not os.path.exists(out_path_merged):
        os.mkdir(out_path_merged)
    out_path_merged = os.path.join(out_path_merged, PREFIX_FN)

    f_names, f_paths = collect_file_names(out_dir_batches)

    uniques_filtered, root_id_hashes = filter_uniques(f_paths, out_path_merged, min_count, coverage)

    lexicon, lexicon_root_ids = merge_and_filter_lexicon(uniques_filtered, root_id_hashes, f_paths, out_path_merged)

    #id_offset_mapping = {o: i for i, o in enumerate(root_id_hashes)}

    filter_and_convert_data_batches(lexicon, lexicon_root_ids, f_names, out_dir_batches, out_dir_batches_converted)

    lexicon = lexicon_add_vecs(lexicon, out_path_merged)

    forest_merged = merge_converted_batches(f_names, out_dir_batches_converted, out_path_merged)
    forest_merged.set_lexicon(lexicon)
    forest_merged.set_lexicon_roots(lexicon_root_ids)

    if use_see_also_counts:
        root_seealso_counts = collect_root_seealso_counts(forest_merged, out_path_merged)
    else:
        root_seealso_counts = None

    root_context_sizes = collect_root_context_sizes(forest_merged, out_path_merged, root_seealso_counts)

    return forest_merged, out_path_merged


@plac.annotations(
    merged_forest_path=('path to merged forest', 'option', 'o', str),
    split_count=('count of produced index files', 'option', 'c', int),
    start_root=('start root index', 'option', 'b', int),
    end_root=('end root index', 'option', 'e', int),
    step_root=('root step', 'option', 's', int),
    suffix=('file name suffix', 'option', 'f', str),
)
def create_index_files(merged_forest_path, split_count=2, start_root=0, end_root=-1, step_root=1, suffix='', *unused):
    logger_fh = logging.FileHandler(os.path.join(merged_forest_path, '../..', 'corpus-indices.log'))
    logger_fh.setLevel(logging.INFO)
    logger_fh.setFormatter(logging.Formatter(LOGGING_FORMAT))
    logger.addHandler(logger_fh)

    logger.info('split_count=%i out_path=%s' % (split_count, merged_forest_path))

    root_pos = numpy_load('%s.%s' % (merged_forest_path, FE_ROOT_POS), assert_exists=True)
    #logger.info('total number of indices: %i' % len(root_pos))
    if end_root <= 0:
        end_root = len(root_pos)
    else:
        assert end_root * step_root <= len(root_pos), 'end_root * step_root: %i is larger then available trees: %i' \
                                                      % (end_root * step_root, len(root_pos))
        end_root = end_root * step_root
    logger.info('use trees [%i:%i] (step: %i; total number of trees: %i)'
                % (start_root, end_root, step_root, len(root_pos)))
    indices = np.arange(start=start_root*step_root, stop=end_root, step=step_root, dtype=DTYPE_IDX)
    np.random.shuffle(indices)
    if len(suffix) > 0:
        suffix = suffix + '.'
    for i, split in enumerate(np.array_split(indices, split_count)):
        numpy_dump('%s.idx.%s%i' % (merged_forest_path, suffix, i), split)