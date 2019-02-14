import csv
import io

import numpy as np
from spacy.strings import StringStore, hash_string

import constants
import logging
import os

from preprocessing import read_data, without_prefix, PREFIX_LEX
from constants import vocab_manual, DTYPE_COUNT, DTYPE_HASH, DTYPE_IDX, DTYPE_VECS, LOGGING_FORMAT, \
    UNKNOWN_EMBEDDING, LINK_TYPES, TYPE_LEXEME, SEPARATOR
from sequence_trees import Forest, concatenate_graphs
from mytools import numpy_dump, numpy_load, numpy_exists, getOrAdd

logger = logging.getLogger('lexicon')
logger.setLevel(logging.DEBUG)
#logger_streamhandler = logging.StreamHandler()
#logger_streamhandler.setLevel(logging.DEBUG)
#logger_streamhandler.setFormatter(logging.Formatter(LOGGING_FORMAT))
#logger.addHandler(logger_streamhandler)
#logger.propagate = False


# DEPRECATED
# TODO: adapt for StringStore
def sort_and_cut_and_fill_dict_DEP(seq_data, vecs, strings, types, count_threshold=1):
    logger.info('sort, cut and fill embeddings ...')
    new_max_size = len(strings)
    logger.info('initial vecs shape: %s ' % str(vecs.shape))
    logger.info('initial strings size: %i' % len(strings))
    # count types
    logger.debug('calculate counts ...')
    counts = np.zeros(shape=new_max_size, dtype=DTYPE_COUNT)
    for d in seq_data:
        counts[d] += 1

    logger.debug('argsort ...')
    sorted_indices = np.argsort(counts)

    # take mean and variance from previous vectors
    vecs_mean = np.mean(vecs, axis=0)
    vecs_variance = np.var(vecs, axis=0)
    new_vecs = np.zeros(shape=(new_max_size, vecs.shape[1]), dtype=vecs.dtype)
    # new_vecs = np.random.standard_normal(size=(new_max_size, vecs.shape[1])) * 0.1
    new_counts = np.zeros(shape=new_max_size, dtype=DTYPE_COUNT)
    #new_types = [None] * new_max_size
    new_types = np.zeros(shape=(new_max_size,), dtype=DTYPE_HASH)
    converter = -np.ones(shape=new_max_size, dtype=DTYPE_IDX)

    logger.debug('process reversed(sorted_indices) ...')
    new_idx = 0
    new_idx_unknown = -1
    new_count = 0
    added_types = []
    for old_idx in reversed(sorted_indices):
        # keep unknown and save new unknown index
        if types[old_idx] == strings[constants.vocab_manual[constants.UNKNOWN_EMBEDDING]]:
            logger.debug('idx_unknown moved from ' + str(old_idx) + ' to ' + str(new_idx))
            new_idx_unknown = new_idx
        # skip vecs with count < threshold, but keep vecs from vocab_manual
        elif counts[old_idx] < count_threshold and strings[types[old_idx]] not in constants.vocab_manual.values():
            continue
        if old_idx < vecs.shape[0]:
            new_vecs[new_idx] = vecs[old_idx]

        else:
            # init missing vecs with previous vecs distribution
            #if not new_as_one_hot:
            new_vecs[new_idx] = np.random.standard_normal(size=vecs.shape[1]) * vecs_variance + vecs_mean
            #else:
            #    if new_count >= vecs.shape[1]:
            #        logger.warning('Adding more then vecs-size=%i new lex entries with new_as_one_hot=True (use '
            #                        'one-hot encodings). That overrides previously added new fake embeddings!'
            #                        % vecs.shape[1])
            #    new_vecs[new_idx][new_count % vecs.shape[1]] = 1.0
            new_count += 1
            added_types.append(strings[types[old_idx]])
            # print(types[old_idx] + '\t'+str(counts[old_idx]))

        new_types[new_idx] = types[old_idx]
        new_counts[new_idx] = counts[old_idx]
        converter[old_idx] = new_idx
        new_idx += 1

    assert new_idx_unknown >= 0, 'UNKNOWN_EMBEDDING not in types'

    logger.info('new lex_size: ' + str(new_idx))
    logger.debug('added ' + str(new_count) + ' new vecs to vocab')
    logger.debug(added_types)

    # cut arrays
    new_vecs = new_vecs[:new_idx, :]
    new_counts = new_counts[:new_idx]
    new_types = new_types[:new_idx]

    new_strings = StringStore([strings[t] for t in new_types])

    return converter, new_vecs, new_strings, new_counts, new_idx_unknown


def sort_and_cut_dict(seq_data, keep, count_threshold=1):
    logger.debug('sort, cut and fill ...')
    # count types
    logger.debug('calculate counts ...')
    unique, counts = np.unique(seq_data, return_counts=True)
    # add keep
    keep_new = np.array([k for k in keep if k not in unique], dtype=unique.dtype)
    unique = np.concatenate([unique, keep_new])
    counts = np.concatenate([counts, np.zeros(shape=len(keep_new), dtype=counts.dtype)])
    logger.debug('count unique: %i' % len(unique))

    logger.debug('argsort ...')
    sorted_indices = np.argsort(counts)

    new_counts = np.zeros(shape=len(unique), dtype=DTYPE_COUNT)
    new_values = np.zeros(shape=len(unique), dtype=unique.dtype)
    converter = -np.ones(shape=len(unique), dtype=DTYPE_IDX)

    logger.debug('process reversed(sorted_indices) ...')
    new_idx = 0
    removed = []
    for old_idx in reversed(sorted_indices):
        if counts[old_idx] < count_threshold and unique[old_idx] not in keep:
            removed.append(unique[old_idx])
            continue
        new_values[new_idx] = unique[old_idx]
        new_counts[new_idx] = counts[old_idx]
        converter[old_idx] = new_idx
        new_idx += 1

    # cut arrays
    new_counts = new_counts[:new_idx]
    new_values = new_values[:new_idx]
    converter = converter[:new_idx]

    logger.debug('removed %i lexicon entries' % len(removed))

    return new_values, removed, converter, new_counts


def fill_vecs(vecs, new_size):
    if len(vecs) == new_size:
        return None
    elif len(vecs) < new_size:
        # take mean and variance from previous vectors
        vecs_mean = np.mean(vecs, axis=0)
        vecs_variance = np.var(vecs, axis=0)
        new_vecs = np.zeros(shape=(new_size - len(vecs), vecs.shape[1]), dtype=vecs.dtype)
        for i in range(len(new_vecs)):
            new_vecs[i] = np.random.standard_normal(size=vecs.shape[1]) * vecs_variance + vecs_mean
        vecs = np.concatenate([vecs, new_vecs])
        return vecs
    else:
        raise IndexError('new_size=%i < len(vecs)=%i' % (new_size, len(vecs)))


def add_and_get_idx(vecs, types, new_type, new_vec=None, overwrite=False):
    if new_vec is None and overwrite:
        vecs_mean = np.mean(vecs, axis=0)
        vecs_variance = np.var(vecs, axis=0)
        new_vec = np.random.standard_normal(size=vecs.shape[1]) * vecs_variance + vecs_mean
    if new_type in types:
        idx = types.index(new_type)
    else:
        types.append(new_type)
        vecs = np.concatenate([vecs, np.zeros(shape=(1, vecs.shape[1]), dtype=vecs.dtype)])
        idx = len(types) - 1
    if overwrite:
        vecs[idx] = new_vec
    return vecs, types, idx


# TODO: adapt for StringStore
# deprecated
def merge_dicts(vecs1, types1, vecs2, types2, add=True, remove=True):
    """
    Replace all embeddings in vecs1 which are contained in vecs2 (indexed via types).
    If remove=True remove the embeddings not contained in vecs2.
    If add=True add the embeddings from vecs2, which are not already in vecs1.

    Inplace modification of vecs1 and types1!

    :param vecs1: embeddings from first dict
    :param types1: types from first dict
    :param vecs2: embeddings from second dict
    :param types2: types from second dict
    :param remove: if remove=True remove the embeddings not contained in vecs2
    :param add: if add=True add the embeddings from vecs2, which are not already in vecs1
    :return: the modified embeddings and types
    """
    assert vecs1.shape[0] == len(types1), 'count of embeddings in vecs1 = %i does not equal length of types1 = %i' \
                                          % (vecs1.shape[0], len(types1))
    assert vecs2.shape[0] == len(types2), 'count of embeddings in vecs2 = %i does not equal length of types2 = %i' \
                                          % (vecs2.shape[0], len(types2))
    logger.info('size of dict1: %i' % len(types1))
    logger.info('size of dict2: %i' % len(types2))
    mapping2 = mapping_from_list(types2)
    logger.debug(len(mapping2))
    logger.debug(np.array_equal(vecs1, vecs2))
    logger.debug(types1 == types2)

    indices_delete = []
    indices2_added = []
    indices2_added_debug = []
    for idx, t in enumerate(types1):
        indices2_added_debug.append(idx)
        if t in mapping2:
            idx2 = mapping2[t]
            types1[idx] = types2[idx2]
            vecs1[idx] = vecs2[idx2]
            if add:
                indices2_added.append(idx2)
        else:
            if remove:
                indices_delete.append(idx)

    if remove:
        for idx in reversed(indices_delete):
            del types1[idx]

        vecs1 = np.delete(vecs1, indices_delete, axis=0)
        logger.info('removed %i entries from dict1' % len(indices_delete))

    if add:
        indices_types2 = sorted(range(len(types2)))
        indices_types2_set = set(indices_types2)
        indices2_added = sorted(indices2_added)
        logger.debug(indices_types2 == indices2_added)
        logger.debug(indices_types2 == indices2_added_debug)
        logger.debug(indices2_added_debug == indices2_added)

        types2_indices_add = list(indices_types2_set.difference(indices2_added))

        types1.extend([types2[idx] for idx in types2_indices_add])
        vecs1 = np.concatenate((vecs1, vecs2[types2_indices_add]), axis=0)
        logger.info('added %i entries to dict1' % len(types2_indices_add))
    return vecs1, types1


def get_dict_from_vocab(vocab):
    #manual_vocab_reverted = revert_mapping_to_map(constants.vocab_manual)
    #manual_vocab_reverted = {constants.vocab_manual[k]: k for k in constants.vocab_manual}
    manual_vocab_values = set(constants.vocab_manual.values())
    size = len(vocab) + len(manual_vocab_values)
    # vecs = np.zeros(shape=(size, vocab.vectors_length), dtype=np.float32)
    vecs = np.random.standard_normal(size=(size, vocab.vectors_length)) * 0.1

    # add manual vocab at first
    # the vecs remain standard_normal samples (see above)
    types = list(manual_vocab_values)
    i = len(types)
    for lexeme in vocab:
        l = TYPE_LEXEME + SEPARATOR + lexeme.orth_
        # exclude entities which are in vocab_manual to avoid collisions
        if l in manual_vocab_values:
            logger.warn(
                'found token in parser vocab with orth_="' + l + '", which was already added from manual vocab: "' + ', '.join(
                    manual_vocab_values) + '", skip!')
            continue
        vecs[i] = lexeme.vector
        types.append(l)
        i += 1
    # constants.UNKNOWN_IDX=0
    vecs[0] = np.mean(vecs[1:i], axis=0)

    if i < size:
        vecs = vecs[:i]
        types = types[:i]

    return vecs, types


# unused
def read_types(out_path):
    logger.debug('read types from file: ' + out_path + '.type ...')
    with open(out_path + '.type') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t', quotechar='|')
        types = [row[0].decode("utf-8") for row in reader]
    return types


def mapping_from_list(l):
    m = {}
    for i, x in enumerate(l):
        if x in m:
            logger.warn('already in dict: "' + x + '" at idx: ' + str(m[x]))
        m[x] = i
    return m


def revert_mapping_to_list(mapping):
    temp = [None] * len(mapping)
    for key in mapping:
        temp[mapping[key]] = key
    return temp


def revert_mapping_to_map(mapping):
    temp = {}
    for key in mapping:
        temp[mapping[key]] = key
    return temp


# unused
def revert_mapping_to_np(mapping):
    temp = -np.ones(shape=len(mapping), dtype=np.int32)
    for key in mapping:
        temp[mapping[key]] = key
    return temp


# unused
def load(fn):
    logger.debug('load vecs from file: ' + fn + '.vec ...')
    v = np.load(fn + '.vec')
    t = read_types(fn)
    logger.debug('vecs.shape: ' + str(v.shape) + ', len(types): ' + str(len(t)))
    return v, t


# unused
def exist(filename):
    return os.path.isfile('%s.vec' % filename) and os.path.isfile('%s.type' % filename)


# unused
def create_or_read_dict(fn, vocab=None, dont_read=False):
    if os.path.isfile(fn + '.vec') and os.path.isfile(fn + '.type'):
        if dont_read:
            return
        v, t = load(fn)
    else:
        logger.debug('extract word embeddings from spaCy ...')
        v, t = get_dict_from_vocab(vocab)
        dump(fn, vecs=v, types=t)
    return v, t


# unused
def dump(out_path, vecs=None, types=None, counts=None):
    if vecs is not None:
        logger.debug('dump embeddings (shape=' + str(vecs.shape) + ') to: ' + out_path + '.vec ...')
        vecs.dump(out_path + '.vec')
    if types is not None:
        logger.debug('write types (len=' + str(len(types)) + ') to: ' + out_path + '.types ...')
        with open(out_path + '.type', 'wb') as f:
            writer = csv.writer(f, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for t in types:
                writer.writerow([t.encode("utf-8")])
    if counts is not None:
        logger.debug('dump counts (len=' + str(len(counts)) + ') to: ' + out_path + '.count ...')
        counts.dump(out_path + '.count')


FE_TYPES = 'type'
FE_VECS = 'vec'
FE_IDS_VECS_FIXED = 'fix'
FE_STRINGS = 'string'


class Lexicon(object):
    def __init__(self, filename=None, types=None, vecs=None, nlp_vocab=None, strings=None, checkpoint_reader=None,
                 ids_fixed=None, load_vecs=True, load_ids_fixed=True, add_vocab_manual=False, hashes=None):
        """
        Create a Lexicon from file, from types (and optionally from vecs), from spacy vocabulary or from spacy
        StringStore.
        :param filename: load types from file <filename>.type and, if the file exists exists, vecs from <filename>.vec
        :param types: a list of unicode strings
        :param vecs: word embeddings, a numpy array of shape [vocab_size, vector_length]
        :param nlp_vocab: a spacy vocabulary. It has to contain embedding vectors.
        :param strings: a spacy StringStore, see https://spacy.io/api/stringstore
        """
        self._frozen = False
        self._mapping = None
        self._hashes = hashes
        self._strings = strings
        self._vecs = None
        self.init_ids_fixed(filename if load_ids_fixed else None, ids_fixed=ids_fixed, assert_exists=False)
        if filename is not None:
            self._strings = StringStore().from_disk('%s.%s' % (filename, FE_STRINGS))
            #self.init_ids_fixed(filename if load_ids_fixed else None, ids_fixed=ids_fixed, assert_exists=False)
            if add_vocab_manual:
                self.add_all(vocab_manual.values())
            if checkpoint_reader is not None:
                self.init_vecs(checkpoint_reader=checkpoint_reader)
            elif load_vecs and Lexicon.exist(filename, vecs_only=True):
                self.init_vecs(filename=filename)
            else:
                # set dummy vecs
                self.init_vecs()
        elif types is not None:
            #types_dep = types
            self._strings = StringStore(types)
            if add_vocab_manual:
                self.add_all(vocab_manual.values())
            if vecs is not None:
                self._vecs = vecs
                #self.init_ids_fixed(ids_fixed=ids_fixed)
            else:
                # set dummy vecs
                self.init_vecs()
        elif nlp_vocab is not None:
            self._vecs, types = get_dict_from_vocab(nlp_vocab)
            #self.init_ids_fixed(ids_fixed=ids_fixed)
            self._strings = StringStore(types)
            if add_vocab_manual:
                self.add_all(vocab_manual.values())
        #else:
            #self.init_ids_fixed(ids_fixed=ids_fixed)
        #elif string_list is not None:
        #    self._strings = StringStore(string_list)
        #    self.init_vecs()

        # create empty lexicon
        if self._strings is None:
            self._strings = StringStore()
            if add_vocab_manual:
                self.add_all(vocab_manual.values())
            self.init_vecs()

    def copy(self, copy_vecs=True, copy_ids_fixed=True):
        return Lexicon(strings=StringStore(self.strings),
                       ids_fixed=self._ids_fixed.copy() if self._ids_fixed is not None and copy_ids_fixed else None,
                       vecs=self._vecs.copy() if self._vecs is not None and copy_vecs else None)

    # deprecated use StringStore
    @staticmethod
    def read_types(filename):
        logger.debug('read types from file: %s.%s ...' % (filename, FE_TYPES))
        with open('%s.%s' % (filename, FE_TYPES)) as csvfile:
            reader = csv.reader(csvfile, delimiter='\t', quotechar='|')
            types = [row[0].decode("utf-8") for row in reader]
        return types

    # deprecated
    def write_types(self, filename):
        logger.debug('write types (len=%i) to: %s.%s ...' % (len(self.hashes), filename, FE_TYPES))
        with open('%s.%s' % (filename, FE_TYPES), 'wb') as f:
            writer = csv.writer(f, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for t in self.hashes:
                writer.writerow([t.encode("utf-8")])

    def dump(self, filename, strings_only=False):
        if self.strings is not None:
            self.strings.to_disk('%s.%s' % (filename, FE_STRINGS))

        if not strings_only:
            assert self.vecs is not None, 'can not dump vecs, they are None'
            fn_vecs = '%s.%s' % (filename, FE_VECS)
            logger.debug('dump embeddings (shape=%s) to: %s ...' % (str(self.vecs.shape), fn_vecs))
            numpy_dump(fn_vecs, self.vecs)

        if self.ids_fixed is not None and len(self.ids_fixed) > 0:
            u = np.unique(self.ids_fixed)
            assert len(u) == len(self.ids_fixed), 'ids_fixed contains replicated entries'
            numpy_dump('%s.%s' % (filename, FE_IDS_VECS_FIXED), self.ids_fixed)

    @staticmethod
    def exist(filename, types_only=False, vecs_only=False):
        strings_exist = vecs_only or os.path.isfile('%s.%s' % (filename, FE_TYPES)) \
                        or os.path.isfile('%s.%s' % (filename, FE_STRINGS))
        #vecs_exist = types_only or os.path.isfile('%s.%s' % (filename, FE_VECS)) \
        #             or os.path.isfile('%s.%s.npy' % (filename, FE_VECS))
        vecs_exist = types_only or numpy_exists('%s.%s' % (filename, FE_VECS))
        return strings_exist and vecs_exist

    @staticmethod
    def delete(filename, types_only=False):
        if Lexicon.exist(filename, types_only=types_only):
            os.remove('%s.%s' % (filename, FE_TYPES))
            if not types_only:
                assert Lexicon.exist(filename, vecs_only=True), 'can not delete vecs file (%s). it does not exist.' % filename
                if os.path.exists('%s.%s' % (filename, FE_VECS)):
                    os.remove('%s.%s' % (filename, FE_VECS))
                else:
                    os.remove('%s.%s.npy' % (filename, FE_VECS))

    def init_ids_fixed(self, filename=None, ids_fixed=None, assert_exists=False):
        if filename is not None:
            ids_fixed = numpy_load('%s.%s' % (filename, FE_IDS_VECS_FIXED), assert_exists=assert_exists)
            if ids_fixed is not None:
                logger.debug('loaded ids_fixed from "%s.%s"' % (filename, FE_IDS_VECS_FIXED))
                u = np.unique(ids_fixed)
                assert len(u) == len(ids_fixed), 'ids_fixed contains replicated entries'
        if ids_fixed is None:
            ids_fixed = np.zeros(shape=0, dtype=DTYPE_IDX)
        self._ids_fixed = ids_fixed
        if not isinstance(self._ids_fixed, np.ndarray) and self._ids_fixed is not None:
            self._ids_fixed = np.array(self._ids_fixed, dtype=DTYPE_IDX)
        self._ids_fixed_dict = None
        self._ids_var_dict = None
        self._ids_var = None

    def init_vecs(self, filename=None, new_vecs=None, new_vecs_fixed=None, checkpoint_reader=None, vocab=None, dims=None,
                  vocab_prefix=PREFIX_LEX):
        if filename is not None:
            assert self._ids_fixed is not None, 'ids_fixed is None'
            new_vecs = numpy_load('%s.%s' % (filename, FE_VECS), assert_exists=True)
            #self.init_ids_fixed(filename, assert_exists=False)
        elif checkpoint_reader is not None:
            assert self._ids_fixed is not None, 'ids_fixed is None'
            import model_fold
            saved_shapes = checkpoint_reader.get_variable_to_shape_map()
            if model_fold.VAR_NAME_LEXICON_VAR in saved_shapes:
                new_vecs = checkpoint_reader.get_tensor(model_fold.VAR_NAME_LEXICON_VAR)
            if model_fold.VAR_NAME_LEXICON_FIX in saved_shapes:
                new_vecs_fixed = checkpoint_reader.get_tensor(model_fold.VAR_NAME_LEXICON_FIX)
            #assert new_vecs is not None or new_vecs_fixed is not None, 'no vecs and no vecs_fixed found in checkpoint'
            if new_vecs is None and new_vecs_fixed is None:
                logger.warning('No vecs and no vecs_fixed found in checkpoint. Set vecs to None.')
                new_vecs = None
                new_vecs_fixed = None
        elif vocab is not None:
            assert new_vecs_fixed is None, 'no new_vecs_fixed allowed when initializing from vocab. set vecs_fixed ' \
                                           'indices afterwards.'
            assert new_vecs is None, 'no new_vecs is allowed if vocab is given (triggers init_vecs_with_spacy_vocab)'
            self.init_vecs_with_spacy_vocab(vocab=vocab, prefix=vocab_prefix)

        # TODO: check _fixed
        if new_vecs_fixed is not None:
            assert new_vecs_fixed.shape[0] == self.len_fixed, \
                'amount of vecs in new_vecs_fixed=%i is different then len_fixed=%i' \
                % (new_vecs_fixed.shape[0], self.len_fixed)
            count_total = new_vecs.shape[0] + new_vecs_fixed.shape[0]
            assert count_total <= len(self), 'can not set more vecs than amount of existing types ' \
                                             '(len(new_vecs + new_vecs_fixed)==%i > len(types)==%i)' \
                                             % (count_total, len(self))

            self._vecs = np.zeros(shape=(len(self), new_vecs_fixed.shape[1]), dtype=np.float32)
            for idx_source, idx_target in enumerate(self.ids_fixed):
                self._vecs[idx_target] = new_vecs_fixed[idx_source]
            for idx_source, idx_target in enumerate(self.ids_var):
                self._vecs[idx_target] = new_vecs[idx_source]
                # if lex entries were added, cancel after last loaded entry
                if idx_source == len(new_vecs) - 1:
                    if count_total < len(self):
                        logger.warning('pad remaining vecs (%i) with zeros' % (len(self) - count_total))
                    break
        elif new_vecs is not None:
            assert len(new_vecs) <= len(self), 'can not set more vecs than amount of existing ' \
                                                                   'types (len(new_vecs)==%i > len(types)==%i)' \
                                                                   % (len(new_vecs), len(self))
            if new_vecs.shape[0] < len(self):
                logging.debug('pad missing vecs (%i) with zero' % (len(self) - new_vecs.shape[0]))
                self._vecs = np.zeros(shape=(len(self), new_vecs.shape[1]), dtype=np.float32)
                self._vecs[:new_vecs.shape[0], :] = new_vecs
            else:
                self._vecs = new_vecs
        elif dims is not None and dims > 0:
            self._vecs = np.zeros(shape=(0, dims), dtype=DTYPE_VECS)
        else:
            self._vecs = None

        if self._vecs is not None and self._vecs.shape[0] > 0:
            self.freeze()

    def shrink_via_min_count(self, data_hashes, keep_hashes=None, min_count=2):
        # get current fixed hashes
        hashes_fixed = self.hashes[self.ids_fixed]
        hashes_unique, c = np.unique(data_hashes, return_counts=True)
        # keep hashes that occur at least min_count times and entries in hashes_fixed
        mask = np.logical_or(c >= min_count, np.isin(hashes_unique, hashes_fixed), np.isin(hashes_unique, keep_hashes))
        if keep_hashes is not None:
            mask = np.logical_or(mask, np.isin(hashes_unique, keep_hashes))
        hashes_filtered = hashes_unique[mask]
        logger.debug('keep %i of %i lexicon entries (removed %i; fixed entries are kept)'
                     % (len(hashes_filtered), len(self), len(self) - len(hashes_filtered)))
        self.create_subset_with_hashes(hashes=hashes_filtered, create_new=False, add_vocab_manual=True)

    def add_vecs_from_other(self, other, mode='concat', self_to_lowercase=True, flag_added=True):
        assert other.has_vecs, 'other lexicon has no vecs'
        if mode == 'concat':
            dim_other = other.vecs.shape[1]
            # add one dimension to hold the flag "vec added"
            vecs_new = np.zeros(shape=[len(self), dim_other], dtype=self.vecs.dtype if self.has_vecs else DTYPE_VECS)

            ids_added = []
            for i, s in enumerate(self.strings):
                if self_to_lowercase:
                    s = s.lower()
                # hack for SPACY BUG
                if s == u'root':
                    continue

                if s in other.strings:
                    other_hash = hash_string(s)
                    other_idx = other.mapping[other_hash]
                    other_vec = other.vecs[other_idx]
                    vecs_new[i] = other_vec
                    ids_added.append(i)
            if self.has_vecs:
                self._vecs = np.concatenate((self._vecs, vecs_new), axis=1)
            else:
                logger.debug('base lexicon does not contain vecs')
                self._vecs = vecs_new
                self.freeze()
            if flag_added:
                logger.debug('flag vecs that contain new data')
                self.add_flag(indices=ids_added)
            logger.debug('updated %i of %i vecs' % (len(ids_added), len(self)))
            return ids_added
        else:
            raise ValueError('unknown mode=%s' % mode)

    def init_vecs_with_glove_file(self, filename, prefix=u''):
        with io.open(filename, encoding='utf8') as f:
            first_line = f.readline()
            dims = len(first_line.split()) -1
            logger.debug('number of dims: %i' % dims)
            f.seek(0)
            # NOTE: unpack does not work with different column datatypes for genfromtxt
            #_dtypes = 'U%i,' % max_characters + ','.join(['f'] * dims)
            #_loaded = np.genfromtxt(f, dtype='U%i' % max_characters, unpack=True, invalid_raise=False)
            #_loaded = np.genfromtxt(f, dtype='U%i' % max_characters, unpack=True, invalid_raise=False)

            ids_added = {}
            vecs_new = np.zeros(shape=[len(self), dims], dtype=DTYPE_VECS)
            for idx_source, line in enumerate(f):
                parts = line.split(' ')
                if len(parts) != dims + 1:
                    logger.warning('line %i has wrong number of columns (%i), skip it: %s...' % (idx_source, len(parts), line[:100]))
                    continue
                word = prefix + parts[0]
                if word in self.strings:
                    _hash = hash_string(word)
                    idx = self.mapping[_hash]
                    vecs_new[idx] = np.asarray(parts[1:], dtype=DTYPE_VECS)
                    if idx in ids_added:
                        logger.warning('vector for word: %s was already added before (%s)' % (word, ids_added[idx]))
                    ids_added[idx] = word
        logger.debug('set %i vecs (total lex size: %i) with loaded vecs' % (len(ids_added), len(self)))
        self.init_ids_fixed(ids_fixed=sorted(ids_added.keys()))
        self._vecs = vecs_new
        self.freeze()

    def init_vecs_with_spacy_vocab(self, vocab, prefix=u''):
        new_vecs = np.zeros(shape=(len(self), vocab.vectors_length), dtype=vocab.vectors.data.dtype)
        count_added = 0
        not_found = []
        found_indices = []
        for i, s in enumerate(self.strings):
            s_cut = without_prefix(s, prefix)
            if s_cut is not None and vocab.has_vector(s_cut):
                new_vecs[i] = vocab.get_vector(s_cut)
                found_indices.append(i)
                count_added += 1
            else:
                not_found.append(s)
                new_vecs[i] = np.zeros(shape=(vocab.vectors_length,), dtype=vocab.vectors.data.dtype)
        logger.info('added %i vecs from vocab and set %i to zero' % (count_added, len(self) - count_added))
        logger.debug('zero (first 100): %s' % ', '.join(not_found[:100]))

        # set loaded vecs as fixed
        self.init_ids_fixed(ids_fixed=found_indices)
        self._vecs = new_vecs
        self.freeze()

    def sort_and_cut_and_fill_dict(self, data, keep_values, count_threshold=10):
        assert self.frozen is False, 'can not sort and cut frozen lexicon'
        new_values, removed, converter, new_counts = sort_and_cut_dict(seq_data=data, count_threshold=count_threshold,
                                                                       keep=keep_values)
        logger.debug('removed (first 100): %s' % ','.join([self.strings[v] for v in removed][:100]))
        # recreate strings without removed ones, in sorted order
        self._strings = StringStore([self.strings[v] for v in new_values if v not in removed])
        logger.debug('new lexicon size: %i' % len(self.strings))
        self.clear_cached_values()

        ## convert vecs, if available
        #if self.vecs is not None:
        #    vecs_new = np.zeros(shape=(len(converter), self.vec_size), dtype=self.vecs.dtype)
        #    for i in range(len(vecs_new)):
        #        if i < len(converter):
        #            vecs_new[converter[i]] = self.vecs[i]

        ## convert fixed indices
        #self._ids_fixed = set([converter[i] for i in self._ids_fixed])
        #self._ids_fixed_dict = None
        #self._ids_var_dict = None

        #return converter, new_counts

    def get_ids_for_prefix(self, prefix, add_separator=True):
        if add_separator:
            prefix = prefix + SEPARATOR
        res = [(self.mapping[self.strings[s]], s) for s in self.strings if s.startswith(prefix)]
        if add_separator:
            if prefix == TYPE_LEXEME + SEPARATOR:
                logger.debug('add UNKNOWN for prefix %s' % TYPE_LEXEME)
                s = vocab_manual[UNKNOWN_EMBEDDING]
                res.append((self.mapping[self.strings[s]], s))
        if len(res) == 0:
            logger.warning('no indices found for prefix=%s' % prefix)
            return [(), ()]
        return zip(*res)

    def get_hashes_for_prefix(self, prefix):
        res = [self.strings[s] for s in self.strings if s.startswith(prefix)]
        return res

    def get_ids_for_prefixes_or_types(self, prefixes_or_types, data_as_hashes):
        ids = []
        for prefix_or_type in prefixes_or_types:
            prefix_or_types = constants.TYPE_LONG.get(prefix_or_type.strip(), prefix_or_type.strip())
            if isinstance(prefix_or_types, list):
                _ids = [self.get_d(s=s, data_as_hashes=data_as_hashes) for s in
                        prefix_or_types]
            else:
                _ids, _strings = self.get_ids_for_prefix(prefix=prefix_or_types, add_separator=False)
            assert len(_ids) > 0, ' no ids found for indices_prefix: %s' % ', '.join(prefix_or_types)
            logger.info('select %i different ids for prefix: %s' % (len(_ids), prefix_or_type.strip()))
            ids.extend(_ids)
        return ids

    def get_indices(self, indices=None, prefix=None, strings=None, indices_as_blacklist=False):
        if strings is not None:
            if prefix is None:
                indices = [self.mapping[hash_string(s)] for s in strings]
            else:
                indices = [self.mapping[hash_string(prefix + s)] for s in strings]
        elif prefix is not None:
            indices, strings = self.get_ids_for_prefix(prefix)
        if indices is None:
            indices = np.arange(len(self), dtype=DTYPE_IDX)
        elif not isinstance(indices, np.ndarray):
            indices = np.array(indices, dtype=DTYPE_IDX)
        if indices_as_blacklist:
            indices_all = np.arange(len(self), dtype=DTYPE_IDX)
            mask_other = ~np.isin(indices_all, indices)
            indices = indices_all[mask_other]
        return indices

    def set_to_zero(self, *args, **kwargs):
        assert self.vecs is not None, 'vecs is None, can not set to zero'
        indices = self.get_indices(*args, **kwargs)
        for i in indices:
            self._vecs[i] = np.zeros(self._vecs.shape[1], dtype=self._vecs.dtype)
        if len(indices) > 0:
            logger.info('set %i vecs to zero' % len(indices))

    def set_to_onehot(self, *args, **kwargs):
        assert self.vecs is not None, 'vecs is None, can not set to onehot'
        indices = self.get_indices(*args, **kwargs)
        self.set_to_zero(indices=indices)
        for i, idx in enumerate(indices):
            self._vecs[idx][i] = 1.0
        if len(indices) > 0:
            logger.info('set %i vecs to one-hot' % len(indices))

    def set_to_random(self, *args, **kwargs):
        assert self.vecs is not None, 'vecs is None, can not set to random'
        indices = self.get_indices(*args, **kwargs)
        for i in indices:
            self._vecs[i] = np.random.standard_normal(size=self._vecs.shape[1])
        if len(indices) > 0:
            logger.info('set %i vecs to random' % len(indices))

    def set_to_mean(self, *args, **kwargs):
        assert self.vecs is not None, 'vecs is None, can not set to mean'
        indices = self.get_indices(*args, **kwargs)
        indices_all = np.arange(len(self), dtype=DTYPE_IDX)
        mask_other = ~np.isin(indices_all, indices)
        indices_other = indices_all[mask_other]

        vecs_mean = np.mean(self._vecs[indices_other], axis=0)
        vecs_variance = np.var(self._vecs[indices_other], axis=0)

        for i in indices:
            self._vecs[i] = np.random.standard_normal(size=self._vecs.shape[1]) * vecs_variance + vecs_mean
        if len(indices) > 0:
            logger.info('set %i vecs to random' % len(indices))

    def set_to_vecs(self, vecs, *args, **kwargs):
        assert self.vecs is not None, 'can not set vecs because they lexicon vecs are not initialized (None)'
        indices = self.get_indices(*args, **kwargs)
        assert len(indices) == vecs.shape[0], 'number of lexicon indices (%i) does not match number of vecs (%i)' \
                                              % (len(indices), vecs.shape[0])
        assert vecs.shape[-1] == self.dims, \
            'dimensionality of new vecs (%i) does not match lexicon dimensionality (%i)' % (vecs.shape[-1], self.dims)
        assert vecs.dtype == self.vecs.dtype, 'dtype of vecs (%s) does not match dtype of lexicon vecs (%s)' \
                                              % (str(vecs.dtype), str(self.vecs.dtype))
        self._vecs[indices] = vecs
        self.freeze()
        return indices

    def add_flag(self, *args, **kwargs):
        assert self.vecs is not None, 'vecs is None, can not set to zero'
        indices = self.get_indices(*args, **kwargs)
        new_entries = np.zeros(self._vecs.shape[0], dtype=self._vecs.dtype)
        new_entries[indices] = 1.0
        self._vecs = np.concatenate((self._vecs, new_entries.reshape((self._vecs.shape[0], 1))), axis=1)
        if len(indices) > 0:
            logger.info('added flag=1.0 to %i vecs and set 0.0 for %i' % (len(indices), self._vecs.shape[0]-len(indices)))

    def set_man_vocab_vec(self, man_vocab_id, new_vec=None):
        if new_vec is None:
            new_vec = np.zeros(shape=self.vec_size, dtype=self._vecs.dtype)
        idx = self[constants.vocab_manual[man_vocab_id]]
        self._vecs[idx] = new_vec
        return idx

    def pad(self, pad_with='random'):
        assert self.vecs is not None, 'vecs not set, can not pad'
        if len(self.vecs) == len(self):
            pass
        elif len(self.vecs) < len(self):
            new_vecs = np.zeros(shape=(len(self) - len(self.vecs), self.vec_size), dtype=self.vecs.dtype)
            if pad_with == 'random':
                # take mean and variance from previous vectors
                if len(self.vecs) > 0:
                    vecs_mean = np.mean(self.vecs, axis=0)
                    vecs_variance = np.var(self.vecs, axis=0)
                else:
                    vecs_mean = 0.0
                    vecs_variance = 1.0
                for i in range(len(new_vecs)):
                    new_vecs[i] = np.random.standard_normal(size=self.vec_size) * vecs_variance + vecs_mean
            elif pad_with == 'zero':
                pass
            else:
                raise ValueError('Unknown padding type: %s. Use "random" or "zero".' % pad_with)

            self._vecs = np.concatenate([self.vecs, new_vecs])
            #self._dumped_vecs = False
        else:
            raise IndexError('len(self)==len(types)==%i < len(vecs)==%i' % (len(self), len(self.vecs)))

    def merge(self, other, add_entries=True, replace_vecs=True, unknown_prefix_mapping={}):
        """
        Merge other lexicon into this one.
        :param other: the other lexicon that will be merged into this one
        :param add_entries: If True, add all entries from other to this one.
        :param replace_vecs: If True, replace vecs that are in other with there entries.
        :param unknown_prefix_mapping: a mapping { prefix_string -> idx } to map all entries from other with prefix
               that do not occur in self to idx (of self), e.g. use {u'conll:WORD=': 0 } to map words not found to 0.
        :return: a numpy array that can be used to convert data encoded with the other lexicon to the merged one. At
        position i it holds the index of the i'th entry of the other in the merged lexicon. So, converter[d_other]
        produces the new data entry (eventually the entry related to UNKNOWN, if d_other is not in the merged lexicon,
        see add_entries).
        """

        if add_entries:
            size_before = len(self)
            self.add_all(other.strings)
            logger.debug('added %i entries to lexicon. new size: %i' % (len(self) - size_before, len(self)))

        d_unknown = self.get_d(vocab_manual[UNKNOWN_EMBEDDING], data_as_hashes=False)
        d_unknown_other = other.get_d(vocab_manual[UNKNOWN_EMBEDDING], data_as_hashes=False)

        # initialize with unknown
        converter = np.ones(len(other), dtype=DTYPE_IDX) * d_unknown
        for h in self.mapping:
            idx_other = other.mapping.get(h, None)
            if idx_other is not None:
                converter[idx_other] = self.mapping[h]
        # set via prefix(es) to certain unknown placeholder(s)
        unknown_dict = {}
        for h in other.mapping:
            if h not in self.mapping:
                s_other = self.get_s(d=h, data_as_hashes=True)
                for prefix in unknown_prefix_mapping:
                    if s_other.startswith(prefix):
                        unknown_dict.setdefault(unknown_prefix_mapping[prefix], []).append(other.mapping[h])
        for unk in unknown_dict:
            converter[np.array(unknown_dict[unk], dtype=DTYPE_IDX)] = unk

        if other.has_vecs:
            if not self.has_vecs or self.dims == 0:
                self.init_vecs(dims=other.dims)
            else:
                assert self.dims == other.dims, 'dimensions of own vecs (%i) does not match others (%i), can not merge ' \
                                                'lexica' % (self.dims, other.dims)
            self.pad(pad_with='zero')
            if replace_vecs:
                self.vecs[converter] = other.vecs
                # overwrite with correct vec for UNKNOWN (converter was initialized with UNKNOWN, so the new UNKNOWN
                # vec was set with all entries in other but not in self one after teh other, resulting in the last of
                # these vecs assigned to the UNKNOWN position in self)
                self.vecs[d_unknown] = other.vecs[d_unknown_other]
                logger.debug('replaced %i vecs of %i (lexicon merge)' % (len(np.unique(converter)), len(self)))

        return converter

    def replicate_types(self, prefix='', suffix=''):
        assert len(prefix) + len(suffix) > 0, 'please provide a prefix or a suffix.'
        #self._types.extend([prefix + t + suffix for t in self.types])
        for s in self.strings:
            self._strings.add(prefix + s + suffix)
        self.clear_cached_values()

    # self._ids_fixed is a numpy array!
    #def update_fix_ids_and_abs_data(self, new_data):
    #    new_ids_fix = new_data[new_data < 0]
    #    if len(new_ids_fix) > 0:
    #        np.abs(new_data, out=new_data)
    #        self._ids_fixed = self._ids_fixed.update(new_ids_fix.tolist())
    #        self._ids_fixed_dict = None
    #        self._ids_var_dict = None

    def convert_data_hashes_to_indices(self, data, convert_dtype=True):
        s_uk = constants.vocab_manual[constants.UNKNOWN_EMBEDDING]
        #data_new = np.zeros(shape=data.shape, dtype=DTYPE_IDX)
        #assert s_uk in self.strings, '%s not in lexicon' % s_uk
        for i in range(len(data)):
            d = data[i]
            if d in self.mapping:
                data[i] = self.mapping[d]
            else:
                data[i] = self.mapping[self.strings[s_uk]]
        self.freeze()
        if convert_dtype:
            data = data.astype(DTYPE_IDX)
        return data
        #return data_new

    def read_data(self, reader, reader_args={}, expand_dict=True, return_hashes=False, as_graph=False,
                  dont_parse=False, *args, **kwargs):
        if as_graph:
            if dont_parse:
                if expand_dict:
                    idx_unknown = None
                else:
                    assert vocab_manual[UNKNOWN_EMBEDDING] in self.strings, '"%s" is not in StringStore.' \
                                                                       % UNKNOWN_EMBEDDING
                    idx_unknown = self.strings[vocab_manual[UNKNOWN_EMBEDDING]]

                data = []

                graph_in_list = []
                graph_out_list = []
                for data_strings, graph_in, graph_out in reader(**reader_args):
                    current_data = [getOrAdd(self.strings, s, idx_unknown) for s in data_strings]
                    data.extend(current_data)
                    if graph_in is not None:
                        graph_in_list.append(graph_in)
                    if graph_out is not None:
                        graph_out_list.append(graph_out)
                graph_in = concatenate_graphs(graph_in_list) if len(graph_in_list) > 0 else None
                graph_out = concatenate_graphs(graph_out_list) if len(graph_out_list) > 0 else None
            else:
                raise NotImplementedError('lexicon.read_data not implemented for as_graph==True and dont_parse==False')
                data, graph_out, graph_in = read_data_graph(*args, reader=reader, reader_args=reader_args,
                                                            strings=self.strings, expand_dict=expand_dict, **kwargs)
            if expand_dict:
                self.clear_cached_values()
            if return_hashes:
                return Forest(data=data, graph_in=graph_in, graph_out=graph_out, lexicon=self, data_as_hashes=True)
            else:
                return Forest(data=self.convert_data_hashes_to_indices(data), graph_in=graph_in, graph_out=graph_out,
                              lexicon=self, data_as_hashes=False)
        data, parents = read_data(*args, reader=reader, reader_args=reader_args, strings=self.strings,
                                  expand_dict=expand_dict, **kwargs)
        if expand_dict:
            self.clear_cached_values()
        if return_hashes:
            return Forest(data=data, parents=parents, lexicon=self, data_as_hashes=True)
        else:
            return Forest(data=self.convert_data_hashes_to_indices(data), parents=parents, lexicon=self,
                          data_as_hashes=False)

    def is_fixed(self, idx):
        return idx in self.ids_fixed_dict

    def add(self, new_string):
        if new_string not in self.strings:
            self.strings.add(new_string)
            self.clear_cached_values()

    def add_and_get(self, new_string, data_as_hashes):
        self.add(new_string)
        return self.get_d(new_string, data_as_hashes=data_as_hashes)

    def add_all(self, new_strings, prefix=u''):
        for s in new_strings:
            self._strings.add(prefix + s)
        self.clear_cached_values()

    def transform_idx(self, idx, revert=False, d_unknown_replacement=None):
        """
        transform lexicon (vec) index to index for vecs_fix (negative) or vecs_var (positive) index
        :param idx: the index to transform
        :param revert: iff True, encode as "reverted" edge
        :param d_unknown_replacement: data id of that is used if converted idx is not found in idx_var and ids_fixed
        :return: the transformed index
        """
        if idx < 0:
            logger.warning('trying to transform a root_id. set to unknown.')
            return self.get_d(s=vocab_manual[UNKNOWN_EMBEDDING], data_as_hashes=False)

        offset = len(self) if revert else 0
        idx_trans = self.ids_fixed_dict.get(idx, None)
        if idx_trans is not None:
            return idx_trans + len(self.ids_var) + offset
        idx_trans = self.ids_var_dict.get(idx, None)
        if idx_trans is not None:
            return idx_trans + offset

        if d_unknown_replacement is None:
            logger.warning('idx=%i not in ids_fixed or ids_var. Set to UNKNOWN.' % idx)
            return self.get_d(s=vocab_manual[UNKNOWN_EMBEDDING], data_as_hashes=False)
        else:
            return d_unknown_replacement

    def transform_indices(self, indices, revert=False, d_unknown_replacement=None):
        return [self.transform_idx(idx=idx, revert=revert, d_unknown_replacement=d_unknown_replacement) for idx in indices]

    def transform_idx_back(self, idx):
        """
        revert transform_idx: transform index for vecs_fix (negative) or vecs_var (positive) index into lexicon index
        :param idx: the negative (-> fixed vec) or positive (-> var vec) index
        :return: the lexicon index, True iff idx was reverted else False
        """
        reverted = False
        if idx >= len(self):
            reverted = True
            idx -= len(self)
        if idx < len(self.ids_var):
            return self.ids_var[idx], reverted
        return self.ids_fixed[idx - len(self.ids_var)], reverted

    def order_by_hashes(self, hashes):
        logger.warning('order_by_hashes is deprecated. use create_subset_with_hashes instead!')
        assert not self.frozen, 'can not order frozen lexicon by hashes'
        assert len(hashes) == len(self), 'number of hashes (%i) does not match size of lexicon (%i), can not order by ' \
                                         'hashes' % (len(hashes), len(self))
        if self.vecs is not None:
            raise NotImplementedError('order not implemented for vecs that are not None')
        new_strings = StringStore()
        for h in hashes:
            new_strings.add(self.strings[h])
        self._strings = new_strings
        self.clear_cached_values()
        self._hashes = hashes
        self._ids_fixed = None

    def create_subset_with_hashes(self, hashes, keep_vecs=True, keep_fixed=True, add_vocab_manual=False, create_new=True):
        new_strings = StringStore()
        if add_vocab_manual:
            for v in vocab_manual.values():
                new_strings.add(v)
        for h in hashes:
            new_strings.add(self.strings[h])

        new_vecs = None
        new_ids_fixed = None
        new_hashes = None
        if keep_fixed or (keep_vecs and self.vecs is not None and len(self.vecs) > 0):
            new_hashes = np.empty(len(new_strings), dtype=DTYPE_HASH)
            if keep_vecs and self.vecs is not None and len(self.vecs) > 0:
                new_vecs = np.empty(shape=(len(new_strings), self.vecs.shape[1]), dtype=self.vecs.dtype)
            if keep_fixed:
                new_ids_fixed = []
            for target_idx, s in enumerate(new_strings):
                h = Lexicon.hash_string(s)
                new_hashes[target_idx] = h
                source_idx = self.mapping[h]
                if new_vecs is not None:
                    new_vecs[target_idx] = self.vecs[source_idx]
                if new_ids_fixed is not None and source_idx in self.ids_fixed:
                    new_ids_fixed.append(target_idx)
        if new_ids_fixed is not None:
            new_ids_fixed = np.array(new_ids_fixed, dtype=DTYPE_IDX)

        if create_new:
            return Lexicon(strings=new_strings, hashes=new_hashes, vecs=new_vecs, ids_fixed=new_ids_fixed)
        else:
            self.clear_cached_values()
            self._strings = new_strings
            self._hashes = new_hashes
            self._vecs = new_vecs
            self._ids_fixed = new_ids_fixed

    @staticmethod
    def merge_strings(lexica):
        new_strings = StringStore()
        for lex in lexica:
            for s in lex.strings:
                new_strings.add(s)
        new_lex = Lexicon(strings=new_strings)
        return new_lex

    @staticmethod
    def hash_string(string):
        return hash_string(string)

    @staticmethod
    def vocab_prefix(man_vocab_id):
        return constants.vocab_manual[man_vocab_id] + SEPARATOR

    @staticmethod
    def has_vocab_prefix(s, man_vocab_id):
        return s.startswith(Lexicon.vocab_prefix(man_vocab_id))

    def clear_cached_values(self):
        #assert not self.frozen, 'can not modify frozen lexicon'
        self._hashes = None
        self._mapping = None
        self._ids_fixed_dict = None
        self._ids_var_dict = None
        self._ids_var = None

    def freeze(self):
        self._frozen = True

    def unfreeze(self):
        self._frozen = False

    def get_link_types(self, data_as_hashes):
        """
        Get the data of the available link types.
        :return:
        """
        res = []
        for lt in LINK_TYPES:
            d = self.get_d(s=lt, data_as_hashes=data_as_hashes)
            if d != self.get_d(s=vocab_manual[UNKNOWN_EMBEDDING], data_as_hashes=data_as_hashes):
                res.append(d)
        return res

    def get_s(self, d, data_as_hashes):
        if data_as_hashes:
            if d in self.mapping:
                return self.strings[d]
            else:
                return vocab_manual[UNKNOWN_EMBEDDING]
        else:
            if d < len(self):
                return self.strings[self.hashes[d]]
            else:
                return vocab_manual[UNKNOWN_EMBEDDING]

    def get_d(self, s, data_as_hashes):

        if s in self.strings:
            #d = self.strings[s]
            d = hash_string(s)
        else:
            d = self.strings[vocab_manual[UNKNOWN_EMBEDDING]]

        if data_as_hashes:
            return d
        else:
            return self.mapping[d]

    def __getitem__(self, item):
        if type(item) == unicode or type(item) == str:
            try:
                res = self.mapping[self.strings[item]]
            # word doesnt occur in dictionary
            except KeyError:
                res = len(self)
                self._strings.add(item)
                self.clear_cached_values()
                #self._mapping[new_h] = res
                #self._types.append(item)
            return res
        else:
            idx = abs(item)
            if idx >= len(self):
                return constants.vocab_manual[constants.UNKNOWN_EMBEDDING]
            return self.strings[self.hashes[abs(item)]]

    def __len__(self):
        #assert self.vecs is None or self.vecs.shape[0] == 0 or self.vecs.shape[0] == len(self.strings), 'nbr of vecs (%i) does not match nbr of strings (%i)' % (self.vecs.shape[0], len(self.strings))
        return len(self.strings)

    def __contains__(self, item):
        return item in self.strings

    @property
    def vec_size(self):
        assert self._vecs is not None and self._vecs.shape[1] > 0, 'vecs are not set or have length 0'
        return self._vecs.shape[1]

    @property
    def len_fixed(self):
        return len(self._ids_fixed)

    @property
    def len_var(self):
        x = len(self) - self.len_fixed
        #assert x == len(self.ids_var), \
        #    'len(self) - self.len_fixed [%i] does not match len(self.ids_var) [%i]' % (x, len(self.ids_var))
        return x

    @property
    def vecs(self):
        return self._vecs

    @property
    def vecs_fixed(self):
        return self._vecs[self.ids_fixed]

    @property
    def vecs_var(self):
        return self._vecs[self.ids_var]

    @property
    def ids_fixed(self):
        #return np.array(sorted(list(self._ids_fixed)))
        return self._ids_fixed

    @property
    def ids_fixed_dict(self):
        if self._ids_fixed_dict is None:
            logger.debug('lexicon: create ids_fixed_dict from ids_fixed (%i)' % len(self.ids_fixed))
            self._ids_fixed_dict = {k: i for i, k in enumerate(self.ids_fixed)}
        return self._ids_fixed_dict

    @property
    def ids_var_dict(self):
        if self._ids_var_dict is None:
            logger.debug('lexicon: create ids_var_dict from ids_var (%i)' % len(self.ids_var))
            self._ids_var_dict = {k: i for i, k in enumerate(self.ids_var)}
        return self._ids_var_dict

    @property
    def ids_var(self):
        if self._ids_var is None:
            logger.debug('lexicon: create ids_var with ids_fixed (%i)' % len(self.ids_fixed))
            #self._ids_var = np.array([i for i in range(len(self)) if i not in self._ids_fixed])
            mask = np.ones(len(self), dtype=bool)
            mask[self.ids_fixed] = False
            self._ids_var = np.arange(len(self), dtype=DTYPE_IDX)[mask]
            #assert len(self._ids_var) == self.len_var, \
            #    'len(self._ids_var) [%i] does not equal len(self) [%i] - len(self.ids_fix)[%i] == self.len_var [%i]; len(self.ids_fixed) [%i]' \
            #    % (len(self._ids_var), len(self), len(self.ids_fixed), self.len_var, len(self.ids_fixed))
        return self._ids_var

    @property
    def hashes(self):
        # maps positions to hashes
        #return self._types
        if self._hashes is None:
            logger.debug('lexicon: create hashes from strings (%i)' % len(self.strings))
            self._hashes = np.zeros(shape=(len(self.strings),), dtype=DTYPE_HASH)
            for i, s in enumerate(self.strings):
                self._hashes[i] = self.strings[s]
        #assert len(self._hashes) == len(self), 'nbr of hashes [%i] does not match lexicon size [%i]' \
        #                                       % (len(self._hashes), len(self))
        return self._hashes

    @property
    def strings(self):
        return self._strings

    @property
    def mapping(self):
        # maps hashes to positions
        if self._mapping is None:
            logger.debug('lexicon: create mapping from strings (%i)' % len(self.strings))
            #self._mapping = mapping_from_list(self._types)
            self._mapping = {}
            for i, s in enumerate(self.strings):
                self._mapping[self.strings[s]] = i
        #assert len(self._mapping) == len(self), 'nbr of mapping entries [%i] does not match lexicon size [%i]' \
        #                                        % (len(self._mapping), len(self))
        return self._mapping

    @property
    def is_filled(self):
        return self._vecs is not None and len(self) == len(self.vecs)

    @property
    def has_vecs(self):
        return self._vecs is not None

    @property
    def dims(self):
        assert self.has_vecs, 'the lexicon has no vecs, can not get dimensions'
        return self.vecs.shape[1]

    @property
    def frozen(self):
        return self._frozen



# one_hot_types = [u'DEP#det', u'DEP#punct', u'DEP#pobj', u'DEP#ROOT', u'DEP#prep', u'DEP#aux', u'DEP#nsubj',
            # u'DEP#dobj', u'DEP#amod', u'DEP#conj', u'DEP#cc', u'DEP#compound', u'DEP#nummod', u'DEP#advmod', u'DEP#acl',
            # u'DEP#attr', u'DEP#auxpass', u'DEP#expl', u'DEP#nsubjpass', u'DEP#poss', u'DEP#agent', u'DEP#neg', u'DEP#prt',
            # u'DEP#relcl', u'DEP#acomp', u'DEP#advcl', u'DEP#case', u'DEP#npadvmod', u'DEP#xcomp', u'DEP#ccomp', u'DEP#pcomp',
            # u'DEP#oprd', u'DEP#nmod', u'DEP#mark', u'DEP#appos', u'DEP#dep', u'DEP#dative', u'DEP#quantmod', u'DEP#csubj',
            # u'DEP#']