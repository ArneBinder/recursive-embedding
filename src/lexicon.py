import csv

import numpy as np
from spacy.strings import StringStore

import constants
import logging
import os

from preprocessing import read_data, without_prefix, PREFIX_LEX
from constants import DTYPE_COUNT, DTYPE_HASH, DTYPE_IDX, LOGGING_FORMAT
from sequence_trees import Forest
from mytools import numpy_dump, numpy_load, numpy_exists

logger = logging.getLogger('lexicon')
logger.setLevel(logging.DEBUG)
logger_streamhandler = logging.StreamHandler()
logger_streamhandler.setLevel(logging.INFO)
logger_streamhandler.setFormatter(logging.Formatter(LOGGING_FORMAT))
logger.addHandler(logger_streamhandler)


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
    manual_vocab_reverted = revert_mapping_to_map(constants.vocab_manual)
    size = len(vocab) + len(constants.vocab_manual)
    # vecs = np.zeros(shape=(size, vocab.vectors_length), dtype=np.float32)
    vecs = np.random.standard_normal(size=(size, vocab.vectors_length)) * 0.1
    # types_unknown = constants.vocab_manual[constants.UNKNOWN_EMBEDDING]
    # types = [types_unknown]

    # add manual vocab at first
    # the vecs remain zeros
    types = constants.vocab_manual.values()
    # i = 1
    i = len(constants.vocab_manual)
    for lexeme in vocab:
        l = constants.vocab_manual[constants.LEXEME_EMBEDDING] + constants.SEPARATOR + lexeme.orth_
        # exclude entities which are in vocab_manual to avoid collisions
        if l in manual_vocab_reverted:
            logger.warn(
                'found token in parser vocab with orth_="' + l + '", which was already added from manual vocab: "' + ', '.join(
                    manual_vocab_reverted) + '", skip!')
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
    def __init__(self, filename=None, types=None, vecs=None, nlp_vocab=None, strings=None, string_list=None, load_vecs=True):
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
        self._types = None
        self._ids_fixed = set()
        self._ids_fixed_dict = None
        self._ids_var_dict = None
        self._strings = strings
        if filename is not None:
            #if os.path.isfile('%s.%s' % (filename, FE_STRINGS)):
            self._strings = StringStore().from_disk('%s.%s' % (filename, FE_STRINGS))
            #else:
            #    types_dep = Lexicon.read_types(filename)
            #    self._strings = StringStore(types_dep)
            if load_vecs and Lexicon.exist(filename, vecs_only=True):# os.path.isfile('%s.%s' % (filename, FE_VECS)):
                self.init_vecs(filename=filename)
            else:
                # set dummy vecs
                self.init_vecs()
            # already done by init_vecs
            #if os.path.isfile('%s.%s' % (filename, FE_IDS_VECS_FIXED)):
            #    logger.debug('load ids_fixed from "%s.%s"' % (filename, FE_IDS_VECS_FIXED))
            #    self._ids_fixed = set(np.load('%s.%s' % (filename, FE_IDS_VECS_FIXED)))
            #    self._ids_fixed_dict = None
            #    self._ids_var_dict = None
        elif types is not None:
            types_dep = types
            self._strings = StringStore(types_dep)
            if vecs is not None:
                self._vecs = vecs
            else:
                # set dummy vecs
                self.init_vecs()
        elif nlp_vocab is not None:
            self._vecs, types_dep = get_dict_from_vocab(nlp_vocab)
            self._strings = StringStore(types_dep)
        elif string_list is not None:
            self._strings = StringStore(string_list)
            self.init_vecs()

        # create empty lexicon
        if self._strings is None:
            self._strings = StringStore()
            self.init_vecs()

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
        logger.debug('write types (len=%i) to: %s.%s ...' % (len(self.types), filename, FE_TYPES))
        with open('%s.%s' % (filename, FE_TYPES), 'wb') as f:
            writer = csv.writer(f, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for t in self.types:
                writer.writerow([t.encode("utf-8")])

    def dump(self, filename, strings_only=False):
        if self.strings is not None:
            self.strings.to_disk('%s.%s' % (filename, FE_STRINGS))

        if not strings_only:
            assert self.vecs is not None, 'can not dump vecs, they are None'
            fn_vecs = '%s.%s' % (filename, FE_VECS)
            logger.debug('dump embeddings (shape=%s) to: %s ...' % (str(self.vecs.shape), fn_vecs))
            #self.vecs.dump('%s.%s' % (filename, FE_VECS))
            #np.save(fn_vecs, self.vecs)
            numpy_dump(fn_vecs, self.vecs)

        # TODO: check _fixed
        if len(self._ids_fixed) > 0:
            #self.ids_fixed.dump('%s.%s' % (filename, FE_IDS_VECS_FIXED))
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

    def init_vecs(self, filename=None, new_vecs=None, new_vecs_fixed=None, checkpoint_reader=None, vocab=None,
                  vocab_prefix=PREFIX_LEX):
        if filename is not None:
            #fn_vecs = '%s.%s' % (filename, FE_VECS)
            #if not os.path.exists(fn_vecs):
            #    fn_vecs = '%s.%s.npy' % (filename, FE_VECS)
            #    assert os.path.exists(fn_vecs), 'could not find vecs file (%s.%s or %s)' % (filename, FE_VECS, fn_vecs)
            #new_vecs = np.load(fn_vecs)
            new_vecs = numpy_load('%s.%s' % (filename, FE_VECS), assert_exists=True)
            ids_fixed = numpy_load('%s.%s' % (filename, FE_IDS_VECS_FIXED), assert_exists=False)
            if ids_fixed is not None:
                logger.debug('loaded ids_fixed from "%s.%s"' % (filename, FE_IDS_VECS_FIXED))
                self._ids_fixed = set(ids_fixed)
                self._ids_fixed_dict = None
                self._ids_var_dict = None
        elif checkpoint_reader is not None:
            import model_fold
            saved_shapes = checkpoint_reader.get_variable_to_shape_map()
            if model_fold.VAR_NAME_LEXICON_VAR in saved_shapes:
                new_vecs = checkpoint_reader.get_tensor(model_fold.VAR_NAME_LEXICON_VAR)
            if model_fold.VAR_NAME_LEXICON_FIX in saved_shapes:
                new_vecs_fixed = checkpoint_reader.get_tensor(model_fold.VAR_NAME_LEXICON_FIX)
            assert new_vecs is not None or new_vecs_fixed is not None, 'no vecs and no vecs_fixed found in checkpoint'
        elif vocab is not None:
            assert new_vecs_fixed is None, 'no new_vecs_fixed allowed when initializing from vocab. set vecs_fixed ' \
                                           'indices afterwards.'
            new_vecs = np.zeros(shape=(len(self), vocab.vectors_length), dtype=vocab.vectors.data.dtype)
            count_added = 0
            not_found = []
            found_indices = []
            for i, s in enumerate(self.strings):
                s_cut = without_prefix(s, vocab_prefix)
                if s_cut is not None and vocab.has_vector(s_cut):
                    new_vecs[i] = vocab.get_vector(s_cut)
                    found_indices.append(i)
                    count_added += 1
                else:
                    not_found.append(s)
                    new_vecs[i] = np.zeros(shape=(vocab.vectors_length,), dtype=vocab.vectors.data.dtype)
            logger.info('added %i vecs from vocab and set %i to zero' % (count_added, len(self) - count_added))
            logger.debug('zero (first 100): %s' % ', '.join(not_found[:100]))

            # fix loaded vecs
            self._ids_fixed = np.array(found_indices, dtype=DTYPE_IDX)

        # TODO: check _fixed
        if new_vecs_fixed is not None:
            assert new_vecs_fixed.shape[0] == self.len_fixed, \
                'amount of vecs in new_vecs_fixed=%i is different then len_fixed=%i' \
                % (new_vecs_fixed.shape[0], self.len_fixed)
            count_total = new_vecs.shape[0] + new_vecs_fixed.shape[0]
            assert count_total <= len(self), 'can not set more vecs than amount of existing types ' \
                                             '(len(new_vecs + new_vecs_fixed)==%i > len(types)==%i)' \
                                             % (count_total, len(self))

            self._vecs = np.zeros(shape=(count_total, new_vecs_fixed.shape[1]), dtype=np.float32)
            for idx_source, idx_target in enumerate(self.ids_fixed):
                self._vecs[idx_target] = new_vecs_fixed[idx_source]
            for idx_source, idx_target in enumerate(self.ids_var):
                self._vecs[idx_target] = new_vecs[idx_source]
        else:
            assert new_vecs is None or len(new_vecs) <= len(self), 'can not set more vecs than amount of existing ' \
                                                                   'types (len(new_vecs)==%i > len(types)==%i)' \
                                                                   % (len(new_vecs), len(self))
            self._vecs = new_vecs

        if self._vecs is not None:
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

    def get_ids_for_prefix(self, prefix):
        res = [self.mapping[self.strings[s]] for s in self.strings if s.startswith(prefix + constants.SEPARATOR)]
        if len(res) == 0:
            logger.warning('no indices found for prefix=%s' % prefix)
        return res

    def get_indices(self, indices=None, prefix=None, indices_as_blacklist=False):
        assert indices is not None or prefix is not None, 'please provide indices or a prefix'
        if prefix is not None:
            indices = self.get_ids_for_prefix(prefix)
        if indices is None:
            indices = np.arange(len(self), dtype=DTYPE_IDX)
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

    def set_man_vocab_vec(self, man_vocab_id, new_vec=None):
        if new_vec is None:
            new_vec = np.zeros(shape=self.vec_size, dtype=self._vecs.dtype)
        idx = self[constants.vocab_manual[man_vocab_id]]
        self._vecs[idx] = new_vec
        return idx

    def pad(self):
        assert self.vecs is not None, 'vecs not set, can not pad'
        if len(self.vecs) == len(self):
            pass
        elif len(self.vecs) < len(self):
            # take mean and variance from previous vectors
            if len(self.vecs) > 0:
                vecs_mean = np.mean(self.vecs, axis=0)
                vecs_variance = np.var(self.vecs, axis=0)
            else:
                vecs_mean = 0.0
                vecs_variance = 1.0
            new_vecs = np.zeros(shape=(len(self) - len(self.vecs), self.vec_size), dtype=self.vecs.dtype)
            for i in range(len(new_vecs)):
                new_vecs[i] = np.random.standard_normal(size=self.vec_size) * vecs_variance + vecs_mean
            self._vecs = np.concatenate([self.vecs, new_vecs])
            #self._dumped_vecs = False
        else:
            raise IndexError('len(self)==len(types)==%i < len(vecs)==%i' % (len(self), len(self.vecs)))

    # TODO: adapt for StringStore! does not work like this!
    def merge(self, other, add=True, remove=True):
        #self._vecs, types_dep = merge_dicts(vecs1=self.vecs, types1=self.types, vecs2=other.vecs, types2=other.types, add=add, remove=remove)
        #self._strings = StringStore(types_dep)
        if self.frozen or other.frozen:
            raise NotImplementedError('merging of frozen lexicons not implemented')

        if add and remove:
            self._strings = StringStore(other.strings)
        elif add and not remove:
            self.add_all(other.strings)
        elif not add and remove:
            new_strings = StringStore()
            for s in self.strings:
                if s in other.strings:
                    new_strings.add(s)
            self._strings = new_strings
        elif not add and not remove:
            pass

        self.clear_cached_values()

        #self._mapping = None
        #self.clear_cached_values()
        #self._dumped_vecs = False
        #self._dumped_types = False
        #if add:
        #    converter_other = [self.mapping[t] for t in other.types]
        #else:
        #    converter_other = None
        #return converter_other

    def replicate_types(self, prefix='', suffix=''):
        assert len(prefix) + len(suffix) > 0, 'please provide a prefix or a suffix.'
        #self._types.extend([prefix + t + suffix for t in self.types])
        for s in self.strings:
            self._strings.add(prefix + s + suffix)
        self.clear_cached_values()

    def update_fix_ids_and_abs_data(self, new_data):
        new_ids_fix = new_data[new_data < 0]
        if len(new_ids_fix) > 0:
            np.abs(new_data, out=new_data)
            self._ids_fixed = self._ids_fixed.update(new_ids_fix.tolist())
            self._ids_fixed_dict = None
            self._ids_var_dict = None

    def convert_data_hashes_to_indices(self, data, id_offset_mapping):
        s_uk = constants.vocab_manual[constants.UNKNOWN_EMBEDDING]
        #data_new = np.zeros(shape=data.shape, dtype=DTYPE_IDX)
        #assert s_uk in self.strings, '%s not in lexicon' % s_uk
        for i in range(len(data)):
            d = data[i]
            if d in id_offset_mapping:
                data[i] = len(self) + id_offset_mapping[d]
            elif d in self.mapping:
                data[i] = self.mapping[d]
            else:
                data[i] = self.mapping[self.strings[s_uk]]
        self.freeze()
        return data.astype(DTYPE_IDX)
        #return data_new

    def read_data(self, return_hashes=False, *args, **kwargs):
        data, parents = read_data(*args, strings=self.strings, **kwargs)
        self.clear_cached_values()
        if return_hashes:
            return Forest(data=data, parents=parents, lexicon=self, data_as_hashes=True)
        else:
            return Forest(data=self.convert_data_hashes_to_indices(data, id_offset_mapping={}), parents=parents, lexicon=self,
                          data_as_hashes=False)

    def is_fixed(self, idx):
        return idx in self._ids_fixed

    def add_all(self, new_strings):
        for s in new_strings:
            self._strings.add(s)
        self.clear_cached_values()

    @staticmethod
    def vocab_prefix(man_vocab_id):
        return constants.vocab_manual[man_vocab_id] + constants.SEPARATOR

    @staticmethod
    def has_vocab_prefix(s, man_vocab_id):
        return s.startswith(Lexicon.vocab_prefix(man_vocab_id))

    def clear_cached_values(self):
        assert not self.frozen, 'can not modify frozen lexicon'
        self._types = None
        self._mapping = None

    def freeze(self):
        self._frozen = True

    def unfreeze(self):
        self._frozen = False

    def get_s(self, d, data_as_hashes):
        if data_as_hashes:
            if d in self.mapping:
                return self.strings[d]
            else:
                return constants.vocab_manual[constants.UNKNOWN_EMBEDDING]
        else:
            if d < len(self):
                return self.strings[self.types[d]]
            else:
                return constants.vocab_manual[constants.UNKNOWN_EMBEDDING]

    def get_d(self, s, data_as_hashes):

        if s in self.strings:
            d = self.strings[s]
        else:
            d = self.strings[constants.vocab_manual[constants.UNKNOWN_EMBEDDING]]

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
            return self.strings[self.types[abs(item)]]

    def __len__(self):
        return len(self.strings)

    @property
    def vec_size(self):
        assert self._vecs is not None and self._vecs.shape[1] > 0, 'vecs are not set or have length 0'
        return self._vecs.shape[1]

    @property
    def len_fixed(self):
        return len(self._ids_fixed)

    @property
    def len_var(self):
        return len(self) - self.len_fixed

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
        return np.array(sorted(list(self._ids_fixed)))

    @property
    def ids_fixed_dict(self):
        if self._ids_fixed_dict is None:
            self._ids_fixed_dict = {k: i for i, k in enumerate(self.ids_fixed)}
        return self._ids_fixed_dict

    @property
    def ids_var_dict(self):
        if self._ids_var_dict is None:
            self._ids_var_dict = {k: i for i, k in enumerate(self.ids_var)}
        return self._ids_var_dict

    @property
    def ids_var(self):
        return np.array([i for i in range(len(self)) if i not in self._ids_fixed])

    @property
    def types(self):
        # maps positions to hashes
        #return self._types
        if self._types is None:
            self._types = np.zeros(shape=(len(self.strings),), dtype=DTYPE_HASH)
            for i, s in enumerate(self.strings):
                self._types[i] = self.strings[s]
        return self._types

    @property
    def strings(self):
        return self._strings

    @property
    def mapping(self):
        # maps hashes to positions
        if self._mapping is None:
            #self._mapping = mapping_from_list(self._types)
            self._mapping = {}
            for i, s in enumerate(self.strings):
                self._mapping[self.strings[s]] = i
        return self._mapping

    @property
    def is_filled(self):
        return self._vecs is not None and len(self) == len(self.vecs)

    @property
    def has_vecs(self):
        return self._vecs is not None

    @property
    def frozen(self):
        return self._frozen



# one_hot_types = [u'DEP#det', u'DEP#punct', u'DEP#pobj', u'DEP#ROOT', u'DEP#prep', u'DEP#aux', u'DEP#nsubj',
            # u'DEP#dobj', u'DEP#amod', u'DEP#conj', u'DEP#cc', u'DEP#compound', u'DEP#nummod', u'DEP#advmod', u'DEP#acl',
            # u'DEP#attr', u'DEP#auxpass', u'DEP#expl', u'DEP#nsubjpass', u'DEP#poss', u'DEP#agent', u'DEP#neg', u'DEP#prt',
            # u'DEP#relcl', u'DEP#acomp', u'DEP#advcl', u'DEP#case', u'DEP#npadvmod', u'DEP#xcomp', u'DEP#ccomp', u'DEP#pcomp',
            # u'DEP#oprd', u'DEP#nmod', u'DEP#mark', u'DEP#appos', u'DEP#dep', u'DEP#dative', u'DEP#quantmod', u'DEP#csubj',
            # u'DEP#']