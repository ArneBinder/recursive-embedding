import csv

import numpy as np

import constants
import logging
import os

import preprocessing
import sequence_trees as sequ_trees


def sort_and_cut_and_fill_dict(seq_data, vecs, types, count_threshold=1):
    logging.info('sort, cut and fill embeddings ...')
    new_max_size = len(types)
    logging.info('initial vecs shape: ' + str(vecs.shape))
    logging.info('initial types size: ' + str(len(types)))
    # count types
    logging.debug('calculate counts ...')
    counts = np.zeros(shape=new_max_size, dtype=np.int32)
    for d in seq_data:
        counts[d] += 1

    logging.debug('argsort ...')
    sorted_indices = np.argsort(counts)

    # take mean and variance from previous vectors
    vecs_mean = np.mean(vecs, axis=0)
    vecs_variance = np.var(vecs, axis=0)
    new_vecs = np.zeros(shape=(new_max_size, vecs.shape[1]), dtype=vecs.dtype)
    # new_vecs = np.random.standard_normal(size=(new_max_size, vecs.shape[1])) * 0.1
    new_counts = np.zeros(shape=new_max_size, dtype=np.int32)
    new_types = [None] * new_max_size
    converter = -np.ones(shape=new_max_size, dtype=np.int32)

    logging.debug('process reversed(sorted_indices) ...')
    new_idx = 0
    new_idx_unknown = -1
    new_count = 0
    added_types = []
    for old_idx in reversed(sorted_indices):
        # keep unknown and save new unknown index
        if types[old_idx] == constants.vocab_manual[constants.UNKNOWN_EMBEDDING]:
            logging.debug('idx_unknown moved from ' + str(old_idx) + ' to ' + str(new_idx))
            new_idx_unknown = new_idx
        # skip vecs with count < threshold, but keep vecs from vocab_manual
        elif counts[old_idx] < count_threshold and types[old_idx] not in constants.vocab_manual.values():
            continue
        if old_idx < vecs.shape[0]:
            new_vecs[new_idx] = vecs[old_idx]

        else:
            # init missing vecs with previous vecs distribution
            #if not new_as_one_hot:
            new_vecs[new_idx] = np.random.standard_normal(size=vecs.shape[1]) * vecs_variance + vecs_mean
            #else:
            #    if new_count >= vecs.shape[1]:
            #        logging.warning('Adding more then vecs-size=%i new lex entries with new_as_one_hot=True (use '
            #                        'one-hot encodings). That overrides previously added new fake embeddings!'
            #                        % vecs.shape[1])
            #    new_vecs[new_idx][new_count % vecs.shape[1]] = 1.0
            new_count += 1
            added_types.append(types[old_idx])
            # print(types[old_idx] + '\t'+str(counts[old_idx]))

        new_types[new_idx] = types[old_idx]
        new_counts[new_idx] = counts[old_idx]
        converter[old_idx] = new_idx
        new_idx += 1

    assert new_idx_unknown >= 0, 'UNKNOWN_EMBEDDING not in types'

    logging.info('new lex_size: ' + str(new_idx))
    logging.debug('added ' + str(new_count) + ' new vecs to vocab')
    logging.debug(added_types)

    # cut arrays
    new_vecs = new_vecs[:new_idx, :]
    new_counts = new_counts[:new_idx]
    new_types = new_types[:new_idx]

    return converter, new_vecs, new_types, new_counts, new_idx_unknown


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
    logging.info('size of dict1: %i' % len(types1))
    logging.info('size of dict2: %i' % len(types2))
    mapping2 = mapping_from_list(types2)
    logging.debug(len(mapping2))
    logging.debug(np.array_equal(vecs1, vecs2))
    logging.debug(types1 == types2)

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
        logging.info('removed %i entries from dict1' % len(indices_delete))

    if add:
        indices_types2 = sorted(range(len(types2)))
        indices_types2_set = set(indices_types2)
        indices2_added = sorted(indices2_added)
        logging.debug(indices_types2 == indices2_added)
        logging.debug(indices_types2 == indices2_added_debug)
        logging.debug(indices2_added_debug == indices2_added)

        types2_indices_add = list(indices_types2_set.difference(indices2_added))

        types1.extend([types2[idx] for idx in types2_indices_add])
        vecs1 = np.concatenate((vecs1, vecs2[types2_indices_add]), axis=0)
        logging.info('added %i entries to dict1' % len(types2_indices_add))
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
            logging.warn(
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


def read_types(out_path):
    logging.debug('read types from file: ' + out_path + '.type ...')
    with open(out_path + '.type') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t', quotechar='|')
        types = [row[0].decode("utf-8") for row in reader]
    return types


def mapping_from_list(l):
    m = {}
    for i, x in enumerate(l):
        if x in m:
            logging.warn('already in dict: "' + x + '" at idx: ' + str(m[x]))
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


def load(fn):
    logging.debug('load vecs from file: ' + fn + '.vec ...')
    v = np.load(fn + '.vec')
    t = read_types(fn)
    logging.debug('vecs.shape: ' + str(v.shape) + ', len(types): ' + str(len(t)))
    return v, t


def exist(filename):
    return os.path.isfile('%s.vec' % filename) and os.path.isfile('%s.type' % filename)


def create_or_read_dict(fn, vocab=None, dont_read=False):
    if os.path.isfile(fn + '.vec') and os.path.isfile(fn + '.type'):
        if dont_read:
            return
        v, t = load(fn)
    else:
        logging.debug('extract word embeddings from spaCy ...')
        v, t = get_dict_from_vocab(vocab)
        dump(fn, vecs=v, types=t)
    return v, t


def dump(out_path, vecs=None, types=None, counts=None):
    if vecs is not None:
        logging.debug('dump embeddings (shape=' + str(vecs.shape) + ') to: ' + out_path + '.vec ...')
        vecs.dump(out_path + '.vec')
    if types is not None:
        logging.debug('write types (len=' + str(len(types)) + ') to: ' + out_path + '.types ...')
        with open(out_path + '.type', 'wb') as f:
            writer = csv.writer(f, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for t in types:
                writer.writerow([t.encode("utf-8")])
    if counts is not None:
        logging.debug('dump counts (len=' + str(len(counts)) + ') to: ' + out_path + '.count ...')
        counts.dump(out_path + '.count')


def vocab_prefix(man_vocab_id):
    return constants.vocab_manual[man_vocab_id] + constants.SEPARATOR


def has_vocab_prefix(s, man_vocab_id):
    return s.startswith(vocab_prefix(man_vocab_id))


class Lexicon(object):
    def __init__(self, filename=None, vecs=None, types=None, nlp_vocab=None):
        self._dummy_vec_size = 300
        self._filename = filename
        self._mapping = None
        self._dumped_vecs = False
        self._dumped_types = False
        if filename is not None:
            #self._vecs, self._types = load(filename)
            self._types = read_types(filename)
            self._dumped_types = True
            if os.path.isfile('%s.vec' % filename):
                self._vecs = np.load('%s.vec' % filename)
                self._dumped_vecs = True
            else:
                # set dummy vecs
                self.init_vecs()
        elif types is not None:
            self._types = types
            if vecs is not None:
                self._vecs = vecs
            else:
                # set dummy vecs
                self.init_vecs()
        elif nlp_vocab is not None:
            self._vecs, self._types = get_dict_from_vocab(nlp_vocab)
            #print(self.filled)
        else:
            raise ValueError('Not enouth arguments to instantiate Lexicon object. Please provide a filename or (vecs array and types list) or a nlp_vocab.')

    def init_vecs(self, new_vecs=None):
        if new_vecs is None:
            new_vecs = np.zeros(shape=(0, self._dummy_vec_size), dtype=np.float32)
            self._dumped_vecs = True
        else:
            self._dumped_vecs = False
        if not self._dumped_vecs:
            logging.warning('overwrite unsaved vecs')
        assert len(new_vecs) <= len(self), 'can not set more vecs than amount of existing types (len(new_vecs)==%i > len(types)==%i)' % (len(new_vecs), len(self))
        self._vecs = new_vecs

    def dump(self, filename, types_only=False):
        dump_vecs = (not self._dumped_vecs or filename != self._filename) and not types_only
        dump_types = not self._dumped_types or filename != self._filename
        dump(filename,
             vecs=self.vecs if dump_vecs else None,
             types=self.types if dump_types else None)
        self._dumped_vecs = dump_vecs or len(self.vecs) == 0
        self._dumped_types = True
        self._filename = filename

    # compatibility
    def set_types_with_mapping(self, mapping):
        self._types = revert_mapping_to_list(mapping)
        self._dumped_types = False

    def sort_and_cut_and_fill_dict(self, data, count_threshold=1):
        converter, self._vecs, self._types, new_counts, new_idx_unknown = sort_and_cut_and_fill_dict(seq_data=data,
                                                                                                     vecs=self._vecs,
                                                                                                     types=self._types,
                                                                                                     count_threshold=count_threshold)
        self._mapping = None
        self._dumped_vecs = False
        self._dumped_types = False
        return converter, new_counts, new_idx_unknown

    def get_ids_for_prefix(self, prefix):
        res = [self[t] for t in self._types if t.startswith(prefix + constants.SEPARATOR)]
        if len(res) == 0:
            logging.warning('no indices found for prefix=%s' % prefix)
        return res

    def set_to_zero(self, indices=None, prefix=None):
        assert indices is not None or prefix is not None, 'please provide indices or a prefix'
        if indices is None:
            indices = self.get_ids_for_prefix(prefix)
        for i in indices:
            self._vecs[i] = np.zeros(self._vecs.shape[1], dtype=self._vecs.dtype)
        if len(indices) > 0:
            logging.info('set %i vecs to zero' % len(indices))
            self._dumped_vecs = False

    def set_to_onehot(self, indices=None, prefix=None):
        assert indices is not None or prefix is not None, 'please provide indices or a prefix'
        if indices is None:
            indices = self.get_ids_for_prefix(prefix)
        self.set_to_zero(indices=indices)
        for i, idx in enumerate(indices):
            self._vecs[idx][i] = 1.0
        if len(indices) > 0:
            logging.info('set %i vecs to one-hot' % len(indices))
            self._dumped_vecs = False

    def set_man_vocab_vec(self, man_vocab_id, new_vec=None):
        if new_vec is None:
            new_vec = np.zeros(shape=self.vec_size, dtype=self._vecs.dtype)
        idx = self[constants.vocab_manual[man_vocab_id]]
        self._vecs[idx] = new_vec
        self._dumped_vecs = False
        return idx

    def pad(self):
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
            self._dumped_vecs = False
        else:
            raise IndexError('len(self)==len(types)==%i < len(vecs)==%i' % (len(self), len(self.vecs)))

    def merge(self, other, add=True, remove=True):
        self._vecs, self._types = merge_dicts(vecs1=self.vecs, types1=self.types, vecs2=other.vecs, types2=other.types, add=add, remove=remove)
        self._mapping = None
        self._dumped_vecs = False
        self._dumped_types = False
        if add:
            converter_other = [self.mapping[t] for t in other.types]
        else:
            converter_other = None
        return converter_other

    def read_data(self, *args, **kwargs):
        data, parents = preprocessing.read_data(*args, data_maps=self.mapping, **kwargs)
        self._types = revert_mapping_to_list(self.mapping)
        self._dumped_types = False
        return sequ_trees.Forest(data=data, parents=parents)

    def __getitem__(self, item):
        if type(item) == unicode or type(item) == str:
            try:
                res = self.mapping[item]
            # word doesnt occur in dictionary
            except KeyError:
                res = len(self)
                self._mapping[item] = res
                self._types.append(item)
            return res
        else:
            return self.types[item]

    def __len__(self):
        return len(self.types)

    @property
    def vec_size(self):
        return self._vecs.shape[1]

    @property
    def vecs(self):
        return self._vecs

    @property
    def types(self):
        return self._types

    @property
    def mapping(self):
        if self._mapping is None:
            self._mapping = mapping_from_list(self._types)
        return self._mapping

    @property
    def is_filled(self):
        return len(self) == len(self.vecs)



# one_hot_types = [u'DEP#det', u'DEP#punct', u'DEP#pobj', u'DEP#ROOT', u'DEP#prep', u'DEP#aux', u'DEP#nsubj',
            # u'DEP#dobj', u'DEP#amod', u'DEP#conj', u'DEP#cc', u'DEP#compound', u'DEP#nummod', u'DEP#advmod', u'DEP#acl',
            # u'DEP#attr', u'DEP#auxpass', u'DEP#expl', u'DEP#nsubjpass', u'DEP#poss', u'DEP#agent', u'DEP#neg', u'DEP#prt',
            # u'DEP#relcl', u'DEP#acomp', u'DEP#advcl', u'DEP#case', u'DEP#npadvmod', u'DEP#xcomp', u'DEP#ccomp', u'DEP#pcomp',
            # u'DEP#oprd', u'DEP#nmod', u'DEP#mark', u'DEP#appos', u'DEP#dep', u'DEP#dative', u'DEP#quantmod', u'DEP#csubj',
            # u'DEP#']