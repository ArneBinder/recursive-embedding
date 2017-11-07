import csv

import numpy as np

import constants
import logging
import os


def load_data_and_parents(fn):
    data = np.load('%s.data' % fn)
    parents = np.load('%s.parent' % fn)
    return data, parents


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
    assert vecs1.shape[0] == len(types1), 'count of embeddings in vecs1 = ' + vecs1.shape[0] + \
                                          ' does not equal length of types1 = ' + str(len(types1))
    assert vecs2.shape[0] == len(types2), 'count of embeddings in vecs2 = ' + vecs2.shape[0] + \
                                          ' does not equal length of types2 = ' + str(len(types2))
    logging.info('size of dict1: ' + str(len(types1)))
    logging.info('size of dict2: ' + str(len(types2)))
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
        logging.info('removed ' + str(len(indices_delete)) + ' entries from dict1')

    if add:
        indices_types2 = sorted(range(len(types2)))
        indices_types2_set = set(indices_types2)
        indices2_added = sorted(indices2_added)
        logging.debug(indices_types2 == indices2_added)
        logging.debug(indices_types2 == indices2_added_debug)
        logging.debug(indices2_added_debug == indices2_added)

        types2_indices_add = list(indices_types2_set.difference(indices2_added))

        types1.extend([types2[idx] for idx in types2_indices_add])
        vecs1 = np.append(vecs1, vecs2[types2_indices_add], axis=0)
        logging.info('added ' + str(len(types2_indices_add)) + ' entries to dict1')
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
        # exclude entities which are in vocab_manual to avoid collisions
        if lexeme.orth_ in manual_vocab_reverted:
            logging.warn(
                'found token in parser vocab with orth_="' + lexeme.orth_ + '", which was already added from manual vocab: "' + ', '.join(
                    manual_vocab_reverted) + '", skip!')
            continue
        vecs[i] = lexeme.vector
        types.append(lexeme.orth_)
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


def read_dict(fn):
    logging.debug('load vecs from file: ' + fn + '.vec ...')
    v = np.load(fn + '.vec')
    t = read_types(fn)
    logging.debug('vecs.shape: ' + str(v.shape) + ', len(types): ' + str(len(t)))
    return v, t


def create_or_read_dict(fn, vocab=None, dont_read=False):
    if os.path.isfile(fn + '.vec') and os.path.isfile(fn + '.type'):
        if dont_read:
            return
        v, t = read_dict(fn)
    else:
        logging.debug('extract word embeddings from spaCy ...')
        v, t = get_dict_from_vocab(vocab)
        write_dict(fn, vecs=v, types=t)
    return v, t


def write_dict(out_path, vecs=None, types=None, counts=None):
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