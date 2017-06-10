#import codecs
import csv
import os
#import pickle

import logging
import numpy as np
import spacy

import constants
#import preprocessing
#import tools


TSV_COLUMN_NAME_LABEL = 'label'
TSV_COLUMN_NAME_ID = 'id_orig'


#def write_dict(out_path, mapping, vecs, vocab_nlp=None, vocab_manual=None):
#    logging.info('dump embeddings to: ' + out_path + '.vec ...')
#    vecs.dump(out_path + '.vec')
#    logging.info('dump mappings to: ' + out_path + '.mapping ...')
#    with open(out_path + '.mapping', "wb") as f:
#        pickle.dump(mapping, f)
#    logging.info('vecs.shape: ' + str(vecs.shape) + ', len(mapping): ' + str(len(mapping)))
#
#    if vocab_nlp is not None:
#        write_dict_plain_token(out_path, mapping, vocab_nlp)

def write_dict(out_path, vecs=None, types=None, counts=None):
    if vecs is not None:
        logging.info('dump embeddings (shape=' + str(vecs.shape) + ') to: ' + out_path + '.vec ...')
        vecs.dump(out_path + '.vec')
    if types is not None:
        logging.info('write types (len='+str(len(types))+') to: ' + out_path + '.types ...')
        with open(out_path + '.type', 'wb') as f:
            writer = csv.writer(f, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for t in types:
                writer.writerow([t.encode("utf-8")])
    if counts is not None:
        logging.info('dump counts (len='+str(len(counts))+') to: ' + out_path + '.count ...')
        counts.dump(out_path + '.count')


def create_or_read_dict(fn, vocab=None, dont_read=False):
    if os.path.isfile(fn+'.vec') and os.path.isfile(fn+'.type'):
        if dont_read:
            return
        logging.info('load vecs from file: '+fn + '.vec ...')
        v = np.load(fn+'.vec')
        t = read_types(fn)
        logging.info('vecs.shape: ' + str(v.shape) + ', len(types): ' + str(len(t)))
    else:
        logging.info('extract word embeddings from spaCy ...')
        v, t = get_dict_from_vocab(vocab)
        write_dict(fn, vecs=v, types=t)
    return v, t


def revert_mapping_to_map(mapping):
    temp = {}
    for key in mapping:
        temp[mapping[key]] = key
    return temp


def revert_mapping_to_list(mapping):
    temp = [None] * len(mapping)
    for key in mapping:
        temp[mapping[key]] = key
    return temp


def revert_mapping_to_np(mapping):
    temp = -np.ones(shape=len(mapping), dtype=np.int32)
    for key in mapping:
        temp[mapping[key]] = key
    return temp


def read_types(out_path):
    logging.info('read types from file: ' + out_path + '.type ...')
    with open(out_path + '.type') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t', quotechar='|')
        types = [row[0].decode("utf-8") for row in reader]
    return types


def mapping_from_list(l):
    m = {}
    for i, x in enumerate(l):
        m[x] = i
    return m


# convert deprecated format
def tsv_to_ids_and_types(fn):
    ids = []
    with open(fn + '.tsv') as csvfile:
        print('read type strings from ' + fn + '.tsv ...')
        reader = csv.DictReader(csvfile, delimiter='\t', quotechar='|')
        with open(fn + '.type', 'wb') as f:
            writer = csv.writer(f, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for row in reader:
                ids.append(int(row[TSV_COLUMN_NAME_ID]))
                writer.writerow([row[TSV_COLUMN_NAME_LABEL]])

    print('len(ids)='+str(len(ids)))
    print('convert and dump ids...')
    ids_np = np.array(ids)
    ids_np.dump(fn + '.id')


#debug
def move_to_front(fn, idx):
    ids = np.load(fn + '.id.bk')
    vecs = np.load(fn + '.vec.bk')
    with open(fn + '.type.bk') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t', quotechar='|')
        types = [row[0] for row in reader]
    data = np.load(fn + '.data.bk')
    print(len(ids))
    print(len(vecs))
    print(len(types))
    print(len(data))

    #converter = np.zeros(shape=len(ids), dtype=np.int32)

    new_ids = np.zeros(shape=ids.shape, dtype=ids.dtype)
    new_vecs = np.zeros(shape=vecs.shape, dtype=vecs.dtype)
    new_types = [None] * len(ids)

    for i in range(idx):
        new_ids[i+1] = ids[i]
        new_vecs[i+1] = vecs[i]
        new_types[i+1] = types[i]

    new_ids[0] = ids[idx]
    new_vecs[0] = vecs[idx]
    new_types[0] = types[idx]

    for i in range(idx+1, len(ids)):
        new_ids[i] = ids[i]
        new_vecs[i] = vecs[i]
        new_types[i] = types[i]

    new_data = np.zeros(shape=data.shape, dtype=data.dtype)
    for i, d in enumerate(data):
        if d < idx:
            new_data[i] = data[i] + 1
        elif d == idx:
            new_data[i] = 0
        else:
            new_data[i] = data[i]

    new_ids.dump(fn + '.id')
    new_vecs.dump(fn + '.vec')
    new_data.dump(fn + '.data')
    with open(fn + '.type', 'wb') as f:
        writer = csv.writer(f, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for t in new_types:
            writer.writerow([t])


def get_dict_from_vocab(vocab):
    manual_vocab_reverted = revert_mapping_to_map(constants.vocab_manual)
    size = len(vocab)
    vecs = np.zeros(shape=(size, vocab.vectors_length), dtype=np.float32)
    types_unknown = constants.vocab_manual[constants.UNKNOWN_EMBEDDING]
    types = [types_unknown]
    i = 1
    for lexeme in vocab:
        # exclude entities which are in vocab_manual to avoid collisions
        if lexeme.orth_ in manual_vocab_reverted:
            logging.warn('found token in parser vocab with orth_="'+lexeme.orth_+'", which is already in manual vocab: "'+', '.join(manual_vocab_reverted)+'", skip!')
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
    logging.info('size of dict1: '+str(len(types1)))
    logging.info('size of dict2: ' + str(len(types2)))
    mapping2 = mapping_from_list(types2)

    indices_delete = []
    indices2_added = []
    for idx, t in enumerate(types1):
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
        types2_indices_add = list(set(range(len(types2))).difference(indices2_added))

        types1.extend([types2[idx] for idx in types2_indices_add])
        vecs1 = np.append(vecs1, vecs2[types2_indices_add], axis=0)
        logging.info('added ' + str(len(types2_indices_add)) + ' entries to dict1')
    return vecs1, types1


# deprected
def calc_ids_from_types(types, vocab=None):
    manual_vocab_reverted = revert_mapping_to_map(constants.vocab_manual)
    vocab_added = {}
    ids = np.ndarray(shape=(len(types), ), dtype=np.int32)
    if vocab is None:
        parser = spacy.load('en')
        vocab = parser.vocab
    for i, t in enumerate(types):
        if t in manual_vocab_reverted:
            ids[i] = manual_vocab_reverted[t]
            logging.debug('add vocab manual id='+str(ids[i]) + ' for type='+t)
        else:
            ids[i] = vocab[t].orth
        assert ids[i] not in vocab_added, 'type='+t+' exists more then one time in types at pos=' + str(vocab_added[ids[i]]) + ' and at pos=' + str(i)
        vocab_added[ids[i]] = i
    return ids


def make_parent_dir(fn):
    out_dir = os.path.abspath(os.path.join(fn, os.pardir))
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)