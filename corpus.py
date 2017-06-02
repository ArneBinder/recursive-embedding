import codecs
import csv
import os
import pickle

import logging
import numpy as np
import spacy

import constants
import preprocessing
import tools


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

def write_dict(out_path, vecs, types=None, counts=None):
    logging.info('dump embeddings to: ' + out_path + '.vec ...')
    vecs.dump(out_path + '.vec')
    if types is not None:
        logging.info('write types to: ' + out_path + '.types ...')
        with open(out_path + '.type', 'wb') as f:
            writer = csv.writer(f, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for t in types:
                writer.writerow([t.encode("utf-8")])
    if counts is not None:
        logging.info('dump counts to: ' + out_path + '.count ...')
        counts.dump(out_path + '.count')
    logging.info('vecs.shape: ' + str(vecs.shape) + ', len(types): ' + str(len(types)))


def create_or_read_dict(fn, vocab=None):
    if os.path.isfile(fn+'.vec') and os.path.isfile(fn+'.type') and os.path.isfile(fn+'.id'):
        logging.info('load vecs from file: '+fn + '.vec ...')
        v = np.load(fn+'.vec')
        logging.info('read types from file: ' + fn + '.type ...')
        t = read_types(fn)
        logging.info('vecs.shape: ' + str(v.shape) + ', len(types): ' + str(len(t)))
    else:
        out_dir = os.path.abspath(os.path.join(fn, os.pardir))
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        logging.info('extract word embeddings from spaCy ...')
        v, t = get_dict_from_vocab(vocab)
        write_dict(fn, v, t)
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
            logging.warn('found token in vocab with orth_="'+lexeme.orth_+'", which is already in manual vocab: "'+', '.join(manual_vocab_reverted)+'", skip!')
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


# TODO: test this!
def replace_dict(vecs1, types1, vecs2, types2):
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
            indices2_added.append(idx2)
            types1[idx] = types2[idx2]
            vecs1[idx] = vecs2[idx2]
        else:
            indices_delete.append(idx)

    for idx in indices_delete:
        del types1[idx]

    vecs1 = np.delete(vecs1, indices_delete, 0)
    logging.info('removed ' + str(len(indices_delete)) + ' entries from dict1')

    types2_indices_add = list(set(range(len(types2))).difference(indices2_added))

    types1.extend([types2[idx] for idx in types2_indices_add])
    vecs1 = np.append(vecs1, vecs2[types2_indices_add], axis=0)
    logging.info('added ' + str(len(types2_indices_add)) + ' entries to dict1')
    return vecs1, types1


# TODO: test this!
def merge_into_dict(vecs1, types1, vecs2, types2):
    assert vecs1.shape[0] == len(types1), 'count of embeddings in vecs1 = ' + vecs1.shape[0] + \
                                          ' does not equal length of types1 = ' + str(len(types1))
    assert vecs2.shape[0] == len(types2), 'count of embeddings in vecs2 = ' + vecs2.shape[0] + \
                                          ' does not equal length of types2 = ' + str(len(types2))
    logging.info('size of dict1: ' + str(len(types1)))
    logging.info('size of dict2: ' + str(len(types2)))

    mapping2 = mapping_from_list(types2)

    indices2_added = []
    for idx, t in enumerate(types1):
        if t in mapping2:
            idx2 = mapping2[t]
            indices2_added.append(idx2)
            types1[idx] = types2[idx2]
            vecs1[idx] = vecs2[idx2]

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
