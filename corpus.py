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

def write_dict(out_path, ids, vecs, types=None, counts=None):
    logging.info('dump embeddings to: ' + out_path + '.vec ...')
    vecs.dump(out_path + '.vec')
    logging.info('dump ids to: ' + out_path + '.id ...')
    ids.dump(out_path + '.id')
    if types is not None:
        logging.info('write types to: ' + out_path + '.types ...')
        with open(out_path + '.type', 'wb') as f:
            writer = csv.writer(f, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for t in types:
                writer.writerow([t.encode("utf-8")])
    if counts is not None:
        logging.info('dump counts to: ' + out_path + '.count ...')
        counts.dump(out_path + '.count')
    logging.info('vecs.shape: ' + str(vecs.shape) + ', len(ids): ' + str(len(ids)))


def create_or_read_dict(fn, vocab=None):
    if os.path.isfile(fn+'.vec') and os.path.isfile(fn+'.type') and os.path.isfile(fn+'.id'):
        logging.info('load vecs from file: '+fn + '.vec ...')
        v = np.load(fn+'.vec')
        logging.info('load ids from file: ' + fn + '.id ...')
        i = np.load(fn+'.id')
        logging.info('read types from file: ' + fn + '.type ...')
        t = read_types(fn)
        logging.info('vecs.shape: ' + str(v.shape) + ', len(ids): ' + str(len(i)))
    else:
        out_dir = os.path.abspath(os.path.join(fn, os.pardir))
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        logging.info('extract word embeddings from spaCy ...')
        v, i, t = get_dict_from_vocab(vocab)
        write_dict(fn, i, v, t)
    return v, i, t


def revert_mapping(mapping):
    temp = {}
    for key in mapping:
        temp[mapping[key]] = key
    return temp


def revert_mapping_to_list(mapping):
    temp = [None] * len(mapping)
    for key in mapping:
        temp[mapping[key]] = key
    return temp


def revert_mapping_np(mapping):
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
                #f.write(row[TSV_COLUMN_NAME_LABEL] + '\n')

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
    manual_vocab_reverted = revert_mapping(constants.vocab_manual)
    # add unknown
    #unknown_idx = vocab[constants.vocab_manual[constants.UNKNOWN_EMBEDDING]].orth
    # subtract 1, implementation of len() for vocab is incorrect
    size = len(vocab)
    #print(size)
    vecs = np.zeros(shape=(size, vocab.vectors_length), dtype=np.float32)
    ids = -np.ones(shape=(size, ), dtype=np.int32)
    # constants.UNKNOWN_IDX=0
    types_unknown = constants.vocab_manual[constants.UNKNOWN_EMBEDDING]
    types = [types_unknown]
    # constants.UNKNOWN_EMBEDDING=0
    ids[0] = constants.UNKNOWN_EMBEDDING
    i = 1
    for lexeme in vocab:
        # exclude entities which are in vocab_manual to avoid collisions
        if lexeme.orth_ in manual_vocab_reverted:
            logging.warn('found token in vocab with orth_="'+lexeme.orth_+'", which is already in manual vocab: "'+', '.join(manual_vocab_reverted)+'", skip!')
            #size -= 1
            continue
        vecs[i] = lexeme.vector
        ids[i] = lexeme.orth
        types.append(lexeme.orth_)
        i += 1
    #print(i)
    # constants.UNKNOWN_IDX=0
    vecs[0] = np.mean(vecs[1:i], axis=0)

    #print(len(ids))
    #print(vecs.shape)
    #print(len(types))

    # cut, if orth id was in vocab
    if i < size:
        vecs = vecs[:i]
        ids = ids[:i]
        types = types[:i]

    #print(len(ids))
    #print(vecs.shape)
    #print(len(types))

    return vecs, ids, types


def calc_ids_from_types(types, vocab=None):
    manual_vocab_reverted = revert_mapping(constants.vocab_manual)
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
