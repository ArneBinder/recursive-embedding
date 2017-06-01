import codecs
import csv
import os
import pickle

import logging
import numpy as np

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
        with codecs.open(out_path + '.type', 'w', 'utf-8') as f:
            for t in types:
                f.write(t + '\n')
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
        t = list(read_types(fn))
        logging.info('vecs.shape: ' + str(v.shape) + ', len(ids): ' + str(len(i)))
    else:
        out_dir = os.path.abspath(os.path.join(fn, os.pardir))
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        logging.info('extract word embeddings from spaCy ...')
        v, i, t = preprocessing.get_word_embeddings(vocab)
        write_dict(fn, i, v, t)
    return v, i, t


def revert_mapping(mapping):
    temp = {}
    for key in mapping:
        temp[mapping[key]] = key
    return temp


def revert_mapping_np(mapping):
    temp = -np.ones(shape=len(mapping), dtype=np.int32)
    for key in mapping:
        temp[mapping[key]] = key
    return temp


def read_types(out_path):
    with codecs.open(out_path + '.type', 'r', 'utf-8') as f:
        for line in f:
            yield line.rstrip('\n')


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
        with open(fn + '.type', 'w') as f:
            for row in reader:
                ids.append(int(row[TSV_COLUMN_NAME_ID]))
                f.write(row[TSV_COLUMN_NAME_LABEL] + '\n')

    print('convert and dump ids...')
    ids_np = np.array(ids)
    ids_np.dump(fn + '.id')