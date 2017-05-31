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


def write_dict(out_path, mapping, vecs, vocab_nlp=None, vocab_manual=None):
    logging.info('dump embeddings to: ' + out_path + '.vecs ...')
    vecs.dump(out_path + '.vecs')
    logging.info('dump mappings to: ' + out_path + '.mapping ...')
    with open(out_path + '.mapping', "wb") as f:
        pickle.dump(mapping, f)
    logging.info('vecs.shape: ' + str(vecs.shape) + ', len(mapping): ' + str(len(mapping)))

    if vocab_nlp is not None:
        write_dict_plain_token(out_path, mapping, vocab_nlp)


def create_or_read_dict(fn, vocab=None):
    if os.path.isfile(fn+'.vecs') and os.path.isfile(fn+'.mapping'):
        logging.info('load vecs from file: '+fn + '.vecs ...')
        v = np.load(fn+'.vecs')
        logging.info('load mapping from file: ' + fn + '.mapping ...')
        m = pickle.load(open(fn+'.mapping', "rb"))
        logging.info('vecs.shape: ' + str(v.shape) + ', len(mapping): ' + str(len(m)))
    else:
        out_dir = os.path.abspath(os.path.join(fn, os.pardir))
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        logging.info('extract word embeddings from spaCy ...')
        v, m = preprocessing.get_word_embeddings(vocab)
        logging.info('vecs.shape: ' + str(v.shape) + ', len(mapping): ' + str(len(m)))
        logging.info('dump vecs to: ' + fn + '.vecs ...')
        v.dump(fn + '.vecs')
        logging.info('dump mappings to: ' + fn + '.mapping ...')
        with open(fn + '.mapping', "wb") as f:
            pickle.dump(m, f)
    return v, m


def revert_mapping(mapping):
    temp = {}
    for key in mapping:
        temp[mapping[key]] = key
    return temp


def revert_mapping_np(mapping):
    temp = np.zeros(shape=(len(mapping)), dtype=np.int32)
    for key in mapping:
        temp[mapping[key]] = key
    return temp


def write_dict_plain_token(out_path, mapping, spacy_vocab):
    logging.info('write tsv dict: ' + out_path + '.tsv ...')
    rev_map = revert_mapping_np(mapping)
    with open(out_path + '.tsv', 'wb') as csvfile:
        fieldnames = [TSV_COLUMN_NAME_LABEL, TSV_COLUMN_NAME_ID]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter='\t', quotechar='|',
                                quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        for i in range(len(rev_map)):
            id_orig = rev_map[i]
            if id_orig >= 0:
                label = spacy_vocab[id_orig].orth_
            else:
                label = constants.vocab_manual[id_orig]
            writer.writerow({'label': label.encode("utf-8"), 'id_orig': str(id_orig)})


def create_or_read_dict_types_string(fn, mapping=None, spacy_vocab=None):
    if not os.path.isfile(fn + '.tsv'):
        assert mapping is not None and spacy_vocab is not None, 'no mapping and/or spacy_vocab defined'
        write_dict_plain_token(fn, mapping, spacy_vocab)
    with open(fn + '.tsv') as csvfile:
        logging.info('read type strings from ' + fn + '.tsv ...')
        reader = csv.DictReader(csvfile, delimiter='\t', quotechar='|')
        for row in reader:
            yield row[TSV_COLUMN_NAME_LABEL]




