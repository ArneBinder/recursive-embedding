#import codecs
import csv
import os
#import pickle

import logging
import numpy as np
import spacy
import ntpath

import constants
#import preprocessing
#import tools
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

def write_dict(out_path, vecs=None, types=None, counts=None):
    if vecs is not None:
        logging.debug('dump embeddings (shape=' + str(vecs.shape) + ') to: ' + out_path + '.vec ...')
        vecs.dump(out_path + '.vec')
    if types is not None:
        logging.debug('write types (len='+str(len(types))+') to: ' + out_path + '.types ...')
        with open(out_path + '.type', 'wb') as f:
            writer = csv.writer(f, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for t in types:
                writer.writerow([t.encode("utf-8")])
    if counts is not None:
        logging.debug('dump counts (len='+str(len(counts))+') to: ' + out_path + '.count ...')
        counts.dump(out_path + '.count')


def create_or_read_dict(fn, vocab=None, dont_read=False):
    if os.path.isfile(fn+'.vec') and os.path.isfile(fn+'.type'):
        if dont_read:
            return
        logging.debug('load vecs from file: '+fn + '.vec ...')
        v = np.load(fn+'.vec')
        t = read_types(fn)
        logging.debug('vecs.shape: ' + str(v.shape) + ', len(types): ' + str(len(t)))
    else:
        logging.debug('extract word embeddings from spaCy ...')
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
    logging.debug('read types from file: ' + out_path + '.type ...')
    with open(out_path + '.type') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t', quotechar='|')
        types = [row[0].decode("utf-8") for row in reader]
    return types


def mapping_from_list(l):
    m = {}
    for i, x in enumerate(l):
        if x in m:
            logging.warn('already in dict: "'+x+'" at idx: '+str(m[x]))
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
    size = len(vocab) + len(constants.vocab_manual)
    vecs = np.zeros(shape=(size, vocab.vectors_length), dtype=np.float32)
    #types_unknown = constants.vocab_manual[constants.UNKNOWN_EMBEDDING]
    #types = [types_unknown]

    # add manual vocab at first
    # the vecs remain zeros
    types = constants.vocab_manual.values()
    #i = 1
    i = len(constants.vocab_manual)
    for lexeme in vocab:
        # exclude entities which are in vocab_manual to avoid collisions
        if lexeme.orth_ in manual_vocab_reverted:
            logging.warn('found token in parser vocab with orth_="'+lexeme.orth_+'", which was already added from manual vocab: "'+', '.join(manual_vocab_reverted)+'", skip!')
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

    vecs_mean = np.mean(vecs, axis=0)
    new_vecs = np.zeros(shape=(new_max_size, vecs.shape[1]), dtype=vecs.dtype)
    new_counts = np.zeros(shape=new_max_size, dtype=np.int32)
    new_types = [None] * new_max_size
    converter = -np.ones(shape=new_max_size, dtype=np.int32)

    logging.debug('process reversed(sorted_indices) ...')
    new_idx = 0
    new_idx_unknown = -1
    for old_idx in reversed(sorted_indices):
        # keep unknown and save new unknown index
        if types[old_idx] == constants.vocab_manual[constants.UNKNOWN_EMBEDDING]:
            logging.debug('idx_unknown moved from ' + str(old_idx) + ' to ' + str(new_idx))
            new_idx_unknown = new_idx
        # keep pre-initialized vecs (count==0), but skip other vecs with count < threshold
        elif counts[old_idx] < count_threshold:
            continue
        if old_idx < vecs.shape[0]:
            new_vecs[new_idx] = vecs[old_idx]

        else:
            # init missing vecs with mean
            new_vecs[new_idx] = vecs_mean

        new_types[new_idx] = types[old_idx]
        new_counts[new_idx] = counts[old_idx]
        converter[old_idx] = new_idx
        new_idx += 1

    assert new_idx_unknown >= 0, 'UNKNOWN_EMBEDDING not in types'

    logging.info('new lex_size: '+str(new_idx))

    # cut arrays
    new_vecs = new_vecs[:new_idx, :]
    new_counts = new_counts[:new_idx]
    new_types = new_types[:new_idx]

    return converter, new_vecs, new_types, new_counts, new_idx_unknown


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


@tools.fn_timer
def convert_texts(in_filename, out_filename, init_dict_filename, sentence_processor, parser, reader,  # mapping, vecs,
                  max_articles=10000, max_depth=10, batch_size=100, article_offset=0, count_threshold=2,
                  concat_mode='sequence', inner_concat_mode='tree'):
    parent_dir = os.path.abspath(os.path.join(out_filename, os.pardir))
    out_base_name = ntpath.basename(out_filename)
    if not os.path.isfile(out_filename + '.data') \
            or not os.path.isfile(out_filename + '.parent') \
            or not os.path.isfile(out_filename + '.vec') \
            or not os.path.isfile(out_filename + '.depth') \
            or not os.path.isfile(out_filename + '.count'):
        if not parser:
            logging.info('load spacy ...')
            parser = spacy.load('en')
            parser.pipeline = [parser.tagger, parser.entity, parser.parser]
        # get vecs and types and save it at out_filename
        if init_dict_filename:
            vecs, types = create_or_read_dict(init_dict_filename)
            write_dict(out_filename, vecs=vecs, types=types)
        else:
            create_or_read_dict(out_filename, vocab=parser.vocab, dont_read=True)

        # parse
        parse_texts(out_filename=out_filename, in_filename=in_filename, reader=reader, parser=parser,
                    sentence_processor=sentence_processor, max_articles=max_articles,
                    batch_size=batch_size, concat_mode=concat_mode, inner_concat_mode=inner_concat_mode,
                    article_offset=article_offset, add_reader_args={})
        # merge batches
        preprocessing.merge_numpy_batch_files(out_base_name + '.parent', parent_dir)
        preprocessing.merge_numpy_batch_files(out_base_name + '.depth', parent_dir)
        seq_data = preprocessing.merge_numpy_batch_files(out_base_name + '.data', parent_dir)
    else:
        logging.debug('load data from file: ' + out_filename + '.data ...')
        seq_data = np.load(out_filename + '.data')

    logging.info('data size: '+str(len(seq_data)))

    if not os.path.isfile(out_filename + '.converter') \
            or not os.path.isfile(out_filename + '.new_idx_unknown')\
            or not os.path.isfile(out_filename + '.lex_size'):
        if not parser:
            logging.info('load spacy ...')
            parser = spacy.load('en')
            parser.pipeline = [parser.tagger, parser.entity, parser.parser]
        vecs, types = create_or_read_dict(out_filename, parser.vocab)
        # sort and filter vecs/mappings by counts
        converter, vecs, types, counts, new_idx_unknown = sort_and_cut_and_fill_dict(seq_data, vecs, types,
                                                                                     count_threshold=count_threshold)
        # write out vecs, mapping and tsv containing strings
        write_dict(out_filename, vecs=vecs, types=types, counts=counts)
        logging.debug('dump converter to: ' + out_filename + '.converter ...')
        converter.dump(out_filename + '.converter')
        logging.debug('dump new_idx_unknown to: ' + out_filename + '.new_idx_unknown ...')
        np.array(new_idx_unknown).dump(out_filename + '.new_idx_unknown')
        logging.debug('dump lex_size to: ' + out_filename + '.lex_size ...')
        np.array(len(types)).dump(out_filename + '.lex_size')

    if os.path.isfile(out_filename + '.converter') and os.path.isfile(out_filename + '.new_idx_unknown'):
        logging.debug('load converter from file: ' + out_filename + '.converter ...')
        converter = np.load(out_filename + '.converter')
        logging.debug('load new_idx_unknown from file: ' + out_filename + '.new_idx_unknown ...')
        new_idx_unknown = np.load(out_filename + '.new_idx_unknown')
        logging.debug('load lex_size from file: ' + out_filename + '.lex_size ...')
        lex_size = np.load(out_filename + '.lex_size')
        logging.debug('lex_size=' + str(lex_size))
        logging.info('convert data ...')
        count_unknown = 0
        for i, d in enumerate(seq_data):
            new_idx = converter[d]
            if 0 <= new_idx < lex_size:
                seq_data[i] = new_idx
            # set to UNKNOWN
            else:
                seq_data[i] = new_idx_unknown  # 0 #new_idx_unknown #mapping[constants.UNKNOWN_EMBEDDING]
                count_unknown += 1
        logging.info('set ' + str(count_unknown) + ' data points to UNKNOWN')

        logging.debug('dump data to: ' + out_filename + '.data ...')
        seq_data.dump(out_filename + '.data')
        logging.debug('delete converter, new_idx_unknown and lex_size ...')
        os.remove(out_filename + '.converter')
        os.remove(out_filename + '.new_idx_unknown')
        os.remove(out_filename + '.lex_size')

    logging.debug('load depths from file: ' + out_filename + '.depth ...')
    seq_depths = np.load(out_filename + '.depth')
    preprocessing.calc_depths_collected(out_filename, parent_dir, max_depth, seq_depths)

    logging.debug('load parents from file: ' + out_filename + '.parent ...')
    seq_parents = np.load(out_filename + '.parent')
    logging.debug('collect roots ...')
    roots = []
    for i, parent in enumerate(seq_parents):
        if parent == 0:
            roots.append(i)
    logging.debug('dump roots to: ' + out_filename + '.root ...')
    np.array(roots, dtype=np.int32).dump(out_filename + '.root')

    # preprocessing.rearrange_children_indices(out_filename, parent_dir, max_depth, max_articles, batch_size)
    # preprocessing.concat_children_indices(out_filename, parent_dir, max_depth)

    # logging.info('load and concatenate child indices batches ...')
    # for current_depth in range(1, max_depth + 1):
    #    if not os.path.isfile(out_filename + '.children.depth' + str(current_depth)):
    #        preprocessing.merge_numpy_batch_files(out_base_name + '.children.depth' + str(current_depth), parent_dir)

    # for re-usage
    return parser


def parse_texts(out_filename, in_filename, reader, parser, sentence_processor, max_articles, batch_size, concat_mode,
                inner_concat_mode, article_offset, add_reader_args={}):
    types = read_types(out_filename)
    mapping = mapping_from_list(types)
    logging.info('parse articles ...')
    for offset in range(0, max_articles, batch_size):
        # all or none: otherwise the mapping lacks entries!
        if not os.path.isfile(out_filename + '.data.batch' + str(offset)) \
                or not os.path.isfile(out_filename + '.parent.batch' + str(offset)) \
                or not os.path.isfile(out_filename + '.depth.batch' + str(offset)):
            logging.info('parse articles for offset=' + str(offset) + ' ...')
            current_reader_args = {
                    'filename': in_filename,
                    'max_articles': min(batch_size, max_articles),
                    'skip': offset + article_offset
                }
            current_reader_args.update(add_reader_args)
            current_seq_data, current_seq_parents, current_seq_depths = preprocessing.read_data(
                reader=reader,
                sentence_processor=sentence_processor,
                parser=parser,
                data_maps=mapping,
                args=current_reader_args,
                # max_depth=max_depth,
                batch_size=batch_size,
                concat_mode=concat_mode,
                inner_concat_mode=inner_concat_mode,
                calc_depths=True,
                # child_idx_offset=child_idx_offset
            )
            write_dict(out_filename, types=revert_mapping_to_list(mapping))
            logging.info('dump data, parents and depths ...')
            current_seq_data.dump(out_filename + '.data.batch' + str(offset))
            current_seq_parents.dump(out_filename + '.parent.batch' + str(offset))
            current_seq_depths.dump(out_filename + '.depth.batch' + str(offset))