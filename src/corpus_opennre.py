import json
import logging
import numpy as np
import os
from os.path import join
from functools import partial

import plac
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix

from src.constants import TYPE_RELATION, LOGGING_FORMAT, TYPE_DATASET, SEPARATOR, TYPE_LEXEME, TYPE_POS_TAG, \
    TYPE_DEPENDENCY_RELATION, DTYPE_HASH, TYPE_CONTEXT, TYPE_NAMED_ENTITY, DTYPE_OFFSET, TYPE_SENTENCE
from src.corpus import merge_batches, save_class_ids, create_index_files, DIR_BATCHES
from src.lexicon import Lexicon
from src.mytools import make_parent_dir, numpy_load
from src.sequence_trees import Forest, concatenate_structures

logger = logging.getLogger('corpus_opennre')
logger.setLevel(logging.DEBUG)
logger_streamhandler = logging.StreamHandler()
logger_streamhandler.setLevel(logging.DEBUG)
logger_streamhandler.setFormatter(logging.Formatter(LOGGING_FORMAT))
logger.addHandler(logger_streamhandler)


DATA_TO_TYPE = {
    'word': TYPE_LEXEME,
    'label': TYPE_RELATION,
    'stanford_pos': TYPE_POS_TAG,
    'stanford_deprel': TYPE_DEPENDENCY_RELATION
}


def distances_to_pos_and_length(distances, max_len):
    _pos = np.argwhere(distances == max_len)
    step = _pos[1:, 0] - _pos[:-1, 0]
    step = np.append([1], step)
    indices = _pos[step == 1][:, 1]
    u, c = np.unique(_pos[:, 0], return_counts=True)
    assert len(indices) == len(c), 'number of indices does not match number of counts'
    return indices, c


def construct_batch(in_path, out_path, fn, lexicon, data2id, id2data, id_prefix, root_hash, context_hash, entity_hash,
                    sentence_hash, head_root=-1):
    # load: _word, _len, _pos1, _pos2, _label, _stanford_head,
    # and if available: _stanford_deprel, _stanford_pos

    data = {}
    data['word'] = numpy_load(join(in_path, fn + '_word'))
    data['label'] = numpy_load(join(in_path, fn + '_label'))

    annotation_keys = list(set(data2id) - set(data))
    for annotation in annotation_keys:
        data[annotation] = numpy_load(join(in_path, '%s_%s' % (fn, annotation)))

    data_converted = {}
    for k in data:
        logger.info('convert %s ...' % k)
        data_flat = data[k].flatten()
        dc = [lexicon.get_d(DATA_TO_TYPE[k] + SEPARATOR + id2data[k][d], data_as_hashes=True) for d in data_flat]
        data_converted[k] = np.array(dc, dtype=DTYPE_HASH).reshape(data[k].shape)

    # structural data
    max_len = data['word'].shape[-1]
    length = numpy_load(join(in_path, fn + '_len'))
    pos1, len1 = distances_to_pos_and_length(numpy_load(join(in_path, fn + '_pos1')), max_len)
    pos2, len2 = distances_to_pos_and_length(numpy_load(join(in_path, fn + '_pos2')), max_len)
    head = numpy_load(join(in_path, fn + '_stanford_head'))

    # construct the data and graph
    data_keys = ['word'] + annotation_keys
    new_data_list = []
    new_parent_list = []
    new_graph_list = []
    for i in range(len(data_converted['word'])):
        added = 0
        graph_tuples = []
        id_string = id_prefix + SEPARATOR + fn + SEPARATOR + str(i)
        lexicon.add(id_string)
        id_hash = lexicon.get_d(id_string, data_as_hashes=True)
        l = length[i]
        heads = head[i][:l]

        # meta data
        _new_data = np.array([root_hash, id_hash, context_hash], dtype=DTYPE_HASH)
        new_data_list.append(_new_data)
        added += len(_new_data)
        graph_tuples.extend([(0, 1), (0, 2)])
        len_meta = added
        pos_pos1 = pos1[i] * len(data_keys) + len_meta
        pos_pos2 = pos2[i] * len(data_keys) + len_meta
        # real data (per token)
        _new_data = np.stack([data_converted[k][i][:l] for k in data_keys], axis=-1).flatten()
        new_data_list.append(_new_data)
        added += len(_new_data)
        root = None
        for i2, h in enumerate(heads):
            pos_self = i2 * len(data_keys) + len_meta
            # head? connect with context node
            if h == head_root:
                root = pos_self
            else:
                pos_head = h * len(data_keys) + len_meta
                graph_tuples.append((pos_head, pos_self))
            graph_tuples.extend([(pos_self, pos_self + i3 + 1) for i3, k in enumerate(data_keys[1:])])

        # remaining data (entities, relation)
        _new_data = np.array([sentence_hash, entity_hash, entity_hash, data_converted['label'][i]], dtype=DTYPE_HASH)
        graph_tuples.append((2, added))
        graph_tuples.append((added, root))
        graph_tuples.append((pos_pos1, added + 1))
        graph_tuples.append((pos_pos2, added + 2))
        graph_tuples.append((pos_pos1, added + 3))
        # TODO: check direction
        graph_tuples.append((added + 3, pos_pos2))
        new_data_list.append(_new_data)
        added += len(_new_data)

        row, col = zip(*graph_tuples)
        current_graph = coo_matrix((np.ones(len(graph_tuples), dtype=bool), (row, col))).transpose().tocsc()
        new_graph_list.append(current_graph)
        #_new_parents = np.ones(added, dtype=DTYPE_OFFSET) * -1
        #_new_parents[0] = 0
        #new_parent_list.append(_new_parents)

    forest = Forest(data=np.concatenate(new_data_list), structure=concatenate_structures(new_graph_list),
                    lexicon=lexicon, data_as_hashes=True)
    forest.set_root_data_by_offset()
    forest.dump(join(out_path, fn))
    lexicon.dump(join(out_path, fn), strings_only=True)
    return forest


@plac.annotations(
    in_path=('corpora input folder', 'option', 'i', str),
    out_path=('corpora output folder', 'option', 'o', str),
    sentence_processor=('sentence processor', 'option', 'p', str),
    dataset_id=('id or name of the dataset', 'option', 'd', str),
    unused='not used parameters'
)
def parse(in_path, out_path, sentence_processor=None, dataset_id='OPENNRE', *unused):
    if sentence_processor is None or sentence_processor.strip() == '':
        sentence_processor = 'process_sentence1'
    # default to process_sentence1
    #if sentence_processor.strip() == 'process_sentence1':
    #    record_reader = partial(reader, key_rel=None, keys_annot=())
    #elif sentence_processor.strip() == 'process_sentence3':
    #    record_reader = partial(reader, key_rel=None, keys_annot=(KEY_STANFORD_DEPREL,))
    #elif sentence_processor.strip() == 'process_sentence11':
    #    record_reader = partial(reader, key_rel=KEY_STANFORD_DEPREL, keys_annot=())
    #elif sentence_processor.strip() == 'process_sentence12':
    #    record_reader = partial(reader, key_rel=KEY_STANFORD_DEPREL, keys_annot=(KEY_STANFORD_POS,))
    #else:
    #    raise NotImplementedError('sentence_processor: %s not implemented for parsing tacred' % sentence_processor)
    logger.info('use %s' % sentence_processor)
    _config = json.load(open(join(in_path, 'config')))
    lexicon = Lexicon(add_vocab_manual=True)
    data2id = {'word': _config['word2id'], 'label': _config['relation2id']}
    data2id.update(_config['annotations2id'])
    id2data = {}
    for k in data2id:
        lexicon.add_all(data2id[k], prefix=DATA_TO_TYPE[k] + SEPARATOR)
        id2data[k] = [None] * len(data2id[k])
        for k2, v in data2id[k].items():
            id2data[k][v] = k2

    root_string = TYPE_DATASET + SEPARATOR + dataset_id
    root_hash = lexicon.add_and_get(root_string, data_as_hashes=True)
    entity_hash = lexicon.add_and_get(TYPE_NAMED_ENTITY, data_as_hashes=True)
    context_hash = lexicon.add_and_get(TYPE_CONTEXT, data_as_hashes=True)
    sentence_hash = lexicon.add_and_get(TYPE_SENTENCE, data_as_hashes=True)
    # TODO: add vecs to lexicon
    vec = numpy_load(join(in_path, 'vec'))
    file_names = ['dev', 'test', 'train']
    for fn in file_names:
        logger.info('create forest for %s ...' % fn)
        #if not os.path.exists(os.path.join(in_path, fn + '.jsonl')):
        #    logger.warning('could not find file: %s. skip it.' % os.path.join(in_path, fn + '.jsonl'))
        #    continue
        try:
            out_base_name = os.path.join(out_path, DIR_BATCHES, fn.split('/')[0])
            make_parent_dir(out_base_name)
            #process_records(records=(json.loads(s) for s in open(os.path.join(in_path, fn + '.jsonl')).readlines()),
            #                out_base_name=out_base_name,
            #                record_reader=partial(record_reader, root_string=TYPE_DATASET + SEPARATOR + dataset_id),
            #                concat_mode=None, as_graph=True, dont_parse=True)
            construct_batch(in_path, join(out_path, DIR_BATCHES), fn, lexicon, data2id, id2data, id_prefix=root_string,
                            root_hash=root_hash, context_hash=context_hash, entity_hash=entity_hash,
                            sentence_hash=sentence_hash)
        except IOError as e:
            logger.warning(e)
            continue
    logger.info('done.')


@plac.annotations(
    mode=('processing mode', 'positional', None, str, ['PARSE', 'PARSE_DUMMY', 'MERGE', 'CREATE_INDICES', 'ALL',
                                                       'ANNOTATE', 'CONVERT_OPENNRE']),
    args='the parameters for the underlying processing method')
def main(mode, *args):
    if mode == 'PARSE':
        plac.call(parse, args)
    elif mode == 'MERGE':
        forest_merged, out_path_merged = plac.call(merge_batches, args)
        relation_ids, relation_strings = forest_merged.lexicon.get_ids_for_prefix(TYPE_RELATION)
        save_class_ids(dir_path=out_path_merged, prefix_type=TYPE_RELATION, classes_ids=relation_ids,
                       classes_strings=relation_strings)
    elif mode == 'CREATE_INDICES':
        plac.call(create_index_files, args)
    elif mode == 'ALL':
        plac.call(main, ('PARSE',) + args)
        plac.call(main, ('MERGE',) + args)
        plac.call(main, ('CREATE_INDICES', '--end-root', '22584', '--split-count', '1', '--suffix', 'dev') + args)
        plac.call(main, ('CREATE_INDICES', '--start-root', '22584', '--end-root', '38041', '--split-count', '1',
                         '--suffix', 'test') + args)
        plac.call(main, ('CREATE_INDICES', '--start-root', '38041', '--split-count', '4', '--suffix', 'train') + args)
    #elif mode == 'ANNOTATE':
    #    plac.call(annotate_file_w_stanford, args)
    #elif mode == 'CONVERT_OPENNRE':
    #    plac.call(convert_to_opennre_format, args)
    else:
        raise ValueError('unknown mode')


if __name__ == '__main__':
    plac.call(main)
    logger.info('done')