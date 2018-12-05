import json
import logging
import numpy as np
import os
from os.path import join

import plac
from scipy.sparse import coo_matrix

from constants import TYPE_RELATION, LOGGING_FORMAT, TYPE_DATASET, SEPARATOR, TYPE_LEXEME, TYPE_POS_TAG, \
    TYPE_DEPENDENCY_RELATION, DTYPE_HASH, TYPE_CONTEXT, TYPE_NAMED_ENTITY, TYPE_SENTENCE, DTYPE_VECS
from corpus import save_class_ids, create_index_files, DIR_BATCHES, DIR_MERGED, \
    annotate_file_w_stanford
from lexicon import Lexicon
from mytools import make_parent_dir, numpy_load
from sequence_trees import Forest, concatenate_structures

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


def lowest_common_ancestor(indices_start, heads, head_root=-1):

    idx_root = np.nonzero(heads == head_root)[0]
    indices_to_root = [[]] * len(indices_start)
    for i, idx_start in enumerate(indices_start):
        idx = idx_start
        while idx != idx_root:
            indices_to_root[i].append(idx)
            idx = heads[idx]

    idx = idx_root
    for i in range(2, min([len(l) for l in indices_to_root])):
        if len(set(map(lambda x: x[-i], indices_to_root))) == 1:
            idx = indices_to_root[0][-i]
        else:
            break
    return idx


def get_argument_root(pos, length, heads, head_root=-1):
    pos_list = [pos + j for j in range(length)]
    if len(pos_list) == 1:
        return pos_list[0]
    lca = lowest_common_ancestor(indices_start=pos_list, heads=heads, head_root=head_root)
    return lca


def construct_batch(in_path, out_path, fn, lexicon, data2id, id2data, id_prefix, root_hash, context_hash, entity_hash,
                    sentence_hash, head_root=-1):
    # load: _word, _len, _pos1, _pos2, _label, _stanford_head,
    # and if available: _stanford_deprel, _stanford_pos

    relation_mention_id = numpy_load(join(in_path, fn + '_relation_mention_id'))

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
    new_graph_list = []
    lexicon_root_data = Lexicon()
    id_hashes = []
    for i in range(len(data_converted['word'])):
        added = 0
        graph_tuples = []
        l = length[i]
        heads = head[i][:l]
        arg1_root = get_argument_root(pos=pos1[i], length=len1[i], heads=heads)
        arg2_root = get_argument_root(pos=pos2[i], length=len2[i], heads=heads)
        if not (pos1[i] <= arg1_root < pos1[i] + len1[i]):
            logger.warning('ID:%s arg1_root (%i) outside entity span [%i:%i]'
                           % (relation_mention_id[i], arg1_root, pos1[i], pos1[i] + len1[i]))
        if not (pos2[i] <= arg2_root < pos2[i] + len2[i]):
            logger.warning('ID:%s arg2_root (%i) outside entity span [%i:%i]'
                           % (relation_mention_id[i], arg2_root, pos2[i], pos2[i] + len2[i]))
        if arg1_root == arg2_root:
            logger.warning(
                'ID:%s arg1_root == arg2_root (%i). Skip it!' % (relation_mention_id[i], arg1_root))
            continue

        id_string = id_prefix + SEPARATOR + fn + SEPARATOR + relation_mention_id[i]
        id_hash = lexicon_root_data.add_and_get(id_string, data_as_hashes=True)
        id_hashes.append(id_hash)

        # meta data
        _new_data = np.array([root_hash, id_hash, context_hash], dtype=DTYPE_HASH)
        new_data_list.append(_new_data)
        added += len(_new_data)
        graph_tuples.extend([(0, 1), (0, 2)])
        len_meta = added

        pos_arg1 = arg1_root * len(data_keys) + len_meta
        pos_arg2 = arg2_root * len(data_keys) + len_meta
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
        graph_tuples.append((pos_arg1, added + 1))
        graph_tuples.append((pos_arg2, added + 2))
        graph_tuples.append((pos_arg1, added + 3))
        graph_tuples.append((added + 3, pos_arg2))
        new_data_list.append(_new_data)
        added += len(_new_data)

        row, col = zip(*graph_tuples)
        current_graph = coo_matrix((np.ones(len(graph_tuples), dtype=bool), (row, col))).transpose().tocsc()
        new_graph_list.append(current_graph)

    forest = Forest(data=np.concatenate(new_data_list), structure=concatenate_structures(new_graph_list),
                    lexicon_roots=lexicon_root_data,
                    data_as_hashes=True)
    forest.set_root_data_by_offset()
    lexicon_root_data.order_by_hashes(forest.root_data)
    forest.set_lexicon_roots(lexicon_root_data)
    if out_path is not None:
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        forest.dump(join(out_path, fn))
        lexicon.dump(join(out_path, fn), strings_only=True)
        lexicon_root_data.dump(join(out_path, '%s.root.id' % fn), strings_only=True)
    return forest


@plac.annotations(
    in_path=('corpora input folder', 'option', 'i', str),
    out_path=('corpora output folder', 'option', 'o', str),
    sentence_processor=('sentence processor', 'option', 'p', str),
    dataset_id=('id or name of the dataset', 'option', 'd', str),
    dump_batches=('dump batches', 'flag', 'b', bool),
    unused='not used parameters'
)
def parse(in_path, out_path, sentence_processor=None, dataset_id='OPENNRE', dump_batches=False, *unused):
    out_path_merged = join(out_path, DIR_MERGED)
    if not os.path.exists(out_path_merged):
        os.makedirs(out_path_merged)
    logger_fh = logging.FileHandler(os.path.join(out_path, 'corpus-parse-merge.log'))
    logger_fh.setLevel(logging.DEBUG)
    logger_fh.setFormatter(logging.Formatter(LOGGING_FORMAT))
    logger.addHandler(logger_fh)

    if sentence_processor is None or sentence_processor.strip() == '':
        sentence_processor = 'process_sentence1'
    #TODO: add other sentence_processors
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

    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    file_names = ['dev', 'test', 'train']
    data_graph_list = []
    for fn in file_names:
        logger.info('create forest for %s ...' % fn)
        try:
            current_data_graph = construct_batch(in_path, join(out_path, DIR_BATCHES) if dump_batches else None,
                                                 fn, lexicon, data2id, id2data,
                                                 id_prefix=root_string, root_hash=root_hash, context_hash=context_hash,
                                                 entity_hash=entity_hash, sentence_hash=sentence_hash)
            logger.info('created graph with %i components' % len(current_data_graph.roots))
            data_graph_list.append(current_data_graph)
        except IOError as e:
            logger.warning('%s. Skip "%s".' % (str(e), fn))
            continue

    logger.info('merge ...')
    data_graph = Forest.concatenate(data_graph_list)
    data_graph.set_lexicon(lexicon)
    logger.info('convert hashes to indices...')
    data_graph.hashes_to_indices()

    # add vecs to lexicon
    logger.info('add vecs ...')
    vec = numpy_load(join(in_path, 'vec'))
    strings = id2data['word']
    # init zero vecs
    lexicon.init_vecs(new_vecs=np.zeros(shape=[len(lexicon), vec.shape[-1]], dtype=DTYPE_VECS))
    # set vecs
    indices_strings = lexicon.set_to_vecs(vecs=vec, strings=strings, prefix=TYPE_LEXEME + SEPARATOR)
    # fix these vecs (except special entries)
    indices_strings_special = lexicon.get_indices(strings=["<START_TOKEN>", "<UNK_TOKEN>", "<PAD_TOKEN>"],
                                                  prefix=TYPE_LEXEME + SEPARATOR)
    indices_fix = indices_strings[~np.isin(indices_strings, indices_strings_special)]
    lexicon.init_ids_fixed(ids_fixed=indices_fix)

    out_path_merged_forest = join(out_path_merged, 'forest')
    data_graph.dump(out_path_merged_forest)
    lexicon.dump(out_path_merged_forest)
    data_graph.lexicon_roots.dump('%s.root.id' % out_path_merged_forest, strings_only=True)
    return data_graph, out_path_merged_forest


@plac.annotations(
    mode=('processing mode', 'positional', None, str, ['PARSE', 'CREATE_INDICES', 'ALL', 'ANNOTATE']),
    args='the parameters for the underlying processing method')
def main(mode, *args):
    if mode == 'PARSE':
        forest_merged, out_path_merged = plac.call(parse, args)
        #elif mode == 'MERGE':
        #forest_merged, out_path_merged = plac.call(merge_batches, args)
        relation_ids, relation_strings = forest_merged.lexicon.get_ids_for_prefix(TYPE_RELATION)
        save_class_ids(dir_path=out_path_merged, prefix_type=TYPE_RELATION, classes_ids=relation_ids,
                       classes_strings=relation_strings)
        return out_path_merged
    elif mode == 'CREATE_INDICES':
        plac.call(create_index_files, args)
    elif mode == 'ALL':
        out_path_merged = plac.call(main, ('PARSE',) + args)
        #plac.call(main, ('MERGE',) + args)
        plac.call(main, ('CREATE_INDICES', '--end-root', '2707', '--split-count', '1', '--suffix', 'test', '--merged-forest-path', out_path_merged) + args)
        plac.call(main, ('CREATE_INDICES', '--start-root', '2707', '--split-count', '4', '--suffix', 'train', '--merged-forest-path', out_path_merged) + args)
    elif mode == 'ANNOTATE':
        plac.call(annotate_file_w_stanford, args)
    else:
        raise ValueError('unknown mode')


if __name__ == '__main__':
    plac.call(main)
    logger.info('done')