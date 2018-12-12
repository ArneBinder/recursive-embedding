import io
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
    assert len(indices) == len(c), \
        'number of indices [%i] does not match number of counts [%i]' % (len(indices), len(c))
    return indices, c


def lowest_common_ancestor(indices_start, heads, idx_root):
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


def get_argument_root(pos, length, heads, idx_root):
    pos_list = [pos + j for j in range(length)]
    if len(pos_list) == 1:
        return pos_list[0]
    lca = lowest_common_ancestor(indices_start=pos_list, heads=heads, idx_root=idx_root)
    return lca


def construct_batch(in_path, out_path, fn, lexicon, id2data, id_prefix, root_hash, context_hash, entity_hash,
                    sentence_hash, annotation_keys, target_offset=0, head_root=-1, discard_relations=False):
    # load: _word, _len, _pos1, _pos2, _label, _stanford_head,
    # and if available: _stanford_deprel, _stanford_pos

    if discard_relations:
        logger.warning('discard relations')

    relation_mention_id = numpy_load(join(in_path, fn + '_relation_mention_id'))

    data = {}
    data['word'] = numpy_load(join(in_path, fn + '_word'))
    data['label'] = numpy_load(join(in_path, fn + '_label'))

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
    # TODO: check error when including that for NYT
    if not discard_relations:
        pos1, len1 = distances_to_pos_and_length(numpy_load(join(in_path, fn + '_pos1')), max_len)
        pos2, len2 = distances_to_pos_and_length(numpy_load(join(in_path, fn + '_pos2')), max_len)
    head = numpy_load(join(in_path, fn + '_stanford_head'))

    # construct the data and graph
    data_keys = ['word'] + annotation_keys
    new_data_list = []
    new_graph_list = []
    lexicon_root_data = Lexicon()
    skipped = 0
    LEN_META = 3 # root_hash, id_hash, context_hash
    for i in range(len(data_converted['word'])):
        added = 0
        graph_tuples = []
        try:
            l = length[i]
            heads = head[i][:l]
            if not discard_relations:
                indices_root = np.nonzero(heads == head_root)[0]
                assert len(indices_root) == 1, 'number of roots has to be 1, but is %i' % len(indices_root)
                try:
                    arg1_root = get_argument_root(pos=pos1[i], length=len1[i], heads=heads, idx_root=indices_root[0])
                    arg2_root = get_argument_root(pos=pos2[i], length=len2[i], heads=heads, idx_root=indices_root[0])
                except IndexError as e:
                    raise AssertionError('could not get argument root: %s' % (str(e)))
                #if not (pos1[i] <= arg1_root < pos1[i] + len1[i]):
                #    logger.warning('ID:%s (#%i) arg1_root (%i) outside entity span [%i:%i]'
                #                   % (relation_mention_id[i], i, arg1_root, pos1[i], pos1[i] + len1[i]))
                #if not (pos2[i] <= arg2_root < pos2[i] + len2[i]):
                #    logger.warning('ID:%s (#%i) arg2_root (%i) outside entity span [%i:%i]'
                #                   % (relation_mention_id[i], i, arg2_root, pos2[i], pos2[i] + len2[i]))
                assert arg1_root is not None, 'could not get arg1_root (None)'
                assert arg2_root is not None, 'could not get arg2_root (None)'
                try:
                    assert arg1_root != arg2_root, 'arg1_root == arg2_root (%i)' % arg1_root
                except TypeError as e:
                    raise e
                pos_arg1 = arg1_root * len(data_keys) + LEN_META
                pos_arg2 = arg2_root * len(data_keys) + LEN_META

            id_string = id_prefix + SEPARATOR + fn + SEPARATOR + relation_mention_id[i]
            # it is ok to already add the hash to lexicon_root_data because it gets cleaned later via root_data
            id_hash = lexicon_root_data.add_and_get(id_string, data_as_hashes=True)

            # meta data
            _new_data = np.array([root_hash, id_hash, context_hash], dtype=DTYPE_HASH)
            new_data_list.append(_new_data)
            added += len(_new_data)
            graph_tuples.extend([(0, 1), (0, 2)])

            # real data (per token)
            _new_data = np.stack([data_converted[k][i][:l] for k in data_keys], axis=-1).flatten()
            new_data_list.append(_new_data)
            added += len(_new_data)
            root = None
            for i2, h in enumerate(heads):
                pos_self = i2 * len(data_keys) + LEN_META
                # head? connect with context node
                if h == head_root:
                    root = pos_self + target_offset
                else:
                    assert h < len(heads), 'head points outside'
                    pos_head = h * len(data_keys) + LEN_META
                    graph_tuples.append((pos_head, pos_self + target_offset))
                graph_tuples.extend([(pos_self, pos_self + i3) if i3 != target_offset else (pos_self + i3, pos_self) for i3 in range(1, len(data_keys))])
            assert root is not None, 'root not found'

            # remaining data (entities, relation)
            graph_tuples.append((2, added))
            graph_tuples.append((added, root))
            _new_dat_list = [sentence_hash]
            if not discard_relations:
                _new_dat_list.extend([entity_hash, entity_hash, data_converted['label'][i]])
                graph_tuples.append((pos_arg1, added + 1))
                graph_tuples.append((pos_arg2, added + 2))
                graph_tuples.append((pos_arg1, added + 3))
                graph_tuples.append((added + 3, pos_arg2))
            _new_data = np.array(_new_dat_list, dtype=DTYPE_HASH)
            added += len(_new_data)

            row, col = zip(*graph_tuples)
            current_graph = coo_matrix((np.ones(len(graph_tuples), dtype=bool), (row, col))).transpose().tocsc()
            new_data_list.append(_new_data)
            new_graph_list.append(current_graph)
        except AssertionError as e:
            logger.warning('ID:%s (#%i) %s. Skip record.' % (relation_mention_id[i], i, str(e)))
            skipped += 1

    logger.info('skipped %i of %i records' % (skipped, len(data_converted['word'])))
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
    annotations=('comma separated list of annotation keys', 'option', 'a', str),
    target_offset=('offset to link descendant elements to', 'option', 't', int),
    discard_relations=('do not use relation data', 'option', 'r', str),
    unused='not used parameters'
)
def parse(in_path, out_path, sentence_processor=None, dataset_id='OPENNRE', dump_batches=False,
          annotations='stanford_deprel,stanford_pos', target_offset=0, discard_relations='False', *unused):
    discard_relations = discard_relations.lower().strip() == 'true'
    out_path_merged = join(out_path, DIR_MERGED)
    if not os.path.exists(out_path_merged):
        os.makedirs(out_path_merged)
    logger_fh = logging.FileHandler(os.path.join(out_path, 'corpus-parse-merge.log'))
    logger_fh.setLevel(logging.DEBUG)
    logger_fh.setFormatter(logging.Formatter(LOGGING_FORMAT))
    logger.addHandler(logger_fh)

    if sentence_processor is None or sentence_processor.strip() == '':
        sentence_processor = 'process_sentence1'
    # default to process_sentence1
    if sentence_processor.strip() == 'process_sentence1':
        annotations = ''
        target_offset = 0
    elif sentence_processor.strip() == 'process_sentence3':
        annotations = 'stanford_deprel'
        target_offset = 0
    elif sentence_processor.strip() == 'process_sentence5':
        annotations = 'stanford_deprel,stanford_pos'
        target_offset = 0
    elif sentence_processor.strip() == 'process_sentence11':
        annotations = 'stanford_deprel'
        target_offset = 1
    elif sentence_processor.strip() == 'process_sentence12':
        annotations = 'stanford_deprel,stanford_pos'
        target_offset = 1
    else:
        raise NotImplementedError('sentence_processor: %s not implemented for parsing opennre' % sentence_processor)
    logger.info('use %s' % sentence_processor)
    _config = json.load(open(join(in_path, 'config')))
    lexicon = Lexicon(add_vocab_manual=True)
    data2id = {'word': _config['word2id'], 'label': _config['relation2id']}
    annotation_keys = [annot.strip() for annot in annotations.split(',') if annot.strip() != '']
    for annot in annotation_keys:
        data2id[annot.strip()] = _config['annotations2id'][annot.strip()]
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
        logger.info('create data for %s ...' % fn)
        try:
            current_data_graph = construct_batch(in_path, join(out_path, DIR_BATCHES) if dump_batches else None,
                                                 fn, lexicon, id2data=id2data, annotation_keys=annotation_keys,
                                                 target_offset=target_offset,
                                                 id_prefix=root_string, root_hash=root_hash, context_hash=context_hash,
                                                 entity_hash=entity_hash, sentence_hash=sentence_hash,
                                                 discard_relations=discard_relations)
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
    mode=('processing mode', 'positional', None, str, ['PARSE', 'CREATE_INDICES', 'ALL_SEMEVAL', 'ALL_TACRED',
                                                       'ANNOTATE', 'CONVERT_NYT']),
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
    elif mode == 'ALL_SEMEVAL':
        out_path_merged = plac.call(main, ('PARSE',) + args)
        #plac.call(main, ('MERGE',) + args)
        plac.call(main, ('CREATE_INDICES', '--end-root', '2714', '--split-count', '1', '--suffix', 'test', '--merged-forest-path', out_path_merged) + args)
        plac.call(main, ('CREATE_INDICES', '--start-root', '2714', '--split-count', '4', '--suffix', 'train', '--merged-forest-path', out_path_merged) + args)
    elif mode == 'ALL_TACRED':
        out_path_merged = plac.call(main, ('PARSE',) + args)
        #plac.call(main, ('MERGE',) + args)
        plac.call(main, ('CREATE_INDICES', '--end-root', '22461', '--split-count', '1', '--suffix', 'dev', '--merged-forest-path', out_path_merged) + args)
        plac.call(main, ('CREATE_INDICES', '--start-root', '22461', '--end-root', str(22461 + 15426), '--split-count', '1', '--suffix', 'test', '--merged-forest-path', out_path_merged) + args)
        plac.call(main, ('CREATE_INDICES', '--start-root', str(22461 + 15426), '--split-count', '4', '--suffix', 'train', '--merged-forest-path', out_path_merged) + args)
    elif mode == 'ANNOTATE':
        plac.call(annotate_file_w_stanford, args)
    elif mode == 'CONVERT_NYT':
        plac.call(convert_nyt, args)
    else:
        raise ValueError('unknown mode')


def find_sub_list(sl, l):
    sll = len(sl)
    for ind in (i for i, e in enumerate(l) if e == sl[0]):
        if l[ind:ind+sll] == sl:
            return ind, sll
    return None


def convert_nyt_record(id, record):
    # tokens, entities, id, label
    tokens = record['sentence'].split()
    tokens_head = record['head']['word'].split()
    tokens_tail = record['tail']['word'].split()
    entitiy_head = find_sub_list(tokens_head, tokens)
    assert entitiy_head is not None, '%s not found in tokens: %s' % (str(tokens_head), str(tokens))
    entity_tail = find_sub_list(tokens_tail, tokens)
    assert entity_tail is not None, '%s not found in tokens: %s' % (str(tokens_head), str(tokens))
    res = {
        'id': id,
        'tokens': tokens,
        'entities': [entitiy_head, entity_tail],
        'label': record['relation'],
        'entities_types': [record['head']['type'].split(','), record['head']['type'].split(',')]
    }
    return res


@plac.annotations(
    in_path=('corpora input folder', 'option', 'i', str),
    server_url=('stanford CoreNLP server url', 'option', 'u', str),
)
def convert_nyt(in_path, server_url='http://localhost:9000'):
    out_path = join(in_path, 'annotated')
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    file_names = ['test', 'dev', 'train']
    #file_names = ['test-reading-friendly-1000']
    for fn in file_names:
        _fn = join(in_path, fn)
        _fn_in = '%s.json' % _fn
        _fn_jl = '%s.jsonl' % _fn
        if not os.path.exists(_fn_jl):
            if not os.path.exists(_fn_in):
                logger.warning('%s does not exist. Skip it.' % _fn_in)
                continue
            logger.info('create %s ...' % _fn_jl)
            records = json.load(open(_fn_in))
            records_converted = [convert_nyt_record('%s/%i' % (fn, i), r) for i, r in enumerate(records)]
            io.open(_fn_jl, 'w', encoding='utf8').writelines((json.dumps(r, ensure_ascii=False) + u'\n' for r in records_converted))
            logger.info('%s written' % _fn_jl)
        if not os.path.exists(_fn_jl):
            logger.warning('%s does not exist. Skipt it.' % _fn_jl)
            continue
        out_fn = join(out_path, '%s.jsonl' % fn)
        if os.path.exists(out_fn):
            logger.info('annotated file %s already exists' % out_fn)
        else:
            annotate_file_w_stanford(fn_in=_fn_jl, fn_out=out_fn, server_url=server_url)


if __name__ == '__main__':
    plac.call(main)
    logger.info('done')