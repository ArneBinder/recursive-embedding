import io
import json
import logging
import os
from functools import partial

import numpy as np
from datetime import datetime

from scipy.sparse import csr_matrix, coo_matrix
import plac

from constants import LOGGING_FORMAT, TYPE_RELATION, TYPE_DATASET, SEPARATOR, TYPE_POS_TAG, TYPE_LEXEME, \
    TYPE_DEPENDENCY_RELATION, TYPE_NAMED_ENTITY, TYPE_SENTENCE, TYPE_CONTEXT, PREFIX_TACRED, JSONLD_TYPE, JSONLD_ID, \
    TACRED_SUBJECT, TACRED_OBJECT, TACRED_RELATION, RDF_PREFIXES_MAP
from corpus import process_records, merge_batches, create_index_files, save_class_ids, DIR_BATCHES, \
    annotate_file_w_stanford, KEY_STANFORD_POS, KEY_STANFORD_TOKENS, KEY_STANFORD_DEPREL, KEY_STANFORD_RELATION, \
    KEY_ID, KEY_STANFORD_HEAD
from mytools import make_parent_dir
from corpus_rdf import parse_to_rdf

logger = logging.getLogger('corpus_tacred')
logger.setLevel(logging.DEBUG)
logger_streamhandler = logging.StreamHandler()
logger_streamhandler.setLevel(logging.DEBUG)
logger_streamhandler.setFormatter(logging.Formatter(LOGGING_FORMAT))
logger.addHandler(logger_streamhandler)

KEY_ENTITIES = "entities"

KEY_PREFIX_MAPPING = {
    KEY_STANFORD_POS: TYPE_POS_TAG,
    KEY_STANFORD_TOKENS: TYPE_LEXEME,
    KEY_STANFORD_DEPREL: TYPE_DEPENDENCY_RELATION,
    KEY_STANFORD_RELATION: TYPE_RELATION,
    KEY_ENTITIES: TYPE_NAMED_ENTITY
}

RELATION_NA = 'no_relation'

DUMMY_RECORD = {
    KEY_STANFORD_POS: ["IN", "DT", "JJ", "NN", ",", "NNP", "NNP", "NNP", "NNP", "NNP", "MD", "VB", "NN", ",", "VBG", "NNP", "NNP", "WP", "VBZ", "VBG", "TO", "VB", "DT", "NN", "NN", "."],
    KEY_STANFORD_HEAD: [4, 4, 4, 12, 12, 10, 10, 10, 10, 12, 12, 0, 12, 12, 12, 17, 15, 20, 20, 17, 22, 20, 25, 25, 22, 12],
    KEY_STANFORD_RELATION: "per:title",
    KEY_STANFORD_TOKENS: ["At", "the", "same", "time", ",", "Chief", "Financial", "Officer", "Douglas", "Flint", "will", "become", "chairman", ",", "succeeding", "Stephen", "Green", "who", "is", "leaving", "to", "take", "a", "government", "job", "."],
    KEY_ENTITIES: [[8, 10], [12, 13]],
    "address": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26],
    KEY_STANFORD_DEPREL: ["case", "det", "amod", "nmod", "punct", "compound", "compound", "compound", "compound", "nsubj", "aux", "ROOT", "xcomp", "punct", "xcomp", "compound", "dobj", "nsubj", "aux", "acl:relcl", "mark", "xcomp", "det", "compound", "dobj", "punct"],
    KEY_ID: "e7798fb926b9403cfcd2"
}


def reader_rdf(base_path, file_name):
    with io.open(os.path.join(base_path, file_name), encoding='utf8') as f:
        loaded = json.load(f)
    n = 0
    for record_loaded in loaded:
        record_id = RDF_PREFIXES_MAP[PREFIX_TACRED] + u'%s/%s' % (file_name, record_loaded['id'])
        token_annotations = [{JSONLD_ID: record_id + u'#r1',
                              JSONLD_TYPE: [u'%s=%s' % (TACRED_RELATION, record_loaded['relation'])],
                              TACRED_SUBJECT: [{JSONLD_ID: record_id + u'#s1_%i' % idx} for idx
                                               in range(record_loaded['subj_start'] + 1, record_loaded['subj_end'] + 2)],
                              TACRED_OBJECT: [{JSONLD_ID: record_id + u'#s1_%i' % idx} for idx
                                              in range(record_loaded['obj_start'] + 1, record_loaded['obj_end'] + 2)],
                              }]
        token_features = {k: record_loaded[k] for k in ['token', 'stanford_pos', 'stanford_ner', 'stanford_deprel', 'stanford_head']}
        record = {'record_id': record_id,
                  'token_features': token_features,
                  'token_annotations': token_annotations
                 }
        yield record
        n += 1
    logger.info('read %i records from %s' % (n, file_name))


def reader(records, key_main=KEY_STANFORD_TOKENS, key_rel=KEY_STANFORD_DEPREL, keys_annot=(KEY_STANFORD_POS,),
           root_string=TYPE_DATASET + SEPARATOR + 'TACRED', key_entity=KEY_ENTITIES
           ):
    nbr_total = 0
    nbr_failed = 0
    keys_check = set(keys_annot)
    keys_check.add(KEY_STANFORD_HEAD)
    if key_rel is not None:
        keys_check.add(key_rel)

    dep_edge_offset = 0 if key_rel is None else 1

    for r in records:
        try:
            nbr_total += 1
            for k in keys_check:
                assert len(r[key_main]) == len(r[k]), \
                    'number of %s: %i != number of %s: %i (dif: %i)'\
                    % (key_main, len(r[key_main]), k, len(r[k]), len(r[k]) - len(r[key_main]))
            entities = r[key_entity]
            entities_end = {e[1]-1: e for e in entities}
            data_strings = [root_string, root_string + SEPARATOR + r[KEY_ID], TYPE_CONTEXT]
            edges = [(0, 1), (0, 2)]
            start_positions = []
            entity_positions = []
            root = None

            for i, entry in enumerate(r[key_main]):
                start_pos = len(data_strings)
                start_positions.append(start_pos)
                data_strings.append(KEY_PREFIX_MAPPING[key_main] + SEPARATOR + entry)
                if key_rel is not None:
                    edges.append((len(data_strings), len(data_strings) - 1))
                    data_strings.append(KEY_PREFIX_MAPPING[key_rel] + SEPARATOR + r[key_rel][i])
                for j, k_annot in enumerate(keys_annot):
                    edges.append((start_pos, len(data_strings)))
                    data_strings.append(KEY_PREFIX_MAPPING[k_annot] + SEPARATOR + r[k_annot][i])
                if i in entities_end:
                    entity_positions.append(len(data_strings))
                    for j in range(entities_end[i][0], entities_end[i][1]):
                        edges.append((start_positions[j], len(data_strings)))
                    data_strings.append(KEY_PREFIX_MAPPING[key_entity])
            for i, head in enumerate(r[KEY_STANFORD_HEAD]):
                if head != 0:
                    edges.append((start_positions[head - 1], start_positions[i] + dep_edge_offset))
                else:
                    root = start_positions[i] + dep_edge_offset

            assert root is not None, 'ROOT not found'
            edges.append((2, len(data_strings)))
            edges.append((len(data_strings), root))
            data_strings.append(TYPE_SENTENCE)

            edges.append((entity_positions[0], len(data_strings)))
            edges.append((len(data_strings), entity_positions[1]))
            data_strings.append(KEY_PREFIX_MAPPING[KEY_STANFORD_RELATION] + SEPARATOR + r[KEY_STANFORD_RELATION])

            rows_and_cols = np.array(edges).T
            graph_out = csr_matrix(coo_matrix((np.ones(len(edges), dtype=bool), (rows_and_cols[1], rows_and_cols[0]))))
            #_graph_out = graph_out.toarray()
            yield data_strings, None, graph_out
        except Exception as e:
            nbr_failed += 1
            logger.warning('ID:%s %s' % (r[KEY_ID], str(e)))
    logger.info('successfully read %i records (%i records failed)' % (nbr_total - nbr_failed, nbr_failed))


@plac.annotations(
    out_base_name=('corpora output base file name', 'option', 'o', str)
)
def parse_dummy(out_base_name):
    print(out_base_name)
    make_parent_dir(out_base_name)
    process_records(records=[DUMMY_RECORD], out_base_name=out_base_name, record_reader=reader, concat_mode=None,
                    as_graph=True, dont_parse=True)


@plac.annotations(
    in_path=('corpora input folder', 'option', 'i', str),
    out_path=('corpora output folder', 'option', 'o', str),
    sentence_processor=('sentence processor', 'option', 'p', str),
    dataset_id=('id or name of teh dataset', 'option', 'd', str),
    unused='not used parameters'
)
def parse(in_path, out_path, sentence_processor=None, dataset_id='TACRED', *unused):
    if sentence_processor is None or sentence_processor.strip() == '':
        sentence_processor = 'process_sentence1'
    # default to process_sentence1
    if sentence_processor.strip() == 'process_sentence1':
        record_reader = partial(reader, key_rel=None, keys_annot=())
    elif sentence_processor.strip() == 'process_sentence3':
        record_reader = partial(reader, key_rel=None, keys_annot=(KEY_STANFORD_DEPREL,))
    elif sentence_processor.strip() == 'process_sentence11':
        record_reader = partial(reader, key_rel=KEY_STANFORD_DEPREL, keys_annot=())
    elif sentence_processor.strip() == 'process_sentence12':
        record_reader = partial(reader, key_rel=KEY_STANFORD_DEPREL, keys_annot=(KEY_STANFORD_POS,))
    else:
        raise NotImplementedError('sentence_processor: %s not implemented for parsing tacred' % sentence_processor)
    logger.info('use %s' % sentence_processor)
    file_names = ['dev', 'test', 'train']
    for fn in file_names:
        logger.info('create forest for %s ...' % fn)
        if not os.path.exists(os.path.join(in_path, fn + '.jsonl')):
            logger.warning('could not find file: %s. skip it.' % os.path.join(in_path, fn + '.jsonl'))
            continue
        out_base_name = os.path.join(out_path, DIR_BATCHES, fn.split('/')[0])
        make_parent_dir(out_base_name)
        process_records(records=(json.loads(s) for s in open(os.path.join(in_path, fn + '.jsonl')).readlines()),
                        out_base_name=out_base_name,
                        record_reader=partial(record_reader, root_string=TYPE_DATASET + SEPARATOR + dataset_id),
                        concat_mode=None, as_graph=True, dont_parse=True)
    logger.info('done.')


def record_to_opennre_format(record, relation_na=RELATION_NA):
    annots = record[KEY_ENTITIES]
    head_words = ' '.join(record[KEY_STANFORD_TOKENS][annots[0][0]:annots[0][1]])
    tail_words = ' '.join(record[KEY_STANFORD_TOKENS][annots[1][0]:annots[1][1]])
    id = record[KEY_ID]
    res = {
        'sentence': ' '.join(record[KEY_STANFORD_TOKENS]),
        'head': {'word': head_words, 'id': '%s/%i/%i' % (id, annots[0][0], annots[0][1])},
        'tail': {'word': tail_words, 'id': '%s/%i/%i' % (id, annots[1][0], annots[1][1])},
        'relation': record[KEY_STANFORD_RELATION] if record[KEY_STANFORD_RELATION] != relation_na else 'NA',
        'id': id
    }
    return res


@plac.annotations(
    in_path=('corpora input folder', 'option', 'i', str),
    out_path=('corpora output folder', 'option', 'o', str),
    relation_na=('use this relation as N/A', 'option', 'n', str),
)
def convert_to_opennre_format(in_path, out_path, relation_na=RELATION_NA):
    file_names = {'train.jsonl': 'train.jsonl',
                  'test.jsonl': 'test.jsonl'}

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    relations_set = set()
    for fn in file_names:
        logger.info('process: %s ...' % fn)
        if os.path.exists(os.path.join(out_path, file_names[fn])):
            logger.info('already precessed. skip it.')
            continue
        records_converted = []
        with open(os.path.join(in_path, fn)) as f:
            for line in f.readlines():
                record = json.loads(line)
                record_converted = record_to_opennre_format(record, relation_na=relation_na)
                relations_set.add(record_converted['relation'])
                records_converted.append(record_converted)
        json.dump(records_converted, open(os.path.join(out_path, file_names[fn]), 'w'), indent=2)

    rel2id = {r: i for i, r in enumerate(['NA'] + sorted([r for r in relations_set if r != 'NA']))}
    json.dump(rel2id, open(os.path.join(out_path, 'rel2id.json'), 'w'), indent=2)


@plac.annotations(
    in_path=('corpora input folder', 'option', 'i', str),
    out_path=('corpora output folder', 'option', 'o', str),
)
def parse_rdf(in_path, out_path):
    file_names = {'data/json/train.json': 'train.jsonl',
                  'data/json/dev.json': 'dev.jsonl',
                  'data/json/test.json': 'test.jsonl'}
    parse_to_rdf(in_path=in_path, out_path=out_path, reader_rdf=reader_rdf, parser=None, file_names=file_names)


@plac.annotations(
    mode=('processing mode', 'positional', None, str, ['PARSE', 'PARSE_DUMMY', 'MERGE', 'CREATE_INDICES', 'ALL',
                                                       'ANNOTATE', 'CONVERT_OPENNRE', 'PARSE_RDF']),
    args='the parameters for the underlying processing method')
def main(mode, *args):
    if mode == 'PARSE_DUMMY':
        plac.call(parse_dummy, args)
    elif mode == 'PARSE':
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
    elif mode == 'ANNOTATE':
        plac.call(annotate_file_w_stanford, args)
    elif mode == 'CONVERT_OPENNRE':
        plac.call(convert_to_opennre_format, args)
    elif mode == 'PARSE_RDF':
        plac.call(parse_rdf, args)
    else:
        raise ValueError('unknown mode')


if __name__ == '__main__':
    plac.call(main)
    logger.info('done')
