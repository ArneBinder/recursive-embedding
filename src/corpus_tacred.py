import json
import logging
import os

import numpy as np
from datetime import datetime

from scipy.sparse import csr_matrix, coo_matrix
import plac

from constants import LOGGING_FORMAT, TYPE_RELATION, TYPE_DATASET, SEPARATOR, TYPE_POS_TAG, TYPE_LEXEME, \
    TYPE_DEPENDENCY_RELATION, TYPE_NAMED_ENTITY, TYPE_SENTENCE, TYPE_CONTEXT
from corpus import process_records, merge_batches, create_index_files, save_class_ids, DIR_BATCHES
from mytools import make_parent_dir

logger = logging.getLogger('corpus_tacred')
logger.setLevel(logging.DEBUG)
logger_streamhandler = logging.StreamHandler()
logger_streamhandler.setLevel(logging.DEBUG)
logger_streamhandler.setFormatter(logging.Formatter(LOGGING_FORMAT))
logger.addHandler(logger_streamhandler)

TYPE_TACRED = TYPE_DATASET + SEPARATOR + 'TACRED'

KEY_PREFIX_MAPPING = {
    "stanford_pos": TYPE_POS_TAG,
    "tokens": TYPE_LEXEME,
    "stanford_deprel": TYPE_DEPENDENCY_RELATION,
    "label": TYPE_RELATION,
    "id": TYPE_TACRED,
    "entities": TYPE_NAMED_ENTITY
}

DUMMY_RECORD = {
    "stanford_pos": ["IN", "DT", "JJ", "NN", ",", "NNP", "NNP", "NNP", "NNP", "NNP", "MD", "VB", "NN", ",", "VBG", "NNP", "NNP", "WP", "VBZ", "VBG", "TO", "VB", "DT", "NN", "NN", "."],
    "stanford_head": [4, 4, 4, 12, 12, 10, 10, 10, 10, 12, 12, 0, 12, 12, 12, 17, 15, 20, 20, 17, 22, 20, 25, 25, 22, 12],
    "label": "per:title",
    "tokens": ["At", "the", "same", "time", ",", "Chief", "Financial", "Officer", "Douglas", "Flint", "will", "become", "chairman", ",", "succeeding", "Stephen", "Green", "who", "is", "leaving", "to", "take", "a", "government", "job", "."],
    "entities": [[8, 10], [12, 13]],
    "address": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26],
    "stanford_deprel": ["case", "det", "amod", "nmod", "punct", "compound", "compound", "compound", "compound", "nsubj", "aux", "ROOT", "xcomp", "punct", "xcomp", "compound", "dobj", "nsubj", "aux", "acl:relcl", "mark", "xcomp", "det", "compound", "dobj", "punct"],
    "id": "e7798fb926b9403cfcd2"
}


def stanford_depgraph_to_dict(dgraph, k_map=None, types=None):
    res = {}
    for i, node in dgraph.nodes.items():
        # skip first (dummy) element
        if node['address'] == 0:
            continue
        for k, v in node.items():
            if (k in k_map or k_map is None) and (types is None or type(v) in types):
                res.setdefault(k_map[k], []).append(v)
    return res


def annotate_file_w_stanford(fn_in='/mnt/DATA/ML/data/corpora_in/tacred/tacred-jsonl/dev_10.jsonl',
                             fn_out='/mnt/DATA/ML/data/corpora_in/tacred/tacred-jsonl/annot_dev_10.jsonl',
                             server_url='http://localhost:9000'):
    from nltk.parse.corenlp import CoreNLPDependencyParser
    t_start = datetime.now()
    dep_parser = CoreNLPDependencyParser(url=server_url)
    print('process %s ...' % fn_in)
    with open(fn_in) as f_in:
        with open(fn_out, 'w') as f_out:
            for line in f_in.readlines():
                jsl = json.loads(line)
                parses = dep_parser.parse(jsl['tokens'])
                annots = None
                for parse in parses:
                    if annots is not None:
                        print('ID:%s\tfound two parses' % jsl['id'])
                        break
                    annots = stanford_depgraph_to_dict(parse, types=(int, unicode),
                                                       k_map={'tag': 'stanford_pos',
                                                              'head': 'stanford_head',
                                                              'rel': 'stanford_deprel',
                                                              'word': 'tokens_stanford',
                                                              'address': 'address'})
                assert annots is not None, 'found no parses'
                if jsl['tokens'] != annots['tokens_stanford']:
                    print('ID:%s\ttokens do not match after parsing' % jsl['id'])
                del annots['tokens_stanford']
                jsl.update(annots)
                f_out.write(json.dumps(jsl) + '\n')
    print('time: %s' % str(datetime.now() - t_start))


def reader(records, key_main="tokens", key_rel="stanford_deprel", keys_annot=("stanford_pos", ),
           root_string=TYPE_TACRED, key_entity="entities"
           #keys_meta=(TYPE_RELATION,), key_id=TYPE_SEMEVAL2010TASK8_ID,
           #root_text_string=TYPE_CONTEXT
           ):
    nbr_total = 0
    nbr_failed = 0
    for r in records:
        try:
            nbr_total += 1
            assert len(r[key_main]) == len(r[key_rel]), \
                'number of %s: %i != number of %s: %i (dif: %i)'\
                % (key_main, len(r[key_main]), key_rel, len(r[key_rel]), len(r[key_rel]) - len(r[key_main]))
            entities = r[key_entity]
            entities_end = {e[1]-1: e for e in entities}
            data_strings = [root_string, KEY_PREFIX_MAPPING['id'] + SEPARATOR + r['id'], TYPE_CONTEXT]
            edges = [(0, 1), (0, 2)]
            start_positions = []
            entity_positions = []
            root = None

            for i, entry in enumerate(r[key_main]):
                start_positions.append(len(data_strings))
                data_strings.append(KEY_PREFIX_MAPPING[key_main] + SEPARATOR + entry)
                edges.append((len(data_strings), len(data_strings) - 1))
                data_strings.append(KEY_PREFIX_MAPPING[key_rel] + SEPARATOR + r[key_rel][i])
                for j, k_annot in enumerate(keys_annot):
                    edges.append((len(data_strings) - 2 - j, len(data_strings)))
                    data_strings.append(KEY_PREFIX_MAPPING[k_annot] + SEPARATOR + r[k_annot][i])
                if i in entities_end:
                    entity_positions.append(len(data_strings))
                    for j in range(entities_end[i][0], entities_end[i][1]):
                        edges.append((start_positions[j], len(data_strings)))
                    data_strings.append(KEY_PREFIX_MAPPING[key_entity])
            for i, head in enumerate(r['stanford_head']):
                if head != 0:
                    edges.append((start_positions[head - 1], start_positions[i] + 1))
                else:
                    root = start_positions[i] + 1

            assert root is not None, 'ROOT not found'
            edges.append((2, len(data_strings)))
            edges.append((len(data_strings), root))
            data_strings.append(TYPE_SENTENCE)

            edges.append((entity_positions[0], len(data_strings)))
            edges.append((len(data_strings), entity_positions[1]))
            data_strings.append(KEY_PREFIX_MAPPING["label"] + SEPARATOR + r["label"])

            rows_and_cols = np.array(edges).T
            graph_out = csr_matrix(coo_matrix((np.ones(len(edges), dtype=bool), (rows_and_cols[1], rows_and_cols[0]))))
            #_graph_out = graph_out.toarray()
            yield data_strings, None, graph_out
        except Exception as e:
            nbr_failed += 1
            logger.warning('ID:%s %s' % (r['id'], str(e)))
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
    unused='not used parameters'
)
def parse(in_path, out_path, *unused):

    file_names = ['dev', 'test', 'train']
    for fn in file_names:
        logger.info('create forest for %s ...' % fn)
        if not os.path.exists(os.path.join(in_path, fn + '.jsonl')):
            logger.warning('could not find file: %s. skip it.' % os.path.join(in_path, fn + '.jsonl'))
            continue
        out_base_name = os.path.join(out_path, DIR_BATCHES, fn.split('/')[0])
        make_parent_dir(out_base_name)
        process_records(records=(json.loads(s) for s in open(os.path.join(in_path, fn + '.jsonl')).readlines()),
                        out_base_name=out_base_name, record_reader=reader, concat_mode=None, as_graph=True,
                        dont_parse=True)
    logger.info('done.')


@plac.annotations(
    mode=('processing mode', 'positional', None, str, ['PARSE', 'PARSE_DUMMY', 'MERGE', 'CREATE_INDICES', 'ALL']),
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

        #relation_ids, relation_strings = forest_merged.lexicon.get_ids_for_prefix(TYPE_RELATION_DIRECTION)
        #save_class_ids(dir_path=out_path_merged, prefix_type=TYPE_RELATION_DIRECTION, classes_ids=relation_ids,
        #               classes_strings=relation_strings)
    elif mode == 'CREATE_INDICES':
        plac.call(create_index_files, args)
    elif mode == 'ALL':
        plac.call(main, ('PARSE',) + args)
        plac.call(main, ('MERGE',) + args)
        plac.call(main, ('CREATE_INDICES', '--end-root', '22584', '--split-count', '1', '--suffix', 'dev') + args)
        plac.call(main, ('CREATE_INDICES', '--start-root', '22584', '--end-root', '38041', '--split-count', '1',
                         '--suffix', 'test') + args)
        plac.call(main, ('CREATE_INDICES', '--start-root', '38041', '--split-count', '4', '--suffix', 'train') + args)
    else:
        raise ValueError('unknown mode')


if __name__ == '__main__':
    plac.call(main)
