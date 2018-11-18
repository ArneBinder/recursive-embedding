import json
import logging
from datetime import datetime

from scipy.sparse import csr_matrix, csc_matrix, lil_matrix, coo_matrix
import plac
import spacy


import preprocessing
from constants import LOGGING_FORMAT, TYPE_RELATION, TYPE_DATASET, SEPARATOR, TYPE_POS_TAG, TYPE_LEXEME, \
    TYPE_DEPENDENCY_RELATION, TYPE_NAMED_ENTITY
from corpus import process_records, merge_batches, create_index_files, save_class_ids
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


def reader(records, key_main="tokens", keys_annot=("stanford_pos", "stanford_deprel"),
           root_string=TYPE_TACRED, key_entity="entities"
           #keys_meta=(TYPE_RELATION,), key_id=TYPE_SEMEVAL2010TASK8_ID,
           #root_text_string=TYPE_CONTEXT
           ):
    for r in records:
        entities = r[key_entity]
        entities_end = [e[-1]-1 for e in entities]
        size = len(r[key_main]) * (1 + len(keys_annot) + len(entities) + 2)
        graph_out = coo_matrix((size, size), dtype=bool)
        data_strings = [root_string, KEY_PREFIX_MAPPING['id'] + SEPARATOR + r['id']]
        mapping = {}
        start_positions = []

        for i, entry in enumerate(r[key_main]):
            start_positions.append(len(data_strings))
            mapping[len(data_strings)] = i
            data_strings.append(KEY_PREFIX_MAPPING[key_main] + SEPARATOR + entry)
            for k_annot in keys_annot:
                mapping[len(data_strings)] = i
                data_strings.append(KEY_PREFIX_MAPPING[k_annot] + SEPARATOR + r[k_annot][i])
            if i in entities_end:
                mapping[len(data_strings)] = i
                data_strings.append(KEY_PREFIX_MAPPING[key_entity])
            # TODO: add graph structure
        yield r


@plac.annotations(
    out_base_name=('corpora output base file name', 'option', 'o', str),
    sentence_processor=('sentence processor', 'option', 'p', str),
)
def parse_dummy(out_base_name, sentence_processor=None):
    print(out_base_name)
    make_parent_dir(out_base_name)
    if sentence_processor is not None and sentence_processor.strip() != '':
        _sentence_processor = getattr(preprocessing, sentence_processor.strip())
    else:
        _sentence_processor = preprocessing.process_sentence1
    parser = spacy.load('en')
    process_records(records=[DUMMY_RECORD], out_base_name=out_base_name, record_reader=reader, parser=parser,
                    sentence_processor=_sentence_processor, concat_mode=None)


@plac.annotations(
    mode=('processing mode', 'positional', None, str, ['PARSE', 'PARSE_DUMMY', 'TEST', 'MERGE', 'CREATE_INDICES', 'ALL']),
    args='the parameters for the underlying processing method')
def main(mode, *args):
    if mode == 'PARSE_DUMMY':
        plac.call(parse_dummy, args)
    elif mode == 'TEST':
        raise NotImplementedError("mode == 'TEST' not yet implemented")
        res = list(plac.call(read_file, args))
    elif mode == 'PARSE':
        raise NotImplementedError("mode == 'PARSE' not yet implemented")
        plac.call(parse, args)
    elif mode == 'MERGE':
        raise NotImplementedError("mode == 'MERGE' not yet implemented")
        forest_merged, out_path_merged = plac.call(merge_batches, args)
        relation_ids, relation_strings = forest_merged.lexicon.get_ids_for_prefix(TYPE_RELATION)
        save_class_ids(dir_path=out_path_merged, prefix_type=TYPE_RELATION, classes_ids=relation_ids,
                       classes_strings=relation_strings)

        #relation_ids, relation_strings = forest_merged.lexicon.get_ids_for_prefix(TYPE_RELATION_DIRECTION)
        #save_class_ids(dir_path=out_path_merged, prefix_type=TYPE_RELATION_DIRECTION, classes_ids=relation_ids,
        #               classes_strings=relation_strings)
    elif mode == 'CREATE_INDICES':
        raise NotImplementedError("mode == 'CREATE_INDICES' not yet implemented")
        plac.call(create_index_files, args)
    elif mode == 'ALL':
        raise NotImplementedError("mode == 'ALL' not yet implemented")
        plac.call(main, ('PARSE',) + args)
        plac.call(main, ('MERGE',) + args)
        plac.call(main, ('CREATE_INDICES', '--end-root', '2717', '--split-count', '1', '--suffix', 'test') + args)
        plac.call(main, ('CREATE_INDICES', '--start-root', '2717', '--split-count', '4', '--suffix', 'train') + args)
    else:
        raise ValueError('unknown mode')


if __name__ == '__main__':
    plac.call(main)