import json
from datetime import datetime


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


