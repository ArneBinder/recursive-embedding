import io
import json
import logging
import os

import numpy as np

import spacy
# from datetime import datetime
import plac
from nltk.parse.corenlp import CoreNLPDependencyParser

from sequence_trees import graph_out_from_children_dict, Forest
from constants import DTYPE_IDX, PREFIX_REC_EMB, PREFIX_CONLL, PREFIX_NIF, PREFIX_TACRED, PREFIX_SICK, \
    PREFIX_IMDB, PREFIX_SEMEVAL, REC_EMB_GLOBAL_ANNOTATION, REC_EMB_HAS_GLOBAL_ANNOTATION, REC_EMB_RECORD, \
    REC_EMB_HAS_PARSE, REC_EMB_HAS_PARSE_ANNOTATION, REC_EMB_HAS_CONTEXT, REC_EMB_USED_PARSER, \
    REC_EMB_SUFFIX_GLOBAL_ANNOTATION, REC_EMB_SUFFIX_NIF_CONTEXT, NIF_WORD, NIF_NEXT_WORD, NIF_SENTENCE, \
    NIF_NEXT_SENTENCE, NIF_IS_STRING, LOGGING_FORMAT, PREFIX_UNIVERSAL_DEPENDENCIES_ENGLISH, RDF_PREFIXES_MAP, \
    NIF_CONTEXT
from lexicon import Lexicon
from corpus import create_index_files, save_class_ids

logger = logging.getLogger('corpus_rdf')
logger.setLevel(logging.DEBUG)
logger_streamhandler = logging.StreamHandler()
logger_streamhandler.setLevel(logging.DEBUG)
logger_streamhandler.setFormatter(logging.Formatter(LOGGING_FORMAT))
logger.addHandler(logger_streamhandler)

TACRED_CONLL_RECORD = '''
# index	token	subj	subj_type	obj	obj_type	stanford_pos	stanford_ner	stanford_deprel	stanford_head
# id=e7798fb926b9403cfcd2 docid=APW_ENG_20101103.0539 reln=per:title
1	At	_	_	_	_	IN	O	case	4
2	the	_	_	_	_	DT	O	det	4
3	same	_	_	_	_	JJ	O	amod	4
4	time	_	_	_	_	NN	O	nmod	12
5	,	_	_	_	_	,	O	punct	12
6	Chief	_	_	_	_	NNP	O	compound	10
7	Financial	_	_	_	_	NNP	O	compound	10
8	Officer	_	_	_	_	NNP	O	compound	10
9	Douglas	SUBJECT	PERSON	_	_	NNP	PERSON	compound	10
10	Flint	SUBJECT	PERSON	_	_	NNP	PERSON	nsubj	12
11	will	_	_	_	_	MD	O	aux	12
12	become	_	_	_	_	VB	O	ROOT	0
13	chairman	_	_	OBJECT	TITLE	NN	O	xcomp	12
14	,	_	_	_	_	,	O	punct	12
15	succeeding	_	_	_	_	VBG	O	xcomp	12
16	Stephen	_	_	_	_	NNP	PERSON	compound	17
17	Green	_	_	_	_	NNP	PERSON	dobj	15
18	who	_	_	_	_	WP	O	nsubj	20
19	is	_	_	_	_	VBZ	O	aux	20
20	leaving	_	_	_	_	VBG	O	acl:relcl	17
21	to	_	_	_	_	TO	O	mark	22
22	take	_	_	_	_	VB	O	xcomp	20
23	a	_	_	_	_	DT	O	det	25
24	government	_	_	_	_	NN	O	compound	25
25	job	_	_	_	_	NN	O	dobj	22
26	.	_	_	_	_	.	O	punct	12
'''

##### DUMMY DATA #####################

TACRED_RECORD_JSON = u'[{"id": "e7798fb926b9403cfcd2", "docid": "APW_ENG_20101103.0539", "relation": "per:title", "token": ["At", "the", "same", "time", ",", "Chief", "Financial", "Officer", "Douglas", "Flint", "will", "become", "chairman", ",", "succeeding", "Stephen", "Green", "who", "is", "leaving", "to", "take", "a", "government", "job", "."], "subj_start": 8, "subj_end": 9, "obj_start": 12, "obj_end": 12, "subj_type": "PERSON", "obj_type": "TITLE", "stanford_pos": ["IN", "DT", "JJ", "NN", ",", "NNP", "NNP", "NNP", "NNP", "NNP", "MD", "VB", "NN", ",", "VBG", "NNP", "NNP", "WP", "VBZ", "VBG", "TO", "VB", "DT", "NN", "NN", "."], "stanford_ner": ["O", "O", "O", "O", "O", "O", "O", "O", "PERSON", "PERSON", "O", "O", "O", "O", "O", "PERSON", "PERSON", "O", "O", "O", "O", "O", "O", "O", "O", "O"], "stanford_head": [4, 4, 4, 12, 12, 10, 10, 10, 10, 12, 12, 0, 12, 12, 12, 17, 15, 20, 20, 17, 22, 20, 25, 25, 22, 12], "stanford_deprel": ["case", "det", "amod", "nmod", "punct", "compound", "compound", "compound", "compound", "nsubj", "aux", "ROOT", "xcomp", "punct", "xcomp", "compound", "dobj", "nsubj", "aux", "acl:relcl", "mark", "xcomp", "det", "compound", "dobj", "punct"]}, {"id": "e779865fb96bbbcc4ca4", "docid": "APW_ENG_20080229.1401.LDC2009T13", "relation": "no_relation", "token": ["U.S.", "District", "Court", "Judge", "Jeffrey", "White", "in", "mid-February", "issued", "an", "injunction", "against", "Wikileaks", "after", "the", "Zurich-based", "Bank", "Julius", "Baer", "accused", "the", "site", "of", "posting", "sensitive", "account", "information", "stolen", "by", "a", "disgruntled", "former", "employee", "."], "subj_start": 17, "subj_end": 18, "obj_start": 4, "obj_end": 5, "subj_type": "PERSON", "obj_type": "PERSON", "stanford_pos": ["NNP", "NNP", "NNP", "NNP", "NNP", "NNP", "IN", "NNP", "VBD", "DT", "NN", "IN", "NNP", "IN", "DT", "JJ", "NNP", "NNP", "NNP", "VBD", "DT", "NN", "IN", "VBG", "JJ", "NN", "NN", "VBN", "IN", "DT", "JJ", "JJ", "NN", "."], "stanford_ner": ["LOCATION", "O", "O", "O", "PERSON", "PERSON", "O", "O", "O", "O", "O", "O", "ORGANIZATION", "O", "O", "MISC", "O", "PERSON", "PERSON", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O"], "stanford_head": [6, 6, 6, 6, 6, 9, 8, 6, 0, 11, 9, 13, 11, 20, 19, 19, 19, 19, 20, 9, 22, 20, 24, 20, 27, 27, 24, 27, 33, 33, 33, 33, 28, 9], "stanford_deprel": ["compound", "compound", "compound", "compound", "compound", "nsubj", "case", "nmod", "ROOT", "det", "dobj", "case", "nmod", "mark", "det", "amod", "compound", "compound", "nsubj", "advcl", "det", "dobj", "mark", "advcl", "amod", "compound", "dobj", "acl", "case", "det", "amod", "amod", "nmod", "punct"]}, {"id": "e7798ae9c0adbcdc81e7", "docid": "APW_ENG_20090707.0488", "relation": "per:city_of_death", "token": ["PARIS", "2009-07-07", "11:07:32", "UTC", "French", "media", "earlier", "reported", "that", "Montcourt", ",", "ranked", "119", ",", "was", "found", "dead", "by", "his", "girlfriend", "in", "the", "stairwell", "of", "his", "Paris", "apartment", "."], "subj_start": 9, "subj_end": 9, "obj_start": 0, "obj_end": 0, "subj_type": "PERSON", "obj_type": "CITY", "stanford_pos": ["NNP", "CD", "CD", "NNP", "NNP", "NNS", "RBR", "VBD", "IN", "NNP", ",", "VBD", "CD", ",", "VBD", "VBN", "JJ", "IN", "PRP$", "NN", "IN", "DT", "NN", "IN", "PRP$", "NNP", "NN", "."], "stanford_ner": ["LOCATION", "TIME", "TIME", "TIME", "MISC", "O", "O", "O", "O", "PERSON", "O", "O", "NUMBER", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "LOCATION", "O", "O"], "stanford_head": [6, 6, 6, 6, 6, 8, 8, 0, 16, 16, 10, 10, 12, 10, 16, 8, 16, 20, 20, 17, 23, 23, 16, 27, 27, 27, 23, 8], "stanford_deprel": ["compound", "nummod", "nummod", "compound", "compound", "nsubj", "advmod", "ROOT", "mark", "nsubjpass", "punct", "acl", "dobj", "punct", "auxpass", "ccomp", "xcomp", "case", "nmod:poss", "nmod", "case", "det", "nmod", "case", "nmod:poss", "compound", "nmod", "punct"]}]'
TACRED_RECORD_TOKEN_FEATURES = {
    "token": ["At", "the", "same", "time", ",", "Chief", "Financial", "Officer", "Douglas", "Flint", "will", "become",
              "chairman", ",", "succeeding", "Stephen", "Green", "who", "is", "leaving", "to", "take", "a",
              "government", "job", "."],
    "stanford_pos": ["IN", "DT", "JJ", "NN", ",", "NNP", "NNP", "NNP", "NNP", "NNP", "MD", "VB", "NN", ",", "VBG",
                     "NNP", "NNP", "WP", "VBZ", "VBG", "TO", "VB", "DT", "NN", "NN", "."],
    "stanford_ner": ["O", "O", "O", "O", "O", "O", "O", "O", "PERSON", "PERSON", "O", "O", "O", "O", "O", "PERSON",
                     "PERSON", "O", "O", "O", "O", "O", "O", "O", "O", "O"],
    "stanford_deprel": ["case", "det", "amod", "nmod", "punct", "compound", "compound", "compound", "compound", "nsubj",
                        "aux", "ROOT", "xcomp", "punct", "xcomp", "compound", "dobj", "nsubj", "aux", "acl:relcl",
                        "mark", "xcomp", "det", "compound", "dobj", "punct"],
    "stanford_head": [4, 4, 4, 12, 12, 10, 10, 10, 10, 12, 12, 0, 12, 12, 12, 17, 15, 20, 20, 17, 22, 20, 25, 25, 22,
                      12]}

TACRED_RECORD_ID = PREFIX_TACRED + u'train/e7798fb926b9403cfcd2'
TACRED_RECORD_TOKEN_ANNOTATIONS = [{u'@id': TACRED_RECORD_ID + u'#r1',
                                    u'@type': [PREFIX_TACRED + u'vocab#relation:per:title'],
                                    PREFIX_TACRED + u'vocab#subj': [{u'@id': TACRED_RECORD_ID + u'#s1_9'},
                                                                    {u'@id': TACRED_RECORD_ID + u'#s1_10'}],
                                    PREFIX_TACRED + u'vocab#obj': [{u'@id': TACRED_RECORD_ID + u'#s1_13'}],
                                    }]
TACRED_RECORD = {'record_id': TACRED_RECORD_ID,
                 'token_features': TACRED_RECORD_TOKEN_FEATURES,
                 'token_annotations': TACRED_RECORD_TOKEN_ANNOTATIONS
                 }

# NOTE: SICK records are split into A and B records!
SICK_RECORD_ID = PREFIX_SICK + u'SICK_train.txt/1/A'
SICK_RECORD_ANNOTATIONS = {PREFIX_SICK + u'vocab#relatedness_score': [{u'@value': 4.5}],
                           PREFIX_SICK + u'vocab#entailment_judgment': [{u'@value': u'NEUTRAL'}],
                           PREFIX_SICK + u'vocab#other': [{u'@id': PREFIX_SICK + u'SICK_train.txt/1/B'}]
                           }
SICK_RECORD = {'record_id': SICK_RECORD_ID,
               # ATTENTION: punctuation "." was added!
               'context_string': u'A group of kids is playing in a yard and an old man is standing in the background.',
               'global_annotations': SICK_RECORD_ANNOTATIONS
               }

IMDB_RECORD_ID = PREFIX_IMDB + u'train/pos/0_9.txt'
IMDB_RECORD_ANNOTATIONS = {PREFIX_IMDB + u'vocab#sentiment': [{u'@value': u'pos'}],
                           PREFIX_IMDB + u'vocab#rating': [{u'@value': 9}]}
IMDB_RECORD = {'record_id': IMDB_RECORD_ID,
               'context_string': u"Bromwell High is a cartoon comedy. It ran at the same time as some other programs about school life, such as \"Teachers\". My 35 years in the teaching profession lead me to believe that Bromwell High\'s satire is much closer to reality than is \"Teachers\". The scramble to survive financially, the insightful students who can see right through their pathetic teachers\' pomp, the pettiness of the whole situation, all remind me of the schools I knew and their students. When I saw the episode in which a student repeatedly tried to burn down the school, I immediately recalled ......... at .......... High. A classic line: INSPECTOR: I\'m here to sack one of your teachers. STUDENT: Welcome to Bromwell High. I expect that many adults of my age think that Bromwell High is far fetched. What a pity that it isn\'t!",
               'global_annotations': IMDB_RECORD_ANNOTATIONS,
               }

SEMEVAL_RECORD_ID = PREFIX_SEMEVAL + u'TRAIN_FILE.TXT/1'
SEMEVAL_RECORD_CHARACTER_ANNOTATIONS = [{u'@id': SEMEVAL_RECORD_ID + u'#r1',
                                         u'@type': [PREFIX_SEMEVAL + u'vocab#relation:Component-Whole(e2,e1)'],
                                         PREFIX_SEMEVAL + u'vocab#subj': (73, 89),
                                         PREFIX_SEMEVAL + u'vocab#obj': (98, 106),
                                         }]
SEMEVAL_RECORD = {'record_id': SEMEVAL_RECORD_ID,
                  'context_string': u"The system as described above has its greatest application in an arrayed configuration of antenna elements.",
                  'character_annotations': SEMEVAL_RECORD_CHARACTER_ANNOTATIONS,
                  }

##### DUMMY DATA END #####################


# TODO: implement reader for SICK, IMDB, SEMEVAL, TACRED
# DONE: SICK, SEMEVAL, IMDB

# TODO: implement sentence_processors

RDF_PREFIXES_MAP_REV = {short: long for long, short in RDF_PREFIXES_MAP.items()}


def shorten(uri):
    if uri in RDF_PREFIXES_MAP:
        return RDF_PREFIXES_MAP[uri]
    for long, short in RDF_PREFIXES_MAP.items():
        if uri.startswith(long):
            return short + uri[len(long):]
    return uri


def enlarge(uri):
    if u':' not in uri:
        return uri
    parts = uri.split(u':')
    short = parts[0] + u':'
    if short not in RDF_PREFIXES_MAP_REV:
        return uri
    return RDF_PREFIXES_MAP_REV[short] + u':'.join(parts[1:])


def parse_spacy_to_conll(text, nlp=spacy.load('en')):
    doc = nlp(text)
    for sent in doc.sents:

        for i, word in enumerate(sent):
            if word.head.i == word.i:
                head_idx = 0
            else:
                # head_idx = doc[i].head.i + 1
                head_idx = word.head.i - sent[0].i + 1

            yield u"%d\t%s\t%s\t%s\t%s\t%s\t%d\t%s\t%s\t%i" % (
                # data          conllu      notes
                i + 1,  # ID        There's a word.i attr that's position in *doc*
                word,  # FORM
                word.lemma_,  # LEMMA
                word.pos_,  # UPOS      Coarse-grained tag
                word.tag_,  # XPOS      Fine-grained tag
                '_',  # FEATS
                head_idx,  # HEAD (ID or 0)
                word.dep_,  # DEPREL    Relation
                '_',  # DEPS
                word.idx,  # MISC      character offset
            )
        yield u''


def parse_corenlp_to_conll(text, tokens=None, dep_parser=CoreNLPDependencyParser(url='http://localhost:9000')):
    if tokens is not None:
        raise NotImplementedError('coreNLP parse for list of tokens not implemented')
    else:
        parsed_sents = dep_parser.parse_text(text)

    template = u'{i}\t{word}\t{lemma}\t{ctag}\t{tag}\t{feats}\t{head}\t{rel}\t_\t{idx}'
    previous_end = 0
    for sent in parsed_sents:
        for i, node in sorted(sent.nodes.items()):
            if node['tag'] == 'TOP':
                continue
            idx = text.find(node['word'], previous_end)
            # assert idx >= 0, 'word="%s" not found in text="%s"' % (node['word'], text)
            if idx < 0:
                print('WARNING: word="%s" not found in text="%s"' % (node['word'], text))
                idx = '_'
            else:
                previous_end = idx + len(node['word'])
            l = template.format(i=i, idx=idx, **node)
            yield l
        yield u''


def record_to_conll(sentence_record, captions, key_mapping):
    selected_entries = {k: sentence_record[key_mapping[k]] for k in key_mapping if key_mapping[k] in sentence_record}
    l = [dict(zip(selected_entries, t)) for t in zip(*selected_entries.values())]
    for i, d in enumerate(l):
        y = u'\t'.join([str(i + 1)] + [str(d[c]) if c in d else '_' for c in captions])
        yield y
    yield u''


def convert_conll_to_rdf(conll_data, base_uri=RDF_PREFIXES_MAP[PREFIX_UNIVERSAL_DEPENDENCIES_ENGLISH],
                         # ATTENTION: ID column has to be the first column, but should not occur in  "columns"!
                         columns=('WORD', 'LEMMA', 'UPOS', 'POS', 'FEAT', 'HEAD', 'EDGE', 'DEPS', 'MISC')):
    res = []
    conll_sentences = conll_data.split('\n\n')
    sent_id = 1
    previous_sentence = None
    for conll_sent in conll_sentences:
        sent_prefix = u'%ss%i_' % (base_uri, sent_id)
        row_rdf = {u'@id': sent_prefix + u'0', u'@type': [NIF_SENTENCE]}
        res.append(row_rdf)
        if previous_sentence is not None:
            previous_sentence[NIF_NEXT_SENTENCE] = [{u'@id': row_rdf[u'@id']}]
        previous_sentence = row_rdf

        conll_lines = conll_sent.split('\n')
        token_id = 1
        previous_word = None
        for line in conll_lines:
            if line.strip().startswith('#') or line.strip() == '':
                continue

            row = line.split('\t')
            row_dict = {RDF_PREFIXES_MAP[PREFIX_CONLL] + columns[i]:
                            (row[i + 1] if columns[i] != 'HEAD' else (u'@id', sent_prefix + row[i + 1])) for i, k in
                        enumerate(columns)
                        if i + 1 < len(row) and row[i + 1] != '_'}
            row_dict[RDF_PREFIXES_MAP[PREFIX_CONLL] + u'ID'] = row[0]
            row_rdf = _to_rdf(row_dict)
            row_rdf[u'@id'] = sent_prefix + str(row[0])
            row_rdf[u'@type'] = [NIF_WORD]
            res.append(row_rdf)
            if previous_word is not None:
                previous_word[NIF_NEXT_WORD] = [{u'@id': row_rdf[u'@id']}]
            previous_word = row_rdf
            token_id += 1
        sent_id += 1
    return res


def _to_rdf(element):
    if isinstance(element, dict):
        res = {k: _to_rdf(element[k]) for k in element}
    elif isinstance(element, list):
        res = [_to_rdf(e) for e in element]
    # tuple indicates marked (@id or @value) entry
    elif isinstance(element, tuple):
        marker, content = element
        assert marker in [u'@value', u'@id'], 'WARNING: Unknown value marker: %s for content: %s. use ' \
                                              '"@value" or "@id" as first tuple entry to mark value as lateral or id.' \
                                              % (marker, content)
        res = [{marker: content}]
    else:
        res = [{u'@value': element}]
    return res


def get_token_ids_in_span(tokens, start, end, idx_key=RDF_PREFIXES_MAP[PREFIX_CONLL] + 'MISC',
                          word_key=RDF_PREFIXES_MAP[PREFIX_CONLL] + 'WORD'):
    res = []
    for t in tokens:
        if idx_key in t and word_key in t:
            token_start = int(t[idx_key][0][u'@value'])
            token_end = token_start + len(t[word_key][0][u'@value'])
            if start <= token_start < end or start < token_end <= end:
                res.append({u'@id': t[u'@id']})
    return res


def parse_and_convert_record(record_id,
                             context_string=None,
                             global_annotations=None,
                             token_features=None,
                             character_annotations=None,
                             token_annotations=None,
                             parser=None,
                             ):
    conll_columns = ('WORD', 'LEMMA', 'UPOS', 'POS', 'FEAT', 'HEAD', 'EDGE', 'DEPS', 'MISC')
    if isinstance(parser, spacy.language.Language):
        conll_lines = list(parse_spacy_to_conll(context_string, nlp=parser))
    elif isinstance(parser, CoreNLPDependencyParser):
        conll_lines = list(parse_corenlp_to_conll(context_string, dep_parser=parser))
    elif parser is None:
        assert token_features is not None, 'parser==None requires token_features, but it is None'
        conll_lines = list(record_to_conll(token_features, captions=conll_columns,
                                           key_mapping={'WORD': 'token', 'POS': 'stanford_pos',
                                                        'HEAD': 'stanford_head', 'EDGE': 'stanford_deprel'}))
        # conll_columns = ('index', 'token', 'subj', 'subj_type', 'obj', 'obj_type', 'stanford_pos', 'stanford_ner', 'stanford_deprel', 'stanford_head')
    else:
        raise NotImplementedError('parser %s not implemented' % parser)

    tokens_jsonld = convert_conll_to_rdf('\n'.join(conll_lines), base_uri=record_id + u'#', columns=conll_columns)

    res = {u'@id': record_id, u'@type': [REC_EMB_RECORD], REC_EMB_HAS_PARSE: tokens_jsonld,
           REC_EMB_USED_PARSER: [{u'@value': u'%s.%s' % (type(parser).__module__, type(parser).__name__)}]}
    if global_annotations is not None:
        global_annotations[u'@id'] = record_id + REC_EMB_SUFFIX_GLOBAL_ANNOTATION
        global_annotations[u'@type'] = [REC_EMB_GLOBAL_ANNOTATION]
        res[REC_EMB_HAS_GLOBAL_ANNOTATION] = [global_annotations]
    if context_string is not None:
        res[REC_EMB_HAS_CONTEXT] = [{u'@id': record_id + REC_EMB_SUFFIX_NIF_CONTEXT,
                                     u'@type': [NIF_CONTEXT],
                                     NIF_IS_STRING: [{u'@value': context_string}]}]

    # if available, convert character_annotations to token_annotations
    if character_annotations is not None:
        for i, character_annotation in enumerate(character_annotations):
            for k in character_annotation:
                if not k.startswith(u'@'):
                    _tokens = get_token_ids_in_span(tokens_jsonld,
                                                    start=character_annotation[k][0],
                                                    end=character_annotation[k][1])
                    assert len(_tokens) > 0, 'no tokens found for key=%s with indices=%s' % (k, character_annotation[k])
                    # overwrite character offsets with actual tokens
                    character_annotations[i][k] = _tokens
        token_annotations = character_annotations

    if token_annotations is not None:
        res[REC_EMB_HAS_PARSE_ANNOTATION] = token_annotations

    return res


def parse_to_rdf(in_path, out_path, reader_rdf, file_names, parser='spacy'):
    logger.info('load parser...')
    if parser.strip() == 'spacy':
        _parser = spacy.load('en')
    elif parser.strip() == 'corenlp':
        _parser = CoreNLPDependencyParser(url='http://localhost:9000')
    else:
        raise NotImplementedError('parser=%s not implemented' % parser)
    logger.info('loaded parser %s' % type(_parser))
    n_failed = {}
    n_total = {}
    out_path = os.path.join(out_path, parser)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    for fn in file_names:
        fn_out = os.path.join(out_path, file_names[fn])
        logger.info('process file %s, save result to %s' % (fn, fn_out))
        n_failed[fn_out] = 0
        already_processed = {}
        if os.path.exists(fn_out):
            with io.open(fn_out, encoding='utf8') as fout:
                for l in fout.readlines():
                    _l = json.loads(l)
                    already_processed[_l['@id']] = l
        n_total[fn_out] = len(already_processed)
        if len(already_processed) > 0:
            logger.info('found %i already processed records' % len(already_processed))

        with io.open(fn_out, 'w', encoding='utf8') as fout:
            for i, record in enumerate(reader_rdf(in_path, fn)):
                if record['record_id'] in already_processed:
                    parsed_rdf_json = already_processed[record['record_id']]
                else:
                    try:
                        parsed_rdf = parse_and_convert_record(parser=_parser, **record)
                        # NOTE: "+ u'\n'" is important to convert to unicode, because of json bug in python 2
                        # (the ensure_ascii=False flag can produce a mix of unicode and str objects)
                        parsed_rdf_json = json.dumps(parsed_rdf, ensure_ascii=False) + u'\n'
                    except Exception as e:
                        logger.warning('failed to parse record=%s: %s' % (record['record_id'], str(e)))
                        n_failed[fn_out] += 1
                        continue
                fout.write(parsed_rdf_json)
                n_total[fn_out] += 1
        for fn_out in n_failed:
            logger.info('%s: failed to process %i of total %i records' % (fn_out, n_failed[fn_out], n_total[fn_out]))
    logger.debug('done')


def serialize_jsonld_dict(jsonld, offset=0, sort_map={}, discard_predicates=(), discard_types=(),
                          id_as_value_predicates=(), skip_predicates=(), swap_predicates=(), revert_predicates=()):
    """
    Serializes a json-ld dict object.
    Returns a list of all _types_ (additional types are created from value entries),
    a mapping from all ids to the index of the corresponding type int type list and,
    a list of edges (source, target), where source and target may be ids or indices to the final serialization.

    For every triple (subject, predicate, object) the edges (subject, predicate) and (predicate, object) are created,
    but lateral values (have only @value entry) are collapsed with their predicates (e.g. with the predicate types).

    ATTENTION: only the first type per entry is used!

    :param jsonld: input json-ld object
    :param offset: offset for id positions
    :param sort_map: a map from node types to lists of predicates that determine their ordering. If the current type t
        is in sort_map, all predicates not in sort_map[t] are discarded
    :param discard_predicates: these predicates are discarded
    :param discard_types: nodes with these types _including_ their targets are discarded
    :param id_as_value_predicates: for these predicates, take @id values as plain literals
    :param skip_predicates: skip the these predicates e.g. create only one edge (subject, object) instead of
        (subject, predicate) and (predicate, object)
    :param revert_predicates: revert the edges from / to these predicates. If it is also in skip_predicates, create
        (object, subject) instead of (subject, object)
    :param swap_predicates: swap positions in final serialization of subject and object, that are linked by these
        predicates. ATTENTION: works only for collapsed objects (literals)
    :return: types, ids, refs
    """
    ser = []
    ids = {}
    edges = []
    # consider only first type
    _type = u'' + jsonld[u'@type'][0]
    if _type in discard_types:
        return ser, ids, edges
    ser.append(_type)
    ids[u'' + jsonld[u'@id']] = offset
    preds = sort_map.get(_type, sorted(jsonld))
    for pred in preds:
        if pred[0] != u'@' and pred in jsonld:
            object_literals = []
            # edge source: points to predicate, if it should be used, to subject, if predicate is skipped, or is None,
            # if no edge should be created at all
            idx_edge_source = None
            if pred not in discard_predicates:
                if pred in skip_predicates:
                    # point to subject (first entry)
                    idx_edge_source = offset
                else:
                    # point to predicate
                    idx_edge_source = offset + len(ser)
                    ser.append(u'' + pred)

            revert_edge = pred in revert_predicates

            for obj_dict in jsonld[pred]:
                # lateral object
                if u'@value' in obj_dict:
                    # assert idx_edge_source is None or len(ser) == idx_edge_source + 1, \
                    #    'laterals (@value) are mixed with complex elements (@type / @id)'
                    object_literals.append(obj_dict[u'@value'])
                # complex object
                else:
                    # assert len(object_literals) == 0 or pred in id_as_value_predicates, \
                    #    'laterals (@value) are mixed with complex elements (@type / @id)'
                    if pred in id_as_value_predicates:
                        object_literals.append(obj_dict[u'@id'])
                        continue

                    if idx_edge_source is not None:
                        new_edge = (obj_dict[u'@id'], idx_edge_source) if revert_edge else (
                            idx_edge_source, obj_dict[u'@id'])
                        edges.append(new_edge)

                    if u'@type' in obj_dict:
                        obj_ser, obj_ids, obj_edges = serialize_jsonld_dict(obj_dict, offset=len(ser) + offset,
                                                                            sort_map=sort_map,
                                                                            discard_predicates=discard_predicates,
                                                                            discard_types=discard_types,
                                                                            id_as_value_predicates=id_as_value_predicates,
                                                                            skip_predicates=skip_predicates,
                                                                            swap_predicates=swap_predicates,
                                                                            revert_predicates=revert_predicates)
                        ser.extend(obj_ser)
                        ids.update(obj_ids)
                        edges.extend(obj_edges)

            # do not add edge(s), if predicate was skipped
            if idx_edge_source is not None and offset != idx_edge_source:
                # collapse laterals with predicate
                if len(object_literals) > 0:
                    # last element added should be the predicate
                    del ser[-1]
                    new_entries = [u'%s=%s' % (pred, v) for v in object_literals]
                    ser.extend(new_entries)
                    if pred in swap_predicates:
                        assert len(new_entries) == 1, 'can swap only single pred+value, but found %i' % len(new_entries)
                        subj = ser[0]
                        ser[0] = ser[-1]
                        ser[-1] = subj
                    if revert_edge:
                        edges.extend([(idx_edge_source + i, offset) for i in range(len(new_entries))])
                    else:
                        edges.extend([(offset, idx_edge_source + i) for i in range(len(new_entries))])
                else:
                    if revert_edge:
                        edges.append((idx_edge_source, offset))
                    else:
                        edges.append((offset, idx_edge_source))

    return ser, ids, edges


def serialize_jsonld(jsonld, sort_map=None, discard_predicates=(), discard_types=(), id_as_value_predicates=(),
                     skip_predicates=(), swap_predicates=(), revert_predicates=()):
    if sort_map is None:
        sort_map = {REC_EMB_RECORD: [REC_EMB_HAS_CONTEXT, REC_EMB_HAS_GLOBAL_ANNOTATION,
                                     REC_EMB_HAS_PARSE_ANNOTATION,  REC_EMB_USED_PARSER, REC_EMB_HAS_PARSE]}
    if not isinstance(jsonld, list):
        jsonld = [jsonld]
    ser, ids, refs = [], {}, {}
    id_indices = []
    for jsonld_dict in jsonld:
        # see below for: + 1 (offset=len(ser) + 1)
        _ser, _ids, _edges = serialize_jsonld_dict(jsonld_dict, offset=len(ser) + 1, sort_map=sort_map,
                                                   discard_predicates=discard_predicates,
                                                   discard_types=discard_types,
                                                   id_as_value_predicates=id_as_value_predicates,
                                                   skip_predicates=skip_predicates,
                                                   swap_predicates=swap_predicates,
                                                   revert_predicates=revert_predicates)
        _refs = {}
        for s, t in _edges:
            _refs.setdefault(_ids.get(s, s), []).append(_ids.get(t, t))

        ## insert the id directly after the root (offset +1 (above); _ser.insert(1, _id);  _ids[_id] = len(ser); ...)
        _id = u'' + jsonld_dict[u'@id']
        _ser.insert(1, _id)
        _ids[_id] = len(ser)
        _refs[len(ser)] = [len(ser) + 1] + _refs[len(ser) + 1]
        del _refs[len(ser) + 1]
        ## insert end
        id_indices.append(len(ser) + 1)

        ser.extend(_ser)
        ids.update(_ids)
        refs.update(_refs)

    return ser, refs, id_indices


def debug_create_dummy_record_rdf():
    print('load parser...')
    # parser = spacy.load('en')
    parser = CoreNLPDependencyParser(url='http://localhost:9000')
    # parser = None
    print('loaded parser: %s' % str(type(parser)))
    # record = TACRED_RECORD
    # record = SEMEVAL_RECORD
    record = SICK_RECORD
    # record = IMDB_RECORD
    res = parse_and_convert_record(parser=parser, **record)
    # res_str = json.dumps(res)
    print(json.dumps(res, indent=2))
    return res


def debug_load_dummy_record(p='/mnt/DATA/ML/data/corpora_out/IMDB_RDF/spacy/test.jsonl'):
    with io.open(p) as f:
        l = f.readline()
    return json.loads(l)


def convert_jsonld_to_recemb(jsonld, discard_predicates=None, discard_types=None, id_as_value_predicates=None,
                             skip_predicates=None, revert_predicates=None, swap_predicates=None):
    if discard_predicates is None:
        discard_predicates = (RDF_PREFIXES_MAP[PREFIX_CONLL] + u'MISC',
                              RDF_PREFIXES_MAP[PREFIX_CONLL] + u'ID',
                              RDF_PREFIXES_MAP[PREFIX_CONLL] + u'LEMMA',
                              RDF_PREFIXES_MAP[PREFIX_CONLL] + u'POS',
                              RDF_PREFIXES_MAP[PREFIX_NIF] + u'nextWord',
                              RDF_PREFIXES_MAP[PREFIX_REC_EMB] + u'hasContext',
                              # RDF_PREFIXES_MAP[PREFIX_REC_EMB] + u'hasParseAnnotation',
                              RDF_PREFIXES_MAP[PREFIX_NIF] + u'isString',
                              # not needed, just spams the lexicon
                              RDF_PREFIXES_MAP[PREFIX_SICK] + u'vocab#other',
                              )
    if discard_types is None:
        discard_types = (RDF_PREFIXES_MAP[PREFIX_NIF] + u'Context')
    if id_as_value_predicates is None:
        id_as_value_predicates = (RDF_PREFIXES_MAP[PREFIX_SICK] + u'vocab#other')
    if skip_predicates is None:
        skip_predicates = (RDF_PREFIXES_MAP[PREFIX_CONLL] + u'HEAD',
                           RDF_PREFIXES_MAP[PREFIX_NIF] + u'nextSentence',
                           RDF_PREFIXES_MAP[PREFIX_SEMEVAL] + u'vocab#subj',
                           RDF_PREFIXES_MAP[PREFIX_SEMEVAL] + u'vocab#obj',
                           #TODO: add TACRED subj / obj?
                           #RDF_PREFIXES_MAP[PREFIX_TACRED] + u'vocab#subj',
                           #RDF_PREFIXES_MAP[PREFIX_TACRED] + u'vocab#obj'
                           )
    if revert_predicates is None:
        revert_predicates = (RDF_PREFIXES_MAP[PREFIX_CONLL] + u'HEAD',
                             # RDF_PREFIXES_MAP[PREFIX_CONLL] + u'EDGE',
                             RDF_PREFIXES_MAP[PREFIX_NIF] + u'nextSentence',
                             RDF_PREFIXES_MAP[PREFIX_SEMEVAL] + u'vocab#subj',
                             # TODO: add TACRED subj?
                             RDF_PREFIXES_MAP[PREFIX_TACRED] + u'vocab#subj'
                             )
    if swap_predicates is None:
        swap_predicates = (RDF_PREFIXES_MAP[PREFIX_CONLL] + u'WORD')

    ser, refs, ids_indices = serialize_jsonld(jsonld,
                                              discard_predicates=discard_predicates,
                                              discard_types=discard_types,
                                              id_as_value_predicates=id_as_value_predicates,
                                              skip_predicates=skip_predicates,
                                              revert_predicates=revert_predicates,
                                              swap_predicates=swap_predicates)

    # remove all children of parse, except to last sentence element
    idx_hasParse = ser.index(RDF_PREFIXES_MAP[PREFIX_REC_EMB] + u'hasParse')
    idx_last_sentence = len(ser) - ser[::-1].index(RDF_PREFIXES_MAP[PREFIX_NIF] + u'Sentence') - 1
    # link to _last_ sentence
    refs[idx_hasParse] = [idx_last_sentence]

    # create rec-emb
    lex = Lexicon(add_vocab_manual=True)
    data = [Lexicon.hash_string(s) for s in ser]
    lex.add_all(ser)
    graph_out = graph_out_from_children_dict(refs, len(ser))
    recemb = Forest(data=data, data_as_hashes=True, structure=graph_out,
                    root_pos=np.array([0], dtype=DTYPE_IDX))

    return recemb, lex


@plac.annotations(
    in_path=('corpora input path, should contain .jsonl or .jl files', 'option', 'i', str),
    out_path=('corpora output path', 'option', 'o', str),
    glove_file=('glove vector file', 'option', 'g', str),
    word_prefix=('prefix of words in lexicon', 'option', 'w', str),
    classes_prefix=('prefix for lex entries that will be saved as class ids', 'option', 'c', str),
    params_json=('optional parameters as json string', 'option', 'a', str),
)
def convert_corpus_jsonld_to_recemb(in_path, out_path, glove_file='',
                                    word_prefix=RDF_PREFIXES_MAP[PREFIX_CONLL] + u'WORD=', classes_prefix='',
                                    params_json=''):
    params = {}
    if params_json is not None and params_json.strip() != '':
        params = json.loads(params_json)

    recembs_all = []
    lex_all = []
    sizes = []
    for fn in os.listdir(in_path):
        if fn.endswith('.jsonl') or fn.endswith('.jl'):
            path = os.path.join(in_path, fn)
            logger.info('process %s...' % path)
            n = 0
            with io.open(path) as f:
                for line in f:
                    l = line.strip()
                    if l == '' or l[0] == '#':
                        continue
                    jsonld = json.loads(l)
                    recemb, lex = convert_jsonld_to_recemb(jsonld, **params)
                    recembs_all.append(recemb)
                    lex_all.append(lex)
                    n += 1
            logger.info('converted %i records from %s' % (n, path))
            sizes.append((fn, n))
    recemb = Forest.concatenate(recembs_all)
    lex = Lexicon.merge_strings(lex_all)
    recemb.set_lexicon(lex)
    recemb.split_lexicon_to_lexicon_and_lexicon_roots()
    recemb.hashes_to_indices()
    if glove_file is not None and glove_file.strip() != '':
        logger.info('init vecs with glove file: %s...' % glove_file)
        recemb.lexicon.init_vecs_with_glove_file(filename=glove_file, prefix=word_prefix + u'')
    else:
        recemb.lexicon.init_vecs()
    #return recemb, sizes
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    fn = os.path.join(out_path, 'forest')
    recemb.dump(filename=fn)
    recemb.lexicon.dump(filename=fn, strings_only=not recemb.lexicon.has_vecs)
    recemb.lexicon_roots.dump(filename=fn + '.root.id', strings_only=True)
    json.dump(sizes, open(fn+'.sizes.json', 'w'))

    if classes_prefix is not None and classes_prefix.strip() != '':
        classes_ids, classes_strings = recemb.lexicon.get_ids_for_prefix(classes_prefix, add_separator=False)
        save_class_ids(dir_path=fn, prefix_type=classes_prefix, classes_ids=classes_strings,
                       classes_strings=classes_strings)


@plac.annotations(
    out_path=('corpora output path', 'option', 'o', str),
    glove_file=('glove vector file', 'option', 'g', str),
    word_prefix=('prefix of words in lexicon', 'option', 'w', str),
)
def add_glove_vecs(out_path, glove_file, word_prefix=RDF_PREFIXES_MAP[PREFIX_CONLL] + u'WORD='):
    fn = os.path.join(out_path, 'forest')
    lex = Lexicon(filename=fn, load_vecs=False)
    logger.info('init vecs with glove file: %s...' % glove_file)
    lex.init_vecs_with_glove_file(filename=glove_file, prefix=word_prefix + u'')
    lex.dump(fn)


@plac.annotations(
    out_path=('corpora output path', 'option', 'o', str),
    classes_prefix=('prefix for lex entries that will be saved as class ids', 'option', 'c', str),
)
def extract_class_ids(out_path, classes_prefix):
    fn = os.path.join(out_path, 'forest')
    lexicon = Lexicon(filename=fn)
    classes_ids, classes_strings = lexicon.get_ids_for_prefix(classes_prefix, add_separator=False)
    save_class_ids(dir_path=fn, prefix_type=classes_prefix, classes_ids=classes_ids,
                   classes_strings=classes_strings)


def debug_main():
    recemb, lex = convert_jsonld_to_recemb(jsonld=debug_load_dummy_record(p='/mnt/DATA/ML/data/corpora_out/SICK_RDF/spacy/test.jsonl'))
    # visualize
    recemb.set_lexicon(lex)
    recemb.split_lexicon_to_lexicon_and_lexicon_roots()
    recemb.visualize('test.svg')
    forest_str = recemb.get_text_plain()
    print('done')


def debug_main2():
    convert_corpus_jsonld_to_recemb(in_path='/mnt/DATA/ML/data/corpora_out/SICK_RDF/spacy',
                                    out_path='/mnt/DATA/ML/data/corpora_out/SICK_RDF/spacy_recemb')
    #recemb_all.visualize('test.svg')


@plac.annotations(
    merged_forest_path=('path to merged forest', 'option', 'o', str),
    split_count=('count of produced index files', 'option', 'c', int),
    step_root=('root step', 'option', 's', int),
    args='the parameters for the underlying processing method')
def _create_index_files(merged_forest_path, split_count, step_root=1, *args):
    sizes = json.load(open(merged_forest_path+'.sizes.json'))
    start = 0
    for fn, s in sizes:
        end = start + s / step_root
        other_args = ('-o', merged_forest_path, '--split-count', str(split_count if fn != 'test.jsonl' else 1),
                      '--start-root', str(start), '--end-root', str(end), '--step-root', str(step_root),
                      '--suffix', fn.split('.')[0])
        plac.call(create_index_files, args + other_args)
        start = end


@plac.annotations(
    mode=('processing mode', 'positional', None, str, ['CONVERT', 'ADD_VECS', 'CREATE_INDICES', 'EXTRACT_CLASSES']),
    args='the parameters for the underlying processing method')
def main(mode, *args):
    if mode == 'CONVERT':
        plac.call(convert_corpus_jsonld_to_recemb, args)
    elif mode == 'ADD_VECS':
        plac.call(add_glove_vecs, args)
    elif mode == 'CREATE_INDICES':
        plac.call(_create_index_files, args)
    elif mode == 'EXTRACT_CLASSES':
        plac.call(extract_class_ids, args)
    logger.info('done')

    # TODO:
    #  * create index files
    #  *


if __name__ == "__main__":
    plac.call(main)
