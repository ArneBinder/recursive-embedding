from datetime import datetime
import io
import json
import logging
import os
from functools import partial
from multiprocessing import Pool

import numpy as np

import spacy
import plac
from nltk.parse.corenlp import CoreNLPDependencyParser, CoreNLPParser

from sequence_trees import graph_out_from_children_dict, Forest, targets, get_path, get_lca_from_paths
from constants import DTYPE_IDX, PREFIX_REC_EMB, PREFIX_CONLL, PREFIX_NIF, PREFIX_TACRED, PREFIX_SICK, \
    PREFIX_IMDB, PREFIX_SEMEVAL, REC_EMB_GLOBAL_ANNOTATION, REC_EMB_HAS_GLOBAL_ANNOTATION, REC_EMB_RECORD, \
    REC_EMB_HAS_PARSE, REC_EMB_HAS_PARSE_ANNOTATION, REC_EMB_HAS_CONTEXT, REC_EMB_USED_PARSER, \
    REC_EMB_SUFFIX_GLOBAL_ANNOTATION, REC_EMB_SUFFIX_NIF_CONTEXT, NIF_WORD, NIF_NEXT_WORD, NIF_SENTENCE, \
    NIF_NEXT_SENTENCE, NIF_IS_STRING, LOGGING_FORMAT, PREFIX_UNIVERSAL_DEPENDENCIES_ENGLISH, RDF_PREFIXES_MAP, \
    NIF_CONTEXT, SICK_OTHER, JSONLD_ID, JSONLD_TYPE, JSONLD_VALUE, SEMEVAL_SUBJECT, SEMEVAL_OBJECT, TACRED_SUBJECT, \
    TACRED_OBJECT, SEMEVAL_RELATION, TACRED_RELATION, DEBUG, CONLL_WORD, CONLL_EDGE
from lexicon import Lexicon
from corpus import create_index_files, save_class_ids

logger = logging.getLogger('corpus_rdf')
logger.setLevel(logging.DEBUG)
logger_streamhandler = logging.StreamHandler()
logger_streamhandler.setLevel(logging.DEBUG)
logger_streamhandler.setFormatter(logging.Formatter(LOGGING_FORMAT))
logging.getLogger('').addHandler(logger_streamhandler)

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


def parse_spacy_to_conll(text, nlp=spacy.load('en'), do_ner=True):
    doc = nlp(text)
    for sent in doc.sents:
        # initial empty yield to indicate start of a sentence
        yield ()

        for i, word in enumerate(sent):
            if word.head.i == word.i:
                head_idx = 0
            else:
                # head_idx = doc[i].head.i + 1
                head_idx = word.head.i - sent[0].i + 1

            yield (
                # data          conllu      notes
                i + 1,  # ID        There's a word.i attr that's position in *doc*
                word,  # FORM
                word.lemma_,  # LEMMA
                word.pos_,  # UPOS      Coarse-grained tag
                word.tag_,  # XPOS      Fine-grained tag
                word.ent_type_ if word.ent_type_ != u'' and do_ner else None,   # ENTITY TYPE    #u'_',  # FEATS
                head_idx,  # HEAD (ID or 0)
                word.dep_,  # DEPREL    Relation
                None,  # DEPS
                word.idx,  # MISC      character offset
            )


def parse_corenlp_to_conll(text, tokens=None, dep_parser=CoreNLPDependencyParser(url='http://localhost:9000'),
                           ner_tagger=None):
    if tokens is not None:
        raise NotImplementedError('coreNLP parse for list of tokens not implemented')
    else:
        parsed_sents = dep_parser.parse_text(text)

    #template = u'{i}\t{word}\t{lemma}\t{ctag}\t{tag}\t{feats}\t{head}\t{rel}\t_\t{idx}'
    previous_end = 0
    for sent in parsed_sents:
        # initial empty yield to indicate start of a sentence
        yield ()
        i_and_nodes = sorted(sent.nodes.items())
        if ner_tagger is not None:
            tokens_str = [node['word'] for i, node in i_and_nodes if node['tag'] != 'TOP']
            ner_words, ner_tags = zip(*ner_tagger.tag(tokens_str))
            assert len(ner_words) == len(i_and_nodes) - 1, \
                'nbr of tokens returned by NER tagger [%i] does not equal nbr of sentence tokens [%i]' \
                % (len(ner_words), len(i_and_nodes) - 1)
        else:
            ner_tags = [u'O'] * (len(i_and_nodes) - 1)
        for i, node in i_and_nodes:
            if node['tag'] == 'TOP':
                continue
            idx = text.find(node['word'], previous_end)
            # assert idx >= 0, 'word="%s" not found in text="%s"' % (node['word'], text)
            if idx < 0:
                print('WARNING: word="%s" not found in text' % node['word'])
                idx = None
            else:
                previous_end = idx + len(node['word'])
            #l = template.format(i=i, idx=idx, **node)
            #yield l
            yield (
                # data          conllu      notes
                i,  # ID        There's a word.i attr that's position in *doc*
                node[u'word'],  # FORM
                node[u'lemma'],  # LEMMA
                node[u'ctag'],  # UPOS      Coarse-grained tag
                node[u'tag'],  # XPOS      Fine-grained tag
                ner_tags[i-1] if ner_tags[i-1] != u'O' else None,   # ENTITY TYPE    #u'_',  # FEATS
                node[u'head'],  # HEAD (ID or 0)
                node[u'rel'],  # DEPREL    Relation
                None,           # DEPS
                idx,            # MISC      character offset
            )


def record_to_conll(sentence_record, captions, key_mapping):
    selected_entries = {k: sentence_record[key_mapping[k]] for k in key_mapping if key_mapping[k] in sentence_record}
    l = [dict(zip(selected_entries, t)) for t in zip(*selected_entries.values())]

    # initial empty yield to indicate start of a sentence
    yield ()
    for i, d in enumerate(l):
        try:
            # set no ENTITY ("O") to None
            y = [i + 1] + [d.get(c, None) if not (c == 'ENTITY' and d.get(c, None) == 'O') else None for c in captions]
        except Exception as e:
            raise e
        yield y


def convert_conll_to_rdf(conll_data, base_uri=RDF_PREFIXES_MAP[PREFIX_UNIVERSAL_DEPENDENCIES_ENGLISH],
                         # ATTENTION: ID column has to be the first column, but should not occur in  "columns"!
                         columns=('WORD', 'LEMMA', 'UPOS', 'POS', 'ENTITY', 'HEAD', 'EDGE', 'DEPS', 'MISC')):
    res = []
    sent_id = 0
    token_id = 1
    previous_word = None

    def start_sentence(sent_id, _previous_sentence=None):
        _sent_prefix = u'%ss%i_' % (base_uri, sent_id)
        _row_rdf = {JSONLD_ID: _sent_prefix + u'0', JSONLD_TYPE: [NIF_SENTENCE]}
        res.append(_row_rdf)
        if _previous_sentence is not None:
            _previous_sentence[NIF_NEXT_SENTENCE] = [{JSONLD_ID: _row_rdf[JSONLD_ID]}]
        _previous_sentence = _row_rdf
        return _previous_sentence, _sent_prefix

    #previous_sentence, sent_prefix = start_sentence()
    previous_sentence = None
    sent_prefix = None

    for line in conll_data:
        if len(line) == 0 or previous_sentence is None:
            token_id = 1
            previous_word = None
            sent_id += 1
            previous_sentence, sent_prefix = start_sentence(sent_id, previous_sentence)
            # go to next line, if it was a sentence placeholder (not the beginning)
            if len(line) == 0:
                continue

        row = [unicode(e) if e is not None else None for e in line]
        row_dict = {RDF_PREFIXES_MAP[PREFIX_CONLL] + columns[i]:
                        (row[i + 1] if columns[i] != 'HEAD' else (JSONLD_ID, sent_prefix + row[i + 1])) for i, k in
                    enumerate(columns)
                    if i + 1 < len(row) and row[i + 1] is not None}
        row_dict[RDF_PREFIXES_MAP[PREFIX_CONLL] + u'ID'] = row[0]
        row_rdf = _to_rdf(row_dict)
        row_rdf[JSONLD_ID] = sent_prefix + row[0]
        row_rdf[JSONLD_TYPE] = [NIF_WORD]
        res.append(row_rdf)
        if previous_word is not None:
            previous_word[NIF_NEXT_WORD] = [{JSONLD_ID: row_rdf[JSONLD_ID]}]
        previous_word = row_rdf
        token_id += 1

    return res


def _to_rdf(element):
    if isinstance(element, dict):
        res = {k: _to_rdf(element[k]) for k in element}
    elif isinstance(element, list):
        res = [_to_rdf(e) for e in element]
    # tuple indicates marked (@id or @value) entry
    elif isinstance(element, tuple):
        marker, content = element
        assert marker in [JSONLD_VALUE, JSONLD_ID], 'WARNING: Unknown value marker: %s for content: %s. use ' \
                                              '"@value" or "@id" as first tuple entry to mark value as lateral or id.' \
                                              % (marker, content)
        res = [{marker: content}]
    else:
        res = [{JSONLD_VALUE: element}]
    return res


def get_token_ids_in_span(tokens, start, end, idx_key=RDF_PREFIXES_MAP[PREFIX_CONLL] + 'MISC',
                          word_key=CONLL_WORD):
    res = []
    for t in tokens:
        if idx_key in t and word_key in t:
            token_start = int(t[idx_key][0][JSONLD_VALUE])
            token_end = token_start + len(t[word_key][0][JSONLD_VALUE])
            if start <= token_start < end or start < token_end <= end:
                res.append({JSONLD_ID: t[JSONLD_ID]})
    return res


def parse_and_convert_record(record_id,
                             context_string=None,
                             global_annotations=None,
                             token_features=None,
                             character_annotations=None,
                             token_annotations=None,
                             parsers=None,
                             ):
    conll_columns = ('WORD', 'LEMMA', 'UPOS', 'POS', 'ENTITY', 'HEAD', 'EDGE', 'DEPS', 'MISC')
    if parsers is None:
        assert token_features is not None, 'parser==None requires token_features, but it is None'
        conll_lines = list(record_to_conll(token_features, captions=conll_columns,
                                           key_mapping={'WORD': 'token', 'UPOS': 'stanford_pos',
                                                        'HEAD': 'stanford_head', 'EDGE': 'stanford_deprel',
                                                        'ENTITY': 'stanford_ner'}))
        parser_str = u'None'
        # conll_columns = ('index', 'token', 'subj', 'subj_type', 'obj', 'obj_type', 'stanford_pos', 'stanford_ner', 'stanford_deprel', 'stanford_head')
    elif isinstance(parsers[0], spacy.language.Language):
        conll_lines = list(parse_spacy_to_conll(context_string, nlp=parsers[0], do_ner=parsers[1]))
        parser_str = u'%s.%s' % (type(parsers[0]).__module__, type(parsers[0]).__name__)
    elif isinstance(parsers[0], CoreNLPDependencyParser):
        conll_lines = list(parse_corenlp_to_conll(context_string, dep_parser=parsers[0], ner_tagger=parsers[1]))
        parser_str = u'%s.%s' % (type(parsers[0]).__module__, type(parsers[0]).__name__)
    else:
        raise NotImplementedError('parser %s not implemented' % parsers)

    tokens_jsonld = convert_conll_to_rdf(conll_lines, base_uri=record_id + u'#', columns=conll_columns)

    res = {JSONLD_ID: record_id, JSONLD_TYPE: [REC_EMB_RECORD], REC_EMB_HAS_PARSE: tokens_jsonld,
           REC_EMB_USED_PARSER: [{JSONLD_VALUE: parser_str}]}
    if global_annotations is not None:
        global_annotations[JSONLD_ID] = record_id + REC_EMB_SUFFIX_GLOBAL_ANNOTATION
        global_annotations[JSONLD_TYPE] = [REC_EMB_GLOBAL_ANNOTATION]
        res[REC_EMB_HAS_GLOBAL_ANNOTATION] = [global_annotations]
    if context_string is not None:
        res[REC_EMB_HAS_CONTEXT] = [{JSONLD_ID: record_id + REC_EMB_SUFFIX_NIF_CONTEXT,
                                     JSONLD_TYPE: [NIF_CONTEXT],
                                     NIF_IS_STRING: [{JSONLD_VALUE: context_string}]}]

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


def parse_to_rdf(in_path, out_path, reader_rdf, file_names, parser='spacy', no_ner=False):
    t_start = datetime.now()
    logger.info('load parser...')
    if parser is not None:
        if no_ner:
            logger.info('avoid NER')
        if parser.strip() == 'spacy':
            _parsers = (spacy.load('en'), not no_ner)
        elif parser.strip() == 'corenlp':
            if no_ner:
                dep_parser = CoreNLPDependencyParser(url='http://localhost:9000')
                _parsers = (dep_parser, None)
            else:
                # a distinct NER tagger returns more types (and is faster)
                ner_tagger = CoreNLPParser(url='http://localhost:9000', tagtype='ner')
                dep_parser = CoreNLPDependencyParser(url='http://localhost:9000')
                #dep_parser = CoreNLPDependencyParser(url='http://localhost:9000', tagtype='ner')
                #ner_tagger = dep_parser
                _parsers = (dep_parser, ner_tagger)
        else:
            raise NotImplementedError('parser=%s not implemented' % parser)
    else:
        _parsers = None
    logger.info('loaded parser %s' % str(_parsers))

    out_path = os.path.join(out_path, str(parser))
    if no_ner:
        out_path = out_path + '_noner'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    else:
        logger.warning('%s already exists, skip it!' % out_path)
        return

    logger_fh = logging.FileHandler(out_path + '.info.log')
    logger_fh.setLevel(logging.INFO)
    logger_fh.setFormatter(logging.Formatter(LOGGING_FORMAT))
    logging.getLogger('').addHandler(logger_fh)

    logger_fh = logging.FileHandler(out_path + '.debug.log')
    logger_fh.setLevel(logging.DEBUG)
    logger_fh.setFormatter(logging.Formatter(LOGGING_FORMAT))
    logging.getLogger('').addHandler(logger_fh)

    n_failed = {}
    n_total = {}
    for fn in file_names:
        fn_out = os.path.join(out_path, file_names[fn])
        logger.info('process %s, save result to %s' % (fn, fn_out))
        n_failed[fn_out] = 0
        already_processed = {}
        if os.path.exists(fn_out):
            with io.open(fn_out, encoding='utf8') as fout:
                for l in fout.readlines():
                    _l = json.loads(l)
                    already_processed[_l[JSONLD_ID]] = l
        n_total[fn_out] = len(already_processed)
        if len(already_processed) > 0:
            logger.info('found %i already processed records' % len(already_processed))

        with io.open(fn_out, 'w', encoding='utf8') as fout:
            for i, record in enumerate(reader_rdf(in_path, fn)):
                if record['record_id'] in already_processed:
                    parsed_rdf_json = already_processed[record['record_id']]
                else:
                    try:
                        parsed_rdf = parse_and_convert_record(parsers=_parsers, **record)
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
    logger.debug('done (time: %s)' % (datetime.now() - t_start))


def token_id_to_tuple(token_id):
    if token_id is None:
        return None
    return [int(part) for part in token_id.split('#s')[-1].split('_')]


def serialize_jsonld_dict(jsonld, offset=0, sort_map={}, discard_predicates=(), discard_types=(),
                          id_as_value_predicates=(), skip_predicates=(), add_skipped_to_target_predicates=(),
                          swap_predicates=(), revert_predicates=(),
                          offset_predicates={}, replace_literal_predicates={}, min_max_predicates={}):
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
    :param add_skipped_to_target_predicates: add skipped predicates as children of target (not reverted!)
    :param revert_predicates: revert the edges from / to these predicates. If it is also in skip_predicates, create
        (object, subject) instead of (subject, object)
    :param swap_predicates: swap positions in final serialization of subject and object, that are linked by these
        predicates. ATTENTION: works only for collapsed objects (literals)
    :return: types, ids, refs
    """
    ser = []
    ids = {}
    edges = []
    skipped_preds = []
    # consider only first type
    _type = u'' + jsonld[JSONLD_TYPE][0]
    if _type in discard_types:
        return ser, ids, edges, skipped_preds
    ser.append(_type)
    ids[u'' + jsonld[JSONLD_ID]] = offset
    preds = sort_map.get(_type, sorted(jsonld))
    for pred in preds:
        if pred[0] != u'@' and pred in jsonld:
            offsets = offset_predicates.get(pred, (0, 0))
            object_literals = []
            min_id, max_id = min_max_predicates.get(pred, (None, None))
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

            for obj_idx, obj_dict in enumerate(jsonld[pred]):
                # lateral object
                if JSONLD_VALUE in obj_dict:
                    if pred in replace_literal_predicates:
                        obj_dict = jsonld.get(replace_literal_predicates[pred], jsonld[pred])[obj_idx]
                    object_literals.append(obj_dict[JSONLD_VALUE])
                # complex object
                else:
                    if min_id is not None and token_id_to_tuple(obj_dict[JSONLD_ID]) < min_id:
                        continue
                    if max_id is not None and token_id_to_tuple(obj_dict[JSONLD_ID]) > max_id:
                        continue

                    # assert len(object_literals) == 0 or pred in id_as_value_predicates, \
                    #    'laterals (@value) are mixed with complex elements (@type / @id)'
                    if pred in id_as_value_predicates:
                        object_literals.append(obj_dict[JSONLD_ID])
                        continue

                    if idx_edge_source is not None:
                        new_edge = (obj_dict[JSONLD_ID], idx_edge_source) if revert_edge else (
                            idx_edge_source, obj_dict[JSONLD_ID])
                        if offset == idx_edge_source:
                            new_edge = new_edge + offsets
                        else:
                            new_edge = new_edge + (0, offsets[1])
                        edges.append(new_edge)
                        if pred in skip_predicates:
                            skipped_preds.append((pred, new_edge))
                            if pred in add_skipped_to_target_predicates:
                                edges.append((obj_dict[JSONLD_ID], offset + len(ser), 0, 0))
                                ser.append(pred)

                    if JSONLD_TYPE in obj_dict:
                        obj_ser, obj_ids, obj_edges, obj_skipped_preds = serialize_jsonld_dict(obj_dict, offset=len(ser) + offset,
                                                                                               sort_map=sort_map,
                                                                                               discard_predicates=discard_predicates,
                                                                                               discard_types=discard_types,
                                                                                               id_as_value_predicates=id_as_value_predicates,
                                                                                               skip_predicates=skip_predicates,
                                                                                               add_skipped_to_target_predicates=add_skipped_to_target_predicates,
                                                                                               swap_predicates=swap_predicates,
                                                                                               revert_predicates=revert_predicates,
                                                                                               offset_predicates=offset_predicates,
                                                                                               replace_literal_predicates=replace_literal_predicates,
                                                                                               min_max_predicates=min_max_predicates)
                        ser.extend(obj_ser)
                        ids.update(obj_ids)
                        edges.extend(obj_edges)
                        skipped_preds.extend(obj_skipped_preds)

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
                        edges.extend([(idx_edge_source + i, offset) + offsets for i in range(len(new_entries))])
                    else:
                        edges.extend([(offset, idx_edge_source + i) + offsets for i in range(len(new_entries))])
                else:
                    if revert_edge:
                        edges.append((idx_edge_source, offset) + (offsets[0], 0))
                    else:
                        edges.append((offset, idx_edge_source) + (offsets[0], 0))

    return ser, ids, edges, skipped_preds


def serialize_jsonld(jsonld, sort_map=None, restrict_span_with_annots=False, relink_relation=False,
                     offset_predicates={}, use_skipped_predicates_for_target=(),
                     use_skipped_predicates_for_source=(), **kwargs):
    if sort_map is None:
        sort_map = {REC_EMB_RECORD: [REC_EMB_HAS_CONTEXT, REC_EMB_HAS_GLOBAL_ANNOTATION,
                                     REC_EMB_HAS_PARSE_ANNOTATION,  REC_EMB_USED_PARSER, REC_EMB_HAS_PARSE]}
    if not isinstance(jsonld, list):
        jsonld = [jsonld]
    ser, ids, refs = [], {}, {}
    id_indices = []
    for jsonld_dict in jsonld:
        min_max_predicates = {}
        if restrict_span_with_annots:
            assert REC_EMB_HAS_PARSE_ANNOTATION in jsonld_dict, 'restrict_span_with_annots, but no hasParseAnnotation found'
            _subject_ids = jsonld_dict[REC_EMB_HAS_PARSE_ANNOTATION][0].get(SEMEVAL_SUBJECT, []) + \
                          jsonld_dict[REC_EMB_HAS_PARSE_ANNOTATION][0].get(TACRED_SUBJECT, [])
            _object_ids = jsonld_dict[REC_EMB_HAS_PARSE_ANNOTATION][0].get(SEMEVAL_OBJECT, []) + \
                          jsonld_dict[REC_EMB_HAS_PARSE_ANNOTATION][0].get(TACRED_OBJECT, [])
            # get minimal span containing subj and obj
            parse_ids_subj = [token_id_to_tuple(x[JSONLD_ID]) for x in _subject_ids]
            parse_ids_obj = [token_id_to_tuple(x[JSONLD_ID]) for x in _object_ids]
            _sorted = sorted(parse_ids_subj + parse_ids_obj)
            min_max_predicates[REC_EMB_HAS_PARSE] = (_sorted[0], _sorted[-1])
        # see below for: + 1 (offset=len(ser) + 1)
        start_offset = len(ser) + 1
        _ser, _ids, _edges, _skipped_preds = serialize_jsonld_dict(jsonld_dict, offset=start_offset, sort_map=sort_map,
                                                                   min_max_predicates=min_max_predicates,
                                                                   offset_predicates=offset_predicates, **kwargs)
        for pred, (s, t, offset_s, offset_t) in _skipped_preds:
            if pred in use_skipped_predicates_for_target:
                t = _ids.get(t, t)
                assert isinstance(t, int), 'target offset for pred=%s could not be resolved: %s' % (pred, str(t))
                t += offset_t
                if use_skipped_predicates_for_target[pred]:
                    _ser[t - start_offset] = pred
                else:
                    _ser[t - start_offset] += '/' + pred
            if pred in use_skipped_predicates_for_source:
                s = _ids.get(s, s)
                assert isinstance(s, int), 'source offset for pred=%s could not be resolved: %s' % (pred, str(s))
                s += offset_s
                if use_skipped_predicates_for_source[pred]:
                    _ser[s - start_offset] = pred
                else:
                    _ser[s - start_offset] += '/' + pred

        _refs = {}
        for edge in _edges:
            s, t = edge[:2]
            s = _ids.get(s, s)
            t = _ids.get(t, t)
            if not (isinstance(s, int) and isinstance(t, int)):
                # skip edge if it points to unresolved id
                continue
            if len(edge) > 2:
                s = s + edge[2]
                t = t + edge[3]
            _refs.setdefault(s, []).append(t)

        ## insert the id directly after the root (offset +1 (above); _ser.insert(1, _id);  _ids[_id] = len(ser); ...)
        _id = u'' + jsonld_dict[JSONLD_ID]
        _ser.insert(1, _id)
        _ids[_id] = len(ser)
        _refs[len(ser)] = [len(ser) + 1] + _refs[len(ser) + 1]
        del _refs[len(ser) + 1]
        ## insert end
        id_indices.append(len(ser) + 1)

        ser.extend(_ser)
        ids.update(_ids)
        refs.update(_refs)

    idx_hasParse = ser.index(REC_EMB_HAS_PARSE)
    if restrict_span_with_annots:
        parse_annot_indices = refs[idx_hasParse]
        # revert edges in parse_annot_indices
        refs_rev = {}
        for idx_source in parse_annot_indices:
            for idx_target in refs[idx_source]:
                refs_rev.setdefault(idx_target, []).append(idx_source)
        # entries without other incoming entries are root(s)
        roots = [idx for idx in parse_annot_indices if idx not in refs_rev]
        refs[idx_hasParse] = roots
        assert len(roots) > 0, 'no root(s) found'
        #if len(roots) != 1:
        #    logger.warning('wrong number of roots [%i], expected 1' % len(roots))
        graph_out = graph_out_from_children_dict(refs, len(ser))
    else:
        # remove all children of parse, except to last sentence element
        idx_last_sentence = len(ser) - ser[::-1].index(NIF_SENTENCE) - 1
        # link to _last_ sentence
        refs[idx_hasParse] = [idx_last_sentence]

        if relink_relation:
            idx_rel = None
            for idx, s in enumerate(ser):
                if s.startswith(SEMEVAL_RELATION) or s.startswith(TACRED_RELATION):
                    idx_rel = idx
            assert idx_rel is not None, 'relation not found'
            graph_out_dok = graph_out_from_children_dict(refs, len(ser), return_dok=True)
            graph_out = graph_out_dok.tocsc()
            graph_in = graph_out.tocsr()

            indices_subject = targets(graph_in, idx_rel)
            # remove index of rem:hasParseAnnotation
            indices_subject = indices_subject[indices_subject > idx_rel]
            indices_object = targets(graph_out, idx_rel)

            # remove links from / to subj / obj
            graph_out_dok[idx_rel, indices_subject] = False
            graph_out_dok[indices_object, idx_rel] = False
            _graph_out = graph_out_dok.tocsc()
            _graph_in = _graph_out.tocsr()

            paths = {}
            for idx_start in np.concatenate((indices_subject, indices_object)):
                paths[idx_start] = get_path(g=_graph_in, data=ser, idx_start=idx_start, stop_data=REC_EMB_HAS_PARSE)
            idx_subj_lca = get_lca_from_paths([paths[idx] for idx in indices_subject], root=idx_hasParse)
            idx_obj_lca = get_lca_from_paths([paths[idx] for idx in indices_object], root=idx_hasParse)
            for idx_start in [idx_subj_lca, idx_obj_lca]:
                if idx_start not in paths:
                    paths[idx_start] = get_path(g=_graph_in, data=ser, idx_start=idx_start, stop_data=REC_EMB_HAS_PARSE)
            idx_ent_lca = get_lca_from_paths([paths[idx] for idx in [idx_subj_lca, idx_obj_lca]], root=idx_hasParse)
            # link hasParse with LCA of (LCA of subj) & (LCA of obj)
            hasParse_offsets = offset_predicates.get(REC_EMB_HAS_PARSE, (0, 0))
            graph_out_dok[idx_last_sentence, idx_hasParse] = False
            graph_out_dok[idx_ent_lca + hasParse_offsets[1], idx_hasParse] = True
            # link relation with LCA of entries (subj and obj individually)
            graph_out_dok[idx_rel, idx_subj_lca] = True
            graph_out_dok[idx_obj_lca, idx_rel] = True
            graph_out = graph_out_dok.tocsc()
        else:
            graph_out = graph_out_from_children_dict(refs, len(ser))

    return ser, graph_out, id_indices


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
    res = parse_and_convert_record(parsers=parser, **record)
    # res_str = json.dumps(res)
    print(json.dumps(res, indent=2))
    return res


def debug_load_dummy_record(p='/mnt/DATA/ML/data/corpora_out/IMDB_RDF/spacy/test.jsonl'):
    with io.open(p) as f:
        l = f.readline()
    return json.loads(l)


def convert_jsonld_to_recemb(jsonld, **kwargs):

    ser, graph_out, ids_indices = serialize_jsonld(jsonld, **kwargs)

    # create rec-emb
    lex = Lexicon(add_vocab_manual=True)
    data = [Lexicon.hash_string(s) for s in ser]
    lex.add_all(ser)

    recemb = Forest(data=data, data_as_hashes=True, structure=graph_out, root_pos=np.array([0], dtype=DTYPE_IDX))

    return recemb, lex


def process_jsonl_file(fn, in_path, **params):
    recembs_all = []
    lex_all = []
    path = os.path.join(in_path, fn)
    logger.info('process %s...' % path)
    n = 0
    n_failed = 0
    with io.open(path, encoding='utf8') as f:
        for line_i, line in enumerate(f):
            l = line.strip()
            if l == u'' or l[0] == u'#':
                continue
            try:
                jsonld = json.loads(l)
                recemb, lex = convert_jsonld_to_recemb(jsonld, **params)
                recembs_all.append(recemb)
                lex_all.append(lex)
            except Exception as e:
                logger.warning('line %i: failed to process: %s' % (line_i, str(e)))
                n_failed += 1
                continue
            n += 1
    logger.info('successfully converted %i records (%i failed) from %s' % (n, n_failed, path))
    return recembs_all, lex_all, fn, n


@plac.annotations(
    in_path=('corpora input path, should contain .jsonl or .jl files', 'option', 'i', str),
    out_path=('corpora output path', 'option', 'o', str),
    glove_file=('glove vector file', 'option', 'g', str),
    word_prefix=('prefix of words in lexicon', 'option', 'w', str),
    classes_prefix=('prefix for lex entries that will be saved as class ids', 'option', 'c', str),
    min_count=('minimal count a token has to be in the corpus', 'option', 'm', int),
    params_json=('optional parameters as json string', 'option', 'p', str),
    link_via_edges=('tokens via dependency edge nodes', 'flag', 'e', bool),
    mask_with_entity_type=('replace words with entity type, if available', 'flag', 't', bool),
    mask_with_argument_type=('replace words with entity type, if available', 'flag', 'a', bool),
    restrict_span_with_annots=('replace words with entity type, if available', 'flag', 's', bool),
    relink_relation=('re-link relation annotation with LCA of subj / obj entries', 'flag', 'l', bool),
)
def convert_corpus_jsonld_to_recemb(in_path, out_path=None, glove_file='',
                                    word_prefix=CONLL_WORD + u'=', classes_prefix='',
                                    min_count=1, params_json='', link_via_edges=False, mask_with_entity_type=False,
                                    mask_with_argument_type=False,
                                    restrict_span_with_annots=False,
                                    relink_relation=False):
    params = {}
    if params_json is not None and params_json.strip() != '':
        params = json.loads(params_json)

    if 'restrict_span_with_annots' not in params:
        params['restrict_span_with_annots'] = restrict_span_with_annots
    else:
        restrict_span_with_annots = params['restrict_span_with_annots']

    if out_path is None:
        out_path = in_path + '_recemb'
    if mask_with_entity_type:
        out_path += '_ner'
    if mask_with_argument_type:
        out_path += '_arg'
    if restrict_span_with_annots:
        out_path += '_span'
    if link_via_edges:
        out_path += '_edges'
    if relink_relation:
        out_path += '_lca'
    if min_count > 1:
        out_path += '_mc' + str(min_count)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    else:
        if not DEBUG:
            logger.warning('%s already exists, skip it!' % out_path)
            return

    logger_fh = logging.FileHandler(out_path + '.info.log')
    logger_fh.setLevel(logging.INFO)
    logger_fh.setFormatter(logging.Formatter(LOGGING_FORMAT))
    logging.getLogger('').addHandler(logger_fh)

    logger_fh = logging.FileHandler(out_path + '.debug.log')
    logger_fh.setLevel(logging.DEBUG)
    logger_fh.setFormatter(logging.Formatter(LOGGING_FORMAT))
    logging.getLogger('').addHandler(logger_fh)

    # set defaults
    if 'discard_predicates' not in params:
        params['discard_predicates'] = (RDF_PREFIXES_MAP[PREFIX_CONLL] + u'MISC',
                                        RDF_PREFIXES_MAP[PREFIX_CONLL] + u'ID',
                                        RDF_PREFIXES_MAP[PREFIX_CONLL] + u'LEMMA',
                                        # dont mistake for CONLL_POS (= ...UPOS)!
                                        RDF_PREFIXES_MAP[PREFIX_CONLL] + u'POS',
                                        RDF_PREFIXES_MAP[PREFIX_CONLL] + u'ENTITY',
                                        NIF_NEXT_WORD,
                                        REC_EMB_HAS_CONTEXT,
                                        # RDF_PREFIXES_MAP[PREFIX_REC_EMB] + u'hasParseAnnotation',
                                        NIF_IS_STRING,
                                        # not needed, just spams the lexicon
                                        SICK_OTHER,
                                        )
    if 'discard_types' not in params:
        params['discard_types'] = (NIF_CONTEXT,)
    if 'id_as_value_predicates' not in params:
        params['id_as_value_predicates'] = (SICK_OTHER,)
    if 'skip_predicates' not in params:
        params['skip_predicates'] = (RDF_PREFIXES_MAP[PREFIX_CONLL] + u'HEAD',
                                     NIF_NEXT_SENTENCE,
                                     SEMEVAL_SUBJECT,
                                     SEMEVAL_OBJECT,
                                     TACRED_SUBJECT,
                                     TACRED_OBJECT,
                                     )
    if 'add_skipped_to_target_predicates' not in params:
        params['add_skipped_to_target_predicates'] = (SEMEVAL_SUBJECT,
                                                      SEMEVAL_OBJECT,
                                                      TACRED_SUBJECT,
                                                      TACRED_OBJECT,
                                                      )
    if 'revert_predicates' not in params:
        params['revert_predicates'] = (RDF_PREFIXES_MAP[PREFIX_CONLL] + u'HEAD',
                                       NIF_NEXT_SENTENCE,
                                       SEMEVAL_SUBJECT,
                                       TACRED_SUBJECT
                                       )
    if 'swap_predicates' not in params:
        params['swap_predicates'] = (CONLL_WORD,)
    if 'offset_predicates' not in params:
        params['offset_predicates'] = {}
    if 'replace_literal_predicates' not in params:
        params['replace_literal_predicates'] = {}
    if link_via_edges:
        logger.info('link via dependency edges')
        params['revert_predicates'] = params['revert_predicates'] + (CONLL_EDGE,)
        params['offset_predicates'][RDF_PREFIXES_MAP[PREFIX_CONLL] + u'HEAD'] = (0, 1)
        params['offset_predicates'][REC_EMB_HAS_PARSE] = (0, 1)
    if mask_with_entity_type:
        logger.info('replace words with entity type, if available')
        params['replace_literal_predicates'][CONLL_WORD] = RDF_PREFIXES_MAP[PREFIX_CONLL] + u'ENTITY'
    if mask_with_argument_type:
        if mask_with_entity_type:
            logger.info('append argument type (subj / obj) to words, if available')
        else:
            logger.info('replace words with argument type (subj / obj), if available')
        # Set to True, if replacing is requested. Otherwise the argument type will be appended.
        # Replace only, if words were replaced by entity types.
        # ATTENTION: For Semeval, just a few arguments are entities!
        params['use_skipped_predicates_for_target'] = {SEMEVAL_OBJECT: not mask_with_entity_type,
                                                       TACRED_OBJECT: not mask_with_entity_type}
        params['use_skipped_predicates_for_source'] = {SEMEVAL_SUBJECT: not mask_with_entity_type,
                                                       TACRED_SUBJECT: not mask_with_entity_type}

    if restrict_span_with_annots:
        logger.info('restrict span with annots')
        assert not relink_relation, 'can not relink relation if restrict_span_with_annots'
        params['discard_types'] = params['discard_types'] + (NIF_SENTENCE,)

    recembs_all = []
    lex_all = []
    sizes = []
    _process = partial(process_jsonl_file, in_path=in_path, relink_relation=relink_relation, **params)
    fn_jsonl = [fn for fn in os.listdir(in_path) if fn.endswith('.jsonl') or fn.endswith('.jl')]
    if DEBUG:
        res_files = [_process(fn) for fn in fn_jsonl]
    else:
        p = Pool(len(fn_jsonl))
        res_files = p.map(_process, fn_jsonl)
    recemb_files, lex_files, fn_files, sizes_files = zip(*res_files)
    for i, s in enumerate(sizes_files):
        sizes.append((fn_files[i], s))
        recembs_all.extend(recemb_files[i])
        lex_all.extend(lex_files[i])

    logger.debug('concatenate data...')
    recemb = Forest.concatenate(recembs_all)
    logger.debug('merge lexica...')
    lex = Lexicon.merge_strings(lex_all)
    recemb.set_lexicon(lex)
    logger.debug('split lexicon into data and ids...')
    recemb.split_lexicon_to_lexicon_and_lexicon_roots()

    if glove_file is not None and glove_file.strip() != '':
        logger.info('init vecs with glove file: %s...' % glove_file)
        recemb.lexicon.init_vecs_with_glove_file(filename=glove_file, prefix=word_prefix + u'')
    else:
        #logger.warning('Do not filter with min_count because no vecs are added.')
        #recemb.lexicon.init_vecs()
        import spacy
        logger.debug('load spacy...')
        nlp = spacy.load('en')
        logger.info('init vecs from spacy...')
        recemb.lexicon.init_vecs_with_spacy_vocab(vocab=nlp.vocab, prefix=word_prefix + u'')
    if min_count > 1:
        logger.debug('filter lexicon by min_count=%i...' % min_count)
        hashes_classes = None
        if classes_prefix is not None and classes_prefix.strip() != '':
            hashes_classes = recemb.lexicon.get_hashes_for_prefix(prefix=classes_prefix.strip())
            logger.debug('keep %i hashes for classes_prefix=%s' % (len(hashes_classes), classes_prefix.strip()))
        recemb.lexicon.shrink_via_min_count(data_hashes=recemb.data, keep_hashes=hashes_classes, min_count=min_count)
    logger.debug('convert data (hashes to lexicon indices)...')
    recemb.hashes_to_indices()

    fn = os.path.join(out_path, 'forest')
    recemb.dump(filename=fn)
    recemb.lexicon.dump(filename=fn, strings_only=not recemb.lexicon.has_vecs)
    recemb.lexicon_roots.dump(filename=fn + '.root.id', strings_only=True)
    json.dump(sizes, open(fn+'.sizes.json', 'w'))

    if classes_prefix is not None and classes_prefix.strip() != '':
        classes_ids, classes_strings = recemb.lexicon.get_ids_for_prefix(classes_prefix, add_separator=False)
        save_class_ids(dir_path=fn, prefix_type=classes_prefix, classes_ids=classes_ids,
                       classes_strings=classes_strings)


@plac.annotations(
    out_path=('corpora output path', 'option', 'o', str),
    glove_file=('glove vector file', 'option', 'g', str),
    word_prefix=('prefix of words in lexicon', 'option', 'w', str),
)
def add_glove_vecs(out_path, glove_file, word_prefix=CONLL_WORD + u'='):
    fn = os.path.join(out_path, 'forest')
    lex = Lexicon(filename=fn, load_vecs=False)
    logger.info('init vecs with glove file: %s...' % glove_file)
    lex.init_vecs_with_glove_file(filename=glove_file, prefix=word_prefix + u'')
    lex.dump(fn)


@plac.annotations(
    out_path=('corpora output path', 'option', 'o', str),
    word_prefix=('prefix of words in lexicon', 'option', 'w', str),
)
def add_spacy_vecs(out_path, word_prefix=CONLL_WORD + u'=',):
    import spacy
    logger.debug('load spacy...')
    nlp = spacy.load('en')
    fn = os.path.join(out_path, 'forest')
    lex = Lexicon(filename=fn, load_vecs=False)
    logger.info('init vecs from spacy...')
    lex.init_vecs_with_spacy_vocab(vocab=nlp.vocab, prefix=word_prefix)
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
    # add dev indices to train
    if 'dev.jsonl' in dict(sizes):
        split_count /= 2
        logger.info('dataset contains dev set, half the split_count to %i' % (split_count))
    start = 0
    for fn, s in sizes:
        end = start + s / step_root
        suffix = fn.split('.')[0]
        other_args = ('-o', merged_forest_path, '--split-count', str(split_count if fn != 'test.jsonl' else 1),
                      '--start-root', str(start), '--end-root', str(end), '--step-root', str(step_root),
                      '--suffix', suffix)
        plac.call(create_index_files, args + other_args)
        start = end
        if fn == 'dev.jsonl':
            for i in range(split_count):
                fn_src = 'idx.dev.%i.npy' % i
                fn_tar = 'idx.train.%i.npy' % (i + split_count)
                logger.debug('rename %s to %s' % (fn_src, fn_tar))
                os.rename(merged_forest_path+'.' + fn_src, merged_forest_path+'.'+fn_tar)


@plac.annotations(
    mode=('processing mode', 'positional', None, str, ['CONVERT', 'ADD_VECS', 'CREATE_INDICES', 'EXTRACT_CLASSES']),
    args='the parameters for the underlying processing method')
def main(mode, *args):
    t_start = datetime.now()
    if mode == 'CONVERT':
        plac.call(convert_corpus_jsonld_to_recemb, args)
    elif mode == 'ADD_VECS':
        plac.call(add_glove_vecs, args)
    elif mode == 'CREATE_INDICES':
        plac.call(_create_index_files, args)
    elif mode == 'EXTRACT_CLASSES':
        plac.call(extract_class_ids, args)
    logger.debug('done (time: %s)' % (datetime.now() - t_start))


if __name__ == "__main__":
    plac.call(main)
