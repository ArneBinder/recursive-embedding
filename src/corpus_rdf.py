import json
import numpy as np

import spacy
#from datetime import datetime
#import plac
from nltk.parse.corenlp import CoreNLPDependencyParser

from sequence_trees import graph_out_from_children_dict, Forest
from constants import TYPE_CONTEXT, DTYPE_IDX
from lexicon import Lexicon

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

PREFIX_REC_EMB = u'https://github.com/ArneBinder/recursive-embedding#'
PREFIX_CONLL = u'http://ufal.mff.cuni.cz/conll2009-st/task-description.html#'
PREFIX_NIF = u'http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#'

PREFIX_TACRED = u'https://catalog.ldc.upenn.edu/LDC2018T24/'
PREFIX_SICK = u'http://clic.cimec.unitn.it/composes/sick.html/'
PREFIX_IMDB = u'http://ai.stanford.edu/~amaas/data/sentiment/'
# correct url: http://semeval2.fbk.eu/semeval2.php?location=tasks#T11
PREFIX_SEMEVAL = u'http://semeval2.fbk.eu/task8/'

REC_EMB_GLOBAL_ANNOTATION = PREFIX_REC_EMB + u'GlobalAnnotation'
REC_EMB_HAS_GLOBAL_ANNOTATION = PREFIX_REC_EMB + u'hasGlobalAnnotation'
REC_EMB_RECORD = PREFIX_REC_EMB + u'Record'
REC_EMB_HAS_PARSE = PREFIX_REC_EMB + u'hasParse'
REC_EMB_HAS_PARSE_ANNOTATION = PREFIX_REC_EMB + u'hasParseAnnotation'
REC_EMB_HAS_CONTEXT = PREFIX_REC_EMB + u'hasContext'
REC_EMB_USED_PARSER = PREFIX_REC_EMB + u'usedParser'
REC_EMB_SUFFIX_GLOBAL_ANNOTATION = u'#GlobalAnnotation'
REC_EMB_SUFFIX_NIF_CONTEXT = u'?nif=context'

NIF_WORD = PREFIX_NIF + u'Word'
NIF_NEXT_WORD = PREFIX_NIF + u'nextWord'
NIF_SENTENCE = PREFIX_NIF + u'Sentence'
NIF_NEXT_SENTENCE = PREFIX_NIF + u'nextSentence'
NIF_IS_STRING = PREFIX_NIF + u'isString'


TACRED_RECORD_JSON = u'[{"id": "e7798fb926b9403cfcd2", "docid": "APW_ENG_20101103.0539", "relation": "per:title", "token": ["At", "the", "same", "time", ",", "Chief", "Financial", "Officer", "Douglas", "Flint", "will", "become", "chairman", ",", "succeeding", "Stephen", "Green", "who", "is", "leaving", "to", "take", "a", "government", "job", "."], "subj_start": 8, "subj_end": 9, "obj_start": 12, "obj_end": 12, "subj_type": "PERSON", "obj_type": "TITLE", "stanford_pos": ["IN", "DT", "JJ", "NN", ",", "NNP", "NNP", "NNP", "NNP", "NNP", "MD", "VB", "NN", ",", "VBG", "NNP", "NNP", "WP", "VBZ", "VBG", "TO", "VB", "DT", "NN", "NN", "."], "stanford_ner": ["O", "O", "O", "O", "O", "O", "O", "O", "PERSON", "PERSON", "O", "O", "O", "O", "O", "PERSON", "PERSON", "O", "O", "O", "O", "O", "O", "O", "O", "O"], "stanford_head": [4, 4, 4, 12, 12, 10, 10, 10, 10, 12, 12, 0, 12, 12, 12, 17, 15, 20, 20, 17, 22, 20, 25, 25, 22, 12], "stanford_deprel": ["case", "det", "amod", "nmod", "punct", "compound", "compound", "compound", "compound", "nsubj", "aux", "ROOT", "xcomp", "punct", "xcomp", "compound", "dobj", "nsubj", "aux", "acl:relcl", "mark", "xcomp", "det", "compound", "dobj", "punct"]}, {"id": "e779865fb96bbbcc4ca4", "docid": "APW_ENG_20080229.1401.LDC2009T13", "relation": "no_relation", "token": ["U.S.", "District", "Court", "Judge", "Jeffrey", "White", "in", "mid-February", "issued", "an", "injunction", "against", "Wikileaks", "after", "the", "Zurich-based", "Bank", "Julius", "Baer", "accused", "the", "site", "of", "posting", "sensitive", "account", "information", "stolen", "by", "a", "disgruntled", "former", "employee", "."], "subj_start": 17, "subj_end": 18, "obj_start": 4, "obj_end": 5, "subj_type": "PERSON", "obj_type": "PERSON", "stanford_pos": ["NNP", "NNP", "NNP", "NNP", "NNP", "NNP", "IN", "NNP", "VBD", "DT", "NN", "IN", "NNP", "IN", "DT", "JJ", "NNP", "NNP", "NNP", "VBD", "DT", "NN", "IN", "VBG", "JJ", "NN", "NN", "VBN", "IN", "DT", "JJ", "JJ", "NN", "."], "stanford_ner": ["LOCATION", "O", "O", "O", "PERSON", "PERSON", "O", "O", "O", "O", "O", "O", "ORGANIZATION", "O", "O", "MISC", "O", "PERSON", "PERSON", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O"], "stanford_head": [6, 6, 6, 6, 6, 9, 8, 6, 0, 11, 9, 13, 11, 20, 19, 19, 19, 19, 20, 9, 22, 20, 24, 20, 27, 27, 24, 27, 33, 33, 33, 33, 28, 9], "stanford_deprel": ["compound", "compound", "compound", "compound", "compound", "nsubj", "case", "nmod", "ROOT", "det", "dobj", "case", "nmod", "mark", "det", "amod", "compound", "compound", "nsubj", "advcl", "det", "dobj", "mark", "advcl", "amod", "compound", "dobj", "acl", "case", "det", "amod", "amod", "nmod", "punct"]}, {"id": "e7798ae9c0adbcdc81e7", "docid": "APW_ENG_20090707.0488", "relation": "per:city_of_death", "token": ["PARIS", "2009-07-07", "11:07:32", "UTC", "French", "media", "earlier", "reported", "that", "Montcourt", ",", "ranked", "119", ",", "was", "found", "dead", "by", "his", "girlfriend", "in", "the", "stairwell", "of", "his", "Paris", "apartment", "."], "subj_start": 9, "subj_end": 9, "obj_start": 0, "obj_end": 0, "subj_type": "PERSON", "obj_type": "CITY", "stanford_pos": ["NNP", "CD", "CD", "NNP", "NNP", "NNS", "RBR", "VBD", "IN", "NNP", ",", "VBD", "CD", ",", "VBD", "VBN", "JJ", "IN", "PRP$", "NN", "IN", "DT", "NN", "IN", "PRP$", "NNP", "NN", "."], "stanford_ner": ["LOCATION", "TIME", "TIME", "TIME", "MISC", "O", "O", "O", "O", "PERSON", "O", "O", "NUMBER", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "LOCATION", "O", "O"], "stanford_head": [6, 6, 6, 6, 6, 8, 8, 0, 16, 16, 10, 10, 12, 10, 16, 8, 16, 20, 20, 17, 23, 23, 16, 27, 27, 27, 23, 8], "stanford_deprel": ["compound", "nummod", "nummod", "compound", "compound", "nsubj", "advmod", "ROOT", "mark", "nsubjpass", "punct", "acl", "dobj", "punct", "auxpass", "ccomp", "xcomp", "case", "nmod:poss", "nmod", "case", "det", "nmod", "case", "nmod:poss", "compound", "nmod", "punct"]}]'
TACRED_RECORD_TOKEN_FEATURES = {"token": ["At", "the", "same", "time", ",", "Chief", "Financial", "Officer", "Douglas", "Flint", "will", "become", "chairman", ",", "succeeding", "Stephen", "Green", "who", "is", "leaving", "to", "take", "a", "government", "job", "." ], "stanford_pos": [ "IN", "DT", "JJ", "NN", ",", "NNP", "NNP", "NNP", "NNP", "NNP", "MD", "VB", "NN", ",", "VBG", "NNP", "NNP", "WP", "VBZ", "VBG", "TO", "VB", "DT", "NN", "NN", "." ], "stanford_ner": [ "O", "O", "O", "O", "O", "O", "O", "O", "PERSON", "PERSON", "O", "O", "O", "O", "O", "PERSON", "PERSON", "O", "O", "O", "O", "O", "O", "O", "O", "O" ], "stanford_deprel": [ "case", "det", "amod", "nmod", "punct", "compound", "compound", "compound", "compound", "nsubj", "aux", "ROOT", "xcomp", "punct", "xcomp", "compound", "dobj", "nsubj", "aux", "acl:relcl", "mark", "xcomp", "det", "compound", "dobj", "punct" ], "stanford_head": [ 4, 4, 4, 12, 12, 10, 10, 10, 10, 12, 12, 0, 12, 12, 12, 17, 15, 20, 20, 17, 22, 20, 25, 25, 22, 12 ]}

TACRED_RECORD_ID = PREFIX_TACRED + u'train/e7798fb926b9403cfcd2'
TACRED_RECORD_TOKEN_ANNOTATIONS = [{u'@id': TACRED_RECORD_ID + u'#r1',
                                    u'@type': [PREFIX_TACRED + u'vocab#relation:per:title'],
                                    PREFIX_TACRED + u'vocab#subj': [{u'@id': TACRED_RECORD_ID + u'#s1_9'}, {u'@id': TACRED_RECORD_ID + u'#s1_10'}],
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

# TODO: implement reader for SICK, IMDB, SEMEVAL, TARCED

# TODO: implement sentence_processors


def parse_spacy_to_conll(text, nlp=spacy.load('en')):
    doc = nlp(text)
    for sent in doc.sents:

        for i, word in enumerate(sent):
            if word.head is word:
                head_idx = 0
            else:
                #head_idx = doc[i].head.i + 1
                head_idx = word.head.i - sent[0].i + 1

            yield "%d\t%s\t%s\t%s\t%s\t%s\t%d\t%s\t%s\t%i" % (
                # data          conllu      notes
                i + 1,          # ID        There's a word.i attr that's position in *doc*
                word,           # FORM
                word.lemma_,    # LEMMA
                word.pos_,      # UPOS      Coarse-grained tag
                word.tag_,      # XPOS      Fine-grained tag
                '_',            # FEATS
                head_idx,       # HEAD (ID or 0)
                word.dep_,      # DEPREL    Relation
                '_',            # DEPS
                word.idx,       # MISC      character offset
            )
        yield ''


def parse_corenlp_to_conll(text, tokens=None, dep_parser=CoreNLPDependencyParser(url='http://localhost:9000')):
    if tokens is not None:
        raise NotImplementedError('coreNLP parse for list of tokens not implemented')
    else:
        parsed_sents = dep_parser.parse_text(text)

    template = '{i}\t{word}\t{lemma}\t{ctag}\t{tag}\t{feats}\t{head}\t{rel}\t_\t{idx}'
    previous_end = 0
    for sent in parsed_sents:
        for i, node in sorted(sent.nodes.items()):
            if node['tag'] == 'TOP':
                continue
            idx = text.find(node['word'], previous_end)
            #assert idx >= 0, 'word="%s" not found in text="%s"' % (node['word'], text)
            if idx < 0:
                print('WARNING: word="%s" not found in text="%s"' % (node['word'], text))
                idx = '_'
            else:
                previous_end = idx + len(node['word'])
            l = template.format(i=i, idx=idx, **node)
            yield l
        yield ''


def record_to_conll(sentence_record, captions, key_mapping):
    selected_entries = {k: sentence_record[key_mapping[k]] for k in key_mapping if key_mapping[k] in sentence_record}
    l = [dict(zip(selected_entries, t)) for t in zip(*selected_entries.values())]
    for i, d in enumerate(l):
        y = '\t'.join([str(i + 1)] + [str(d[c]) if c in d else '_' for c in captions])
        yield y
    yield ''


def convert_conll_to_rdf(conll_data, base_uri=u'https://github.com/UniversalDependencies/UD_English#',
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
            row_dict = {PREFIX_CONLL + columns[i]:
                            ('' if columns[i] != 'HEAD' else 'id@' + sent_prefix) + row[i+1] for i, k in enumerate(columns)
                        if i+1 < len(row) and row[i+1] != '_'}
            row_dict[PREFIX_CONLL + u'ID'] = row[0]
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
    else:
        # check, if value is marked as
        try:
            parts = element.split(u'@')
            if len(parts) == 1 or parts[0].strip() == u'value':
                res = [{u'@value': element}]
            else:
                if parts[0] == u'id':
                    res = [{u'@' + parts[0]: u'@'.join(parts[1:])}]
                # elif parts[0] == u'type':
                # does not work like this
                #    res =
                else:
                    raise AssertionError(
                        'unknown value marker: %s. use "value@" or "id@" as prefix to mark value as value or id.' %
                        parts[0])
        # if element is not a string (no split method)
        except AttributeError:
            res = [{u'@value': element}]

    #if rdf_type is not None:
    #    res[u'@type'] = rdf_type
    return res


def get_token_ids_in_span(tokens, start, end, idx_key=PREFIX_CONLL + 'MISC', word_key=PREFIX_CONLL + 'WORD'):
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
        conll_lines = list(parse_spacy_to_conll(context_string, parser))
    elif isinstance(parser, CoreNLPDependencyParser):
        conll_lines = list(parse_corenlp_to_conll(context_string, dep_parser=parser))
    elif parser is None:
        assert token_features is not None, 'parser==None requires token_features, but it is None'
        conll_lines = list(record_to_conll(token_features, captions=conll_columns,
                                           key_mapping={'WORD': 'token', 'POS': 'stanford_pos',
                                                      'HEAD': 'stanford_head', 'EDGE': 'stanford_deprel'}))
        #conll_columns = ('index', 'token', 'subj', 'subj_type', 'obj', 'obj_type', 'stanford_pos', 'stanford_ner', 'stanford_deprel', 'stanford_head')
    else:
        raise NotImplementedError('parser %s not implemented' % parser)

    print('convert conll to rdf...')
    tokens_jsonld = convert_conll_to_rdf('\n'.join(conll_lines), base_uri=record_id + u'#', columns=conll_columns)

    res = {u'@id': record_id, u'@type': [REC_EMB_RECORD], REC_EMB_HAS_PARSE: tokens_jsonld,
           REC_EMB_USED_PARSER: [{u'@value': u'%s.%s' % (type(parser).__module__, type(parser).__name__)}]}
    if global_annotations is not None:
        global_annotations[u'@id'] = record_id + REC_EMB_SUFFIX_GLOBAL_ANNOTATION
        global_annotations[u'@type'] = [REC_EMB_GLOBAL_ANNOTATION]
        res[REC_EMB_HAS_GLOBAL_ANNOTATION] = [global_annotations]
    if context_string is not None:
        res[REC_EMB_HAS_CONTEXT] = [{u'@id': record_id + REC_EMB_SUFFIX_NIF_CONTEXT,
                                    u'@type': [TYPE_CONTEXT],
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


def serialize_jsonld_dict(jsonld, discard_predicates=(), offset=0, sort_map={}, save_id_predicates=()):
    """
    Serializes a json-ld dict object.
    Returns a list of all types (additional types are created from value entries),
    a mapping from all ids to the index of the corresponding type int type list and,
    a mapping from all elements that reference other elements as: source index to types list -> id
    ATTENTION: only the first type per entry is used!
    :param jsonld:
    :return: types, ids, refs
    """
    _type = u'' + jsonld[u'@type'][0]
    ser = [_type]
    ids = {u'' + jsonld[u'@id']: offset}
    refs = {}
    preds = sort_map.get(_type, sorted(jsonld))
    for pred in preds:
        if pred not in discard_predicates and pred[0] != u'@' and pred in jsonld:
            idx_pred = len(ser)
            value_objects = []
            ser.append(u'' + pred)
            for obj_dict in jsonld[pred]:
                if u'@type' in obj_dict:
                    assert len(value_objects) == 0, 'laterals (@value) are mixed with complex elements (@type / @id)'
                    obj_ser, obj_ids, ob_refs = serialize_jsonld_dict(obj_dict, discard_predicates=discard_predicates,
                                                                      offset=len(ser)+offset, sort_map=sort_map,
                                                                      save_id_predicates=save_id_predicates)
                    ser.extend(obj_ser)
                    ids.update(obj_ids)
                    refs.update(ob_refs)
                    refs.setdefault(idx_pred + offset, []).append(obj_dict[u'@id'])
                elif u'@id' in obj_dict:
                    assert len(value_objects) == 0, 'laterals (@value) are mixed with complex elements (@type / @id)'
                    if pred in save_id_predicates:
                        refs.setdefault(idx_pred + offset, []).append(len(ser) + offset)
                        ser.append(u'' + obj_dict[u'@id'])
                    else:
                        refs.setdefault(idx_pred + offset, []).append(obj_dict[u'@id'])
                elif u'@value' in obj_dict:
                    assert len(ser) == idx_pred + 1, 'laterals (@value) are mixed with complex elements (@type / @id)'
                    value_objects.append(obj_dict[u'@value'])
                else:
                    raise AssertionError('unknown situation')
            # collapse laterals with predicate
            if len(value_objects) > 0:
                del ser[-1]
                new_entries = [u'%s=%s' % (pred, v) for v in value_objects]
                ser.extend(new_entries)
                refs.setdefault(offset, []).extend(list(range(idx_pred + offset, idx_pred + offset + len(new_entries))))
            else:
                refs.setdefault(offset, []).append(idx_pred + offset)

    return ser, ids, refs


def serialize_jsonld(jsonld, discard_predicates=(), sort_map=None, save_id_predicates=()):
    if sort_map is None:
        sort_map = {REC_EMB_RECORD: [REC_EMB_HAS_CONTEXT, REC_EMB_HAS_GLOBAL_ANNOTATION,
                                     REC_EMB_HAS_PARSE_ANNOTATION, REC_EMB_HAS_PARSE]}
    if not isinstance(jsonld, list):
        jsonld = [jsonld]
    ser, ids, refs = [], {}, {}
    for jsonld_dict in jsonld:
        ## insert the id directly after the root (offset +1; _ser.insert(1, _id);  _ids[_id] = len(ser); ...)
        _ser, _ids, _refs = serialize_jsonld_dict(jsonld_dict, discard_predicates=discard_predicates,
                                                  offset=len(ser) + 1, sort_map=sort_map,
                                                  save_id_predicates=save_id_predicates)
        _id = u'' + jsonld_dict[u'@id']
        _ser.insert(1, _id)
        _ids[_id] = len(ser)
        _refs[len(ser)] = [len(ser) + 1] + _refs[len(ser)+1]
        del _refs[len(ser)+1]
        ## insert end

        for r_key in _refs:
            r_targets = _refs[r_key]
            #_refs[r_key] = [_ids.get(r, r) for r in r_targets]
            new_r_targets = []
            for i, r_t in enumerate(r_targets):
                if isinstance(r_t, int):
                    new_r_targets.append(r_t)
                elif r_t in _ids:
                    new_r_targets.append(_ids[r_t])
                # append not-resolvable refs to predicate
                else:
                    _ser[r_key] = u'%s=%s' % (_ser[r_key], r_t)

            _refs[r_key] = new_r_targets
        ser.extend(_ser)
        ids.update(_ids)
        refs.update(_refs)

    return ser, refs


def main():
    print('load parser...')
    #parser = spacy.load('en')
    parser = CoreNLPDependencyParser(url='http://localhost:9000')
    #parser = None
    print('loaded parser: %s' % str(type(parser)))
    #record = TACRED_RECORD
    #record = SEMEVAL_RECORD
    record = SICK_RECORD
    #record = IMDB_RECORD
    res = parse_and_convert_record(parser=parser, **record)
    res_str = json.dumps(res)
    print(json.dumps(res, indent=2))


def main2():
    #dummy_jsonld = {"https://github.com/ArneBinder/recursive-embedding#hasParseAnnotation": [{"http://semeval2.fbk.eu/task8/vocab#subj": [{"@id": "http://semeval2.fbk.eu/task8/TRAIN_FILE.TXT/1#s1_13"}, {"@id": "http://semeval2.fbk.eu/task8/TRAIN_FILE.TXT/1#s1_14"}], "@id": "http://semeval2.fbk.eu/task8/TRAIN_FILE.TXT/1#r1", "@type": ["http://semeval2.fbk.eu/task8/vocab#relation:Component-Whole(e2,e1)"], "http://semeval2.fbk.eu/task8/vocab#obj": [{"@id": "http://semeval2.fbk.eu/task8/TRAIN_FILE.TXT/1#s1_16"}]}], "https://github.com/ArneBinder/recursive-embedding#hasContext": [{"http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#isString": [{"@value": "The system as described above has its greatest application in an arrayed configuration of antenna elements."}], "@id": "http://semeval2.fbk.eu/task8/TRAIN_FILE.TXT/1?nif=context", "@type": ["http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#Context"]}], "@id": "http://semeval2.fbk.eu/task8/TRAIN_FILE.TXT/1", "@type": ["https://github.com/ArneBinder/recursive-embedding#Record"], "https://github.com/ArneBinder/recursive-embedding#hasParse": [{"@id": "http://semeval2.fbk.eu/task8/TRAIN_FILE.TXT/1#s1_0", "@type": ["http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#Sentence"]}, {"http://ufal.mff.cuni.cz/conll2009-st/task-description.html#ID": [{"@value": "1"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#EDGE": [{"@value": "det"}], "http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#nextWord": [{"@id": "http://semeval2.fbk.eu/task8/TRAIN_FILE.TXT/1#s1_2"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#UPOS": [{"@value": "DT"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#POS": [{"@value": "DT"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#LEMMA": [{"@value": "the"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#MISC": [{"@value": "0"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#HEAD": [{"@id": "http://semeval2.fbk.eu/task8/TRAIN_FILE.TXT/1#s1_2"}], "@id": "http://semeval2.fbk.eu/task8/TRAIN_FILE.TXT/1#s1_1", "@type": ["http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#Word"], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#WORD": [{"@value": "The"}]}, {"http://ufal.mff.cuni.cz/conll2009-st/task-description.html#ID": [{"@value": "2"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#EDGE": [{"@value": "nsubj"}], "http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#nextWord": [{"@id": "http://semeval2.fbk.eu/task8/TRAIN_FILE.TXT/1#s1_3"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#UPOS": [{"@value": "NN"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#POS": [{"@value": "NN"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#LEMMA": [{"@value": "system"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#MISC": [{"@value": "4"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#HEAD": [{"@id": "http://semeval2.fbk.eu/task8/TRAIN_FILE.TXT/1#s1_6"}], "@id": "http://semeval2.fbk.eu/task8/TRAIN_FILE.TXT/1#s1_2", "@type": ["http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#Word"], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#WORD": [{"@value": "system"}]}, {"http://ufal.mff.cuni.cz/conll2009-st/task-description.html#ID": [{"@value": "3"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#EDGE": [{"@value": "mark"}], "http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#nextWord": [{"@id": "http://semeval2.fbk.eu/task8/TRAIN_FILE.TXT/1#s1_4"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#UPOS": [{"@value": "IN"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#POS": [{"@value": "IN"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#LEMMA": [{"@value": "as"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#MISC": [{"@value": "11"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#HEAD": [{"@id": "http://semeval2.fbk.eu/task8/TRAIN_FILE.TXT/1#s1_4"}], "@id": "http://semeval2.fbk.eu/task8/TRAIN_FILE.TXT/1#s1_3", "@type": ["http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#Word"], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#WORD": [{"@value": "as"}]}, {"http://ufal.mff.cuni.cz/conll2009-st/task-description.html#ID": [{"@value": "4"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#EDGE": [{"@value": "acl"}], "http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#nextWord": [{"@id": "http://semeval2.fbk.eu/task8/TRAIN_FILE.TXT/1#s1_5"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#UPOS": [{"@value": "VBN"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#POS": [{"@value": "VBN"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#LEMMA": [{"@value": "describe"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#MISC": [{"@value": "14"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#HEAD": [{"@id": "http://semeval2.fbk.eu/task8/TRAIN_FILE.TXT/1#s1_2"}], "@id": "http://semeval2.fbk.eu/task8/TRAIN_FILE.TXT/1#s1_4", "@type": ["http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#Word"], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#WORD": [{"@value": "described"}]}, {"http://ufal.mff.cuni.cz/conll2009-st/task-description.html#ID": [{"@value": "5"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#EDGE": [{"@value": "advmod"}], "http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#nextWord": [{"@id": "http://semeval2.fbk.eu/task8/TRAIN_FILE.TXT/1#s1_6"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#UPOS": [{"@value": "IN"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#POS": [{"@value": "IN"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#LEMMA": [{"@value": "above"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#MISC": [{"@value": "24"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#HEAD": [{"@id": "http://semeval2.fbk.eu/task8/TRAIN_FILE.TXT/1#s1_4"}], "@id": "http://semeval2.fbk.eu/task8/TRAIN_FILE.TXT/1#s1_5", "@type": ["http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#Word"], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#WORD": [{"@value": "above"}]}, {"http://ufal.mff.cuni.cz/conll2009-st/task-description.html#ID": [{"@value": "6"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#EDGE": [{"@value": "ROOT"}], "http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#nextWord": [{"@id": "http://semeval2.fbk.eu/task8/TRAIN_FILE.TXT/1#s1_7"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#UPOS": [{"@value": "VBZ"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#POS": [{"@value": "VBZ"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#LEMMA": [{"@value": "have"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#MISC": [{"@value": "30"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#HEAD": [{"@id": "http://semeval2.fbk.eu/task8/TRAIN_FILE.TXT/1#s1_0"}], "@id": "http://semeval2.fbk.eu/task8/TRAIN_FILE.TXT/1#s1_6", "@type": ["http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#Word"], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#WORD": [{"@value": "has"}]}, {"http://ufal.mff.cuni.cz/conll2009-st/task-description.html#ID": [{"@value": "7"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#EDGE": [{"@value": "nmod:poss"}], "http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#nextWord": [{"@id": "http://semeval2.fbk.eu/task8/TRAIN_FILE.TXT/1#s1_8"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#UPOS": [{"@value": "PRP$"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#POS": [{"@value": "PRP$"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#LEMMA": [{"@value": "its"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#MISC": [{"@value": "34"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#HEAD": [{"@id": "http://semeval2.fbk.eu/task8/TRAIN_FILE.TXT/1#s1_9"}], "@id": "http://semeval2.fbk.eu/task8/TRAIN_FILE.TXT/1#s1_7", "@type": ["http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#Word"], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#WORD": [{"@value": "its"}]}, {"http://ufal.mff.cuni.cz/conll2009-st/task-description.html#ID": [{"@value": "8"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#EDGE": [{"@value": "amod"}], "http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#nextWord": [{"@id": "http://semeval2.fbk.eu/task8/TRAIN_FILE.TXT/1#s1_9"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#UPOS": [{"@value": "JJS"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#POS": [{"@value": "JJS"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#LEMMA": [{"@value": "greatest"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#MISC": [{"@value": "38"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#HEAD": [{"@id": "http://semeval2.fbk.eu/task8/TRAIN_FILE.TXT/1#s1_9"}], "@id": "http://semeval2.fbk.eu/task8/TRAIN_FILE.TXT/1#s1_8", "@type": ["http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#Word"], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#WORD": [{"@value": "greatest"}]}, {"http://ufal.mff.cuni.cz/conll2009-st/task-description.html#ID": [{"@value": "9"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#EDGE": [{"@value": "dobj"}], "http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#nextWord": [{"@id": "http://semeval2.fbk.eu/task8/TRAIN_FILE.TXT/1#s1_10"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#UPOS": [{"@value": "NN"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#POS": [{"@value": "NN"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#LEMMA": [{"@value": "application"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#MISC": [{"@value": "47"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#HEAD": [{"@id": "http://semeval2.fbk.eu/task8/TRAIN_FILE.TXT/1#s1_6"}], "@id": "http://semeval2.fbk.eu/task8/TRAIN_FILE.TXT/1#s1_9", "@type": ["http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#Word"], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#WORD": [{"@value": "application"}]}, {"http://ufal.mff.cuni.cz/conll2009-st/task-description.html#ID": [{"@value": "10"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#EDGE": [{"@value": "case"}], "http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#nextWord": [{"@id": "http://semeval2.fbk.eu/task8/TRAIN_FILE.TXT/1#s1_11"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#UPOS": [{"@value": "IN"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#POS": [{"@value": "IN"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#LEMMA": [{"@value": "in"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#MISC": [{"@value": "59"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#HEAD": [{"@id": "http://semeval2.fbk.eu/task8/TRAIN_FILE.TXT/1#s1_13"}], "@id": "http://semeval2.fbk.eu/task8/TRAIN_FILE.TXT/1#s1_10", "@type": ["http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#Word"], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#WORD": [{"@value": "in"}]}, {"http://ufal.mff.cuni.cz/conll2009-st/task-description.html#ID": [{"@value": "11"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#EDGE": [{"@value": "det"}], "http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#nextWord": [{"@id": "http://semeval2.fbk.eu/task8/TRAIN_FILE.TXT/1#s1_12"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#UPOS": [{"@value": "DT"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#POS": [{"@value": "DT"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#LEMMA": [{"@value": "a"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#MISC": [{"@value": "62"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#HEAD": [{"@id": "http://semeval2.fbk.eu/task8/TRAIN_FILE.TXT/1#s1_13"}], "@id": "http://semeval2.fbk.eu/task8/TRAIN_FILE.TXT/1#s1_11", "@type": ["http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#Word"], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#WORD": [{"@value": "an"}]}, {"http://ufal.mff.cuni.cz/conll2009-st/task-description.html#ID": [{"@value": "12"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#EDGE": [{"@value": "amod"}], "http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#nextWord": [{"@id": "http://semeval2.fbk.eu/task8/TRAIN_FILE.TXT/1#s1_13"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#UPOS": [{"@value": "VBN"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#POS": [{"@value": "VBN"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#LEMMA": [{"@value": "array"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#MISC": [{"@value": "65"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#HEAD": [{"@id": "http://semeval2.fbk.eu/task8/TRAIN_FILE.TXT/1#s1_13"}], "@id": "http://semeval2.fbk.eu/task8/TRAIN_FILE.TXT/1#s1_12", "@type": ["http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#Word"], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#WORD": [{"@value": "arrayed"}]}, {"http://ufal.mff.cuni.cz/conll2009-st/task-description.html#ID": [{"@value": "13"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#EDGE": [{"@value": "nmod"}], "http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#nextWord": [{"@id": "http://semeval2.fbk.eu/task8/TRAIN_FILE.TXT/1#s1_14"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#UPOS": [{"@value": "NN"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#POS": [{"@value": "NN"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#LEMMA": [{"@value": "configuration"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#MISC": [{"@value": "73"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#HEAD": [{"@id": "http://semeval2.fbk.eu/task8/TRAIN_FILE.TXT/1#s1_9"}], "@id": "http://semeval2.fbk.eu/task8/TRAIN_FILE.TXT/1#s1_13", "@type": ["http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#Word"], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#WORD": [{"@value": "configuration"}]}, {"http://ufal.mff.cuni.cz/conll2009-st/task-description.html#ID": [{"@value": "14"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#EDGE": [{"@value": "case"}], "http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#nextWord": [{"@id": "http://semeval2.fbk.eu/task8/TRAIN_FILE.TXT/1#s1_15"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#UPOS": [{"@value": "IN"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#POS": [{"@value": "IN"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#LEMMA": [{"@value": "of"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#MISC": [{"@value": "87"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#HEAD": [{"@id": "http://semeval2.fbk.eu/task8/TRAIN_FILE.TXT/1#s1_16"}], "@id": "http://semeval2.fbk.eu/task8/TRAIN_FILE.TXT/1#s1_14", "@type": ["http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#Word"], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#WORD": [{"@value": "of"}]}, {"http://ufal.mff.cuni.cz/conll2009-st/task-description.html#ID": [{"@value": "15"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#EDGE": [{"@value": "compound"}], "http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#nextWord": [{"@id": "http://semeval2.fbk.eu/task8/TRAIN_FILE.TXT/1#s1_16"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#UPOS": [{"@value": "NN"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#POS": [{"@value": "NN"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#LEMMA": [{"@value": "antenna"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#MISC": [{"@value": "90"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#HEAD": [{"@id": "http://semeval2.fbk.eu/task8/TRAIN_FILE.TXT/1#s1_16"}], "@id": "http://semeval2.fbk.eu/task8/TRAIN_FILE.TXT/1#s1_15", "@type": ["http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#Word"], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#WORD": [{"@value": "antenna"}]}, {"http://ufal.mff.cuni.cz/conll2009-st/task-description.html#ID": [{"@value": "16"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#EDGE": [{"@value": "nmod"}], "http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#nextWord": [{"@id": "http://semeval2.fbk.eu/task8/TRAIN_FILE.TXT/1#s1_17"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#UPOS": [{"@value": "NNS"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#POS": [{"@value": "NNS"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#LEMMA": [{"@value": "element"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#MISC": [{"@value": "98"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#HEAD": [{"@id": "http://semeval2.fbk.eu/task8/TRAIN_FILE.TXT/1#s1_13"}], "@id": "http://semeval2.fbk.eu/task8/TRAIN_FILE.TXT/1#s1_16", "@type": ["http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#Word"], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#WORD": [{"@value": "elements"}]}, {"http://ufal.mff.cuni.cz/conll2009-st/task-description.html#ID": [{"@value": "17"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#EDGE": [{"@value": "punct"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#UPOS": [{"@value": "."}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#POS": [{"@value": "."}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#LEMMA": [{"@value": "."}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#MISC": [{"@value": "106"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#HEAD": [{"@id": "http://semeval2.fbk.eu/task8/TRAIN_FILE.TXT/1#s1_6"}], "@id": "http://semeval2.fbk.eu/task8/TRAIN_FILE.TXT/1#s1_17", "@type": ["http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#Word"], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#WORD": [{"@value": "."}]}]}
    dummy_jsonld = {"https://github.com/ArneBinder/recursive-embedding#usedParser": [{"@value": "spacy.lang.en.English"}], "https://github.com/ArneBinder/recursive-embedding#hasParse": [{"@id": "http://clic.cimec.unitn.it/composes/sick.html/SICK_train.txt/1/A#s1_0", "@type": ["http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#Sentence"]}, {"http://ufal.mff.cuni.cz/conll2009-st/task-description.html#ID": [{"@value": "1"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#EDGE": [{"@value": "det"}], "http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#nextWord": [{"@id": "http://clic.cimec.unitn.it/composes/sick.html/SICK_train.txt/1/A#s1_2"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#UPOS": [{"@value": "DET"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#POS": [{"@value": "DT"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#LEMMA": [{"@value": "a"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#MISC": [{"@value": "0"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#HEAD": [{"@id": "http://clic.cimec.unitn.it/composes/sick.html/SICK_train.txt/1/A#s1_2"}], "@id": "http://clic.cimec.unitn.it/composes/sick.html/SICK_train.txt/1/A#s1_1", "@type": ["http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#Word"], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#WORD": [{"@value": "A"}]}, {"http://ufal.mff.cuni.cz/conll2009-st/task-description.html#ID": [{"@value": "2"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#EDGE": [{"@value": "nsubj"}], "http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#nextWord": [{"@id": "http://clic.cimec.unitn.it/composes/sick.html/SICK_train.txt/1/A#s1_3"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#UPOS": [{"@value": "NOUN"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#POS": [{"@value": "NN"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#LEMMA": [{"@value": "group"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#MISC": [{"@value": "2"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#HEAD": [{"@id": "http://clic.cimec.unitn.it/composes/sick.html/SICK_train.txt/1/A#s1_6"}], "@id": "http://clic.cimec.unitn.it/composes/sick.html/SICK_train.txt/1/A#s1_2", "@type": ["http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#Word"], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#WORD": [{"@value": "group"}]}, {"http://ufal.mff.cuni.cz/conll2009-st/task-description.html#ID": [{"@value": "3"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#EDGE": [{"@value": "prep"}], "http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#nextWord": [{"@id": "http://clic.cimec.unitn.it/composes/sick.html/SICK_train.txt/1/A#s1_4"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#UPOS": [{"@value": "ADP"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#POS": [{"@value": "IN"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#LEMMA": [{"@value": "of"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#MISC": [{"@value": "8"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#HEAD": [{"@id": "http://clic.cimec.unitn.it/composes/sick.html/SICK_train.txt/1/A#s1_2"}], "@id": "http://clic.cimec.unitn.it/composes/sick.html/SICK_train.txt/1/A#s1_3", "@type": ["http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#Word"], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#WORD": [{"@value": "of"}]}, {"http://ufal.mff.cuni.cz/conll2009-st/task-description.html#ID": [{"@value": "4"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#EDGE": [{"@value": "pobj"}], "http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#nextWord": [{"@id": "http://clic.cimec.unitn.it/composes/sick.html/SICK_train.txt/1/A#s1_5"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#UPOS": [{"@value": "NOUN"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#POS": [{"@value": "NNS"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#LEMMA": [{"@value": "kid"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#MISC": [{"@value": "11"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#HEAD": [{"@id": "http://clic.cimec.unitn.it/composes/sick.html/SICK_train.txt/1/A#s1_3"}], "@id": "http://clic.cimec.unitn.it/composes/sick.html/SICK_train.txt/1/A#s1_4", "@type": ["http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#Word"], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#WORD": [{"@value": "kids"}]}, {"http://ufal.mff.cuni.cz/conll2009-st/task-description.html#ID": [{"@value": "5"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#EDGE": [{"@value": "aux"}], "http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#nextWord": [{"@id": "http://clic.cimec.unitn.it/composes/sick.html/SICK_train.txt/1/A#s1_6"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#UPOS": [{"@value": "VERB"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#POS": [{"@value": "VBZ"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#LEMMA": [{"@value": "be"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#MISC": [{"@value": "16"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#HEAD": [{"@id": "http://clic.cimec.unitn.it/composes/sick.html/SICK_train.txt/1/A#s1_6"}], "@id": "http://clic.cimec.unitn.it/composes/sick.html/SICK_train.txt/1/A#s1_5", "@type": ["http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#Word"], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#WORD": [{"@value": "is"}]}, {"http://ufal.mff.cuni.cz/conll2009-st/task-description.html#ID": [{"@value": "6"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#EDGE": [{"@value": "ROOT"}], "http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#nextWord": [{"@id": "http://clic.cimec.unitn.it/composes/sick.html/SICK_train.txt/1/A#s1_7"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#UPOS": [{"@value": "VERB"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#POS": [{"@value": "VBG"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#LEMMA": [{"@value": "play"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#MISC": [{"@value": "19"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#HEAD": [{"@id": "http://clic.cimec.unitn.it/composes/sick.html/SICK_train.txt/1/A#s1_6"}], "@id": "http://clic.cimec.unitn.it/composes/sick.html/SICK_train.txt/1/A#s1_6", "@type": ["http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#Word"], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#WORD": [{"@value": "playing"}]}, {"http://ufal.mff.cuni.cz/conll2009-st/task-description.html#ID": [{"@value": "7"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#EDGE": [{"@value": "prep"}], "http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#nextWord": [{"@id": "http://clic.cimec.unitn.it/composes/sick.html/SICK_train.txt/1/A#s1_8"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#UPOS": [{"@value": "ADP"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#POS": [{"@value": "IN"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#LEMMA": [{"@value": "in"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#MISC": [{"@value": "27"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#HEAD": [{"@id": "http://clic.cimec.unitn.it/composes/sick.html/SICK_train.txt/1/A#s1_6"}], "@id": "http://clic.cimec.unitn.it/composes/sick.html/SICK_train.txt/1/A#s1_7", "@type": ["http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#Word"], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#WORD": [{"@value": "in"}]}, {"http://ufal.mff.cuni.cz/conll2009-st/task-description.html#ID": [{"@value": "8"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#EDGE": [{"@value": "det"}], "http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#nextWord": [{"@id": "http://clic.cimec.unitn.it/composes/sick.html/SICK_train.txt/1/A#s1_9"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#UPOS": [{"@value": "DET"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#POS": [{"@value": "DT"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#LEMMA": [{"@value": "a"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#MISC": [{"@value": "30"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#HEAD": [{"@id": "http://clic.cimec.unitn.it/composes/sick.html/SICK_train.txt/1/A#s1_9"}], "@id": "http://clic.cimec.unitn.it/composes/sick.html/SICK_train.txt/1/A#s1_8", "@type": ["http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#Word"], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#WORD": [{"@value": "a"}]}, {"http://ufal.mff.cuni.cz/conll2009-st/task-description.html#ID": [{"@value": "9"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#EDGE": [{"@value": "pobj"}], "http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#nextWord": [{"@id": "http://clic.cimec.unitn.it/composes/sick.html/SICK_train.txt/1/A#s1_10"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#UPOS": [{"@value": "NOUN"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#POS": [{"@value": "NN"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#LEMMA": [{"@value": "yard"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#MISC": [{"@value": "32"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#HEAD": [{"@id": "http://clic.cimec.unitn.it/composes/sick.html/SICK_train.txt/1/A#s1_7"}], "@id": "http://clic.cimec.unitn.it/composes/sick.html/SICK_train.txt/1/A#s1_9", "@type": ["http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#Word"], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#WORD": [{"@value": "yard"}]}, {"http://ufal.mff.cuni.cz/conll2009-st/task-description.html#ID": [{"@value": "10"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#EDGE": [{"@value": "cc"}], "http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#nextWord": [{"@id": "http://clic.cimec.unitn.it/composes/sick.html/SICK_train.txt/1/A#s1_11"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#UPOS": [{"@value": "CCONJ"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#POS": [{"@value": "CC"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#LEMMA": [{"@value": "and"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#MISC": [{"@value": "37"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#HEAD": [{"@id": "http://clic.cimec.unitn.it/composes/sick.html/SICK_train.txt/1/A#s1_6"}], "@id": "http://clic.cimec.unitn.it/composes/sick.html/SICK_train.txt/1/A#s1_10", "@type": ["http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#Word"], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#WORD": [{"@value": "and"}]}, {"http://ufal.mff.cuni.cz/conll2009-st/task-description.html#ID": [{"@value": "11"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#EDGE": [{"@value": "det"}], "http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#nextWord": [{"@id": "http://clic.cimec.unitn.it/composes/sick.html/SICK_train.txt/1/A#s1_12"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#UPOS": [{"@value": "DET"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#POS": [{"@value": "DT"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#LEMMA": [{"@value": "an"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#MISC": [{"@value": "41"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#HEAD": [{"@id": "http://clic.cimec.unitn.it/composes/sick.html/SICK_train.txt/1/A#s1_13"}], "@id": "http://clic.cimec.unitn.it/composes/sick.html/SICK_train.txt/1/A#s1_11", "@type": ["http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#Word"], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#WORD": [{"@value": "an"}]}, {"http://ufal.mff.cuni.cz/conll2009-st/task-description.html#ID": [{"@value": "12"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#EDGE": [{"@value": "amod"}], "http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#nextWord": [{"@id": "http://clic.cimec.unitn.it/composes/sick.html/SICK_train.txt/1/A#s1_13"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#UPOS": [{"@value": "ADJ"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#POS": [{"@value": "JJ"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#LEMMA": [{"@value": "old"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#MISC": [{"@value": "44"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#HEAD": [{"@id": "http://clic.cimec.unitn.it/composes/sick.html/SICK_train.txt/1/A#s1_13"}], "@id": "http://clic.cimec.unitn.it/composes/sick.html/SICK_train.txt/1/A#s1_12", "@type": ["http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#Word"], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#WORD": [{"@value": "old"}]}, {"http://ufal.mff.cuni.cz/conll2009-st/task-description.html#ID": [{"@value": "13"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#EDGE": [{"@value": "nsubj"}], "http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#nextWord": [{"@id": "http://clic.cimec.unitn.it/composes/sick.html/SICK_train.txt/1/A#s1_14"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#UPOS": [{"@value": "NOUN"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#POS": [{"@value": "NN"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#LEMMA": [{"@value": "man"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#MISC": [{"@value": "48"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#HEAD": [{"@id": "http://clic.cimec.unitn.it/composes/sick.html/SICK_train.txt/1/A#s1_15"}], "@id": "http://clic.cimec.unitn.it/composes/sick.html/SICK_train.txt/1/A#s1_13", "@type": ["http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#Word"], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#WORD": [{"@value": "man"}]}, {"http://ufal.mff.cuni.cz/conll2009-st/task-description.html#ID": [{"@value": "14"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#EDGE": [{"@value": "aux"}], "http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#nextWord": [{"@id": "http://clic.cimec.unitn.it/composes/sick.html/SICK_train.txt/1/A#s1_15"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#UPOS": [{"@value": "VERB"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#POS": [{"@value": "VBZ"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#LEMMA": [{"@value": "be"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#MISC": [{"@value": "52"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#HEAD": [{"@id": "http://clic.cimec.unitn.it/composes/sick.html/SICK_train.txt/1/A#s1_15"}], "@id": "http://clic.cimec.unitn.it/composes/sick.html/SICK_train.txt/1/A#s1_14", "@type": ["http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#Word"], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#WORD": [{"@value": "is"}]}, {"http://ufal.mff.cuni.cz/conll2009-st/task-description.html#ID": [{"@value": "15"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#EDGE": [{"@value": "conj"}], "http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#nextWord": [{"@id": "http://clic.cimec.unitn.it/composes/sick.html/SICK_train.txt/1/A#s1_16"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#UPOS": [{"@value": "VERB"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#POS": [{"@value": "VBG"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#LEMMA": [{"@value": "stand"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#MISC": [{"@value": "55"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#HEAD": [{"@id": "http://clic.cimec.unitn.it/composes/sick.html/SICK_train.txt/1/A#s1_6"}], "@id": "http://clic.cimec.unitn.it/composes/sick.html/SICK_train.txt/1/A#s1_15", "@type": ["http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#Word"], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#WORD": [{"@value": "standing"}]}, {"http://ufal.mff.cuni.cz/conll2009-st/task-description.html#ID": [{"@value": "16"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#EDGE": [{"@value": "prep"}], "http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#nextWord": [{"@id": "http://clic.cimec.unitn.it/composes/sick.html/SICK_train.txt/1/A#s1_17"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#UPOS": [{"@value": "ADP"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#POS": [{"@value": "IN"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#LEMMA": [{"@value": "in"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#MISC": [{"@value": "64"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#HEAD": [{"@id": "http://clic.cimec.unitn.it/composes/sick.html/SICK_train.txt/1/A#s1_15"}], "@id": "http://clic.cimec.unitn.it/composes/sick.html/SICK_train.txt/1/A#s1_16", "@type": ["http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#Word"], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#WORD": [{"@value": "in"}]}, {"http://ufal.mff.cuni.cz/conll2009-st/task-description.html#ID": [{"@value": "17"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#EDGE": [{"@value": "det"}], "http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#nextWord": [{"@id": "http://clic.cimec.unitn.it/composes/sick.html/SICK_train.txt/1/A#s1_18"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#UPOS": [{"@value": "DET"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#POS": [{"@value": "DT"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#LEMMA": [{"@value": "the"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#MISC": [{"@value": "67"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#HEAD": [{"@id": "http://clic.cimec.unitn.it/composes/sick.html/SICK_train.txt/1/A#s1_18"}], "@id": "http://clic.cimec.unitn.it/composes/sick.html/SICK_train.txt/1/A#s1_17", "@type": ["http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#Word"], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#WORD": [{"@value": "the"}]}, {"http://ufal.mff.cuni.cz/conll2009-st/task-description.html#ID": [{"@value": "18"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#EDGE": [{"@value": "pobj"}], "http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#nextWord": [{"@id": "http://clic.cimec.unitn.it/composes/sick.html/SICK_train.txt/1/A#s1_19"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#UPOS": [{"@value": "NOUN"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#POS": [{"@value": "NN"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#LEMMA": [{"@value": "background"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#MISC": [{"@value": "71"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#HEAD": [{"@id": "http://clic.cimec.unitn.it/composes/sick.html/SICK_train.txt/1/A#s1_16"}], "@id": "http://clic.cimec.unitn.it/composes/sick.html/SICK_train.txt/1/A#s1_18", "@type": ["http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#Word"], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#WORD": [{"@value": "background"}]}, {"http://ufal.mff.cuni.cz/conll2009-st/task-description.html#ID": [{"@value": "19"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#EDGE": [{"@value": "punct"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#UPOS": [{"@value": "PUNCT"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#POS": [{"@value": "."}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#LEMMA": [{"@value": "."}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#MISC": [{"@value": "81"}], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#HEAD": [{"@id": "http://clic.cimec.unitn.it/composes/sick.html/SICK_train.txt/1/A#s1_15"}], "@id": "http://clic.cimec.unitn.it/composes/sick.html/SICK_train.txt/1/A#s1_19", "@type": ["http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#Word"], "http://ufal.mff.cuni.cz/conll2009-st/task-description.html#WORD": [{"@value": "."}]}], "https://github.com/ArneBinder/recursive-embedding#hasContext": [{"http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#isString": [{"@value": "A group of kids is playing in a yard and an old man is standing in the background."}], "@id": "http://clic.cimec.unitn.it/composes/sick.html/SICK_train.txt/1/A?nif=context", "@type": ["http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#Context"]}], "https://github.com/ArneBinder/recursive-embedding#hasGlobalAnnotation": [{"http://clic.cimec.unitn.it/composes/sick.html/vocab#entailment_judgment": [{"@value": "NEUTRAL"}], "@id": "http://clic.cimec.unitn.it/composes/sick.html/SICK_train.txt/1/A#ga", "@type": ["https://github.com/ArneBinder/recursive-embedding#GlobalAnnotation"], "http://clic.cimec.unitn.it/composes/sick.html/vocab#relatedness_score": [{"@value": 4.5}], "http://clic.cimec.unitn.it/composes/sick.html/vocab#other": [{"@id": "http://clic.cimec.unitn.it/composes/sick.html/SICK_train.txt/1/B"}]}], "@id": "http://clic.cimec.unitn.it/composes/sick.html/SICK_train.txt/1/A", "@type": ["https://github.com/ArneBinder/recursive-embedding#Record"]}
    ser, refs = serialize_jsonld(dummy_jsonld,
                                 discard_predicates=(PREFIX_CONLL + u'MISC', PREFIX_CONLL + u'ID',
                                                     PREFIX_CONLL + u'LEMMA', PREFIX_CONLL + u'POS',
                                                     PREFIX_CONLL + u'HEAD',
                                                     PREFIX_NIF + u'nextWord', PREFIX_NIF + u'nextSentence',
                                                     PREFIX_REC_EMB + u'hasContext'),
                                 save_id_predicates=(u'http://clic.cimec.unitn.it/composes/sick.html/vocab#other')
                                 )

    # create rec-emb
    lex = Lexicon()
    lex_ids = Lexicon()
    lex.add_all([ser[0]] + ser[2:])
    lex_ids.add(ser[1])
    data = [lex.get_d(s=s, data_as_hashes=False) if s in lex else -lex_ids.get_d(s=s, data_as_hashes=False) -1 for s in ser]
    graph_out = graph_out_from_children_dict(refs, len(ser))
    forest = Forest(data=data, data_as_hashes=False, lexicon=lex, lexicon_roots=lex_ids, structure=graph_out, root_pos=np.array([1], dtype=DTYPE_IDX))
    # visualize
    forest.visualize('test.svg')
    forest_str = forest.get_text_plain()
    print('done')


if __name__ == "__main__":
    main2()
