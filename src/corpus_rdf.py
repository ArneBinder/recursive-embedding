import json

import spacy
#from datetime import datetime
#import plac
from nltk.parse.corenlp import CoreNLPDependencyParser

from constants import TYPE_CONTEXT


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
REC_EMB_PARSE = PREFIX_REC_EMB + u'Parse'
REC_EMB_HAS_PARSE_ANNOTATION = PREFIX_REC_EMB + u'hasParseAnnotation'
REC_EMB_HAS_CONTEXT = PREFIX_REC_EMB + u'hasContext'
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
                           PREFIX_SICK + u'vocab#entailment_judgment': [{u'@value': u'NEUTRAL'}]}
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
                                         PREFIX_SEMEVAL + u'vocab#subj': (73, 86),
                                         PREFIX_SEMEVAL + u'vocab#obj': (98, 106),
                                        }]
SEMEVAL_RECORD = {'record_id': SEMEVAL_RECORD_ID,
                  'context_string': u"The system as described above has its greatest application in an arrayed configuration of antenna elements.",
                  'character_annotations': SEMEVAL_RECORD_CHARACTER_ANNOTATIONS,
                  }


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
        yield '\n'


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
                res.append([{u'@id': t[u'@id']}])
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

    res = {u'@id': record_id, u'@type': [REC_EMB_RECORD], REC_EMB_PARSE: tokens_jsonld}
    if global_annotations is not None:
        global_annotations[u'@id'] = record_id + u'#ga'
        global_annotations[u'@type'] = [REC_EMB_GLOBAL_ANNOTATION]
        res[REC_EMB_HAS_GLOBAL_ANNOTATION] = [global_annotations]
    if context_string is not None:
        res[REC_EMB_HAS_CONTEXT] = [{u'@id': record_id + u'?nif=context',
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


def main():
    print('load parser...')
    #parser = spacy.load('en')
    parser = CoreNLPDependencyParser(url='http://localhost:9000')
    #parser = None
    print('loaded parser: %s' % str(type(parser)))
    #record = TACRED_RECORD
    record = SEMEVAL_RECORD
    #record = SICK_RECORD
    #record = IMDB_RECORD
    res = parse_and_convert_record(parser=parser, **record)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
