
import numpy as np

LOGGING_FORMAT = '%(asctime)s %(levelname)s %(message)s'

###################### OLD TYPES ###########################

SEPARATOR = u'/'

# BASE TYPES
TYPE_REF = u'http://www.w3.org/2005/11/its/rdf#taIdentRef'
TYPE_DBPEDIA_RESOURCE = u'http://dbpedia.org/resource'
TYPE_CONTEXT = u'http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#Context'
TYPE_PARAGRAPH = u'http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#Paragraph'
TYPE_TITLE = u'http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#Title'
TYPE_SECTION = u'http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#Section'
TYPE_SENTENCE = u'http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#Sentence'
TYPE_PMID = u'http://id.nlm.nih.gov/pubmed/pmid'
TYPE_LEXEME = u'http://purl.org/olia/olia.owl#Lexeme'
TYPE_DEPENDENCY_RELATION = u'http://purl.org/olia/olia-top.owl#DependencyRelation'
#TYPE_MORPHOSYNTACTIC_CATEGORY = u'http://purl.org/olia/olia-top.owl#MorphosyntacticCategory'
#TYPE_BASE_FORM = u'http://purl.org/olia/olia.owl#BaseForm'
TYPE_LEMMA = u'http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#lemma'
TYPE_POS_TAG = u'http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#posTag'
TYPE_NAMED_ENTITY = u'http://purl.org/olia/olia.owl#NamedEntity'
TYPE_PHRASE = u'http://purl.org/olia/olia.owl#Phrase'
TYPE_ID = u'http://www.w3.org/2005/11/its/rdf#id'
TYPE_NIF = u'http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#'
TYPE_MESH = u"http://id.nlm.nih.gov/mesh"

# TODO: choose better!
# TASK SPECIFIC TYPES. ATTENTION: may be used in file names!
# IMDB SENTIMENT
TYPE_RELATEDNESS_SCORE = u'RELATEDNESS_SCORE'
TYPE_ENTAILMENT = u'ENTAILMENT'
# IMDB SENTIMENT
TYPE_POLARITY = u"POLARITY"
TYPE_RATING = u"RATING"
# SEMEVAL2010TASK8 relation extraction
TYPE_RELATION = u"RELATION"

TYPE_DATASET = u"DATASET"

TYPE_ARTIFICIAL = u"ARTIFICIAL"

TYPE_TOKEN = u"TOKEN"

# used just in filter_and_shorten_label
BASE_TYPES = [TYPE_REF, TYPE_DBPEDIA_RESOURCE, TYPE_CONTEXT, TYPE_PARAGRAPH, TYPE_TITLE, TYPE_SECTION, TYPE_SECTION,
              TYPE_SENTENCE, TYPE_PMID, TYPE_LEXEME, TYPE_DEPENDENCY_RELATION, TYPE_LEMMA, TYPE_POS_TAG,
              TYPE_NAMED_ENTITY, TYPE_PHRASE, TYPE_ID, TYPE_NIF, TYPE_RELATEDNESS_SCORE, TYPE_ENTAILMENT,
              TYPE_POLARITY, TYPE_RATING, TYPE_RELATION, TYPE_DATASET]

STRUCTURE_TYPES = [TYPE_PARAGRAPH, TYPE_TITLE, TYPE_SECTION, TYPE_SENTENCE, TYPE_TOKEN]

# CONSTRUCTED TYPES
TYPE_REF_SEEALSO = TYPE_REF + SEPARATOR + u'seeAlso'
TYPE_SECTION_SEEALSO = TYPE_SECTION + SEPARATOR + u'seeAlso'
TYPE_SECTION_ABSTRACT = TYPE_SECTION + SEPARATOR + u'abstract'
TYPE_REF_TUPLE = TYPE_REF + SEPARATOR + u'other'
#TYPE_RELATION_TYPE = TYPE_RELATION + SEPARATOR + u'TYPE'
#TYPE_RELATION_DIRECTION = TYPE_RELATION + SEPARATOR + u'DIRECTION'
TYPE_RELATION_FORWARD = TYPE_RELATION + SEPARATOR + u'FW'
TYPE_RELATION_BACKWARD = TYPE_RELATION + SEPARATOR + u'BW'

LINK_TYPES = [TYPE_REF, TYPE_REF_SEEALSO, TYPE_REF_TUPLE]

# for saved class ids
CLASSES_FNS = {TYPE_MESH: u'MESH',
               TYPE_ENTAILMENT: u'ENTAILMENT',
               TYPE_POLARITY: u"POLARITY",
               TYPE_RELATION: u"RELATION",
               TYPE_LEXEME: u"LEXEME",
               TYPE_POS_TAG: u"POS",
               TYPE_DEPENDENCY_RELATION: u"DEP-REL",
               #STRUCTURE_TYPES: u"STRUCTURE"
               }

##################### OLD TYPES END #################################

##################### RDF BASED FORMAT ##############################

# NOTE: prefixes should end with a separator like "#" or "/"
PREFIX_REC_EMB = u'https://github.com/ArneBinder/recursive-embedding#'
PREFIX_CONLL = u'http://ufal.mff.cuni.cz/conll2009-st/task-description.html#'
PREFIX_NIF = u'http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#'
PREFIX_UNIVERSAL_DEPENDENCIES_ENGLISH = u'https://github.com/UniversalDependencies/UD_English#'
# corpus related prefixes
PREFIX_TACRED = u'https://catalog.ldc.upenn.edu/LDC2018T24/'
PREFIX_SICK = u'http://clic.cimec.unitn.it/composes/sick.html/'
PREFIX_IMDB = u'http://ai.stanford.edu/~amaas/data/sentiment/'
# correct url: http://semeval2.fbk.eu/semeval2.php?location=tasks#T11
PREFIX_SEMEVAL = u'http://semeval2.fbk.eu/task8/'

RDF_PREFIXES_MAP = {PREFIX_REC_EMB: u'rem:',
                    PREFIX_CONLL: u'conll:',
                    PREFIX_NIF: u'nif:',
                    PREFIX_TACRED: u'tac:',
                    PREFIX_SICK: u'sck:',
                    PREFIX_IMDB: u'imdb:',
                    PREFIX_SEMEVAL: u'smvl:',
                    PREFIX_UNIVERSAL_DEPENDENCIES_ENGLISH: u'ude:'}

REC_EMB_GLOBAL_ANNOTATION = RDF_PREFIXES_MAP[PREFIX_REC_EMB] + u'GlobalAnnotation'
REC_EMB_HAS_GLOBAL_ANNOTATION = RDF_PREFIXES_MAP[PREFIX_REC_EMB] + u'hasGlobalAnnotation'
REC_EMB_RECORD = RDF_PREFIXES_MAP[PREFIX_REC_EMB] + u'Record'
REC_EMB_HAS_PARSE = RDF_PREFIXES_MAP[PREFIX_REC_EMB] + u'hasParse'
REC_EMB_HAS_PARSE_ANNOTATION = RDF_PREFIXES_MAP[PREFIX_REC_EMB] + u'hasParseAnnotation'
REC_EMB_HAS_CONTEXT = RDF_PREFIXES_MAP[PREFIX_REC_EMB] + u'hasContext'
REC_EMB_USED_PARSER = RDF_PREFIXES_MAP[PREFIX_REC_EMB] + u'usedParser'
REC_EMB_SUFFIX_GLOBAL_ANNOTATION = u'#GlobalAnnotation'
REC_EMB_SUFFIX_NIF_CONTEXT = u'?nif=context'

NIF_CONTEXT = RDF_PREFIXES_MAP[PREFIX_NIF] + u'Context'
NIF_WORD = RDF_PREFIXES_MAP[PREFIX_NIF] + u'Word'
NIF_NEXT_WORD = RDF_PREFIXES_MAP[PREFIX_NIF] + u'nextWord'
NIF_SENTENCE = RDF_PREFIXES_MAP[PREFIX_NIF] + u'Sentence'
NIF_NEXT_SENTENCE = RDF_PREFIXES_MAP[PREFIX_NIF] + u'nextSentence'
NIF_IS_STRING = RDF_PREFIXES_MAP[PREFIX_NIF] + u'isString'

SICK_VOCAB = RDF_PREFIXES_MAP[PREFIX_SICK] + u'vocab#'
SICK_OTHER = SICK_VOCAB + u'other'
SICK_RELATEDNESS_SCORE = SICK_VOCAB + u'relatedness_score'
SICK_ENTAILMENT_JUDGMENT = SICK_VOCAB + u'entailment_judgment'
IMDB_SENTIMENT = RDF_PREFIXES_MAP[PREFIX_IMDB] + u'vocab#sentiment'
IMDB_RATING = RDF_PREFIXES_MAP[PREFIX_IMDB] + u'vocab#rating'
SEMEVAL_RELATION = RDF_PREFIXES_MAP[PREFIX_SEMEVAL] + u'vocab#relation'
SEMEVAL_SUBJECT = RDF_PREFIXES_MAP[PREFIX_SEMEVAL] + u'vocab#subj'
SEMEVAL_OBJECT = RDF_PREFIXES_MAP[PREFIX_SEMEVAL] + u'vocab#obj'
TACRED_RELATION = RDF_PREFIXES_MAP[PREFIX_TACRED] + u'vocab#relation'
TACRED_SUBJECT = RDF_PREFIXES_MAP[PREFIX_TACRED] + u'vocab#subj'
TACRED_OBJECT = RDF_PREFIXES_MAP[PREFIX_TACRED] + u'vocab#obj'

# these entries have to start with "@" (see corpus_rdf)
JSONLD_ID = u'@id'
JSONLD_VALUE = u'@value'
JSONLD_TYPE = u'@type'
# added for rec-emb
JSONLD_IDX = u'@idx'
JSONLD_DATA = u'@data'

##################### RDF BASED FORMAT END ##########################


FN_TREE_INDICES = 'tree_indices'

# special embeddings (have to be negative to get recognized during visualization #deprecated)
UNKNOWN_EMBEDDING = 0

AGGREGATOR_EMBEDDING = -2
SOURCE_EMBEDDING = -8
IDENTITY_EMBEDDING = -9
ROOT_EMBEDDING = -10
UNIQUE_EMBEDDING = -12
BACK_EMBEDDING = -13
TARGET_EMBEDDING = -15
ANCHOR_EMBEDDING = -16
PADDING_EMBEDDING = -17
BLANKED_EMBEDDING = -18

vocab_manual = {UNKNOWN_EMBEDDING: u'UNKNOWN',
                AGGREGATOR_EMBEDDING: u'AGGREGATOR',
                IDENTITY_EMBEDDING: u'IDENTITY',
                ROOT_EMBEDDING: u'ROOT',
                PADDING_EMBEDDING: u'PADDING',
                TARGET_EMBEDDING: u'TARGET',
                BLANKED_EMBEDDING: u'BLANKED'
                }
# TODO: consider using distinct embedding for blanked  nodes ('BLANKED' instead of 'UNKNOWN')
#vocab_manual[BLANKED_EMBEDDING] = vocab_manual[UNKNOWN_EMBEDDING]

CM_TREE = 'tree'
CM_SEQUENCE = 'sequence'
CM_AGGREGATE = 'aggregate'
concat_modes = [None, CM_SEQUENCE, CM_AGGREGATE, CM_TREE]

default_concat_mode = CM_SEQUENCE
default_inner_concat_mode = CM_TREE

#DTYPE_DATA = np.int64 # use DTYPE_IDX
DTYPE_OFFSET = np.int64
DTYPE_DEPTH = np.int16
DTYPE_HASH = np.uint64
DTYPE_VECS = np.float32
DTYPE_COUNT = np.int32
DTYPE_IDX = np.int64
DTYPE_PROBS = np.float32

KEY_HEAD = 'h'
KEY_CHILDREN = 'c'
KEY_CANDIDATES = 'ca'
KEY_HEAD_CONCAT = 'hc'
KEY_HEAD_STRING = 'hs'

# model types
MT_MULTICLASS = 'multiclass'
#MT_TUPLE_DISCRETE_DEPENDENT = 'tuple_class' ### TODO: use 'multiclass' in .env files!
MT_CANDIDATES = 'reroot'
MT_CANDIDATES_W_REF = 'tuple'
MT_TUPLE_CONTINOUES = 'sim_tuple'

M_INDICES = 'indices'
M_INDICES_SAMPLER = 'indices_sampler'
M_TEST = 'test'
M_TRAIN = 'train'
M_MODEL = 'model'
M_FNAMES = 'fnames'
M_TREES = 'trees'
M_EMBEDDINGS = 'embeddings'
#M_TREE_EMBEDDINGS = 'tree_embeddings'
#M_TARGETS = 'targets'
M_DATA = 'data'
M_IDS = 'ids'
M_TREE_ITER = 'tree_iterator'
M_TREE_ITER_TFIDF = 'tree_iterator_tfidf'
#M_IDS_TARGET = 'ids_target'
M_INDICES_TARGETS = 'indices_targets'
M_BATCH_ITER = 'batch_iter'
M_NEG_SAMPLES = 'neg_samples'
M_MODEL_NEAREST = 'model_nearest'
M_INDEX_FILE_SIZES = 'index_file_sizes'

# TASKS
TASK_MESH_PREDICTION = 'mesh'
TASK_SENTIMENT_PREDICTION = 'sentiment'
TASK_ENTAILMENT_PREDICTION = 'entailment'
TASK_RELATION_EXTRACTION_SEMEVAL = 're/semeval'
TASK_RELATION_EXTRACTION_TACRED = 're/tacred'
TASK_LANGUAGE = 'language'

TYPE_FOR_TASK_OLD = {TASK_MESH_PREDICTION: TYPE_MESH,
                     TASK_SENTIMENT_PREDICTION: TYPE_POLARITY,
                     TASK_ENTAILMENT_PREDICTION: TYPE_ENTAILMENT,
                     TASK_RELATION_EXTRACTION_SEMEVAL: TYPE_RELATION}
TYPE_FOR_TASK = {#TASK_MESH_PREDICTION: TYPE_MESH,
                 TASK_SENTIMENT_PREDICTION: IMDB_SENTIMENT,
                 TASK_ENTAILMENT_PREDICTION: SICK_ENTAILMENT_JUDGMENT,
                 TASK_RELATION_EXTRACTION_SEMEVAL: SEMEVAL_RELATION,
                 TASK_RELATION_EXTRACTION_TACRED: TACRED_RELATION}

TYPE_LONG = {'REL': TYPE_RELATION,
             'RELF': TYPE_RELATION_FORWARD,
             'RELB': TYPE_RELATION_BACKWARD,
             'DEP': TYPE_DEPENDENCY_RELATION,
             #'POS': TYPE_POS_TAG,
             'LEX': TYPE_LEXEME,
             'STR': STRUCTURE_TYPES,
             'DAT': TYPE_DATASET,
             'CON': [TYPE_CONTEXT],
             'SEN': [TYPE_SENTENCE],
             'ENT': [TYPE_NAMED_ENTITY],
             'EDG': RDF_PREFIXES_MAP[PREFIX_CONLL]+u'EDGE=',
             'POS': RDF_PREFIXES_MAP[PREFIX_CONLL]+u'UPOS='
             # collides with TYPE_LEXEME because of UNKNOWN is added for TYPE_LEXEME:
             #'MAN': vocab_manual.values()
             }

# structural assumptions
# general
OFFSET_ID = 1
OFFSET_CONTEXT_ROOT = 2
# dbpedianif
OFFSET_SEEALSO_ROOT = 3
# sick
OFFSET_RELATEDNESS_SCORE_ROOT = 3
OFFSET_ENTAILMENT_ROOT = 5
OFFSET_OTHER_ENTRY_ROOT = 7
# bioasq
OFFSET_MESH_ROOT = 3
OFFSET_POLARITY_ROOT = 3
# semeval2010task8
OFFSET_RELATION_ROOT = 3

OFFSET_CLASS_ROOTS = {
    TYPE_MESH: OFFSET_MESH_ROOT,
    TYPE_ENTAILMENT: OFFSET_ENTAILMENT_ROOT,
    TYPE_POLARITY: OFFSET_POLARITY_ROOT,
    TYPE_RELATION: OFFSET_RELATION_ROOT
}


# global config
RDF_BASED_FORMAT = True
DEBUG = False



#DEPRECATED

# edges
INTER_TREE = 0

# embedding_vectors
NOT_IN_WORD_DICT = -1

# human readable labels
NOT_IN_WORD_DICT_ = u'NOTINDICT'
INTER_TREE_ = u'INTERTREE'


