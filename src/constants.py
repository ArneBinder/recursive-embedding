
import numpy as np

LOGGING_FORMAT = '%(asctime)s %(levelname)s %(message)s'

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

TYPE_ARTIFICIAL = u"ARTIFICIAL"

# used just in filter_and_shorten_label
BASE_TYPES = [TYPE_REF, TYPE_DBPEDIA_RESOURCE, TYPE_CONTEXT, TYPE_PARAGRAPH, TYPE_TITLE, TYPE_SECTION, TYPE_SECTION,
              TYPE_SENTENCE, TYPE_PMID, TYPE_LEXEME, TYPE_DEPENDENCY_RELATION, TYPE_LEMMA, TYPE_POS_TAG,
              TYPE_NAMED_ENTITY, TYPE_PHRASE, TYPE_ID, TYPE_NIF, TYPE_RELATEDNESS_SCORE, TYPE_ENTAILMENT,
              TYPE_POLARITY, TYPE_RATING, TYPE_RELATION]

# CONSTRUCTED TYPES
TYPE_REF_SEEALSO = TYPE_REF + SEPARATOR + u'seeAlso'
TYPE_SECTION_SEEALSO = TYPE_SECTION + SEPARATOR + u'seeAlso'
TYPE_SECTION_ABSTRACT = TYPE_SECTION + SEPARATOR + u'abstract'
TYPE_REF_TUPLE = TYPE_REF + SEPARATOR + u'other'
TYPE_RELATION_TYPE = TYPE_RELATION + SEPARATOR + u'TYPE'
TYPE_RELATION_DIRECTION = TYPE_RELATION + SEPARATOR + u'DIRECTION'

LINK_TYPES = [TYPE_REF, TYPE_REF_SEEALSO, TYPE_REF_TUPLE]

# for saved class ids
CLASSES_FNS = {TYPE_MESH: u'MESH',
               TYPE_ENTAILMENT: u'ENTAILMENT',
               TYPE_POLARITY: u"POLARITY",
               TYPE_RELATION: u"RELATION"}

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

vocab_manual = {UNKNOWN_EMBEDDING: u'UNKNOWN',
                AGGREGATOR_EMBEDDING: u'AGGREGATOR',
                IDENTITY_EMBEDDING: u'IDENTITY',
                ROOT_EMBEDDING: u'ROOT',
                PADDING_EMBEDDING: u'PADDING',
                TARGET_EMBEDDING: u'TARGET',
                }

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
TASK_RELATION_EXTRACTION = 'relation'
TASK_LANGUAGE = 'language'

TYPE_FOR_TASK = {TASK_MESH_PREDICTION: TYPE_MESH,
                 TASK_SENTIMENT_PREDICTION: TYPE_POLARITY,
                 TASK_ENTAILMENT_PREDICTION: TYPE_ENTAILMENT,
                 TASK_RELATION_EXTRACTION: TYPE_RELATION}

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

#DEPRECATED

# edges
INTER_TREE = 0

# embedding_vectors
NOT_IN_WORD_DICT = -1

# human readable labels
NOT_IN_WORD_DICT_ = u'NOTINDICT'
INTER_TREE_ = u'INTERTREE'


