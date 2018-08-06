
import numpy as np

LOGGING_FORMAT = '%(asctime)s %(levelname)s %(message)s'

# special embeddings (have to be negative to get recognized during visualization #deprecated)
UNKNOWN_EMBEDDING = 0

AGGREGATOR_EMBEDDING = -2
#EDGE_EMBEDDING = -3     # deprecated use DEPENDENCY_EMBEDDING
LEXEME_EMBEDDING = -4
ENTITY_EMBEDDING = -5
LEMMA_EMBEDDING = -6
POS_EMBEDDING = -7
SOURCE_EMBEDDING = -8
IDENTITY_EMBEDDING = -9
ROOT_EMBEDDING = -10
DEPENDENCY_EMBEDDING = -11
UNIQUE_EMBEDDING = -12
BACK_EMBEDDING = -13
SENTENCE_EMBEDDING = -14
TARGET_EMBEDDING = -15

vocab_manual = {LEXEME_EMBEDDING: u'http://purl.org/olia/olia.owl#Lexeme', UNKNOWN_EMBEDDING: u'UNKNOWN',
                AGGREGATOR_EMBEDDING: u'AGGREGATOR', ENTITY_EMBEDDING: u'ENTITY',
                LEMMA_EMBEDDING: u'LEMMA', POS_EMBEDDING: u'POS', SOURCE_EMBEDDING: u'SOURCE',
                IDENTITY_EMBEDDING: u'IDENTITY', ROOT_EMBEDDING: u'ROOT', DEPENDENCY_EMBEDDING: u'DEPENDENCY',
                UNIQUE_EMBEDDING: u'UNIQUE', BACK_EMBEDDING: u'BACK',
                SENTENCE_EMBEDDING: u'http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#Sentence',
                TARGET_EMBEDDING: u'TARGET'
                }

SEPARATOR = '/'

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

KEY_HEAD = 'h'
KEY_CHILDREN = 'c'
KEY_CANDIDATES = 'ca'

TYPE_REF_SEEALSO = u'http://www.w3.org/2005/11/its/rdf#taIdentRef/seeAlso'
TYPE_REF = u'http://www.w3.org/2005/11/its/rdf#taIdentRef'
TYPE_ROOT = u'http://dbpedia.org/resource'
TYPE_ANCHOR = u'http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#Context'
TYPE_SECTION_SEEALSO = u'http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#Section/seeAlso'
TYPE_PARAGRAPH = u'http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#Paragraph'
TYPE_TITLE = u'http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#Title'
TYPE_SECTION = u'http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#Section'
TYPE_SENTENCE = vocab_manual[SENTENCE_EMBEDDING]

M_INDICES = 'indices'
M_TEST = 'test'
M_TRAIN = 'train'
M_MODEL = 'model'
M_FNAMES = 'fnames'
M_TREES = 'trees'
#M_TREE_EMBEDDINGS = 'tree_embeddings'
#M_TARGETS = 'targets'
M_DATA = 'data'
M_IDS = 'ids'
M_TREE_ITER = 'tree_iterator'
#M_IDS_TARGET = 'ids_target'
M_INDICES_TARGETS = 'indices_targets'
M_BATCH_ITER = 'batch_iter'
M_NEG_SAMPLES = 'neg_samples'
M_MODEL_NEAREST = 'model_nearest'
M_INDEX_FILE_SIZES = 'index_file_sizes'

FN_TREE_INDICES = 'tree_indices'

#DEPRECATED

# edges
INTER_TREE = 0

# embedding_vectors
NOT_IN_WORD_DICT = -1

# human readable labels
NOT_IN_WORD_DICT_ = u'NOTINDICT'
INTER_TREE_ = u'INTERTREE'


