
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

vocab_manual = {LEXEME_EMBEDDING: u'LEXEME', UNKNOWN_EMBEDDING: u'UNKNOWN',
                AGGREGATOR_EMBEDDING: u'AGGREGATOR', ENTITY_EMBEDDING: u'ENTITY',
                LEMMA_EMBEDDING: u'LEMMA', POS_EMBEDDING: u'POS', SOURCE_EMBEDDING: u'SOURCE',
                IDENTITY_EMBEDDING: u'IDENTITY', ROOT_EMBEDDING: u'ROOT', DEPENDENCY_EMBEDDING: u'DEPENDENCY',
                UNIQUE_EMBEDDING: u'UNIQUE', BACK_EMBEDDING: u'BACK'}

SEPARATOR = '/'

CM_TREE = 'tree'
CM_SEQUENCE = 'sequence'
CM_AGGREGATE = 'aggregate'
concat_modes = [None, CM_SEQUENCE, CM_AGGREGATE, CM_TREE]

default_concat_mode = CM_SEQUENCE
default_inner_concat_mode = CM_TREE

#DEPRECATED

# edges
INTER_TREE = 0

# embedding_vectors
NOT_IN_WORD_DICT = -1

# human readable labels
NOT_IN_WORD_DICT_ = u'NOTINDICT'
INTER_TREE_ = u'INTERTREE'

