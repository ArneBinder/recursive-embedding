
# special embeddings (have to be negative to get recognized during visualization #deprecated)
UNKNOWN_EMBEDDING = 0

AGGREGATOR_EMBEDDING = -2
EDGE_EMBEDDING = -3
TOKEN_EMBEDDING = -4
ENTITY_EMBEDDING = -5
LEMMA_EMBEDDING = -6
POS_EMBEDDING = -7

vocab_manual = {TOKEN_EMBEDDING: u'WORD', EDGE_EMBEDDING: u'EDGE', UNKNOWN_EMBEDDING: u'UNKNOWN',
                AGGREGATOR_EMBEDDING: u'AGGREGATOR', ENTITY_EMBEDDING: u'ENTITY_TYPE',
                LEMMA_EMBEDDING: u'LEMMA', POS_EMBEDDING: u'POS_TAG'}
#vocab_manual = {TOKEN_EMBEDDING: u'TOKEN', EDGE_EMBEDDING: u'DEPENDENCY', UNKNOWN_EMBEDDING: u'UNKNOWN',
#                AGGREGATOR_EMBEDDING: u'AGGREGATOR', ENTITY_EMBEDDING: u'ENTITY',
#                LEMMA_EMBEDDING: u'LEMMA', POS_EMBEDDING: u'POS'}

tree_modes = [None, 'sequence', 'aggregate', 'tree']

#DEPRECATED

# edges
INTER_TREE = 0

# embedding_vectors
NOT_IN_WORD_DICT = -1

# human readable labels
NOT_IN_WORD_DICT_ = u'NOTINDICT'
INTER_TREE_ = u'INTERTREE'


