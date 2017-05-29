
# special embeddings (have to be negative to get recognized during visualization)
UNKNOWN_EMBEDDING = -1
TERMINATOR_EMBEDDING = -2
EDGE_EMBEDDING = -3
WORD_EMBEDDING = -4
ENTITY_TYPE_EMBEDDING = -5
LEMMA_EMBEDDING = -6
POS_TAG_EMBEDDING = -7

vocab_manual = {WORD_EMBEDDING: u'WORD', EDGE_EMBEDDING: u'EDGE', UNKNOWN_EMBEDDING: u'UNKNOWN',
                TERMINATOR_EMBEDDING: u'TERMINATOR', ENTITY_TYPE_EMBEDDING: u'ENTITY_TYPE',
                LEMMA_EMBEDDING: u'LEMMA', POS_TAG_EMBEDDING: u'POS_TAG'}

tree_modes = [None, 'sequence', 'aggregate', 'tree']

#DEPRECATED

# edges
INTER_TREE = 0

# embedding_vectors
NOT_IN_WORD_DICT = -1

# human readable labels
NOT_IN_WORD_DICT_ = u'NOTINDICT'
INTER_TREE_ = u'INTERTREE'


