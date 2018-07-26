# best parameters settings so far

# TREE (F1: 0.63)
{"tree_embedder": "TreeEmbedding_HTU_reduceSUM_mapGRU", "learning_rate": 0.001, "batch_size": 100, "fc_sizes": "2000", "concat_mode": "tree", "state_size": 900, "leaf_fc_size": 0}
# TFIDF (F1: 0.64)
{"tree_embedder": "tfidf", "learning_rate": 0.0001, "batch_size": 100, "fc_sizes": "2000", "concat_mode": "aggregate"}
# BIGRU (F1: 0.50)
{"tree_embedder": "TreeEmbedding_FLATconcat_BIGRU", "learning_rate": 0.0003, "batch_size": 25, "fc_sizes": "1000", "concat_mode": "aggregate", "state_size": 300, "leaf_fc_size": 0}
# SUM (F1: 0.60)
{"tree_embedder": "TreeEmbedding_FLAT_SUM", "learning_rate": 0.0001, "batch_size": 100, "fc_sizes": "1000", "concat_mode": "aggregate", "leaf_fc_size": 600}
