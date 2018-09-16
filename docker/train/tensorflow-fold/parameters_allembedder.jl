# best parameters settings so far

# TREE (F1: 0.63)
{"tree_embedder": "HTU_reduceSUM_mapGRU", "concat_mode": "tree", "leaf_fc_size": 0, "learning_rate": 0.0003, "batch_size": 100}
# TFIDF (F1: 0.64)
{"tree_embedder": "tfidf", "concat_mode": "aggregate", "learning_rate": 0.0001, "batch_size": 100}
# GRU
{"tree_embedder": "FLATconcat_GRU", "concat_mode": "aggregate", "leaf_fc_size": 0, "learning_rate": 0.001, "batch_size": 100}
# SUM (F1: 0.60)
{"tree_embedder": "FLAT_SUM", "concat_mode": "aggregate", "learning_rate": 0.0001, "batch_size": 100}
# TREE + TFIDF
{"tree_embedder": "HTU_reduceSUM_mapGRU", "concat_mode": "tree", "use_tfidf": true, "leaf_fc_size": 0, "learning_rate": 0.001, "batch_size": 100}