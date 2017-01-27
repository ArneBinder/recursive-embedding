from __future__ import print_function
from preprocessing import read_data, slice_forest, articles_from_csv_reader, dummy_str_reader, get_word_embeddings
from visualize import visualize
import numpy as np
import spacy
import constants

dim = 300
edge_count = 60
seq_length = 10

slice_size = 75
max_forest_count = 10

nlp = spacy.load('en')
nlp.pipeline = [nlp.tagger, nlp.parser]

vecs, mapping, human_mapping = get_word_embeddings(nlp.vocab)
# for processing parser output
data_embedding_maps = {constants.WORD_EMBEDDING: mapping}
# for displaying human readable tokens etc.
data_embedding_maps_human = {constants.WORD_EMBEDDING: human_mapping}
# data vectors
data_vecs = {constants.WORD_EMBEDDING: vecs}

data_dir = '/media/arne/DATA/DEVELOPING/ML/data/'
# create data arrays
(seq_data, seq_types, seq_heads, seq_edges), edge_map = \
    read_data(articles_from_csv_reader, nlp, data_embedding_maps, max_forest_count=max_forest_count, max_sen_length=slice_size,
              args={'max_articles': 1, 'filename': data_dir + 'corpora/documents_utf8_filtered_20pageviews.csv'})


# take first 50 token and visualize the dependency graph
sliced_data, sliced_types, sliced_heads, sliced_edges = slice_forest(seq_data, seq_types, seq_heads, seq_edges, 0, 50)
visualize('forest.png', sliced_data, sliced_types, sliced_heads, sliced_edges, data_embedding_maps_human, edge_map)



