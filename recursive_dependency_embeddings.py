from __future__ import print_function
import tensorflow as tf
import sys

dim = 300
edge_count = 60
seq_length = 10


graph = tf.Graph()
with graph.as_default():
    # input
    seq_vecs = tf.placeholder(tf.float32, shape=[seq_length, dim])

    curr_seq_parent_ids = tf.placeholder(tf.int32, shape=[seq_length])
    curr_seq_edges = tf.placeholder(tf.int32, shape=[seq_length])

    # prev_seq_vec = tf.placeholder(tf.float32, shape=[seq_length, dim])
    prev_seq_parent_ids = tf.placeholder(tf.int32, shape=[seq_length])
    prev_seq_edges = tf.placeholder(tf.int32, shape=[seq_length])

    added_edge_id = tf.placeholder(tf.in32, shape=[])

    # embeddings
    edge_weights = tf.Variable(tf.zeros([edge_count, dim, dim]))
    edge_biases = tf.Variable(tf.zeros([edge_count, dim]))

    # scoring
    score_weights = tf.Variable(tf.zeros([2 * dim, 1]))
    score_biases = tf.Variable(tf.zeros([2 * dim, 1]))

    prev_embedding = calc_embedding(prev_seq_parent_ids, prev_seq_edges, seq_vecs, edge_weights, edge_biases)
    correct_embedding = calc_embedding(curr_seq_parent_ids, curr_seq_edges, seq_vecs, edge_weights, edge_biases)
    # seqs = [(curr_seq_parent_ids, curr_seq_edges)] # put correct at id=0
    seqs = possible_seqs(prev_seq_parent_ids, curr_seq_parent_ids, added_edge_id)

    highest_score = -sys.maxint - 1 # -inf
    highest_embedding = None
    # for seq_parents, seq_edges in seqs:
    #	embedding = calc_embedding(seq_parents, seq_edges, seq_vecs, edge_weights, edge_biases)
    #	score = calc_score(embedding, prev_embedding, score_weights_score_biases)
    #	if(score > highest_score):
    #		highest_score = score
    #		highest_embedding = embedding

    loss = calc_loss(correct_embedding, highest_embedding)



    # TODO:
    # * possible_seqs
    # * calc_embedding
    # * calc_score