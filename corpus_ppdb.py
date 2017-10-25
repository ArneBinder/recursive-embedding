import tensorflow as tf

import corpus_simtuple


corpus_simtuple.set_flags(corpus_name='PPDB', fn_train='ppdb-2.0-s-phrasal_1000')

FLAGS = tf.flags.FLAGS


def sentence_reader(filename):
    with open(filename, 'rb') as file:
        for line in file:
            cols = line.split(' ||| ')
            yield cols[1].decode('utf-8')
            yield cols[2].decode('utf-8')


def score_reader(filename):
    num_lines = sum(1 for line in open(filename))
    for _ in range(num_lines):
        yield 1.0


if __name__ == '__main__':
    corpus_simtuple.create_corpus(reader_sentences=sentence_reader, reader_score=score_reader, FLAGS=FLAGS)



