import csv

import tensorflow as tf

import corpus_simtuple

corpus_simtuple.set_flags(corpus_name='SICK', fn_train='sick_train/SICK_train.txt', fn_dev='sick_test_annotated/SICK_test_annotated.txt', output_suffix='_tt')

FLAGS = tf.flags.FLAGS


def sentence_reader(filename):
    with open(filename, 'rb') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        for row in reader:
            yield row['sentence_A'].decode('utf-8') + '.'
            yield row['sentence_B'].decode('utf-8') + '.'


def score_reader(filename):
    with open(filename, 'rb') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        for row in reader:
            yield float(row['relatedness_score'])


if __name__ == '__main__':
    corpus_simtuple.create_corpus(sentence_reader=sentence_reader, score_reader=score_reader, FLAGS=FLAGS)