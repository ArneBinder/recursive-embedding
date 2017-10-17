import csv
import tensorflow as tf

import corpus_simtuple


corpus_simtuple.set_flags(corpus_name='STSBENCH', fn_train='sts-train_fixed.csv', fn_dev='sts-train_fixed.csv', fn_test='sts-train_fixed.csv')

FLAGS = tf.flags.FLAGS

delimiter = '\t'


def sentence_reader(filename):
    with open(filename, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=delimiter)
        for row in reader:
            if len(row) > 6:
                yield row[5].decode('utf-8')
                yield row[6].decode('utf-8')
            else:
                print('ERROR could not read line:')
                print(row)


def score_reader(filename):
    with open(filename, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=delimiter)
        for row in reader:
            yield float(row[4]) / 5.0


if __name__ == '__main__':
    corpus_simtuple.create_corpus(reader_sentences=sentence_reader, reader_score=score_reader, FLAGS=FLAGS)


