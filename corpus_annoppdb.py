import csv
import tensorflow as tf

import corpus_simtuple


corpus_simtuple.set_flags(corpus_name='ANNOPPDB', fn_train='ppdb_all_fixed.txt', fn_dev='ppdb_dev_fixed.txt', fn_test='ppdb_test_fixed.txt')

FLAGS = tf.flags.FLAGS

delimiter = '\t'


def sentence_reader(filename):
    with open(filename, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=delimiter, quotechar='"')
        for row in reader:
            if len(row) > 2:
                yield row[0].decode('utf-8')
                yield row[1].decode('utf-8')
            else:
                print('ERROR could not read line:')
                print(row)


def score_reader(filename):
    with open(filename, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=delimiter, quotechar='"')
        for row in reader:
            yield float(row[2])


if __name__ == '__main__':
    corpus_simtuple.create_corpus(sentence_reader=sentence_reader, score_reader=score_reader, FLAGS=FLAGS)



