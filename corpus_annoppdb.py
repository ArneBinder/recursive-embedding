import csv
import tensorflow as tf

import corpus_simtuple


tf.flags.DEFINE_string('corpus_name',
                       'ANNOPPDB',
                       'name of the corpus (used as source folder and output dir)')

FLAGS = tf.flags.FLAGS

delimiter = '|||'


def sentence_reader(filename):
    with open(filename, 'rb') as file:
        for line in file:
            cols = line.split(delimiter)
            if len(cols) > 2:
                yield cols[0].decode('utf-8')
                yield cols[1].decode('utf-8')
            else:
                print(cols)


def score_reader(filename):
    with open(filename, 'rb') as file:
        for line in file:
            cols = line.split(delimiter)
            yield (float(cols[2].decode('utf-8')) - 1.0) / 4.0


def main(args=None):
    if len(args) > 1:
        file_names = args[1:]
    else:
        file_names = ['ppdb_train.txt', 'ppdb_dev.txt', 'ppdb_test.txt']
    corpus_simtuple.create_corpus(reader_sentences=sentence_reader, reader_score=score_reader,
                                  corpus_name=FLAGS.corpus_name,
                                  file_names=file_names)


if __name__ == '__main__':
    tf.app.run()




