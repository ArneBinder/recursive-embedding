import csv
import tensorflow as tf

import corpus_simtuple


tf.flags.DEFINE_string('corpus_name',
                       'ANNOPPDB',
                       'name of the corpus (used as source folder and output dir)')

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
            yield (float(row[2]) - 1.0) / 4.0


def main(args=None):
    if len(args) > 1:
        file_names = args[1:]
    else:
        file_names = ['ppdb_all_fixed.txt', 'ppdb_dev_fixed.txt', 'ppdb_test_fixed.txt']
    corpus_simtuple.create_corpus(reader_sentences=sentence_reader, reader_score=score_reader,
                                  corpus_name=FLAGS.corpus_name,
                                  file_names=file_names)


if __name__ == '__main__':
    tf.app.run()




