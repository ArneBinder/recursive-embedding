import csv

import tensorflow as tf

import corpus_simtuple

tf.flags.DEFINE_string('corpus_name',
                       'SICK',
                       'name of the corpus (used as source folder and output dir)')

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
            yield (float(row['relatedness_score']) - 1.0) / 4.0


def main(args=None):
    if len(args) > 1:
        file_names = args[1:]
    else:
        file_names = ['sick_train/SICK_train.txt', 'sick_test_annotated/SICK_test_annotated.txt']
    corpus_simtuple.create_corpus(reader_sentences=sentence_reader, reader_score=score_reader,
                                  corpus_name=FLAGS.corpus_name,
                                  file_names=file_names)


if __name__ == '__main__':
    tf.app.run()
