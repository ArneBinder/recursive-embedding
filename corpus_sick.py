import csv

import tensorflow as tf

import constants
import corpus_simtuple

tf.flags.DEFINE_string('corpus_name',
                       'SICK',
                       'name of the corpus (used as source folder and output dir)')

FLAGS = tf.flags.FLAGS


def sentence_reader(filename):
    with open(filename, 'rb') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        for row in reader:
            a = row['sentence_A'].decode('utf-8')
            if a[-1] != u'.':
                a += u'.'
            b = row['sentence_B'].decode('utf-8')
            if b[-1] != u'.':
                b += u'.'
            yield a
            yield b


def score_reader(filename):
    with open(filename, 'rb') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        for row in reader:
            yield (float(row['relatedness_score']) - 1.0) / 4.0


def roots_reader():
    lc = 0
    ANNOT_str = u'TUPLE'
    while True:
        #yield [ANNOT_str, '%s/%s/%i' % (ANNOT_str, prefix, lc)]
        #yield [ANNOT_str, '%s/%s/%i' % (ANNOT_str, prefix, lc)]
        #yield '%s/%s/%i/0' % (ANNOT_str, prefix, lc)
        #yield '%s/%s/%i/1' % (ANNOT_str, prefix, lc)
        #yield '%s/0' % ANNOT_str
        #yield '%s/1' % ANNOT_str
        yield [ANNOT_str, '%s/0' % ANNOT_str]
        yield [ANNOT_str, '%s/1' % ANNOT_str]
        lc += 1


def main(args=None):
    if len(args) > 1:
        file_names = args[1:]
    else:
        file_names = ['sick_train/SICK_train.txt', 'sick_test_annotated/SICK_test_annotated.txt']
    corpus_simtuple.create_corpus(reader_sentences=sentence_reader, reader_scores=score_reader,
                                  corpus_name=FLAGS.corpus_name,
                                  file_names=file_names,
                                  reader_roots=roots_reader,
                                  reader_roots_args={}
                                  )


if __name__ == '__main__':
    tf.app.run()
