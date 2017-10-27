import os
import logging
import tensorflow as tf

import corpus_simtuple

tf.flags.DEFINE_string('corpus_name',
                       'PPDB',
                       'name of the corpus (used as source folder and output dir)')
tf.flags.DEFINE_string('base_filename',
                       'ppdb-2.0-s-phrasal',
                       '')
tf.flags.DEFINE_integer('size',
                        10000,
                        'size of each train file (without negative samples)')
tf.flags.DEFINE_integer('file_count',
                        2,
                        'amount of train files')

FLAGS = tf.flags.FLAGS


def sentence_reader(filename):
    with open(filename, 'rb') as file:
        for line in file:
            cols = line.split(' ||| ')
            yield cols[1].decode('utf-8')
            yield cols[2].decode('utf-8')


def score_reader(filename):
    num_lines = sum(1 for _ in open(filename))
    for _ in range(num_lines):
        yield 1.0


def source_fn(fn):
    return os.path.join(FLAGS.corpora_source_root, FLAGS.corpus_name, fn)


def main(args=None):
    pat = '%0'+str(len(str(FLAGS.file_count * FLAGS.size)))+'d'
    file_names = [('%s_'+pat+'-'+pat) % (FLAGS.base_filename, i*FLAGS.size, (i+1)*FLAGS.size) for i in range(FLAGS.file_count)]
    create = False
    for fn in file_names:
        if not os.path.isfile(source_fn(fn)):
            logging.info('file does not exist "%s"' % source_fn(fn))
            create = True
    if create:
        f_id = 0
        logging.info('(re-)create files from "%s" ...' % source_fn(FLAGS.base_filename))
        with open(source_fn(FLAGS.base_filename)) as fin:
            fout = open(source_fn(file_names[f_id]), "wb")
            for i, line in enumerate(fin):
                fout.write(line)
                if (i + 1) % FLAGS.size == 0:
                    if f_id == len(file_names) - 1:
                        break
                    fout.close()
                    logging.info('created "%s"' % file_names[f_id])
                    f_id += 1
                    fout = open(source_fn(file_names[f_id]), "wb")

            fout.close()
            logging.info('created "%s"' % file_names[f_id])
        if f_id < FLAGS.file_count -1:
            logging.warning('not enough data to create %i files of size %i. Use only created (%i) files to create corpus.' % (FLAGS.file_count, FLAGS.size, f_id+1))
            file_names = file_names[:f_id+1]
    corpus_simtuple.create_corpus(reader_sentences=sentence_reader, reader_score=score_reader,
                                  corpus_name=FLAGS.corpus_name,
                                  file_names=file_names,
                                  output_suffix='_%i' % FLAGS.size)


if __name__ == '__main__':
    tf.app.run()



