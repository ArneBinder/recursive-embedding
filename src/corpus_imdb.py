import os

import plac
import logging

import preprocessing
from mytools import numpy_dump, make_parent_dir
from corpus import process_records, merge_batches, create_index_files, DIR_BATCHES, FE_CLASS_IDS
from constants import TYPE_SECTION, SEPARATOR, LOGGING_FORMAT, TYPE_CONTEXT, TYPE_PARAGRAPH


TYPE_POLARITY = u"POLARITY"
TYPE_RATING = u"RATING"
TYPE_ACLIMDB_ID = u'http://ai.stanford.edu/~amaas/data/sentiment/aclimdb_v1'

logger = logging.getLogger('corpus_imdb')
logger.setLevel(logging.DEBUG)
logger_streamhandler = logging.StreamHandler()
logger_streamhandler.setLevel(logging.DEBUG)
logger_streamhandler.setFormatter(logging.Formatter(LOGGING_FORMAT))
logger.addHandler(logger_streamhandler)


DUMMY_RECORD = {TYPE_POLARITY: u"neg",
                TYPE_RATING: u"2",
                u"content": u"Once again Mr. Costner has dragged out a movie for far longer than necessary. Aside from the terrific sea rescue sequences, of which there are very few I just did not care about any of the characters. Most of us have ghosts in the closet, and Costner's character are realized early on, and then forgotten until much later, by which time I did not care. The character we should really care about is a very cocky, overconfident Ashton Kutcher. The problem is he comes off as kid who thinks he's better than anyone else around him and shows no signs of a cluttered closet. His only obstacle appears to be winning over Costner. Finally when we are well past the half way point of this stinker, Costner tells us all about Kutcher's ghosts. We are told why Kutcher is driven to be the best with no prior inkling or foreshadowing. No magic here, it was all I could do to keep from turning it off an hour in.",
                TYPE_ACLIMDB_ID: u"test/neg/0_2.txt"}


def reader(records, key_text='content', root_string=TYPE_ACLIMDB_ID,
           keys_meta=(TYPE_POLARITY, TYPE_RATING), key_id=TYPE_ACLIMDB_ID,
           root_text_string=TYPE_CONTEXT):
    """

    :param records: dicts containing the textual data and optional meta data
    :param key_text:
    :param root_string:
    :param keys_meta:
    :param key_id:
    :param root_text_string:
    :return:
    """
    count_finished = 0
    count_discarded = 0
    for record in records:
        try:
            prepend_data_strings = [root_string]
            prepend_parents = [0]
            if key_id is not None:
                prepend_data_strings.append(key_id + SEPARATOR + record[key_id])
                prepend_parents.append(-1)

            prepend_data_strings.append(root_text_string)
            prepend_parents.append(-len(prepend_parents))
            text_root_offset = len(prepend_parents) - 1

            for k_meta in keys_meta:
                # add key
                prepend_data_strings.append(k_meta)
                prepend_parents.append(-len(prepend_parents))
                # add value(s)
                # ATTENTION: assumed to be string(s)!
                v_meta = record[k_meta]
                if not isinstance(v_meta, list):
                    v_meta = [v_meta]
                # replace spaces by underscores
                prepend_data_strings.extend([k_meta + SEPARATOR + v.replace(' ', '_') for v in v_meta])
                prepend_parents.extend([-i - 1 for i in range(len(v_meta))])

            prepend_data_strings.append(TYPE_SECTION)
            prepend_parents.append(text_root_offset - len(prepend_parents))
            text_root_offset = len(prepend_parents) - 1

            prepend = (prepend_data_strings, prepend_parents)
            record_data = []

            record_data.append((record[key_text], {'root_type': TYPE_PARAGRAPH, 'prepend_tree': prepend, 'parent_prepend_offset': text_root_offset}))

            # has to be done in the end because the whole record should be discarded at once if an exception is raised
            for d in record_data:
                yield d
            count_finished += 1
        except Warning as e:
            count_discarded += 1
            #logger.debug('failed to process record (%s): %s' % (e.message, str(record)))
            pass
        except Exception as e:
            count_discarded += 1
            logger.warning('failed to process record (%s): %s' % (e.message, str(record)))

    logger.info('discarded %i of %i records' % (count_discarded, count_finished + count_discarded))


@plac.annotations(
    out_base_name=('corpora output base file name', 'option', 'o', str)
)
def parse_dummy(out_base_name):
    print(out_base_name)
    make_parent_dir(out_base_name)
    process_records(records=[DUMMY_RECORD], out_base_name=out_base_name)


def read_files(in_path, subdir, polarities=(u'pos', u'neg')):
    for polarity in polarities:
        in_path_polarity = os.path.join(in_path, subdir, polarity)
        for fn in os.listdir(in_path_polarity):
            full_fn = os.path.join(in_path_polarity, fn)
            with open(full_fn) as f:
                lines = f.readlines()
            assert len(lines) == 1, 'incorrect number of lines: %i, expected one.' % len(lines)
            rating = unicode(os.path.splitext(fn)[0].split('_')[-1])
            text = unicode(lines[-1], 'utf8')
            # simple cleaning of remaining html tags
            text = text.replace(u'<br /><br />', u' ').replace(u'<br />', u' ').replace(u'<i>', u'')\
                .replace(u'</i>', u'').replace(u'<hr>', u'')
            yield {TYPE_POLARITY: polarity, TYPE_ACLIMDB_ID: unicode(os.path.join(subdir, polarity, fn)),
                   TYPE_RATING: rating, u'content': text}


@plac.annotations(
    in_path=('corpora input folder', 'option', 'i', str),
    out_path=('corpora output folder', 'option', 'o', str),
    n_threads=('number of threads for replacement operations', 'option', 't', int),
    parser_batch_size=('parser batch size', 'option', 'b', int)
)
def parse_dirs(in_path, out_path, n_threads=4, parser_batch_size=1000):
    sub_dirs = ['train', 'test']
    for sub_dir in sub_dirs:
        logger.info('create forest for %s ...' % sub_dir)
        out_base_name = os.path.join(out_path, DIR_BATCHES, sub_dir)
        make_parent_dir(out_base_name)
        process_records(records=read_files(in_path, sub_dir), out_base_name=out_base_name, reader=reader,
                        sentence_processor=preprocessing.process_sentence1, n_threads=n_threads,
                        batch_size=parser_batch_size)
        logger.info('done.')


@plac.annotations(
    mode=('processing mode', 'positional', None, str, ['PARSE', 'PARSE_DUMMY', 'MERGE', 'CREATE_INDICES']),
    args='the parameters for the underlying processing method')
def main(mode, *args):
    if mode == 'PARSE_DUMMY':
        plac.call(parse_dummy, args)
    elif mode == 'PARSE':
        plac.call(parse_dirs, args)
    elif mode == 'MERGE':
        forest_merged, out_path_merged = plac.call(merge_batches, args)
        #rating_ids = forest_merged.lexicon.get_ids_for_prefix(TYPE_RATING)
        #logger.info('number of ratings to predict: %i' % len(rating_ids))
        #numpy_dump(filename='%s.%s' % (out_path_merged, FE_CLASS_IDS), ndarray=rating_ids)
        polarity_ids = forest_merged.lexicon.get_ids_for_prefix(TYPE_POLARITY)
        logger.info('number of polarities to predict: %i. save only %i for prediction.'
                    % (len(polarity_ids), len(polarity_ids)-1))
        numpy_dump(filename='%s.%s' % (out_path_merged, FE_CLASS_IDS), ndarray=polarity_ids[:-1])
    elif mode == 'CREATE_INDICES':
        plac.call(create_index_files, args)
    else:
        raise ValueError('unknown mode. use one of PROCESS_DUMMY or PROCESS_SINGLE.')


if __name__ == '__main__':
    plac.call(main)
