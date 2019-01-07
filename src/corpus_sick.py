import csv
import json
import logging
import os
import spacy

import plac
from nltk import CoreNLPDependencyParser

from constants import LOGGING_FORMAT, TYPE_CONTEXT, SEPARATOR, TYPE_PARAGRAPH, TYPE_RELATEDNESS_SCORE, \
    TYPE_ENTAILMENT, TYPE_REF_TUPLE, PREFIX_SICK
from mytools import make_parent_dir, numpy_dump
from corpus import process_records, merge_batches, create_index_files, DIR_BATCHES, FE_CLASS_IDS, save_class_ids
from corpus_rdf import parse_and_convert_record
import preprocessing

logger = logging.getLogger('corpus_sick')
logger.setLevel(logging.DEBUG)
logger_streamhandler = logging.StreamHandler()
logger_streamhandler.setLevel(logging.DEBUG)
logger_streamhandler.setFormatter(logging.Formatter(LOGGING_FORMAT))
logger.addHandler(logger_streamhandler)

TYPE_SICK_ID = u'http://clic.cimec.unitn.it/composes/sick'

KEYS_SENTENCE = (u'sentence_A', u'sentence_B')

DUMMY_RECORD = {
    TYPE_RELATEDNESS_SCORE: u"4.5",
    TYPE_ENTAILMENT: u"NEUTRAL",
    KEYS_SENTENCE[0]: u"A group of kids is playing in a yard and an old man is standing in the background",
    KEYS_SENTENCE[1]: u"A group of boys in a yard is playing and a man is standing in the background",
    TYPE_SICK_ID: u"1"
}


def create_record_rdf(row, record_id_prefix, A_or_B, A_or_B_other):
    global_annotations = {PREFIX_SICK + u'vocab#relatedness_score': [{u'@value': float(row['relatedness_score'])}],
                          PREFIX_SICK + u'vocab#entailment_judgment': [{u'@value': row['entailment_judgment']}],
                          PREFIX_SICK + u'vocab#other': [{u'@id': record_id_prefix + A_or_B_other}]
                          }
    record = {'record_id': record_id_prefix + A_or_B,
              # ATTENTION: punctuation "." is added!
              'context_string': row['sentence_%s' % A_or_B] + u'.',
              'global_annotations': global_annotations
              }
    return record


def reader_rdf(base_path, file_name):
    n = 0
    with open(os.path.join(base_path, file_name)) as tsvin:
        tsv = csv.DictReader(tsvin, delimiter='\t')
        for row in tsv:
            record_id_prefix = u'%s%s/%s/' % (PREFIX_SICK, file_name, row['pair_ID'])
            record_A = create_record_rdf(row, record_id_prefix=record_id_prefix, A_or_B=u'A', A_or_B_other=u'B')
            record_B = create_record_rdf(row, record_id_prefix=record_id_prefix, A_or_B=u'B', A_or_B_other=u'A')
            yield record_A
            yield record_B
            n += 1
    logger.info('read %i records from %s resulting in %i rdf records' % (n, base_path, n * 2))


def reader(records, keys_text=KEYS_SENTENCE, root_string=TYPE_SICK_ID,
           keys_meta=(TYPE_RELATEDNESS_SCORE, TYPE_ENTAILMENT), key_id=TYPE_SICK_ID,
           root_text_string=TYPE_CONTEXT):
    """

    :param records: dicts containing the textual data and optional meta data
    :param keys_text:
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
            record_data = []
            for i, key_text in enumerate(keys_text):
                prepend_data_strings = [root_string]
                prepend_parents = [0]
                if key_id is not None:
                    prepend_data_strings.append(key_id + SEPARATOR + record[key_id] + SEPARATOR + str(i))
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
                    prepend_parents.extend([-j - 1 for j in range(len(v_meta))])

                prepend_data_strings.append(TYPE_REF_TUPLE)
                prepend_parents.append(-len(prepend_parents))

                prepend_data_strings.append(key_id + SEPARATOR + record[key_id] + SEPARATOR + str(1-i))
                prepend_parents.append(-1)

                prepend_data_strings.append(TYPE_PARAGRAPH)
                prepend_parents.append(text_root_offset - len(prepend_parents))
                text_root_offset = len(prepend_parents) - 1

                prepend = (prepend_data_strings, prepend_parents)

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


@plac.annotations(
    out_base_name=('corpora output base file name', 'option', 'o', str),
    sentence_processor=('sentence processor', 'option', 'p', str),
)
def parse_dummy(out_base_name,sentence_processor=None):
    print(out_base_name)
    make_parent_dir(out_base_name)
    if sentence_processor is not None and sentence_processor.strip() != '':
        _sentence_processor = getattr(preprocessing, sentence_processor.strip())
    else:
        _sentence_processor = preprocessing.process_sentence1
    parser = spacy.load('en')
    process_records(records=[DUMMY_RECORD], out_base_name=out_base_name, record_reader=reader, parser=parser,
                    sentence_processor=_sentence_processor, concat_mode=None)


def convert_record(record):
    # input format:
    # pair_ID	sentence_A	sentence_B	relatedness_score	entailment_judgment
    return {
        TYPE_RELATEDNESS_SCORE: unicode(record['relatedness_score']),
        TYPE_ENTAILMENT: unicode(record['entailment_judgment']),
        KEYS_SENTENCE[0]: unicode(record['sentence_A']),
        KEYS_SENTENCE[1]: unicode(record['sentence_B']),
        TYPE_SICK_ID: unicode(record['pair_ID'])
    }


def read_file(file_name):
    n = 0
    with open(file_name) as tsvin:
        tsv = csv.DictReader(tsvin, delimiter='\t')
        for row in tsv:
            yield convert_record(row)
            n += 1
    logger.info('read %i records from %s' % (n, file_name))


@plac.annotations(
    in_path=('corpora input folder', 'option', 'i', str),
    out_path=('corpora output folder', 'option', 'o', str),
    sentence_processor=('sentence processor', 'option', 'p', str),
    n_threads=('number of threads for replacement operations', 'option', 't', int),
    parser_batch_size=('parser batch size', 'option', 'b', int)
)
def parse(in_path, out_path, sentence_processor=None, n_threads=4, parser_batch_size=1000):
    if sentence_processor is not None and sentence_processor.strip() != '':
        _sentence_processor = getattr(preprocessing, sentence_processor.strip())
    else:
        _sentence_processor = preprocessing.process_sentence1
    file_names = ['sick_test_annotated/SICK_test_annotated.txt', 'sick_train/SICK_train.txt']
    parser = spacy.load('en')
    for fn in file_names:
        logger.info('create forest for %s ...' % fn)
        out_base_name = os.path.join(out_path, DIR_BATCHES, fn.split('/')[0])
        make_parent_dir(out_base_name)
        process_records(records=read_file(os.path.join(in_path, fn)), out_base_name=out_base_name, record_reader=reader,
                        parser=parser, sentence_processor=_sentence_processor, concat_mode=None,
                        n_threads=n_threads, batch_size=parser_batch_size)
        logger.info('done.')


@plac.annotations(
    in_path=('corpora input folder', 'option', 'i', str),
    out_path=('corpora output folder', 'option', 'o', str),
    #n_threads=('number of threads for replacement operations', 'option', 't', int),
    #parser_batch_size=('parser batch size', 'option', 'b', int)
    parser=('parser: spacy or corenlp', 'option', 'p', str),
)
def parse_rdf(in_path, out_path, parser='spacy'):
    file_names = {'sick_test_annotated/SICK_test_annotated.txt': 'test.jsonl', 'sick_train/SICK_train.txt': 'train.jsonl'}
    logger.info('load parser...')
    if parser.strip() == 'spacy':
        _parser = spacy.load('en')
    elif parser.strip() == 'corenlp':
        _parser = CoreNLPDependencyParser(url='http://localhost:9000')
    else:
        raise NotImplementedError('parser=%s not implemented' % parser)
    logger.info('loaded parser %s' % type(_parser))
    n_failed = {}
    n_total = {}
    out_path = os.path.join(out_path, parser)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    for fn in file_names:
        fn_out = os.path.join(out_path, file_names[fn])
        logger.info('process file %s, save result to %s' % (fn, fn_out))
        n_failed[fn_out] = 0
        already_processed = {}
        if os.path.exists(fn_out):
            with open(fn_out) as fout:
                for l in fout.readlines():
                    _l = json.loads(l)
                    already_processed[_l['@id']] = l
        n_total[fn_out] = len(already_processed)
        logger.info('read %i already processed records' % len(already_processed))

        with open(fn_out, 'w') as fout:
            for i, record in enumerate(reader_rdf(in_path, fn)):
                if record['record_id'] in already_processed:
                    parsed_rdf_json = already_processed[record['record_id']]
                else:
                    try:
                        parsed_rdf = parse_and_convert_record(parser=_parser, **record)
                        parsed_rdf_json = json.dumps(parsed_rdf, ensure_ascii=False).encode('utf8') + u'\n'
                    except Exception as e:
                        logger.warning('failed to parse record=%s: %s' % (record['record_id'], str(e)))
                        n_failed[fn_out] += 1
                        continue
                fout.write(parsed_rdf_json)
        for fn_out in n_failed:
            logger.info('%s: failed to process %i of total %i records' % (fn_out, n_failed[fn_out], n_total[fn_out]))
    logger.debug('done')


@plac.annotations(
    mode=('processing mode', 'positional', None, str, ['PARSE', 'PARSE_DUMMY', 'MERGE', 'CREATE_INDICES', 'PARSE_RDF']),
    args='the parameters for the underlying processing method')
def main(mode, *args):
    if mode == 'PARSE_DUMMY':
        plac.call(parse_dummy, args)
    elif mode == 'PARSE':
        plac.call(parse, args)
    elif mode == 'MERGE':
        forest_merged, out_path_merged = plac.call(merge_batches, args)
        entailment_ids, entailment_strings = forest_merged.lexicon.get_ids_for_prefix(TYPE_ENTAILMENT)
        #logger.info('number of entailment types to predict: %i.' % len(entailment_ids))
        #numpy_dump(filename='%s.%s.%s' % (out_path_merged, TYPE_ENTAILMENT, FE_CLASS_IDS), ndarray=entailment_ids)
        save_class_ids(dir_path=out_path_merged, prefix_type=TYPE_ENTAILMENT, classes_ids=entailment_ids)
    elif mode == 'CREATE_INDICES':
        plac.call(create_index_files, args + ('--step-root', '2'))
    elif mode == 'PARSE_RDF':
        plac.call(parse_rdf, args)
    else:
        raise ValueError('unknown mode')


if __name__ == '__main__':
    plac.call(main)
