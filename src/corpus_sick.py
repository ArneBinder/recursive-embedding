import logging
import plac

from constants import LOGGING_FORMAT, TYPE_CONTEXT, SEPARATOR, TYPE_PARAGRAPH, TYPE_REF
from mytools import make_parent_dir
from corpus import process_records
import preprocessing

logger = logging.getLogger('corpus_sick')
logger.setLevel(logging.DEBUG)
logger_streamhandler = logging.StreamHandler()
logger_streamhandler.setLevel(logging.DEBUG)
logger_streamhandler.setFormatter(logging.Formatter(LOGGING_FORMAT))
logger.addHandler(logger_streamhandler)

TYPE_RELATEDNESS_SCORE = u'RELATEDNESS_SCORE'
TYPE_ENTAILMENT = u'ENTAILMENT'
TYPE_SICK_ID = u'http://clic.cimec.unitn.it/composes/sick'
TYPE_REF_TUPLE = TYPE_REF + SEPARATOR + u'other'

DUMMY_RECORD = {
    TYPE_RELATEDNESS_SCORE: u"4.5",
    TYPE_ENTAILMENT: u"NEUTRAL",
    u"sentence_1": u"A group of kids is playing in a yard and an old man is standing in the background",
    u"sentence_2": u"A group of boys in a yard is playing and a man is standing in the background",
    TYPE_SICK_ID: u"1"
}


def reader(records, keys_text=(u'sentence_1', u'sentence_2'), root_string=TYPE_SICK_ID,
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
    out_base_name=('corpora output base file name', 'option', 'o', str)
)
def parse_dummy(out_base_name):
    print(out_base_name)
    make_parent_dir(out_base_name)
    process_records(records=[DUMMY_RECORD], out_base_name=out_base_name, reader=reader,
                    sentence_processor=preprocessing.process_sentence1, concat_mode=None)


@plac.annotations(
    mode=('processing mode', 'positional', None, str, ['PARSE', 'PARSE_DUMMY', 'MERGE', 'CREATE_INDICES']),
    args='the parameters for the underlying processing method')
def main(mode, *args):
    if mode == 'PARSE_DUMMY':
        plac.call(parse_dummy, args)
    #elif mode == 'PARSE':
    #    plac.call(parse_dirs, args)
    #elif mode == 'MERGE':
    #    forest_merged, out_path_merged = plac.call(merge_batches, args)
    #    #rating_ids = forest_merged.lexicon.get_ids_for_prefix(TYPE_RATING)
    #    #logger.info('number of ratings to predict: %i' % len(rating_ids))
    #    #numpy_dump(filename='%s.%s' % (out_path_merged, FE_CLASS_IDS), ndarray=rating_ids)
    #    polarity_ids = forest_merged.lexicon.get_ids_for_prefix(TYPE_POLARITY)
    #    logger.info('number of polarities to predict: %i. save only %i for prediction.'
    #                % (len(polarity_ids), len(polarity_ids)-1))
    #    numpy_dump(filename='%s.%s' % (out_path_merged, FE_CLASS_IDS), ndarray=polarity_ids[:-1])
    #elif mode == 'CREATE_INDICES':
    #    plac.call(create_index_files, args)
    else:
        raise ValueError('unknown mode. use one of PROCESS_DUMMY or PROCESS_SINGLE.')


if __name__ == '__main__':
    plac.call(main)
