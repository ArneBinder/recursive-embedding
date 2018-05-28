# coding=utf-8
import json
import logging
import numpy as np
import os
from functools import partial

import plac
import spacy

from constants import TYPE_ANCHOR, TYPE_TITLE, TYPE_SECTION, LOGGING_FORMAT, TYPE_PARAGRAPH
import preprocessing
from lexicon import Lexicon
from corpus import FE_UNIQUE_HASHES, FE_COUNTS
from mytools import numpy_dump, numpy_exists
import corpus
from sequence_trees import Forest

logger = logging.getLogger('corpus_bioasq')
logger.setLevel(logging.DEBUG)
logger_streamhandler = logging.StreamHandler()
logger_streamhandler.setLevel(logging.DEBUG)
logger_streamhandler.setFormatter(logging.Formatter(LOGGING_FORMAT))
logger.addHandler(logger_streamhandler)

KEY_MAPPING = {'journal': u"http://id.nlm.nih.gov/pubmed/journal",
               'meshMajor': u"http://id.nlm.nih.gov/mesh",
               'year': u"http://id.nlm.nih.gov/pubmed/year",
               'abstractText': TYPE_SECTION+u'/abstract',
               'pmid': u'http://id.nlm.nih.gov/pubmed/pmid',
               'title': TYPE_TITLE}


DUMMY_RECORD = {"journal": u"PLoS pathogens",
                "meshMajor": [u"Animals", u"Drug Resistance, Multiple", u"Interferon Type I", u"Killer Cells, Natural",
                                                      u"Klebsiella Infections", u"Klebsiella pneumoniae", u"Macrophages, Alveolar", u"Mice",
                                                      u"Mice, Inbred C57BL", u"Mice, Knockout", u"Receptor Cross-Talk",
                                                      u"Respiratory Tract Infections", u"Signal Transduction"],
                "year": u"2017",
                "abstractText": u"BACKGROUND: In a c-VEP BCI setting, test subjects can have highly varying performances when different pseudorandom sequences are applied as stimulus, and ideally, multiple codes should be supported. On the other hand, repeating the experiment with many different pseudorandom sequences is a laborious process.OBJECTIVE: This study aimed to suggest an efficient method for choosing the optimal stimulus sequence based on a fast test and simple measures to increase the performance and minimize the time consumption for research trials.METHODS: A total of 21 healthy subjects were included in an online wheelchair control task and completed the same task using stimuli based on the m-code, the gold-code, and the Barker-code. Correct\/incorrect identification and time consumption were obtained for each identification. Subject-specific templates were characterized and used in a forward-step first-order model to predict the chance of completion and accuracy score.RESULTS: No specific pseudorandom sequence showed superior accuracy on the group basis. When isolating the individual performances with the highest accuracy, time consumption per identification was not significantly increased. The Accuracy Score aids in predicting what pseudorandom sequence will lead to the best performance using only the templates. The Accuracy Score was higher when the template resembled a delta function the most and when repeated templates were consistent. For completion prediction, only the shape of the template was a significant predictor.CONCLUSIONS: The simple and fast method presented in this study as the Accuracy Score, allows c-VEP based BCI systems to support multiple pseudorandom sequences without increase in trial length. This allows for more personalized BCI systems with better performance to be tested without increased costs.",
                  "pmid": u"291129 52",
                  "title": u"Natural killer cell-intrinsic type I IFN signaling controls Klebsiella pneumoniae growth during lung infection."}
DUMMY_RECORD_CONVERTED = {u"http://id.nlm.nih.gov/pubmed/journal": u"PLoS pathogens",
                      u"http://id.nlm.nih.gov/mesh": [u"Animals", u"Drug Resistance, Multiple", u"Interferon Type I", u"Killer Cells, Natural",
                                                      u"Klebsiella Infections", u"Klebsiella pneumoniae", u"Macrophages, Alveolar", u"Mice",
                                                      u"Mice, Inbred C57BL", u"Mice, Knockout", u"Receptor Cross-Talk",
                                                      u"Respiratory Tract Infections", u"Signal Transduction"],
                            u"http://id.nlm.nih.gov/pubmed/year": u"2017",
                            TYPE_SECTION+u'/abstract': u"BACKGROUND: In a c-VEP BCI setting, test subjects can have highly varying performances when different pseudorandom sequences are applied as stimulus, and ideally, multiple codes should be supported. On the other hand, repeating the experiment with many different pseudorandom sequences is a laborious process.OBJECTIVE: This study aimed to suggest an efficient method for choosing the optimal stimulus sequence based on a fast test and simple measures to increase the performance and minimize the time consumption for research trials.METHODS: A total of 21 healthy subjects were included in an online wheelchair control task and completed the same task using stimuli based on the m-code, the gold-code, and the Barker-code. Correct\/incorrect identification and time consumption were obtained for each identification. Subject-specific templates were characterized and used in a forward-step first-order model to predict the chance of completion and accuracy score.RESULTS: No specific pseudorandom sequence showed superior accuracy on the group basis. When isolating the individual performances with the highest accuracy, time consumption per identification was not significantly increased. The Accuracy Score aids in predicting what pseudorandom sequence will lead to the best performance using only the templates. The Accuracy Score was higher when the template resembled a delta function the most and when repeated templates were consistent. For completion prediction, only the shape of the template was a significant predictor.CONCLUSIONS: The simple and fast method presented in this study as the Accuracy Score, allows c-VEP based BCI systems to support multiple pseudorandom sequences without increase in trial length. This allows for more personalized BCI systems with better performance to be tested without increased costs.",
                            u'http://id.nlm.nih.gov/pubmed/pmid': u'http://id.nlm.nih.gov/pubmed/pmid/29112952',
                            TYPE_TITLE: u"Natural killer cell-intrinsic type I IFN signaling controls Klebsiella pneumoniae growth during lung infection."}


def multisplit(text, sep, sep_postfix=u': ', unlabeled=None):
    lastmatch = i = 0
    matches = [unlabeled]
    rest = []
    while i < len(text):
        for j, _s in enumerate(sep):
            s = _s + sep_postfix
            if text[i:].startswith(s):
                #if i > lastmatch:
                rest.append(text[lastmatch:i])
                matches.append(_s)  # Replace the string containing the matched separator with a tuple of which separator and where in the string the match occured
                lastmatch = i + len(s)
                i += len(s)
                break
        else:
            i += 1
    #if i > lastmatch:
    rest.append(text[lastmatch:i])
    return matches, rest


def reader(records, keys_text, root_string, keys_meta=(), key_id=None, root_text_string=TYPE_ANCHOR,
           paragraph_labels=(), uniform_paragraphs=False):
    """

    :param records: dicts containing the textual data and optional meta data
    :param keys_text:
    :param root_string:
    :param keys_meta:
    :param key_id:
    :param root_text_string:
    :param paragraph_labels:
    :param uniform_paragraphs:
    :return:
    """
    for record in records:
        try:
            prepend_data_strings = [root_string]
            prepend_parents = [0]
            if key_id is not None:
                prepend_data_strings.append(key_id + '/' + record[key_id])
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
                    # skip None values
                    if v_meta is None:
                        continue
                    v_meta = [v_meta]
                prepend_data_strings.extend([k_meta + '/' + v for v in v_meta])
                prepend_parents.extend([-i - 1 for i in range(len(v_meta))])

            prepend_data_strings.append(TYPE_SECTION)
            prepend_parents.append(text_root_offset - len(prepend_parents))
            text_root_offset = len(prepend_parents) - 1

            #if None in prepend_data_strings:
            #    print record

            prepend = (prepend_data_strings, prepend_parents)
            for k_text in keys_text:
                # debug
                if record[k_text] is None:
                    logger.warning('contains None text (@k_text=%s): %s' % (k_text, str(record)))
                # debug end
                matches, rest = multisplit(record[k_text], paragraph_labels)
                for i, text in enumerate(rest):
                    if text == u'':
                        continue
                    if matches[i] is not None:
                        root_type = TYPE_PARAGRAPH
                        if not uniform_paragraphs:
                            root_type += u'/' + matches[i]
                    else:
                        root_type = k_text
                    yield (text, {'root_type': root_type, 'prepend_tree': prepend, 'parent_prepend_offset': text_root_offset})
                    prepend = None
        except Exception as e:
            logger.warning('failed to process record (%s): %s' % (e.message, str(record)))


def read_file(in_file):
    with open(in_file) as f:
        for line in f:
            if line.startswith('{') and line.endswith(',\n'):
                encoded = json.loads(line[:-2])
                yield encoded
            else:
                logger.warning('skip line: %s' % line)


def convert_record(record, mapping=KEY_MAPPING):
    return {mapping[m]: record[m] for m in mapping}


@plac.annotations(
    records=('dicts containing BioASQ data obtained from json', 'option', 'r', str),
    out_base_name=('corpora output base file name', 'option', 'o', str),
)
def process_records(records, out_base_name, parser=spacy.load('en'), batch_size=1000, n_threads=4):
    if not Lexicon.exist(out_base_name, types_only=True) \
            or not Forest.exist(out_base_name) \
            or not numpy_exists('%s.%s' % (out_base_name, FE_UNIQUE_HASHES)) \
            or not numpy_exists('%s.%s' % (out_base_name, FE_COUNTS)):
        _reader = partial(reader,
                          records=(convert_record(r) for r in records),
                          key_id=u'http://id.nlm.nih.gov/pubmed/pmid',
                          keys_text=[TYPE_TITLE, TYPE_SECTION + u'/abstract'],
                          keys_meta=[u"http://id.nlm.nih.gov/pubmed/journal", u"http://id.nlm.nih.gov/pubmed/year",
                                     u"http://id.nlm.nih.gov/mesh"],
                          paragraph_labels=[u'BACKGROUND', u'CONCLUSIONS', u'METHODS', u'OBJECTIVE', u'RESULTS'],
                          root_string=u'http://id.nlm.nih.gov/pubmed/resource')
        logger.debug('parse abstracts ...')
        # forest, lexicon, lexicon_roots = corpus.process_records(parser=nlp, reader=_reader)

        lexicon = Lexicon()
        forest = lexicon.read_data(reader=_reader, sentence_processor=preprocessing.process_sentence1,
                                   parser=parser, batch_size=batch_size, concat_mode='sequence',
                                   inner_concat_mode='tree', expand_dict=True, as_tuples=True,
                                   return_hashes=True, n_threads=n_threads)

        forest.set_children_with_parents()
        roots = forest.roots
        # ids are at one position after roots
        root_ids = forest.data[roots + 1]
        forest.set_root_ids(root_ids=root_ids)

        #out_path = os.path.join(out_path, os.path.basename(in_file))
        lexicon.dump(filename=out_base_name, strings_only=True)
        forest.dump(filename=out_base_name)

        unique, counts = np.unique(forest.data, return_counts=True)
        # unique.dump('%s.%s' % (filename, FE_UNIQUE_HASHES))
        # counts.dump('%s.%s' % (filename, FE_COUNTS))
        numpy_dump('%s.%s' % (out_base_name, FE_UNIQUE_HASHES), unique)
        numpy_dump('%s.%s' % (out_base_name, FE_COUNTS), counts)
    else:
        logger.debug('%s was already processed' % out_base_name)


@plac.annotations(
    out_base_name=('corpora output base file name', 'option', 'o', str)
)
def parse_dummy(out_base_name):
    process_records(records=[DUMMY_RECORD], out_base_name=out_base_name)


@plac.annotations(
    in_file=('corpora input file', 'option', 'i', str),
    out_path=('corpora output folder', 'option', 'o', str),
)
def parse_single(in_file, out_path):
    logger.info('init parser ...')
    parser = spacy.load('en')
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    out_base_name = os.path.join(out_path, os.path.basename(in_file))
    process_records(records=read_file(in_file), out_base_name=out_base_name, parser=parser)


@plac.annotations(
    in_path=('corpora input folder', 'option', 'i', str),
    out_path=('corpora output folder', 'option', 'o', str),
    n_threads=('number of threads for parsing', 'option', 't', int),
    parser_batch_size=('parser batch size', 'option', 'b', int),
)
def parse_batches(in_path, out_path, n_threads=4, parser_batch_size=1000):
    logger_fh = logging.FileHandler(os.path.join(out_path, 'corpus-parse.log'))
    logger_fh.setLevel(logging.DEBUG)
    logger_fh.setFormatter(logging.Formatter(LOGGING_FORMAT))
    logger.addHandler(logger_fh)

    logger.debug('use in_path=%s, out_path=%s, n_threads=%i, parser_batch_size=%i' %
                 (in_path, out_path, n_threads, parser_batch_size))
    logger.info('init parser ...')
    parser = spacy.load('en')
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    for in_file in os.listdir(in_path):
        logger.info('parse file: %s' % os.path.basename(in_file))
        out_base_name = os.path.join(out_path, os.path.basename(in_file))
        process_records(records=read_file(os.path.join(in_path, in_file)), out_base_name=out_base_name, parser=parser, n_threads=n_threads,
                        batch_size=parser_batch_size)


@plac.annotations(
    mode=('processing mode', 'positional', None, str, ['PARSE_DUMMY', 'PARSE_SINGLE', 'PARSE_BATCHES',
                                                       'MERGE_BATCHES']),
    args='the parameters for the underlying processing method')
def main(mode, *args):
    if mode == 'PARSE_DUMMY':
        plac.call(parse_dummy, args)
    elif mode == 'PARSE_SINGLE':
        plac.call(parse_single, args)
    elif mode == 'PARSE_BATCHES':
        plac.call(parse_batches, args)
    elif mode == 'MERGE_BATCHES':
        plac.call(corpus.merge_batches, args)
    else:
        raise ValueError('unknown mode. use one of PROCESS_DUMMY or PROCESS_SINGLE.')


if __name__ == '__main__':
    plac.call(main)
