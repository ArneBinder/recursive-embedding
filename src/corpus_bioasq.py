# coding=utf-8
import json
import logging
import re
import shutil

import numpy as np
import os
from functools import partial
from multiprocessing import Pool

import plac
import spacy

from constants import TYPE_ANCHOR, TYPE_TITLE, TYPE_SECTION, LOGGING_FORMAT, TYPE_PARAGRAPH, SEPARATOR, DTYPE_IDX
import preprocessing
from lexicon import Lexicon
from corpus import FE_UNIQUE_HASHES, FE_COUNTS
from mytools import numpy_dump, numpy_exists, numpy_load
from corpus import DIR_BATCHES, FE_CLASS_IDS, merge_batches
from sequence_trees import Forest, FE_ROOT_POS

logger = logging.getLogger('corpus_bioasq')
logger.setLevel(logging.DEBUG)
logger_streamhandler = logging.StreamHandler()
logger_streamhandler.setLevel(logging.DEBUG)
logger_streamhandler.setFormatter(logging.Formatter(LOGGING_FORMAT))
logger.addHandler(logger_streamhandler)

TYPE_MESH = u"http://id.nlm.nih.gov/mesh"
TYPE_YEAR = u"http://id.nlm.nih.gov/pubmed/year"
TYPE_JOURNAL = u"http://id.nlm.nih.gov/pubmed/journal"
TYPE_PMID = u'http://id.nlm.nih.gov/pubmed/pmid'

KEY_MAPPING = {'journal': TYPE_JOURNAL,
               'meshMajor': TYPE_MESH,
               'year': TYPE_YEAR,
               'abstractText': TYPE_SECTION + SEPARATOR + u'abstract',
               'pmid': TYPE_PMID,
               'title': TYPE_TITLE}

PARAGRAPH_LABELS_UNIFORM = [u'BACKGROUND', u'CONCLUSIONS', u'METHODS', u'OBJECTIVE', u'RESULTS']
PARAGRAPH_LABEL_UNLABELED = u'UNLABELED'


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


def reader(records, keys_text, keys_text_structured, root_string, keys_meta=(), keys_mandatory=(), key_id=None,
           root_text_string=TYPE_ANCHOR, allowed_paragraph_labels=PARAGRAPH_LABELS_UNIFORM):
    """

    :param records: dicts containing the textual data and optional meta data
    :param keys_text:
    :param keys_text_structured:
    :param root_string:
    :param keys_meta:
    :param keys_mandatory:
    :param key_id:
    :param root_text_string:
    :param allowed_paragraph_labels:
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
                    # skip None values
                    if v_meta is None:
                        if k_meta in keys_mandatory:
                            raise Warning('value for mandatory key=%s is None' % k_meta)
                        continue
                    v_meta = [v_meta]
                # replace spaces by underscores
                prepend_data_strings.extend([k_meta + SEPARATOR + v.replace(' ', '_') for v in v_meta])
                prepend_parents.extend([-i - 1 for i in range(len(v_meta))])

            prepend_data_strings.append(TYPE_SECTION)
            prepend_parents.append(text_root_offset - len(prepend_parents))
            text_root_offset = len(prepend_parents) - 1

            #if None in prepend_data_strings:
            #    print record

            prepend = (prepend_data_strings, prepend_parents)

            record_data = []

            for k_text in keys_text:
                # debug
                if record[k_text] is None:
                    if key_id is not None:
                        logger.debug('entry with %s=%s contains None text (@k_text=%s)' % (key_id, record[key_id], k_text))
                    else:
                        logger.debug('entry contains None text (@k_text=%s): %s' % (k_text, str(record)))
                    if k_text in keys_mandatory:
                        raise Warning('value for mandatory key=%s is None' % k_text)
                    continue
                record_data.append((record[k_text], {'root_type': k_text, 'prepend_tree': prepend, 'parent_prepend_offset': text_root_offset}))
                prepend = None

            for k_text in keys_text_structured:
                # debug
                if record[k_text] is None:
                    if key_id is not None:
                        logger.debug('entry with %s=%s contains None text (@k_text=%s)' % (key_id, record[key_id], k_text))
                    else:
                        logger.debug('entry contains None text (@k_text=%s): %s' % (k_text, str(record)))
                    if k_text in keys_mandatory:
                        raise Warning('value for mandatory key=%s is None' % k_text)
                    continue
                # debug end
                matches = re.split('(^|\.)([A-Z][A-Z ]{3,}[A-Z]): ', record[k_text])
                # if no label candidate was found or content before first label is available add UNLABELED label
                if len(matches) < 2 or matches[0] + matches[1] != u'':
                    matches = [PARAGRAPH_LABEL_UNLABELED] + matches + ['']
                else:
                    # otherwise strip empty contents
                    matches = matches[2:] + ['']
                assert len(matches) % 3 == 0, 'wrong amount of matches (has to be a multiple of 3): %s' % str(matches)

                labels, texts = zip(*[(matches[i], matches[i+1]+matches[i+2]) for i in range(0, len(matches), 3)])
                if allowed_paragraph_labels is not None and any(l not in allowed_paragraph_labels for l in labels):
                    raise Warning('one of the labels is not in allowed_paragraph_labels')

                for i, text in enumerate(texts):
                    if text == u'':
                        continue
                    record_data.append((text, {'root_type': TYPE_PARAGRAPH + SEPARATOR + labels[i], 'prepend_tree': prepend, 'parent_prepend_offset': text_root_offset}))
                    prepend = None

            # has to be done in the end because the whole record should be discarded at once if an exception is raised
            for d in record_data:
                yield d
            count_finished += 1
        except Warning:
            count_discarded += 1
            pass
        except Exception as e:
            count_discarded += 1
            logger.warning('failed to process record (%s): %s' % (e.message, str(record)))

    logger.debug('discarded %i of %i records' % (count_discarded, count_finished + count_discarded))


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
                          key_id=TYPE_PMID,
                          keys_text=[TYPE_TITLE],
                          keys_text_structured=[TYPE_SECTION + SEPARATOR + u'abstract'],
                          # ATTENTION: mesh has to be the first meta data because MESH_ROOT_OFFSET has to be fixed, but
                          # journal and year are not mandatory
                          keys_meta=[TYPE_MESH, TYPE_JOURNAL, TYPE_YEAR],
                          keys_mandatory=[TYPE_MESH, TYPE_SECTION + SEPARATOR + u'abstract'],
                          allowed_paragraph_labels=PARAGRAPH_LABELS_UNIFORM,
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


def prepare_file(fn, in_path, out_path, mappings):
    logger.debug('process %s...' % fn)
    out_fn = os.path.join(out_path, os.path.basename(fn))
    if os.path.exists(out_fn):
        logger.debug('%s was already processed' % os.path.basename(fn))
        return
    with open(os.path.join(in_path, fn), 'r') as f:
        data = f.read()
    for m in mappings:
        data = data.replace('.' + m[0] + ':', '.' + m[1] + ':')
        data = data.replace('"' + m[0] + ':', '"' + m[1] + ':')

    with open(out_fn, 'w') as f:
        f.write(data)


@plac.annotations(
    in_path=('corpora input folder', 'option', 'i', str),
    mapping_file=('path to the file containing abstract label mappings '
                  '(see https://structuredabstracts.nlm.nih.gov/downloads.shtml)', 'option', 'm', str),
    n_threads=('number of threads for replacement operations', 'option', 't', int),
)
def prepare_batches(in_path, mapping_file, n_threads=4):
    # move all files to backup
    bk_path = in_path+'_bk'
    if not os.path.exists(bk_path):
        logger.info('create backup folder: %s' % bk_path)
        shutil.move(in_path, bk_path)
        os.mkdir(in_path)

    out_path = in_path
    in_path = bk_path
    file_names = os.listdir(in_path)

    # get mappings
    with open(mapping_file, 'r') as f:
        content = f.readlines()
    # reverse to replace long mappings first
    mappings = [tuple(x.split('|')[:2]) for x in reversed(content)]

    p = Pool(n_threads)
    p.map(partial(prepare_file, in_path=in_path, out_path=out_path, mappings=mappings), file_names)


@plac.annotations(
    in_path=('corpora input folder', 'option', 'i', str),
    out_path=('corpora output folder', 'option', 'o', str),
    n_threads=('number of threads for parsing', 'option', 't', int),
    parser_batch_size=('parser batch size', 'option', 'b', int),
)
def parse_batches(in_path, out_path, n_threads=4, parser_batch_size=1000):
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    logger_fh = logging.FileHandler(os.path.join(out_path, 'corpus-parse.log'))
    logger_fh.setLevel(logging.DEBUG)
    logger_fh.setFormatter(logging.Formatter(LOGGING_FORMAT))
    logger.addHandler(logger_fh)

    out_path = os.path.join(out_path, DIR_BATCHES)
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    logger.debug('use in_path=%s, out_path=%s, n_threads=%i, parser_batch_size=%i' %
                 (in_path, out_path, n_threads, parser_batch_size))
    logger.info('init parser ...')
    parser = spacy.load('en')

    for in_file in os.listdir(in_path):
        logger.info('parse file: %s' % os.path.basename(in_file))
        out_base_name = os.path.join(out_path, os.path.basename(in_file))
        process_records(records=read_file(os.path.join(in_path, in_file)), out_base_name=out_base_name, parser=parser,
                        n_threads=n_threads, batch_size=parser_batch_size)


@plac.annotations(
    merged_forest_path=('path to merged forest', 'option', 'o', str),
    split_count=('count of produced index files', 'option', 'c', int)
)
def create_index_files(merged_forest_path, split_count=2):
    logger_fh = logging.FileHandler(os.path.join(merged_forest_path, '../..', 'corpus-indices.log'))
    logger_fh.setLevel(logging.INFO)
    logger_fh.setFormatter(logging.Formatter(LOGGING_FORMAT))
    logger.addHandler(logger_fh)

    logger.info('split_count=%i out_path=%s' % (split_count, merged_forest_path))

    root_pos = numpy_load('%s.%s' % (merged_forest_path, FE_ROOT_POS), assert_exists=True)

    logger.info('total number of indices: %i' % len(root_pos))
    indices = np.arange(len(root_pos), dtype=DTYPE_IDX)
    np.random.shuffle(indices)
    for i, split in enumerate(np.array_split(indices, split_count)):
        numpy_dump('%s.idx.%i' % (merged_forest_path, i), split)


@plac.annotations(
    mode=('processing mode', 'positional', None, str, ['PARSE_DUMMY', 'PREPARE_BATCHES', 'PARSE_SINGLE', 'PARSE_BATCHES',
                                                       'MERGE_BATCHES', 'CREATE_INDICES']),
    args='the parameters for the underlying processing method')
def main(mode, *args):
    if mode == 'PARSE_DUMMY':
        plac.call(parse_dummy, args)
    elif mode == 'PREPARE_BATCHES':
        plac.call(prepare_batches, args)
    elif mode == 'PARSE_SINGLE':
        plac.call(parse_single, args)
    elif mode == 'PARSE_BATCHES':
        plac.call(parse_batches, args)
    elif mode == 'MERGE_BATCHES':
        forest_merged, out_path_merged = plac.call(merge_batches, args)
        mesh_ids = forest_merged.lexicon.get_ids_for_prefix(TYPE_MESH)
        numpy_dump(filename='%s.%s' % (out_path_merged, FE_CLASS_IDS), ndarray=mesh_ids)
    elif mode == 'CREATE_INDICES':
        plac.call(create_index_files, args)
    else:
        raise ValueError('unknown mode. use one of PROCESS_DUMMY or PROCESS_SINGLE.')


if __name__ == '__main__':
    plac.call(main)
