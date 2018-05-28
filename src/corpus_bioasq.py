# coding=utf-8
import json
import logging
import os
from functools import partial

import plac
import spacy

import corpus
from constants import TYPE_ANCHOR, TYPE_TITLE, TYPE_SECTION, LOGGING_FORMAT, TYPE_PARAGRAPH

logger = logging.getLogger('corpus_bioasq')
logger.setLevel(logging.DEBUG)
logger_streamhandler = logging.StreamHandler()
logger_streamhandler.setLevel(logging.DEBUG)
logger_streamhandler.setFormatter(logging.Formatter(LOGGING_FORMAT))
logger.addHandler(logger_streamhandler)



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
        prepend_data_strings = [root_string]
        prepend_parents = [0]
        if key_id is not None:
            prepend_data_strings.append(record[key_id])
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
            prepend_data_strings.extend(v_meta)
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


def read_file(in_file):
    #for file in os.listdir(in_path):
    with open(in_file) as f:
        for line in f:
            if line.startswith('{') and line.endswith(',\n'):
                encoded = json.loads(line[:-2])
                converted = convert_record(encoded)
                yield converted
            else:
                logger.warning('skip line: %s' % line)


def convert_record(record, mapping={'journal': u"http://id.nlm.nih.gov/pubmed/journal",
                                                             'meshMajor': u"http://id.nlm.nih.gov/mesh",
                                                             'year': u"http://id.nlm.nih.gov/pubmed/year",
                                                             'abstractText': TYPE_SECTION+u'/abstract',
                                                             'pmid': u'http://id.nlm.nih.gov/pubmed/pmid',
                                                             'title': TYPE_TITLE
                                                             }):
    result = {}
    for m in mapping:
        result[mapping[m]] = record[m]
    return result


@plac.annotations(
    out_path=('corpora out path', 'option', 'o', str),
)
def process_dummy(out_path):
    logger.debug('init spacy ...')
    nlp = spacy.load('en')

    record_converted = convert_record(DUMMY_RECORD)
    _reader = partial(reader, records=[record_converted], key_id=u'http://id.nlm.nih.gov/pubmed/pmid',
                      keys_text=[TYPE_TITLE, TYPE_SECTION+u'/abstract'],
                      keys_meta=[u"http://id.nlm.nih.gov/pubmed/journal", u"http://id.nlm.nih.gov/pubmed/year", u"http://id.nlm.nih.gov/mesh"],
                      paragraph_labels=[u'BACKGROUND', u'CONCLUSIONS', u'METHODS', u'OBJECTIVE', u'RESULTS'],
                      root_string=u'http://id.nlm.nih.gov/pubmed/resource')
    logger.debug('parse abstracts ...')
    forest, lexicon, lexicon_roots = corpus.process_records(parser=nlp, reader=_reader)
    lexicon.dump(filename=out_path, strings_only=True)
    forest.dump(filename=out_path)


@plac.annotations(
    out_path=('corpora out path', 'option', 'o', str),
    in_file=('corpora input file', 'option', 'i', str),
)
def process_single(in_file, out_path, nlp=spacy.load('en')):

    _reader = partial(reader, records=read_file(in_file), key_id=u'http://id.nlm.nih.gov/pubmed/pmid',
                      keys_text=[TYPE_TITLE, TYPE_SECTION+u'/abstract'],
                      keys_meta=[u"http://id.nlm.nih.gov/pubmed/journal", u"http://id.nlm.nih.gov/pubmed/year", u"http://id.nlm.nih.gov/mesh"],
                      paragraph_labels=[u'BACKGROUND', u'CONCLUSIONS', u'METHODS', u'OBJECTIVE', u'RESULTS'],
                      root_string=u'http://id.nlm.nih.gov/pubmed/resource')
    #logger.debug('init spacy ...')
    #nlp = spacy.load('en')
    logger.debug('parse abstracts ...')
    forest, lexicon, lexicon_roots = corpus.process_records(parser=nlp, reader=_reader)
    out_path += os.path.basename(in_file)
    lexicon.dump(filename=out_path, strings_only=True)
    forest.dump(filename=out_path)


@plac.annotations(
    mode=('processing mode', 'positional', None, str, ['PROCESS_DUMMY', 'PROCESS_SINGLE']),
    args='the parameters for the underlying processing method')
def main(mode, *args):
    if mode == 'PROCESS_DUMMY':
        plac.call(process_dummy, args)
    elif mode == 'PROCESS_SINGLE':
        plac.call(process_single, args)
    else:
        raise ValueError('unknown mode. use one of PROCESS_DUMMY or PROCESS_SINGLE.')


if __name__ == '__main__':
    plac.call(main)
