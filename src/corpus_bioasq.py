# coding=utf-8
import logging
from functools import partial

import plac
import spacy

import corpus
from constants import TYPE_ANCHOR, TYPE_TITLE, TYPE_SECTION, LOGGING_FORMAT

logger = logging.getLogger('corpus_bioasq')
logger.setLevel(logging.DEBUG)
logger_streamhandler = logging.StreamHandler()
logger_streamhandler.setLevel(logging.DEBUG)
logger_streamhandler.setFormatter(logging.Formatter(LOGGING_FORMAT))
logger.addHandler(logger_streamhandler)


def reader(records, keys_text, root_string, keys_meta=[], key_id=None, root_text_string=TYPE_ANCHOR):
    """
    :param records: dicts containing the textual data and optional meta data
    :param keys_text:
    :param root_string:
    :param keys_meta:
    :param key_id:
    :param root_text_string:
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
                v_meta = [v_meta]
            prepend_data_strings.extend(v_meta)
            prepend_parents.extend([-i - 1 for i in range(len(v_meta))])

        prepend = (prepend_data_strings, prepend_parents)
        for k_text in keys_text:
            yield (record[k_text], {'root_type': k_text,
                                    #'annotations': refs.get(terminal_uri_strings[i], None),
                                    'prepend_tree': prepend,
                                    'parent_prepend_offset': text_root_offset})
            prepend = None

    # TODO:
    #  DONE 1) construct prepend tuple (prepend_data, prepend_parents) from meta data
    #  2) split text (by paragraph labels)


@plac.annotations(
    out_path=('corpora out path', 'option', 'o', str),
)
def process_dummy(out_path):
    logger.debug('init spacy ...')
    nlp = spacy.load('en')
    records = [{"journal": "PLoS pathogens",
                "meshMajor": ["Animals","Drug Resistance, Multiple","Interferon Type I","Killer Cells, Natural","Klebsiella Infections","Klebsiella pneumoniae","Macrophages, Alveolar","Mice","Mice, Inbred C57BL","Mice, Knockout","Receptor Cross-Talk","Respiratory Tract Infections","Signal Transduction"],
                "year":"2017",
                "abstractText": "Klebsiella pneumoniae is a significant cause of nosocomial pneumonia and an alarming pathogen owing to the recent isolation of multidrug resistant strains. Understanding of immune responses orchestrating K. pneumoniae clearance by the host is of utmost importance. Here we show that type I interferon (IFN) signaling protects against lung infection with K. pneumoniae by launching bacterial growth-controlling interactions between alveolar macrophages and natural killer (NK) cells. Type I IFNs are important but disparate and incompletely understood regulators of defense against bacterial infections. Type I IFN receptor 1 (Ifnar1)-deficient mice infected with K. pneumoniae failed to activate NK cell-derived IFN-γ production. IFN-γ was required for bactericidal action and the production of the NK cell response-amplifying IL-12 and CXCL10 by alveolar macrophages. Bacterial clearance and NK cell IFN-γ were rescued in Ifnar1-deficient hosts by Ifnar1-proficient NK cells. Consistently, type I IFN signaling in myeloid cells including alveolar macrophages, monocytes and neutrophils was dispensable for host defense and IFN-γ activation. The failure of Ifnar1-deficient hosts to initiate a defense-promoting crosstalk between alveolar macrophages and NK cell was circumvented by administration of exogenous IFN-γ which restored endogenous IFN-γ production and restricted bacterial growth. These data identify NK cell-intrinsic type I IFN signaling as essential driver of K. pneumoniae clearance, and reveal specific targets for future therapeutic exploitations.",
                "pmid":"29112952",
                "title":"Natural killer cell-intrinsic type I IFN signaling controls Klebsiella pneumoniae growth during lung infection."}]
    records_converted = [{u"http://id.nlm.nih.gov/pubmed/journal": u"PLoS pathogens",
                          u"http://id.nlm.nih.gov/mesh": [u"Animals", u"Drug Resistance, Multiple", u"Interferon Type I", u"Killer Cells, Natural",
                                                          u"Klebsiella Infections", u"Klebsiella pneumoniae", u"Macrophages, Alveolar", u"Mice",
                                                          u"Mice, Inbred C57BL", u"Mice, Knockout", u"Receptor Cross-Talk",
                                                          u"Respiratory Tract Infections", u"Signal Transduction"],
                          u"http://id.nlm.nih.gov/pubmed/year": u"2017",
                          TYPE_SECTION+u'/abstract': u"Klebsiella pneumoniae is a significant cause of nosocomial pneumonia and an alarming pathogen owing to the recent isolation of multidrug resistant strains. Understanding of immune responses orchestrating K. pneumoniae clearance by the host is of utmost importance. Here we show that type I interferon (IFN) signaling protects against lung infection with K. pneumoniae by launching bacterial growth-controlling interactions between alveolar macrophages and natural killer (NK) cells. Type I IFNs are important but disparate and incompletely understood regulators of defense against bacterial infections. Type I IFN receptor 1 (Ifnar1)-deficient mice infected with K. pneumoniae failed to activate NK cell-derived IFN-γ production. IFN-γ was required for bactericidal action and the production of the NK cell response-amplifying IL-12 and CXCL10 by alveolar macrophages. Bacterial clearance and NK cell IFN-γ were rescued in Ifnar1-deficient hosts by Ifnar1-proficient NK cells. Consistently, type I IFN signaling in myeloid cells including alveolar macrophages, monocytes and neutrophils was dispensable for host defense and IFN-γ activation. The failure of Ifnar1-deficient hosts to initiate a defense-promoting crosstalk between alveolar macrophages and NK cell was circumvented by administration of exogenous IFN-γ which restored endogenous IFN-γ production and restricted bacterial growth. These data identify NK cell-intrinsic type I IFN signaling as essential driver of K. pneumoniae clearance, and reveal specific targets for future therapeutic exploitations.",
                          u'http://id.nlm.nih.gov/pubmed/pmid': u'http://id.nlm.nih.gov/pubmed/pmid/29112952',
                          TYPE_TITLE: u"Natural killer cell-intrinsic type I IFN signaling controls Klebsiella pneumoniae growth during lung infection."}]
    _reader = partial(reader, records=records_converted, key_id=u'http://id.nlm.nih.gov/pubmed/pmid',
                      keys_text=[TYPE_TITLE, TYPE_SECTION+u'/abstract'],
                      keys_meta=[u"http://id.nlm.nih.gov/pubmed/journal", u"http://id.nlm.nih.gov/pubmed/year", u"http://id.nlm.nih.gov/mesh"],
                      root_string=u'http://id.nlm.nih.gov/pubmed/resource')
    logger.debug('parse abstracts ...')
    forest, lexicon, lexicon_roots = corpus.process_records(parser=nlp, reader=_reader)
    lexicon.dump(filename=out_path, strings_only=True)
    forest.dump(filename=out_path)


@plac.annotations(
    mode=('processing mode', 'positional', None, str, ['PROCESS_DUMMY']),
    args='the parameters for the underlying processing method')
def main(mode, *args):
    if mode == 'PROCESS_DUMMY':
        plac.call(process_dummy, args)
    else:
        raise ValueError('unknown mode. use one of CREATE_BATCHES or MERGE_BATCHES.')


if __name__ == '__main__':
    plac.call(main)
