# coding: utf-8
#from __future__ import unicode_literals
import Queue
import logging
import threading
from datetime import datetime, timedelta
import time
import plac
import numpy as np

import os
from threading import Thread

import spacy
from spacy.strings import hash_string
from rdflib.graph import ConjunctiveGraph as Graph
from rdflib.store import Store
from rdflib.plugin import get as plugin
from rdflib.term import URIRef
from rdflib import Namespace
from rdflib.namespace import RDF, RDFS

from lexicon import Lexicon, FE_STRINGS
from sequence_trees import Forest, FE_ROOT_ID
from constants import DTYPE_HASH, DTYPE_COUNT, TYPE_REF, UNKNOWN_EMBEDDING, vocab_manual, TYPE_ROOT, TYPE_ANCHOR, \
    TYPE_SECTION_SEEALSO, TYPE_PARAGRAPH, TYPE_TITLE, TYPE_REF_SEEALSO, DTYPE_IDX, LOGGING_FORMAT
import preprocessing
from mytools import numpy_dump, numpy_load

"""
prerequisites:
    set up / install:
      * virtuoso docker image: 
            see https://joernhees.de/blog/2015/11/23/setting-up-a-linked-data-mirror-from-rdf-dumps-dbpedia-2015-04-freebase-wikidata-linkedgeodata-with-virtuoso-7-2-1-and-docker-optional/
      * unixODBC
      * virtuoso odbc driver (libvirtodbc0) included in libvirtodbc0_7.2_amd64.deb (e.g. from https://github.com/Dockerizing/triplestore-virtuoso7)
            wget https://github.com/Dockerizing/triplestore-virtuoso7/raw/master/libvirtodbc0_7.2_amd64.deb
            sudo apt install /PATH/TO/libvirtodbc0_7.2_amd64.deb
      * rdflib:
            pip install rdflib
      * virtuoso-python: see https://github.com/maparent/virtuoso-python
            git clone git@github.com:maparent/virtuoso-python.git && cd virtuoso-python
            pip install -r requirements.txt
            python setup.py install
      * fix connection encoding: due libvirtodbc0 knows only utf-8, the connections settings in virtuoso.vstore of the 
        virtuoso-python package has to be adapted. Change the line ~234 that is commented out below with the other one:
            #connection.setencoding(unicode, 'utf-32LE', pyodbc.SQL_WCHAR)
            connection.setencoding(unicode, 'utf-8', pyodbc.SQL_CHAR)
        and further change line ~353:
            #log.log(9, "query: \n" + str(q))
            log.log(9, "query: \n" + unicode(q))
        
    
    settings:            
        ~/.odbc.ini (or /etc/odbc.ini) containing:
        "
        [VOS]
        Description = Open Virtuoso
        Driver      = /usr/lib/odbc/virtodbc_r.so
        Address     = localhost:1111
        Locale      = en.UTF-8
        "
    
    start virtuoso docker image (use 8gb of ram):
        docker run -it -p 8890:8890 -p 1111:1111 -v ~/virtuoso_db:/var/lib/virtuoso-opensource-7 -e "NumberOfBuffers=$((8*85000))" joernhees/virtuoso
        
    # debug
    # docker rm test; docker build -t test -f docker/create_corpus/dbpedia-nif/Dockerfile . && docker run --name test --net="host" -v /mnt/WIN/ML/data/corpora:/root/corpora_out -it test bash  
        
    ATTENTION:
        DO NOT PUT A FILE virtuoso.py IN THE SAME FOLDER!
"""
NIF = Namespace("http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#")
DBR = Namespace("http://dbpedia.org/resource/")
DBO = Namespace("http://dbpedia.org/ontology/")
ITSRDF = Namespace("http://www.w3.org/2005/11/its/rdf#")
ns_dict = {'nif': NIF, 'dbr': DBR, 'itsrdf': ITSRDF, 'rdf': RDF, 'rdfs': RDFS, 'dbo': DBO}

PREFIX_CONTEXT = '?dbpv=2016-10&nif=context'

#FE_RESOURCE_HASHES = 'resource.hash'
FE_ROOT_ID_FAILED = 'root.id.failed'
FE_FAILED = 'failed'
FE_UNIQUE_HASHES = 'unique.hash'
FE_COUNTS = 'count'
FE_UNIQUE_HASHES_FILTERED = 'unique.hash.filtered'
FE_UNIQUE_HASHES_DISCARDED = 'unique.hash.discarded'
FE_UNIQUE_COUNTS_FILTERED = 'count.hash.filtered'
FE_UNIQUE_COUNTS_DISCARDED = 'count.hash.discarded'
FE_ROOT_SEEALSO_COUNT = 'root.seealso.count'
FE_ROOT_CONTEXT_SIZE = 'root.context.size'

DIR_BATCHES = 'batches'
DIR_BATCHES_CONVERTED = 'batches_converted'
DIR_MERGED = 'merged'

PREFIX_FN = 'forest'

logger = logging.getLogger('corpus_dbpedia_nif')
logger.setLevel(logging.DEBUG)
logger_streamhandler = logging.StreamHandler()
logger_streamhandler.setLevel(logging.INFO)
logger_streamhandler.setFormatter(logging.Formatter(LOGGING_FORMAT))
logger.addHandler(logger_streamhandler)
#logger.addHandler(logging.FileHandler('../virtuoso_test.log', mode='w', encoding='utf-8'))
logger.propagate = False

logger_virtuoso = logging.getLogger('virtuoso.vstore')
logger_virtuoso.setLevel(logging.INFO)
# TODO: change back
#logger_virtuoso.addHandler(logging.FileHandler('../virtuoso_new.log', mode='w', encoding='utf-8'))
virtuoso_sh = logging.StreamHandler()
virtuoso_sh.setFormatter(logger_streamhandler.formatter)
logger_virtuoso.addHandler(virtuoso_sh)
logger_virtuoso.propagate = False


def query_context(graph, initBindings=None, cursor=None):
    q_str = (u'CONSTRUCT {'
             '  ?s ?p ?o .'
             '  ?context ?context_p ?context_o .'
             '} WHERE {'
             '  ?context ?context_p ?context_o .'
             '  ?s nif:referenceContext ?context .'
             '  ?s ?p ?o .'
             '} '
             )
    #logger.debug(q_str)
    res = graph.query(q_str, cursor=cursor, initNs=ns_dict, initBindings=initBindings)
    return res


def query_see_also_links_DEP(graph, initBindings=None):
    q_str = (#'SELECT ?context ?linkRef '
             u'SELECT ?linkRef '
             'WHERE {'
             '  ?context a nif:Context . '
             '  ?seeAlsoSection a nif:Section . '
             '  ?seeAlsoSection nif:superString+ ?context . '
             '  ?seeAlsoSection nif:beginIndex ?beginIndex . '
             '  ?seeAlsoSection nif:endIndex ?endIndex . '
             '  ?link nif:superString+ ?seeAlsoSection . '
             '  ?link <http://www.w3.org/2005/11/its/rdf#taIdentRef> ?linkRef . '
             '  ?context nif:isString ?contextStr . '
             '  FILTER (?endIndex - ?beginIndex > 0) '
             '  FILTER (STRLEN(?contextStr) >= ?endIndex) '
             '  BIND(SUBSTR(STR(?contextStr), ?beginIndex + 1, ?endIndex - ?beginIndex) AS ?seeAlsoSectionStr) '
             '  FILTER (STRSTARTS(STR(?seeAlsoSectionStr), "See also")) '
             #'  BIND(IRI(CONCAT(STR(?linkRef), "?dbpv=2016-10&nif=context")) AS ?linkRefContext) '
             '} '
             )
    #logger.debug(q_str)
    res = graph.query(q_str, initNs=ns_dict, initBindings=initBindings)
    return res


def query_see_also_links(graph, context, cursor=None):#initBindings=None):
    q_str = (u'SELECT ?linkRef WHERE' 
             '{'
             '  <'+context+'> a nif:Context . '
             '  ?seeAlsoSection a nif:Section . '
             '  ?seeAlsoSection nif:superString+ <'+context+'> . '
             '  ?seeAlsoSection nif:beginIndex ?beginIndex . '
             '  ?seeAlsoSection nif:endIndex ?endIndex . '
             '  ?link nif:superString+ ?seeAlsoSection . '
             '  ?link <http://www.w3.org/2005/11/its/rdf#taIdentRef> ?linkRef . '
             '  <'+context+'> nif:isString ?contextStr . '
             '  FILTER (?beginIndex < ?endIndex) . '
             '  FILTER (STRLEN(?contextStr) >= ?endIndex) . '
             '  BIND(SUBSTR(STR(?contextStr), ?beginIndex + 1, ?endIndex - ?beginIndex) AS ?seeAlsoSectionStr) '
             '  FILTER (STRSTARTS(STR(?seeAlsoSectionStr), "See also"^^xsd:string)) . '
             #'  FILTER (STRSTARTS(STR(?seeAlsoSectionStr), "See also")) . '
             '} '
             )
    #logger.debug(q_str)
    res = graph.query(q_str, cursor=cursor, initNs=ns_dict)#, initBindings=initBindings)
    return res


def query_first_section_structure_DEP(graph, initBindings=None):
    q_str = (
        u'construct {'
        ' ?s ?p ?o .'
        #' ?superString nif:subString ?s .'
        ' ?context ?context_p ?context_o .'
        #' ?s nif:superOffset ?superOffset .'
        #' ?s nif:superString ?superString .'
        '} '
        #'select distinct ?p '
        'where {'
        ' ?context ?context_p ?context_o .'
        ' VALUES ?context_p {nif:beginIndex nif:endIndex rdf:type nif:isString } .'
        ' ?s nif:referenceContext ?context .'
        ' ?context nif:firstSection ?firstSection .'
        ' ?firstSection nif:beginIndex 0 .'
        ' ?s nif:superString* ?firstSection .'
        ' ?s ?p ?o .'
        ' VALUES ?p {nif:beginIndex nif:endIndex nif:superString itsrdf:taIdentRef rdf:type} .'
        #' ?s nif:superString ?superString .'
        #' ?superString nif:beginIndex ?superOffset .'
        #' ?s itsrdf:taIdentRef ?ref .'
        #' ?s ?p2 ?oo2 .'
        
        #' FILTER (?p != nif:referenceContext)'
        '}')
    #logger.debug(q_str)
    res = graph.query(q_str, initNs=ns_dict, initBindings=initBindings)
    #print(type(res))
    return res


def query_first_section_structure(graph, context, cursor=None):
    q_str = (
        u'CONSTRUCT {'
        ' ?s ?p ?o .'
        ' <'+context+'> ?context_p ?context_o .'
        '} WHERE {'
        ' <'+context+'> ?context_p ?context_o .'
        ' VALUES ?context_p {nif:beginIndex nif:endIndex rdf:type nif:isString } .'
        ' ?s nif:referenceContext <'+context+'> .'
        ' <'+context+'> nif:firstSection ?firstSection .'
        ' ?firstSection nif:beginIndex "0"^^xsd:nonNegativeInteger .'
        #' ?firstSection nif:beginIndex 0 .'
        ' ?s nif:superString* ?firstSection .'
        ' ?s ?p ?o .'
        ' VALUES ?p {nif:beginIndex nif:endIndex nif:superString itsrdf:taIdentRef rdf:type} .'
        '}')
    #logger.debug(q_str)
    res = graph.query(q_str, cursor=cursor, initNs=ns_dict)#, initBindings=initBindings)
    # print(type(res))
    return res


def tree_from_sorted_parent_triples(sorted_parent_triples, root_id_str,
                                    see_also_refs,
                                    see_also_ref_type=TYPE_REF_SEEALSO,
                                    root_type=TYPE_ROOT,
                                    anchor_type=TYPE_ANCHOR,
                                    terminal_types=None,
                                    see_also_section_type=TYPE_SECTION_SEEALSO
                                    ):
    """
    Constructs a tree from triples.
    :param sorted_parent_triples: list of triples (uri, uri_type, uri_parent)
                                  sorted by ?parent_beginIndex DESC(?parent_endIndex) ?beginIndex DESC(?endIndex))
    :param lexicon: the lexicon used to convert the uri strings into integer ids
    :param root_id_str: uri string added as direct root child, e.g. "http://dbpedia.org/resource/Damen_Group"
    :param root_type: uri string used as root data, e.g. "http://dbpedia.org/resource"
    :param anchor_type: the tree represented in sorted_parent_triples will be anchored via this uri string to the
                        root_type node, e.g. "http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#Context"
    :param terminal_types: uri strings that are considered as terminals, i.e. are used as roots of parsed string trees
    :return: the tree data strings, parents, position mappings ({str(uri): offset}) and list of terminal types
    """
    if terminal_types is None:
        terminal_types = [TYPE_PARAGRAPH, TYPE_TITLE]
    temp_data = [root_type, root_id_str, anchor_type, see_also_section_type]
    temp_parents = [0, -1, -2, -3]
    for see_also_ref, in see_also_refs:
        temp_data.append(see_also_ref_type)
        temp_parents.append(3 - len(temp_parents))
        temp_data.append(see_also_ref.toPython())
        temp_parents.append(-1)
    pre_len = len(temp_data)
    positions = {}
    parent_uris = {}
    terminal_parent_positions = {}
    terminal_types_list = []

    for uri, uri_type, uri_parent in sorted_parent_triples:
        uri_str = uri.toPython()
        uri_type_str = uri_type.toPython()
        uri_parent_str = uri_parent.toPython()
        if len(temp_data) == pre_len:
            positions[uri_parent_str] = 2
        parent_uris[uri_str] = uri_parent_str
        positions[uri_str] = len(temp_data)
        if uri_type_str in terminal_types:
            terminal_types_list.append(uri_type_str)
            terminal_parent_positions[uri_str] = positions[parent_uris[uri_str]]
        else:
            temp_data.append(uri_type_str)
            temp_parents.append(positions[uri_parent_str] - len(temp_parents))

    return temp_data, temp_parents, terminal_parent_positions, terminal_types_list


def process_link_refs(link_refs, ref_type_str=TYPE_REF):
    refs = {}
    # tuple format: ?superString ?target ?superOffset ?beginIndex ?endIndex ?type
    for ref_tuple in link_refs:
        super_string = ref_tuple[0].toPython()
        target_str = ref_tuple[1].toPython()
        # target_id = lexicon[target]
        offset = int(ref_tuple[2])
        begin_index = int(ref_tuple[3])
        end_index = int(ref_tuple[4])

        ref_list = refs.get(super_string, [])
        ref_list.append((begin_index - offset, end_index - offset, [ref_type_str, target_str], [0, -1]))
        refs[super_string] = ref_list

    return refs


def prepare_context_datas(graph, nif_context_strings, cursor):
    nif_context_datas = []
    query_datas, failed = query_context_datas(graph, nif_context_strings, cursor)

    for nif_context_str, query_data in query_datas:
        try:
            see_also_refs, children_typed, terminals, link_refs, nif_context_content_str = query_data
            tree_context_data_strings, tree_context_parents, terminal_parent_positions, terminal_types = tree_from_sorted_parent_triples(
                sorted_parent_triples=children_typed,
                see_also_refs=see_also_refs,
                root_id_str=nif_context_str[:-len(PREFIX_CONTEXT)])

            terminal_uri_strings, terminal_strings = zip(
                *[(uri.toPython(), nif_context_content_str[int(begin):int(end)]) for uri, begin, end in terminals])

            refs = process_link_refs(link_refs)

            nif_context_datas.append((tree_context_data_strings, tree_context_parents, terminal_strings,
                                      terminal_uri_strings, terminal_types, refs, terminal_parent_positions,
                                      nif_context_str))
        except Exception as e:
            failed.append((nif_context_str, e))
    return nif_context_datas, failed


def create_context_forest_DEP(nif_context_data, nlp, lexicon, n_threads=1):

    def terminal_reader():
        tree_context_data, tree_context_parents, terminal_strings, terminal_uri_strings, terminal_types, refs, \
        terminal_parent_positions, nif_context = tuple(nif_context_data)
        prepend = (tree_context_data, tree_context_parents)
        for i in range(len(terminal_strings)):
            yield (terminal_strings[i], {'root_type': terminal_types[i],
                                         'annotations': refs.get(terminal_uri_strings[i], None),
                                         'prepend_tree': prepend,
                                         'parent_prepend_offset': terminal_parent_positions[terminal_uri_strings[i]]})
            prepend = None

    forest = lexicon.read_data(reader=terminal_reader, sentence_processor=preprocessing.process_sentence1,
                               parser=nlp, batch_size=10000, concat_mode='sequence', inner_concat_mode='tree',
                               expand_dict=True, as_tuples=True, return_hashes=True, n_threads=n_threads)
    return forest


def create_contexts_forest(nif_context_datas, nlp, lexicon, n_threads=1, batch_size=1000):
    #tree_contexts = []
    #resource_hashes = []
    #resource_hashes_failed = []
    failed = []

    def terminal_reader():
        for nif_context_data in nif_context_datas:
            tree_context_data, tree_context_parents, terminal_strings, terminal_uri_strings, terminal_types, refs, \
            terminal_parent_positions, nif_context_str = tuple(nif_context_data)
            #res_hash = hash_string(context[:-len(PREFIX_CONTEXT)])
            try:
                prepend = (tree_context_data, tree_context_parents)
                for i in range(len(terminal_strings)):
                    yield (terminal_strings[i], {'root_type': terminal_types[i],
                                                 'annotations': refs.get(terminal_uri_strings[i], None),
                                                 'prepend_tree': prepend,
                                                 'parent_prepend_offset': terminal_parent_positions[
                                                     terminal_uri_strings[i]]})
                    prepend = None
            except Exception as e:
                failed.append((nif_context_str, e))

    forest = lexicon.read_data(reader=terminal_reader, sentence_processor=preprocessing.process_sentence1,
                               parser=nlp, batch_size=batch_size, concat_mode='sequence',
                               inner_concat_mode='tree', expand_dict=True, as_tuples=True,
                               return_hashes=True, n_threads=n_threads)
    forest.set_children_with_parents()
    roots = forest.roots
    # ids are at one position after roots
    root_ids = forest.data[roots + 1]
    # forest.set_root_ids(root_ids=np.array(resource_hashes, dtype=forest.data.dtype))
    forest.set_root_ids(root_ids=root_ids)

    #if len(tree_contexts) > 0:
    #    forest = Forest.concatenate(tree_contexts)
    #    forest.set_children_with_parents()
    #    roots = forest.roots
    #    # ids are at one position after roots
    #    root_ids = forest.data[roots+1]
    #    #forest.set_root_ids(root_ids=np.array(resource_hashes, dtype=forest.data.dtype))
    #    forest.set_root_ids(root_ids=root_ids)
    #else:
    #    forest = None

    return forest, failed


def query_context_datas(graph, context_strings, cursor):
    #logger.setLevel(logging.DEBUG)
    #logger_virtuoso.setLevel(logging.DEBUG)

    #context_uris = [URIRef(context_str) for context_str in context_strings]
    #logger.debug('len(context_uris)=%i' % len(context_uris))

    # ATTENTION: the following is BUGGY
    #contexts_res = query_context(graph, initBindings={'context': context_uris})
    #_graph = Graph()
    #for triple in contexts_res:
    #    _graph.add(triple)
    #    logger.debug(triple)
    _graph = graph

    query_datas = []
    failed = []
    #for context_uri in context_uris:
    for context_str in context_strings:
        try:
            t_start = datetime.now()
            res_context_seealsos = query_see_also_links(_graph, context=context_str, cursor=cursor) #initBindings={'context': context_uri})
            #logger.debug('len(res_context_seealsos)=%i' % len(res_context_seealsos))
            res_context = query_first_section_structure(_graph, context=context_str, cursor=cursor) #{'context': context_uri})
            #logger.debug('len(res_context)=%i' % len(res_context))

            g_structure = Graph()
            for triple in res_context:
                g_structure.add(triple)
                #logger.debug(triple)

            children_typed = g_structure.query(
                u'SELECT DISTINCT ?child ?type_child ?parent WHERE {?parent a ?type . VALUES ?type {nif:Section nif:Context} . ?child a ?type_child . ?parent nif:beginIndex ?parent_beginIndex . ?parent nif:endIndex ?parent_endIndex . ?child nif:superString ?parent . ?child nif:beginIndex ?child_beginIndex . ?child nif:endIndex ?child_endIndex .} ORDER BY ?parent_beginIndex DESC(?parent_endIndex) ?child_beginIndex DESC(?child_endIndex)',
                initNs=ns_dict)

            terminals = g_structure.query(
                u'SELECT DISTINCT ?terminal ?beginIndex ?endIndex WHERE {?terminal nif:beginIndex ?beginIndex . ?terminal nif:endIndex ?endIndex . ?terminal a ?type . VALUES ?type {nif:Title nif:Paragraph}} ORDER BY ?beginIndex DESC(?endIndex)',
                initNs=ns_dict)
            refs = g_structure.query(
                u'SELECT DISTINCT ?superString ?target ?superOffset ?beginIndex ?endIndex ?type WHERE {?ref itsrdf:taIdentRef ?target . ?ref nif:superString ?superString . ?ref nif:beginIndex ?beginIndex . ?ref nif:endIndex ?endIndex . ?superString nif:beginIndex ?superOffset . ?ref a ?type . }',
                initNs=ns_dict)
            context_content = g_structure.value(subject=URIRef(context_str), predicate=NIF.isString, any=False)
            assert context_content is not None, 'context_content is None'
            context_content_str = context_content.toPython()

            #if logger.level <= logging.DEBUG:
            if False:
                # debug
                logger.info('exec query: %s' % str(datetime.now() - t_start))

                for triple in g_structure:
                    logger.debug(triple)

                t_start = datetime.now()
                logger.debug("see also's:")
                for row in res_context_seealsos:
                    logger.debug(row)

                logger.debug(
                    'children (ordered by ?parent_beginIndex DESC(?parent_endIndex) ?child_beginIndex DESC(?child_endIndex)):')
                for row in children_typed:
                    logger.debug(row)

                logger.debug('terminals (ordered by ?beginIndex DESC(?endIndex)):')
                for row in terminals:
                    logger.debug(row)

                logger.debug('refs:')
                for row in refs:
                    logger.debug(row)

                logger.debug('str:')
                logger.debug(context_content_str)

                logger.info('print result: %s' % str(datetime.now() - t_start))

            #return res_context_seealsos, children_typed, terminals, refs, context_content_str
            query_datas.append((context_str, (res_context_seealsos, children_typed, terminals, refs, context_content_str)))
        except Exception as e:
            failed.append((context_str, e))

        graph.commit()

    return query_datas, failed


def test_context_tree(graph, context=URIRef(u"http://dbpedia.org/resource/Damen_Group?dbpv=2016-10&nif=context")):
    logger.setLevel(logging.DEBUG)
    logger_virtuoso.setLevel(logging.DEBUG)

    # create empty lexicon
    lexicon = Lexicon(types=[])
    logger.debug('lexicon size: %i' % len(lexicon))

    #res_context_seealsos, children_typed, terminals, refs, context_str = query_context_data(graph, context)

    logger.info('load spacy ...')
    t_start = datetime.now()
    nlp = spacy.load('en')
    logger.info('loaded spacy: %s' % str(datetime.now() - t_start))

    nif_context_data = prepare_context_data(graph, context)
    tree_context = create_context_forest(nif_context_data, lexicon=lexicon, nlp=nlp)
    logger.info('leafs: %i' % len(tree_context))

    tree_context.visualize('../tmp.svg')  # , start=0, end=100)


def test_query_context(graph, context=URIRef(u"http://dbpedia.org/resource/Damen_Group?dbpv=2016-10&nif=context")):
    logger.setLevel(logging.DEBUG)
    logger_virtuoso.setLevel(logging.DEBUG)

    g_context = query_context(graph, initBindings={'context': context})
    i = 0
    for triple in g_context:
        print(triple)
        i += 1
    logger.debug('triple_count=%i' % i)


def save_current_forest(forest, failed, lexicon, filename, t_parse, t_query):

    if forest is not None:
        forest.dump(filename)
        lexicon.dump(filename, strings_only=True)
        unique, counts = np.unique(forest.data, return_counts=True)
        #unique.dump('%s.%s' % (filename, FE_UNIQUE_HASHES))
        #counts.dump('%s.%s' % (filename, FE_COUNTS))
        numpy_dump('%s.%s' % (filename, FE_UNIQUE_HASHES), unique)
        numpy_dump('%s.%s' % (filename, FE_COUNTS), counts)
        logger.info('%s: t_query=%s t_parse=%s (failed: %i, forest_size: %i, lexicon size: %i)'
                    % (filename, str(t_query), str(t_parse), len(failed), len(forest), len(lexicon)))
    else:
        logger.info('%s: t_query=%s t_parse=%s (failed: %i, forest_size: %i, lexicon size: %i)'
                    % (filename, str(t_query), str(t_parse), len(failed), 0, len(lexicon)))
    if failed is not None and len(failed) > 0:
        with open('%s.%s' % (filename, FE_FAILED), 'w', ) as f:
            for context_uri_str, e in failed:
                f.write((context_uri_str + u'\t' + str(e) + u'\n').encode('utf8'))
    elif os.path.exists('%s.%s' % (filename, FE_FAILED)):
        os.remove('%s.%s' % (filename, FE_FAILED))
        #assert len(resource_hashes_failed) > 0, 'entries in "failed" list, but resource_hashes_failed is empty'
        #resource_hashes_failed.dump('%s.%s' % (filename, FE_ROOT_ID_FAILED))
    #else:
    #    assert len(resource_hashes_failed) == 0, 'entries in resource_hashes_failed, but "failed" list is empty'


class ThreadParse(threading.Thread):
    """Threaded Parsing"""

    def __init__(self, queue, out_path, nlp=None):
        threading.Thread.__init__(self)
        self._queue = queue
        self._out_path = out_path
        self._nlp = nlp
        if self._nlp is None:
            logger.info('load spacy ...')
            self._nlp = spacy.load('en')

    def run(self):
        while True:
            begin_idx, nif_context_datas, failed, t_query = self._queue.get()
            fn = os.path.join(self._out_path, '%s.%i' % (PREFIX_FN, begin_idx))
            try:
                parse_context_batch(nif_context_datas=nif_context_datas, failed=failed, nlp=self._nlp,
                                    filename=fn, t_query=t_query)
            except Exception as e:
                print('%s: failed' % str(e))
            self._queue.task_done()


class ThreadQuery(threading.Thread):
    """Threaded Querying"""

    def __init__(self, queue, queue_out, graph=None, default_graph_uri="http://dbpedia.org/nif"):
        threading.Thread.__init__(self)
        self._queue = queue
        self._queue_out = queue_out
        self._graph = graph
        if self._graph is None:
            logger.info('set up connection ...')
            Virtuoso = plugin("Virtuoso", Store)
            store = Virtuoso("DSN=VOS;UID=dba;PWD=dba;WideAsUTF16=Y")
            self._graph = Graph(store, identifier=URIRef(default_graph_uri))

    def run(self):
        while True:
            begin_idx, contexts = self._queue.get()
            t_start = datetime.now()
            failed = []
            nif_context_datas = []
            for context in contexts:
                try:
                    nif_context_data = prepare_context_data(self._graph, context)
                    nif_context_datas.append(nif_context_data)
                except Exception as e:
                    failed.append((context, e))
            self._queue_out.put((begin_idx, nif_context_datas, failed, datetime.now() - t_start))
            self._queue.task_done()


def parse_context_batch_DEP(nif_context_datas, failed, nlp, filename, t_query):

    t_start = datetime.now()
    lexicon = Lexicon()
    #tree_contexts = []
    #resource_hashes = []
    #resource_hashes_failed = []

    #for nif_context_data in nif_context_datas:
    #    context = nif_context_data[-1]
    #    res_hash = hash_string(context[:-len(PREFIX_CONTEXT)])
    #    try:
    #        tree_context = create_context_forest(nif_context_data, lexicon=lexicon, nlp=nlp)
    #        tree_context.set_children_with_parents()
    #        tree_contexts.append(tree_context)
    #        resource_hashes.append(res_hash)
    #    except Exception as e:
    #        failed.append((context, e))
    #        resource_hashes_failed.append(res_hash)
    tree_contexts, resource_hashes, failed_parse = create_contexts_forest(nif_context_datas, lexicon=lexicon, nlp=nlp)
    failed.extend(failed_parse)

    if len(tree_contexts) > 0:
        forest = Forest.concatenate(tree_contexts)
        forest.set_root_ids(root_ids=np.array(resource_hashes, dtype=forest.data.dtype))
    else:
        forest = None

    save_current_forest(forest=forest,
                        failed=failed, #resource_hashes_failed=np.array(resource_hashes_failed, dtype=DTYPE_HASH),
                        lexicon=lexicon, filename=filename, t_parse=datetime.now()-t_start, t_query=t_query)


def connect_graph():
    while True:
        try:
            Virtuoso = plugin("Virtuoso", Store)
            store = Virtuoso("DSN=VOS;UID=dba;PWD=dba;WideAsUTF16=Y")
            default_graph_uri = "http://dbpedia.org/nif"
            graph = Graph(store, identifier=URIRef(default_graph_uri))
            break
        except Exception as e:
            logger.warn('connection failed: %s wait %i seconds ...' % (str(e), 10))
            time.sleep(10)
    return graph, store


@plac.annotations(
    out_path=('corpora out path', 'option', 'o', str),
    batch_size=('amount of articles to query before fed to parser', 'option', 'b', int),
    #num_threads=('total number of threads used for querying and parsing', 'option', 't', int),
    start_offset=('index of articles to start with', 'option', 's', int),
    batch_count=('if batch_count > 0 process only this amount of articles', 'option', 'c', int)
)
def process_prepare(out_path='/root/corpora_out/DBPEDIANIF-test', batch_size=1000, start_offset=0, batch_count=0):
    if not os.path.exists(out_path):
        logger.info('checked existence of: %s' % out_path)
        os.mkdir(out_path)
    out_path = os.path.join(out_path, str(batch_size))
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    logger_fh = logging.FileHandler(os.path.join(out_path, 'corpus-dbpedia-nif-prepare.log'))
    logger_fh.setLevel(logging.INFO)
    logger_fh.setFormatter(logging.Formatter(LOGGING_FORMAT))
    logger.addHandler(logger_fh)
    logger.info('batch-size=%i start-offset=%i batch-count=%i out_path=%s'
                % (batch_size, start_offset, batch_count, out_path))

    out_path = os.path.join(out_path, DIR_BATCHES)
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    logger.info('THREAD MAIN: set up connection ...')
    graph, _ = connect_graph()
    logger.info('THREAD MAIN: connected')

    t_start = datetime.now()
    logger.info('THREAD MAIN: dump context strings ...')
    current_contexts = []
    batch_start = 0
    current_batch_count = 0
    for i, context in enumerate(graph.subjects(RDF.type, NIF.Context)):
        if i < start_offset:
            continue
        if i % batch_size == 0:
            if len(current_contexts) > 0:
                fn = os.path.join(out_path, '%s-%i' % (PREFIX_FN, batch_start))
                if not Lexicon.exist(fn, types_only=True):
                    # q_query.put((batch_start, current_contexts))
                    lexicon = Lexicon(string_list=current_contexts)
                    lexicon.dump(filename=fn, strings_only=True)

                    current_batch_count += 1
                    current_contexts = []
                    if current_batch_count >= batch_count > 0:
                        break
                else:
                    current_contexts = []
            batch_start = i
        current_contexts.append(context.toPython())

    fn = os.path.join(out_path, '%s-%i' % (PREFIX_FN, batch_start))
    if len(current_contexts) > 0 and not (Forest.exist(fn) and Lexicon.exist(fn, types_only=True)):
        #q_query.put((batch_start, current_contexts))
        lexicon = Lexicon(string_list=current_contexts)
        lexicon.dump(filename=fn, strings_only=True)

    print('%s finished' % str(datetime.now() - t_start))


@plac.annotations(
    out_path=('corpora out path', 'option', 'o', str),
    batch_size=('amount of articles to query before fed to parser', 'option', 's', int),
    batch_size_parse=('batch_size used by spacy.pipe for parsing', 'option', 'b', int),
    num_threads_query=('number of threads used for querying', 'option', 'q', int),
    num_threads_parse=('number of threads used for parsing', 'option', 'p', int),
    num_threads_parse_pipe=('number of threads used by spacy.pipe for parsing', 'option', 't', int),
    #start_offset=('index of articles to start with', 'option', 's', int),
    #batch_count=('if batch_count > 0 process only this amount of articles', 'option', 'c', int)
)
def process_contexts_multi(out_path='/root/corpora_out/DBPEDIANIF-test', batch_size=1000, num_threads_parse=2,
                           num_threads_query=4, num_threads_parse_pipe=1, batch_size_parse=1000):
    #assert num_threads >= 2, 'require at least num_threads==2 (one for querying and one for parsing)'

    if not os.path.exists(out_path):
        logger.info('checked existence of: %s' % out_path)
        os.mkdir(out_path)
    out_path = os.path.join(out_path, str(batch_size))
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    logger_fh = logging.FileHandler(os.path.join(out_path, 'corpus-dbpedia-nif-batches.log'))
    logger_fh.setLevel(logging.INFO)
    logger_fh.setFormatter(logging.Formatter(LOGGING_FORMAT))
    logger.addHandler(logger_fh)
    logger_fh_debug = logging.FileHandler(os.path.join(out_path, 'corpus-dbpedia-nif-batches.debug.log'))
    logger_fh_debug.setLevel(logging.DEBUG)
    logger_fh_debug.setFormatter(logging.Formatter(LOGGING_FORMAT))
    logger.addHandler(logger_fh_debug)

    logger.info('batch-size=%i batch-size-parse=%i num-threads-query=%i num-threads-parse=%i num-threads-parse-pipe=%i out_path=%s'
                % (batch_size, batch_size_parse, num_threads_query, num_threads_parse, num_threads_parse_pipe, out_path))

    out_path = os.path.join(out_path, DIR_BATCHES)
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    q_query = Queue.Queue(maxsize=0)
    q_parse = Queue.Queue(maxsize=0)

    def do_query(_q_in, _q_out, thread_id):
        logger.info('THREAD %i QUERY: set up connection ...' % thread_id)
        _g, store = connect_graph()
        logger.info('THREAD %i QUERY: connected' % thread_id)
        while True:
            fn = _q_in.get()
            logger.debug('THREAD %i QUERY: start job.    queue size: %i.' % (thread_id, q_query.qsize()))
            t_start = datetime.now()
            lexicon = Lexicon(filename=fn)
            #failed = []
            nif_context_datas, failed = prepare_context_datas(_g, lexicon.strings, store.cursor())
            store.commit()
            #for context_str in lexicon.strings:
            #    try:
            #        nif_context_data = prepare_context_data(_g, context_str)
            #        nif_context_datas.append(nif_context_data)
            #    except Exception as e:
            #        failed.append((context_str, e))
            #assert len(nif_context_datas) > 0, '(QUERY) nif_context_datas is empty'
            t_delta = datetime.now() - t_start
            _q_out.put((fn, nif_context_datas, failed, t_delta))
            _q_in.task_done()
            logger.debug('THREAD %i QUERY: finished job. queue size: %i. t_delta: %s' % (thread_id, q_query.qsize(),
                                                                                         str(t_delta)))

    def do_parse(_q, _path_out, thread_id, _nlp):

        while True:
            fn, nif_context_datas, failed, t_query = _q.get()
            logger.debug('THREAD %i PARSE: start job.    queue size: %i.' % (thread_id, q_parse.qsize()))
            t_start = datetime.now()
            #fn = os.path.join(out_path, '%s.%i' % (PREFIX_FN, begin_idx))
            try:
                assert len(nif_context_datas) > 0, '(PARSE) nif_context_datas is empty'
                lexicon = Lexicon()
                #parse_context_batch(nif_context_datas=nif_context_datas, failed=failed, nlp=_nlp,
                #                    filename=fn, t_query=t_query)
                forest, failed_parse = create_contexts_forest(nif_context_datas, lexicon=lexicon, nlp=_nlp,
                                                              n_threads=num_threads_parse_pipe,
                                                              batch_size=batch_size_parse)
                failed.extend(failed_parse)

                save_current_forest(forest=forest,
                                    failed=failed,
                                    # resource_hashes_failed=np.array(resource_hashes_failed, dtype=DTYPE_HASH),
                                    lexicon=lexicon, filename=fn, t_parse=datetime.now() - t_start,
                                    t_query=t_query)
            except Exception as e:
                print('%s: failed' % str(e))
            _q.task_done()
            logger.debug('THREAD %i PARSE: finished job. queue size: %i. t_delta: %s' % (thread_id, q_parse.qsize(),
                                                                                        str(datetime.now() - t_start)))

    for i in range(num_threads_query):
        worker_query = Thread(target=do_query, args=(q_query, q_parse, i))
        worker_query.setDaemon(True)
        worker_query.start()

    for i in range(num_threads_parse):
        thread_id_parse = i + num_threads_query
        logger.info('THREAD %i PARSE: load spacy ...' % thread_id_parse)
        _nlp = spacy.load('en')
        logger.info('THREAD %i PARSE: loaded' % thread_id_parse)
        worker_parse = Thread(target=do_parse, args=(q_parse, out_path, thread_id_parse, _nlp))
        worker_parse.setDaemon(True)
        worker_parse.start()

    t_start = datetime.now()
    logger.info('THREAD MAIN: fill query queue ...')
    for file in os.listdir(out_path):
        if file.endswith('.' + FE_STRINGS):
            fn = os.path.join(out_path, file[:-len('.' + FE_STRINGS)])
            if not Forest.exist(fn):
                q_query.put(fn)

    q_query.join()
    q_parse.join()
    print('%s finished' % str(datetime.now() - t_start))


def _collect_file_names(out_path_batches):
    logger.info('collect file names ...')
    t_start = datetime.now()
    suffix = '.' + FE_STRINGS
    l = len(suffix)
    f_names = []
    for file in os.listdir(out_path_batches):
        if file.endswith(suffix) and Forest.exist(os.path.join(out_path_batches, file[:-l])):
            f_names.append(file[:-l])
    f_names = sorted(f_names, key=lambda fn: int(fn[len(PREFIX_FN)+1:]))
    f_paths = [os.path.join(out_path_batches, f) for f in f_names]
    logger.info('finished. %s' % str(datetime.now()-t_start))
    return f_names, f_paths


def _collect_counts_merged(f_paths):
    logger.info('collect counts ...')
    t_start = datetime.now()
    counts_merged = {}
    for fn in f_paths:
        #counts = np.load('%s.%s' % (fn, FE_COUNTS))
        #uniques = np.load('%s.%s' % (fn, FE_UNIQUE_HASHES))
        counts = numpy_load('%s.%s' % (fn, FE_COUNTS), assert_exists=True)
        uniques = numpy_load('%s.%s' % (fn, FE_UNIQUE_HASHES), assert_exists=True)
        for i, c in enumerate(counts):
            _c = counts_merged.get(uniques[i], 0)
            counts_merged[uniques[i]] = _c + c
    logger.info('finished. %s' % str(datetime.now() - t_start))
    return counts_merged


def _collect_root_ids(f_paths, out_path_merged):
    logger.info('collect root_ids ...')
    fn_root_ids = '%s.%s' % (out_path_merged, FE_ROOT_ID)
    if os.path.exists(fn_root_ids):
        logger.info('found root_ids (%s). load from file.' % fn_root_ids)
        #return np.load(fn_root_ids)
        return numpy_load(fn_root_ids, assert_exists=True)

    t_start = datetime.now()
    root_ids = []
    for fn in f_paths:
        #root_ids.append(np.load('%s.%s' % (fn, FE_ROOT_ID)))
        root_ids.append(numpy_load('%s.%s' % (fn, FE_ROOT_ID), assert_exists=True))
    root_ids = np.concatenate(root_ids)

    count_root_ids_unique = len(np.unique(root_ids))
    assert len(root_ids) == count_root_ids_unique, '%i root ids are duplicated' \
                                                   % (len(root_ids) - count_root_ids_unique)
    #root_ids.dump(fn_root_ids)
    numpy_dump(fn_root_ids, root_ids)
    logger.info('finished. %s' % str(datetime.now()-t_start))
    return root_ids


def _filter_uniques(f_paths, min_count, min_count_root_id, out_path_merged):

    fn_uniques_filtered = '%s.%s' % (out_path_merged, FE_UNIQUE_HASHES_FILTERED)
    fn_uniques_discarded = '%s.%s' % (out_path_merged, FE_UNIQUE_HASHES_DISCARDED)
    fn_counts_filtered = '%s.%s' % (out_path_merged, FE_UNIQUE_COUNTS_FILTERED)
    fn_counts_discarded = '%s.%s' % (out_path_merged, FE_UNIQUE_COUNTS_DISCARDED)
    if os.path.exists(fn_uniques_filtered):
        logger.info('found uniques_filtered (%s). load from file.' % fn_uniques_filtered)
        assert os.path.exists(fn_uniques_discarded), 'found uniques_filtered (%s), but misses files for ' \
                                                     'uniques_discarded (%s).' % (fn_uniques_filtered,
                                                                                  fn_uniques_discarded)
        assert os.path.exists(fn_counts_filtered), 'found uniques_filtered (%s), but misses files for ' \
                                                   'counts_filtered (%s).' % (fn_uniques_filtered, fn_counts_filtered)
        assert os.path.exists(fn_counts_discarded), 'found uniques_filtered (%s), but misses files for ' \
                                                    'counts_discarded (%s).' % (fn_uniques_filtered,
                                                                                fn_counts_discarded)
        #return np.load(fn_uniques_filtered)
        return numpy_load(fn_uniques_filtered, assert_exists=True)


    counts_merged = _collect_counts_merged(f_paths)
    root_ids = _collect_root_ids(f_paths, out_path_merged)
    root_ids_set = set(root_ids)
    assert len(root_ids_set) == len(root_ids), 'root_ids contains %i duplicates' % (len(root_ids) - len(root_ids_set))

    logger.info('filter uniques by count ...')
    t_start = datetime.now()
    uniques_filtered = np.zeros(shape=len(counts_merged.keys()), dtype=DTYPE_HASH)
    uniques_discarded = np.zeros(shape=len(counts_merged.keys()), dtype=DTYPE_HASH)
    counts_filtered = np.zeros(shape=len(counts_merged.keys()), dtype=DTYPE_COUNT)
    counts_discarded = np.zeros(shape=len(counts_merged.keys()), dtype=DTYPE_COUNT)
    i_filtered = 0
    i_discarded = 0
    for u in counts_merged.keys():
        #if counts_merged[u] >= min_count or (u in root_ids and counts_merged[u] >= min_count_root_id >= 0):
        #if u not in root_ids and counts_merged[u] >= min_count:
        if (u not in root_ids_set and counts_merged[u] >= min_count) \
                or (u in root_ids_set and counts_merged[u] >= min_count_root_id >= 0):
            uniques_filtered[i_filtered] = u
            counts_filtered[i_filtered] = counts_merged[u]
            i_filtered += 1
        else:
            uniques_discarded[i_discarded] = u
            counts_discarded[i_discarded] = counts_merged[u]
            i_discarded += 1
    uniques_filtered = uniques_filtered[:i_filtered]
    uniques_discarded = uniques_discarded[:i_discarded]
    counts_filtered = counts_filtered[:i_filtered]
    counts_discarded = counts_discarded[:i_discarded]
    #uniques_filtered.dump(fn_uniques_filtered)
    #uniques_discarded.dump(fn_uniques_discarded)
    #counts_filtered.dump(fn_counts_filtered)
    #counts_discarded.dump(fn_counts_discarded)
    numpy_dump(fn_uniques_filtered, uniques_filtered)
    numpy_dump(fn_uniques_discarded, uniques_discarded)
    numpy_dump(fn_counts_filtered, counts_filtered)
    numpy_dump(fn_counts_discarded, counts_discarded)

    logger.info('finished. %s' % str(datetime.now() - t_start))
    return uniques_filtered, root_ids


def _merge_and_filter_lexicon(uniques_filtered, f_paths, out_path_merged):
    logger.info('merge and filter lexicon ...')
    fn_lexicon_discarded = '%s.discarded' % out_path_merged
    if Lexicon.exist(filename=out_path_merged, types_only=True):
        logger.info('found lexicon (%s). load from file.' % out_path_merged)
        assert Lexicon.exist(filename=fn_lexicon_discarded, types_only=True), \
            'found lexicon (%s), but misses lexicon_discarded (%s).' % (out_path_merged, fn_lexicon_discarded)
        # Note: Load with vecs to skip _lexicon_add_vecs, eventually.
        return Lexicon(filename=out_path_merged)
    t_start = datetime.now()
    uniques_filtered_set = set(uniques_filtered)
    lexicon = Lexicon()
    lexicon.add_all(vocab_manual.values())
    lexicon_discarded = Lexicon()
    for fn in f_paths:
        lex = Lexicon(filename=fn)
        for s in lex.strings:
            h = hash_string(s)
            if h in uniques_filtered_set:
                lexicon.strings.add(s)
            else:
                lexicon_discarded.strings.add(s)
    lexicon.dump(filename=out_path_merged, strings_only=True)
    lexicon_discarded.dump(filename=fn_lexicon_discarded, strings_only=True)
    logger.info('finished. %s' % str(datetime.now() - t_start))
    return lexicon


def _filter_and_convert_data_batches(lexicon, id_offset_mapping, f_names, out_dir_batches, out_dir_batches_converted):
    logger.info('filter and convert batches ...')
    t_start = datetime.now()
    assert vocab_manual[UNKNOWN_EMBEDDING] in lexicon.strings or not lexicon.frozen, 'UNKNOWN_EMBEDDING not in ' \
                                                                                     'lexicon, but it is frozen'
    lexicon.strings.add(vocab_manual[UNKNOWN_EMBEDDING])
    count_skipped = 0
    for fn in f_names:
        fn_path_in = os.path.join(out_dir_batches, fn)
        fn_path_out = os.path.join(out_dir_batches_converted, fn)
        if Forest.exist(filename=fn_path_out):
            count_skipped += 1
            continue
        forest = Forest(filename=fn_path_in, lexicon=lexicon)
        forest.hashes_to_indices(id_offset_mapping)
        forest.dump(filename=fn_path_out)
    logger.info('finished (processed: %i, skipped: %i). %s' % (len(f_names) - count_skipped, count_skipped,
                                                               str(datetime.now() - t_start)))


def _lexicon_add_vecs(lexicon, out_path_merged):
    logger.info('add vecs ...')
    if lexicon.has_vecs:
        logger.info('lexicon has vecs already.')
        return lexicon
    t_start = datetime.now()
    logger.info('lexicon size: %i' % len(lexicon))
    logger.info('load spacy ...')
    nlp = spacy.load('en')
    lexicon.init_vecs(vocab=nlp.vocab)
    logger.info('lexicon fixed size: %i' % len(lexicon.ids_fixed))
    lexicon.set_to_random(indices=lexicon.ids_fixed, indices_as_blacklist=True)
    lexicon.dump(filename=out_path_merged)
    logger.info('finished. %s' % str(datetime.now() - t_start))
    return lexicon


def _merge_converted_batches(f_names, out_dir_batches_converted, out_path_merged):
    logger.info('merge converted batches ...')
    if Forest.exist(out_path_merged):
        logger.info('found forest_merged (%s). load from file.' % out_path_merged)
        return Forest(filename=out_path_merged)
    t_start = datetime.now()
    forests = []
    for fn in f_names:
        fn_path_out = os.path.join(out_dir_batches_converted, fn)
        forests.append(Forest(filename=fn_path_out))
    forest_merged = Forest.concatenate(forests)
    forest_merged.dump(filename=out_path_merged)
    logger.info('finished. %s' % str(datetime.now() - t_start))
    return forest_merged


def _collect_root_seealso_counts(forest_merged, out_path_merged):
    logger.info('collect root seealso counts ...')
    fn_root_seealso_counts = '%s.%s' % (out_path_merged, FE_ROOT_SEEALSO_COUNT)
    if os.path.exists(fn_root_seealso_counts):
        logger.info('found root_seealso_counts (%s). load from file.' % fn_root_seealso_counts)
        #return np.load(fn_root_seealso_counts)
        return numpy_load(fn_root_seealso_counts, assert_exists=True)
    t_start = datetime.now()
    root_seealso_counts = forest_merged.get_children_counts(forest_merged.roots + 3)
    #root_seealso_counts.dump(fn_root_seealso_counts)
    numpy_dump(fn_root_seealso_counts, root_seealso_counts)
    logger.info('finished. %s' % str(datetime.now()-t_start))
    return root_seealso_counts


def _collect_root_context_sizes(forest_merged, root_seealso_counts, out_path_merged):
    logger.info('collect root context sizes ...')
    fn_root_context_sizes = '%s.%s' % (out_path_merged, FE_ROOT_CONTEXT_SIZE)
    if os.path.exists(fn_root_context_sizes):
        logger.info('found root_context_sizes (%s). load from file.' % fn_root_context_sizes)
        #return np.load(fn_root_seealso_counts)
        return numpy_load(fn_root_context_sizes, assert_exists=True)
    t_start = datetime.now()
    #root_seealso_counts = forest_merged.get_children_counts(forest_merged.roots + 3)
    #root_seealso_counts.dump(fn_root_seealso_counts)

    # get node counts of roots by root positions
    root_shifted = np.concatenate([forest_merged.roots[1:], [len(forest_merged)]])
    root_length = root_shifted - forest_merged.roots
    root_context_sizes = (root_length - (root_seealso_counts * 2 + 1)) - 3

    numpy_dump(fn_root_context_sizes, root_context_sizes)
    logger.info('finished. %s' % str(datetime.now()-t_start))
    return root_context_sizes


@plac.annotations(
    out_path=('corpora out path', 'option', 'o', str),
    min_count=('minimal count a token has to occur to stay in the lexicon', 'option', 'c', int),
    min_count_root_id=('minimal count a root_id has to occur to stay in the lexicon', 'option', 'r', int),
)
def process_merge_batches(out_path, min_count=1, min_count_root_id=1):
    logger_fh = logging.FileHandler(os.path.join(out_path, 'corpus-dbpedia-nif-merge.log'))
    logger_fh.setLevel(logging.INFO)
    logger_fh.setFormatter(logging.Formatter(LOGGING_FORMAT))
    logger.addHandler(logger_fh)

    #logger_lexicon = logging.getLogger('lexicon')
    #logger_lexicon_fh = logging.FileHandler(os.path.join(out_path, 'corpus-dbpedia-nif-merge.log'))
    #logger_lexicon_fh.setLevel(logging.INFO)
    #logger_lexicon_fh.setFormatter(logging.Formatter(LOGGING_FORMAT))
    #logger_lexicon.addHandler(logger_lexicon_fh)

    logger.info('min_count=%i min_count_root_id=%i out_path=%s' % (min_count, min_count_root_id, out_path))

    out_dir_batches = os.path.join(out_path, DIR_BATCHES)
    out_dir_batches_converted = os.path.join(out_path, DIR_BATCHES_CONVERTED)
    if not os.path.exists(out_dir_batches_converted):
        os.mkdir(out_dir_batches_converted)
    out_path_merged = os.path.join(out_path, DIR_MERGED)
    if not os.path.exists(out_path_merged):
        os.mkdir(out_path_merged)
    out_path_merged = os.path.join(out_path_merged, PREFIX_FN)

    f_names, f_paths = _collect_file_names(out_dir_batches)

    uniques_filtered, root_ids = _filter_uniques(f_paths, min_count, min_count_root_id, out_path_merged)

    lexicon = _merge_and_filter_lexicon(uniques_filtered, f_paths, out_path_merged)

    id_offset_mapping = {o: i for i, o in enumerate(root_ids)}

    _filter_and_convert_data_batches(lexicon, id_offset_mapping, f_names, out_dir_batches, out_dir_batches_converted)

    lexicon = _lexicon_add_vecs(lexicon, out_path_merged)

    forest_merged = _merge_converted_batches(f_names, out_dir_batches_converted, out_path_merged)

    root_seealso_counts = _collect_root_seealso_counts(forest_merged, out_path_merged)

    root_context_sizes = _collect_root_context_sizes(forest_merged, root_seealso_counts, out_path_merged)


@plac.annotations(
    mode=('processing mode', 'positional', None, str, ['PREPARE_BATCHES', 'CREATE_BATCHES', 'MERGE_BATCHES',
                                                       'CREATE_INDICES']),
    args='the parameters for the underlying processing method')
def main(mode, *args):
    if mode == 'CREATE_BATCHES':
        plac.call(process_contexts_multi, args)
    elif mode == 'MERGE_BATCHES':
        plac.call(process_merge_batches, args)
    elif mode == 'PREPARE_BATCHES':
        plac.call(process_prepare, args)
    elif mode == 'CREATE_INDICES':
        plac.call(create_index_files, args)
    else:
        raise ValueError('unknown mode. use one of CREATE_BATCHES or MERGE_BATCHES.')


@plac.annotations(
    merged_forest_path=('path to merged forest', 'option', 'o', str),
    seealso_min=('minimal required count of seeAlso links, discard other articles', 'option', 'm', int),
    seealso_max=('maximal allowed count of seeAlso links, discard other articles', 'option', 'M', int),
    split_count=('count of produced index files', 'option', 'c', int)
)
def create_index_files(merged_forest_path, split_count=2, seealso_min=1, seealso_max=50):
    logger_fh = logging.FileHandler(os.path.join(merged_forest_path, '../..', 'corpus-dbpedia-nif-indices.log'))
    logger_fh.setLevel(logging.INFO)
    logger_fh.setFormatter(logging.Formatter(LOGGING_FORMAT))
    logger.addHandler(logger_fh)

    logger.info('split_count=%i seealso_min=%i seealso_max=%i out_path=%s' % (split_count, seealso_min, seealso_max,
                                                                              merged_forest_path))

    #seealso_counts = np.load('%s.root.seealso.count' % merged_forest_path)
    seealso_counts = numpy_load('%s.root.seealso.count' % merged_forest_path, assert_exists=True)
    # roots = np.load('%s.root.pos' % p)
    indices_filtered = np.arange(len(seealso_counts), dtype=DTYPE_IDX)[(seealso_counts >= seealso_min)
                                                                       & (seealso_counts <= seealso_max)]
    logger.info('count of filtered indices: %i' % len(indices_filtered))

    np.random.shuffle(indices_filtered)
    for i, split in enumerate(np.array_split(indices_filtered, split_count)):
        #split.dump('%s.idx.%i' % (merged_forest_path, i))
        numpy_dump('%s.idx.%i' % (merged_forest_path, i), split)


if __name__ == '__main__':
    import getpass

    username = getpass.getuser()
    logger.info('username=%s' % username)

    plac.call(main)

    #logger.info('set up connection ...')
    #Virtuoso = plugin("Virtuoso", Store)
    #store = Virtuoso("DSN=VOS;UID=dba;PWD=dba;WideAsUTF16=Y")
    #default_graph_uri = "http://dbpedia.org/nif"
    #g = Graph(store, identifier=URIRef(default_graph_uri))
    #logger.info('connected')
    #test_query_context(g)

    #test_context_tree(g)
    #test_context_tree(g, context=URIRef(u'http://dbpedia.org/resource/1958_US–UK_Mutual_Defence_Agreement?dbpv=2016-10&nif=context'))
    #test_utf8_context(g)

    #process_all_contexts_new(g)
    #test_process_all_contexts_parallel(g)



