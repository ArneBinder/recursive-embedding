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
from toolz import partition_all
#from joblib import Parallel, delayed

from lexicon import Lexicon
from sequence_trees import Forest, tree_from_sorted_parent_triples
from constants import DTYPE_HASH
import preprocessing

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

FE_RESOURCE_HASHES = 'resource.hash'
FE_RESOURCE_HASHES_FAILED = 'resource.hash.failed'
FE_FAILED = 'failed'
FE_ROOTS = 'root'
FE_UNIQUE_HASHES = 'unique.hash'
FE_COUNTS = 'count'

logger = logging.getLogger('corpus_dbpedia_nif')
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())
#logger.addHandler(logging.FileHandler('../virtuoso_test.log', mode='w', encoding='utf-8'))
logger.propagate = False

logger_virtuoso = logging.getLogger('virtuoso.vstore')
logger_virtuoso.setLevel(logging.INFO)
logger_virtuoso.addHandler(logging.FileHandler('../virtuoso.log', mode='w', encoding='utf-8'))
logger_virtuoso.propagate = False


def query_see_also_links(graph, initBindings=None):
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
             'LIMIT 1000 '
             'OFFSET 0 '
             )
    logging.debug(q_str)
    res = graph.query(q_str, initNs=ns_dict, initBindings=initBindings)
    return res


def query_first_section_structure(graph, initBindings=None):
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
    logger.debug(q_str)
    res = graph.query(q_str, initNs=ns_dict, initBindings=initBindings)
    #print(type(res))
    return res


def prepare_context_data(graph, nif_context,
                         ref_type_str=u"http://www.w3.org/2005/11/its/rdf#taIdentRef", max_see_also_refs=50):
    see_also_refs, children_typed, terminals, link_refs, nif_context_str = query_context_data(graph, nif_context)
    if len(see_also_refs) > max_see_also_refs:
        see_also_refs = []
    tree_context_data_strings, tree_context_parents, terminal_parent_positions, terminal_types = tree_from_sorted_parent_triples(
        children_typed,
        see_also_refs=see_also_refs,
        root_id=unicode(nif_context)[:-len(PREFIX_CONTEXT)])
    terminal_uri_strings, terminal_strings = zip(
        *[(unicode(uri), nif_context_str[int(begin):int(end)]) for uri, begin, end in terminals])

    refs = {}
    # tuple format: ?superString ?target ?superOffset ?beginIndex ?endIndex ?type
    for ref_tuple in link_refs:
        super_string = unicode(ref_tuple[0])
        target_str = unicode(ref_tuple[1])
        #target_id = lexicon[target]
        offset = int(ref_tuple[2])
        begin_index = int(ref_tuple[3])
        end_index = int(ref_tuple[4])

        ref_list = refs.get(super_string, [])
        ref_list.append((begin_index - offset, end_index - offset, [ref_type_str, target_str], [0, -1]))
        refs[super_string] = ref_list

    return tree_context_data_strings, \
           tree_context_parents, \
           terminal_strings, \
           terminal_uri_strings, \
           terminal_types, \
           refs, \
           terminal_parent_positions, \
           nif_context


def create_context_forest(nif_context_data, nlp, lexicon, n_threads=1):

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


def query_context_data(graph, context):
    t_start = datetime.now()
    res_context_seealsos = query_see_also_links(graph, initBindings={'context': context})
    res_context = query_first_section_structure(graph, {'context': context})
    #for i, row in enumerate(res_context):
    #    logger.debug(row)

    g_structure = Graph()
    for triple in res_context:
        g_structure.add(triple)

    children_typed = g_structure.query(
        u'SELECT DISTINCT ?child ?type_child ?parent WHERE {?parent a ?type . VALUES ?type {nif:Section nif:Context} . ?child a ?type_child . ?parent nif:beginIndex ?parent_beginIndex . ?parent nif:endIndex ?parent_endIndex . ?child nif:superString ?parent . ?child nif:beginIndex ?child_beginIndex . ?child nif:endIndex ?child_endIndex .} ORDER BY ?parent_beginIndex DESC(?parent_endIndex) ?child_beginIndex DESC(?child_endIndex)',
        initNs=ns_dict)

    terminals = g_structure.query(
        u'SELECT DISTINCT ?terminal ?beginIndex ?endIndex WHERE {?terminal nif:beginIndex ?beginIndex . ?terminal nif:endIndex ?endIndex . ?terminal a ?type . VALUES ?type {nif:Title nif:Paragraph}} ORDER BY ?beginIndex DESC(?endIndex)',
        initNs=ns_dict)
    refs = g_structure.query(
        u'SELECT DISTINCT ?superString ?target ?superOffset ?beginIndex ?endIndex ?type WHERE {?ref itsrdf:taIdentRef ?target . ?ref nif:superString ?superString . ?ref nif:beginIndex ?beginIndex . ?ref nif:endIndex ?endIndex . ?superString nif:beginIndex ?superOffset . ?ref a ?type . }',
        initNs=ns_dict)
    context_str = unicode(g_structure.value(subject=context, predicate=NIF.isString, any=False))

    if logger.level <= logging.DEBUG:
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
        logger.debug(context_str)

        logger.info('print result: %s' % str(datetime.now() - t_start))

    return res_context_seealsos, children_typed, terminals, refs, context_str


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


def save_current_forest(i, forest, resource_hashes, failed, resource_hashes_failed, lexicon, filename, t_parse, t_query):

    if forest is not None:
        forest.dump(filename)
        lexicon.dump(filename, strings_only=True)
        roots = forest.roots
        roots.dump('%s.%s' % (filename, FE_ROOTS))
        resource_hashes.dump('%s.%s' % (filename, FE_RESOURCE_HASHES))
        # consistency check
        resource_hashes_check = forest.data[roots + 1]
        assert np.array_equal(resource_hashes_check, resource_hashes), 'data[roots+1] does not match resource_hashes'
        # consistency check end
        unique, counts = np.unique(forest.data, return_counts=True)
        unique.dump('%s.%s' % (filename, FE_UNIQUE_HASHES))
        counts.dump('%s.%s' % (filename, FE_COUNTS))
        logger.info('%i: t_query=%s t_parse=%s (failed: %i, forest_size: %i, lexicon size: %i)'
                    % (i, str(t_query), str(t_parse), len(failed), len(forest), len(lexicon)))
    else:
        logger.info('%i: t_query=%s t_parse=%s (failed: %i, forest_size: %i, lexicon size: %i)'
                    % (i, str(t_query), str(t_parse), len(failed), 0, len(lexicon)))
    if failed is not None and len(failed) > 0:
        with open('%s.%s' % (filename, FE_FAILED), 'w', ) as f:
            for uri, e in failed:
                f.write((unicode(uri) + u'\t' + unicode(e) + u'\n').encode('utf8'))
        assert len(resource_hashes_failed) > 0, 'entries in "failed" list, but resource_hashes_failed is empty'
        resource_hashes_failed.dump('%s.%s' % (filename, FE_RESOURCE_HASHES_FAILED))
    else:
        assert len(resource_hashes_failed) == 0, 'entries in resource_hashes_failed, but "failed" list is empty'


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
            fn = os.path.join(self._out_path, 'forest-%i' % begin_idx)
            try:
                parse_context_batch(nif_context_datas=nif_context_datas, failed=failed, nlp=self._nlp, begin_idx=begin_idx,
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


def parse_context_batch(nif_context_datas, failed, nlp, begin_idx, filename, t_query):

    t_start = datetime.now()
    lexicon = Lexicon()
    tree_contexts = []
    resource_hashes = []
    resource_hashes_failed = []

    for nif_context_data in nif_context_datas:
        context = nif_context_data[-1]
        res_hash = hash_string(unicode(context)[:-len(PREFIX_CONTEXT)])
        try:
            #nif_context_data = prepare_context_data(graph, context)
            tree_context = create_context_forest(nif_context_data, lexicon=lexicon, nlp=nlp)
            tree_context.children_dict_to_arrays()
            tree_contexts.append(tree_context)
            resource_hashes.append(res_hash)
        except Exception as e:
            failed.append((context, e))
            resource_hashes_failed.append(res_hash)

    if len(tree_contexts) > 0:
        forest = Forest.concatenate(tree_contexts)
    else:
        forest = None

    save_current_forest(i=begin_idx + len(nif_context_datas) + len(failed), forest=forest,
                        resource_hashes=np.array(resource_hashes, dtype=DTYPE_HASH), failed=failed,
                        resource_hashes_failed=np.array(resource_hashes_failed, dtype=DTYPE_HASH), lexicon=lexicon,
                        filename=filename, t_parse=datetime.now()-t_start, t_query=t_query)


@plac.annotations(
    out_path=('corpora out path', 'option', 'o', str),
    batch_size=('amount of articles to query before fed to parser', 'option', 'b', int),
    num_threads=('total number of threads used for querying and parsing', 'option', 't', int),
    start_offset=('index of articles to start with', 'option', 's', int),
    batch_count=('if batch_count > 0 process only this amount of articles', 'option', 'c', int)
)
def process_contexts_multi(out_path='/root/corpora_out/DBPEDIANIF-test', batch_size=1000, num_threads=2, start_offset=0,
                           batch_count=0):
    assert num_threads >= 2, 'require at least num_threads==2 (one for querying and one for parsing)'
    logger.info('batch-size=%i num-threads=%i start-offset=%i batch-count=%i' % (batch_size, num_threads, start_offset, batch_count))

    if not os.path.exists(out_path):
        os.mkdir(out_path)
    out_path = os.path.join(out_path, str(batch_size))
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    #logger.info('out_path=%s' % out_path)
    #logger.warn(os.path.exists(out_path))

    q_query = Queue.Queue(maxsize=100)
    q_parse = Queue.Queue(maxsize=100)

    logger.info('THREAD MAIN: set up connection ...')
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
    logger.info('THREAD MAIN: connected')

    def do_query(_q_in, _q_out, thread_id):
        logger.info('THREAD %i QUERY: set up connection ...' % thread_id)
        store = Virtuoso("DSN=VOS;UID=dba;PWD=dba;WideAsUTF16=Y")
        _g = Graph(store, identifier=URIRef(default_graph_uri))
        logger.info('THREAD %i QUERY: connected' % thread_id)
        while True:
            begin_idx, contexts = _q_in.get()
            t_start = datetime.now()
            failed = []
            nif_context_datas = []
            for context in contexts:
                try:
                    nif_context_data = prepare_context_data(_g, context)
                    nif_context_datas.append(nif_context_data)
                except Exception as e:
                    failed.append((context, e))
            _q_out.put((begin_idx, nif_context_datas, failed, datetime.now() - t_start))
            _q_in.task_done()

    def do_parse(_q, _path_out, thread_id):
        logger.info('THREAD %i PARSE: load spacy ...' % thread_id)
        _nlp = spacy.load('en')
        logger.info('THREAD %i PARSE: loaded' % thread_id)

        while True:
            begin_idx, nif_context_datas, failed, t_query = _q.get()
            fn = os.path.join(out_path, 'forest-%i' % begin_idx)
            try:
                parse_context_batch(nif_context_datas=nif_context_datas, failed=failed, nlp=_nlp, begin_idx=begin_idx,
                                    filename=fn, t_query=t_query)
            except Exception as e:
                print('%s: failed' % str(e))
            _q.task_done()

    for i in range(num_threads / 2):
        worker_query = Thread(target=do_query, args=(q_query, q_parse, i * 2))
        worker_query.setDaemon(True)
        worker_query.start()

        worker_parse = Thread(target=do_parse, args=(q_parse, out_path, i * 2 + 1))
        worker_parse.setDaemon(True)
        worker_parse.start()

    t_start = datetime.now()
    logger.info('THREAD MAIN: fill query queue ...')
    current_contexts = []
    resource_hashes = []
    batch_start = 0
    current_batch_count = 0
    for i, context in enumerate(graph.subjects(RDF.type, NIF.Context)):
        if i < start_offset:
            continue
        if i % batch_size == 0:
            if len(current_contexts) > 0:
                fn = os.path.join(out_path, 'forest-%i' % batch_start)
                if not (Forest.exist(fn) and Lexicon.exist(fn, types_only=True)):
                    q_query.put((batch_start, current_contexts))
                    current_batch_count += 1
                    current_contexts = []
                    resource_hashes = []
                    if current_batch_count >= batch_count > 0:
                        break
                else:
                    # consistency check of previously processed batches
                    loaded_resource_hashes = np.load('%s.%s' % (fn, FE_RESOURCE_HASHES))
                    if os.path.isfile('%s.%s' % (fn, FE_RESOURCE_HASHES_FAILED)):
                        loaded_resource_hashes_failed = np.load('%s.%s' % (fn, FE_RESOURCE_HASHES_FAILED))
                        hashes_prev = np.sort(np.concatenate([loaded_resource_hashes, loaded_resource_hashes_failed]))
                    else:
                        hashes_prev = np.sort(loaded_resource_hashes)
                    hashes_current = np.sort(np.array(resource_hashes, dtype=hashes_prev.dtype))
                    assert np.array_equal(hashes_prev, hashes_current), 'order has changed. batch %i does not match ' \
                                                                        'previously calculated one.' % current_batch_count
                    # consistency check end
                    current_contexts = []
                    resource_hashes = []
            batch_start = i
        current_contexts.append(context)
        resource_hashes.append(hash_string(unicode(context)[:-len(PREFIX_CONTEXT)]))

    fn = os.path.join(out_path, 'forest-%i' % batch_start)
    if len(current_contexts) > 0 and not (Forest.exist(fn) and Lexicon.exist(fn, types_only=True)):
        q_query.put((batch_start, current_contexts))

    q_query.join()
    q_parse.join()
    print('%s finished' % str(datetime.now() - t_start))


if __name__ == '__main__':
    #test_connect_utf8()
    #process_contexts_multi()
    plac.call(process_contexts_multi)
    #logger.info('set up connection ...')
    #Virtuoso = plugin("Virtuoso", Store)
    #store = Virtuoso("DSN=VOS;UID=dba;PWD=dba;WideAsUTF16=Y")
    #default_graph_uri = "http://dbpedia.org/nif"
    #g = Graph(store, identifier=URIRef(default_graph_uri))
    #logger.info('connected')

    #test_context_tree(g)
    #test_context_tree(g, context=URIRef(u'http://dbpedia.org/resource/1958_US–UK_Mutual_Defence_Agreement?dbpv=2016-10&nif=context'))
    #test_utf8_context(g)

    #process_all_contexts_new(g)
    #test_process_all_contexts_parallel(g)



