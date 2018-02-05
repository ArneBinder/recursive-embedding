# coding: utf-8
#from __future__ import unicode_literals

import logging
from datetime import datetime, timedelta

import os
import spacy
from rdflib.graph import ConjunctiveGraph as Graph
from rdflib.store import Store
from rdflib.plugin import get as plugin
from rdflib.term import URIRef
from rdflib import Namespace
from rdflib.namespace import RDF, RDFS
from toolz import partition_all
from joblib import Parallel, delayed

from lexicon import Lexicon
from sequence_trees import Forest, tree_from_sorted_parent_triples
from src import preprocessing

"""
prerequisites:
    set up / install:
      * virtuoso docker image: 
            see https://joernhees.de/blog/2015/11/23/setting-up-a-linked-data-mirror-from-rdf-dumps-dbpedia-2015-04-freebase-wikidata-linkedgeodata-with-virtuoso-7-2-1-and-docker-optional/
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
        virtuoso-python package has to be adapted. Change the line that is commented out below with the other one:
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
        
    ATTENTION:
        DO NOT PUT A FILE virtuoso.py IN THE SAME FOLDER!
"""
NIF = Namespace("http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#")
DBR = Namespace("http://dbpedia.org/resource/")
DBO = Namespace("http://dbpedia.org/ontology/")
ITSRDF = Namespace("http://www.w3.org/2005/11/its/rdf#")
ns_dict = {'nif': NIF, 'dbr': DBR, 'itsrdf': ITSRDF, 'rdf': RDF, 'rdfs': RDFS, 'dbo': DBO}

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


def create_context_tree(nlp, lexicon, children_typed, terminals, context, context_str, see_also_refs, link_refs,
                        link_ref_type=u"http://www.w3.org/2005/11/its/rdf#taIdentRef", max_see_also_refs=50):
    t_start = datetime.now()
    if len(see_also_refs) > max_see_also_refs:
        see_also_refs = []
    tree_context, terminal_parent_positions, terminal_types = tree_from_sorted_parent_triples(children_typed,
                                                                                              see_also_refs=see_also_refs,
                                                                                              lexicon=lexicon,
                                                                                              root_id=unicode(context)[:-len('?dbpv=2016-10&nif=context')])
    if logger.level <= logging.DEBUG:
        tree_context.visualize('../tmp_structure.svg')
    logger.debug('created forest_struct: %s' % str(datetime.now() - t_start))
    t_start = datetime.now()

    terminal_uri_strings, terminal_strings = zip(*[(unicode(uri), context_str[int(begin):int(end)]) for uri, begin, end in terminals])

    def terminal_reader():
        for s in terminal_strings:
            yield s

    def reader_roots():
        for s in terminal_types:
            yield s

    ref_type_id = lexicon[unicode(link_ref_type)]

    def link_reader():
        """
        :return: a generator yielding tuples (start_char, end_char, link_annotation_data, link_annotation_parents) for
                 the respective docs and None otherwise
        """
        refs = {}
        # tuple format: ?superString ?target ?superOffset ?beginIndex ?endIndex ?type
        for ref_tuple in link_refs:
            super_string = unicode(ref_tuple[0])
            target = unicode(ref_tuple[1])
            target_id = lexicon[target]
            offset = int(ref_tuple[2])
            begin_index = int(ref_tuple[3])
            end_index = int(ref_tuple[4])

            ref_list = refs.get(super_string, [])
            ref_list.append((begin_index - offset, end_index - offset, [ref_type_id, target_id], [0, -1]))
            refs[super_string] = ref_list
        for s in terminal_uri_strings:
            yield refs.get(s, None)

    logger.debug('parse data ...')
    forest_terminals = lexicon.read_data(reader=terminal_reader, sentence_processor=preprocessing.process_sentence1,
                                         reader_roots=reader_roots, reader_annotations=link_reader,
                                         parser=nlp, batch_size=10000, concat_mode='sequence', inner_concat_mode='tree',
                                         expand_dict=True)
    logger.debug('parsed data: %s' % str(datetime.now() - t_start))
    t_start = datetime.now()
    if logger.level <= logging.DEBUG:
        forest_terminals.visualize('../tmp_terminals.svg')

    # link terminal roots to (virtual) parents
    for i, root in enumerate(forest_terminals.roots):
        uri_string = terminal_uri_strings[i]
        #parent_uri = parent_uris[uri_string]
        uri_pos = terminal_parent_positions[uri_string]
        forest_terminals.parents[root] = uri_pos - (len(tree_context) + root)

    # append to tree_context
    tree_context.append(forest_terminals)
    logger.debug('added terminals: %s' % str(datetime.now() - t_start))

    #children = tree_context.children
    #roots = tree_context.roots
    return tree_context


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

    res_context_seealsos, children_typed, terminals, refs, context_str = query_context_data(graph, context)

    logger.info('load spacy ...')
    t_start = datetime.now()
    nlp = spacy.load('en_core_web_md')
    logger.info('loaded spacy: %s' % str(datetime.now() - t_start))
    tree_context = create_context_tree(lexicon=lexicon, nlp=nlp, children_typed=children_typed, terminals=terminals,
                                       see_also_refs=res_context_seealsos, link_refs=refs, context_str=context_str,
                                       context=context)
    logger.info('leafs: %i' % len(tree_context))

    tree_context.visualize('../tmp.svg')  # , start=0, end=100)


def process_all_contexts(graph, out_path='/mnt/WIN/ML/data/corpora/DBPEDIANIF', steps_save=1000):
    logger.setLevel(logging.INFO)

    if not os.path.exists(out_path):
        os.mkdir(out_path)

    # create empty lexicon
    lexicon = Lexicon(types=[])
    logger.info('lexicon size: %i' % len(lexicon))

    logger.info('load spacy ...')
    t_start = datetime.now()
    nlp = spacy.load('en_core_web_md')
    logger.info('loaded spacy: %s' % str(datetime.now() - t_start))

    logger.info('start parsing ...')
    failed = []
    t_start_batch = datetime.now()
    t_query = timedelta(microseconds=0)
    t_parse = timedelta(microseconds=0)
    skip = False
    i = 0
    for i, context in enumerate(graph.subjects(RDF.type, NIF.Context)):
        if i % steps_save == 0:
            if i > 0:
                logger.info('%i: %s (failed: %i, forest_size: %i) t_query: %s t_parse: %s%s'
                            % (i, str(datetime.now() - t_start_batch), len(failed), len(forest), str(t_query), str(t_parse), ' LOADED' if skip else ''))
                t_query = timedelta(microseconds=0)
                t_parse = timedelta(microseconds=0)
                if not skip:
                    fn = os.path.join(out_path, 'forest-%i' % i)
                    forest.dump(fn)
                    lexicon.dump(fn, types_only=True)
                    with open(fn+'.failed', 'w', ) as f:
                        #f.writelines([(unicode(uri) + u'\t' + unicode(e)).encode('utf8') for uri, e in failed])
                        for uri, e in failed:
                            f.write((unicode(uri) + u'\t' + unicode(e) + u'\n').encode('utf8'))
                    fn_prev = os.path.join(out_path, 'forest-%i' % (i - steps_save))
                    Lexicon.delete(fn_prev, types_only=True)

            # check, if next ones are already available
            try:
                fn_next = os.path.join(out_path, 'forest-%i' % (i+steps_save))
                #lexicon = Lexicon(filename=fn_next)
                forest = Forest(filename=fn_next)
                skip = True
            except IOError:
                skip = False
                forest = Forest(data=[], parents=[], lexicon=lexicon)
                failed = []

            t_start_batch = datetime.now()
        #contexts.append(context)
        if not skip:
            try:
                t_start = datetime.now()
                res_context_seealsos, children_typed, terminals, refs, context_str = query_context_data(graph, context)
                t_query += datetime.now() - t_start
                t_start = datetime.now()
                tree_context = create_context_tree(lexicon=lexicon, nlp=nlp, children_typed=children_typed, terminals=terminals,
                                                   see_also_refs=res_context_seealsos, link_refs=refs, context_str=context_str,
                                                   context=context)
                t_parse += datetime.now() - t_start
                forest.append(tree_context)
                #logger.debug('leafs: %i' % len(tree_context))
            except Exception as e:
                failed.append((context, e))

    # save remaining
    if i % steps_save > 0:
        fn = os.path.join(out_path, 'forest-%i' % i)
        forest.dump(fn)
        lexicon.dump(fn, types_only=True)


def test_utf8_context(graph, context=URIRef(u"http://dbpedia.org/resource/1958_US–UK_Mutual_Defence_Agreement?dbpv=2016-10&nif=context")):
    logger.setLevel(logging.DEBUG)
    logger_virtuoso.setLevel(logging.DEBUG)
    #for s, p, o in graph.triples((context, None, None)):
    #    print(s, p, o)
    #test_context_tree(graph, context)
    res = graph.query(u'select distinct ?p ?o where { <http://dbpedia.org/resource/1958_US–UK_Mutual_Defence_Agreement?dbpv=2016-10&nif=context> ?p ?o . } LIMIT 10')
    print(res)


def test_connect_utf8(dns="DSN=VOS;UID=dba;PWD=dba;WideAsUTF16=Y",
                 #q=u'SPARQL select distinct ?p ?o where { <http://dbpedia.org/resource/Damen_Group?dbpv=2016-10&nif=context> ?p ?o . } LIMIT 10'
                 q=u'SPARQL select distinct ?s ?p ?o where { ?s ?p ?o . VALUES ?s {<http://dbpedia.org/resource/1958_US–UK_Mutual_Defence_Agreement?dbpv=2016-10&nif=context>}} LIMIT 10'
                 ):
    import pyodbc

    def _sparql_select(q, cursor, must_close):
        from virtuoso.vstore import EagerIterator, VirtuosoResultRow, Variable, resolve
        logger.debug("_sparql_select")
        results = cursor.execute(q)
        vars = [Variable(col[0]) for col in results.description]
        var_dict = VirtuosoResultRow.prepare_var_dict(vars)

        def f():
            try:
                for r in results:
                    try:
                        yield VirtuosoResultRow([resolve(cursor, x) for x in r],
                                                var_dict)
                    except Exception as e:
                        logger.debug("skip row, because of %s", e)
                        pass
            finally:
                if must_close:
                    cursor.close()

        e = EagerIterator(f())
        e.vars = vars
        e.selectionF = e.vars
        return e

    logger.setLevel(logging.DEBUG)
    logger_virtuoso.setLevel(logging.DEBUG)
    connection = pyodbc.connect(dns)
    connection.setdecoding(pyodbc.SQL_CHAR, 'utf-8', pyodbc.SQL_CHAR)
    connection.setdecoding(pyodbc.SQL_WCHAR, 'utf-32LE', pyodbc.SQL_WCHAR, unicode)
    connection.setdecoding(pyodbc.SQL_WMETADATA, 'utf-32LE', pyodbc.SQL_WCHAR, unicode)
    #connection.setdecoding(pyodbc.SQL_WCHAR, 'utf-8', pyodbc.SQL_WCHAR, unicode)
    #connection.setdecoding(pyodbc.SQL_WMETADATA, 'utf-8', pyodbc.SQL_WCHAR, unicode)

    #connection.setencoding(unicode, 'utf-32LE', pyodbc.SQL_WCHAR)
    connection.setencoding(unicode, 'utf-8', pyodbc.SQL_CHAR)
    connection.setencoding(str, 'utf-8', pyodbc.SQL_CHAR)

    cursor = connection.cursor()
    res = _sparql_select(q, cursor, True)
    for s, p, o in res:
        print(s, p, o)
    #results = cursor.execute(q)
    #cursor.close()
    return res


def process_contexts(nlp, batch_idx, contexts, graph, out_path):
    logger.debug('start batch: %i' % batch_idx)
    t_start = datetime.now()
    lexicon = Lexicon(types=[])
    forest = Forest(data=[], parents=[], lexicon=lexicon)
    failed = []

    t_query = timedelta(microseconds=0)
    t_parse = timedelta(microseconds=0)

    for context in contexts:
        try:
            t_start = datetime.now()
            res_context_seealsos, children_typed, terminals, refs, context_str = query_context_data(graph, context)
            t_query += datetime.now() - t_start
            t_start = datetime.now()
            tree_context = create_context_tree(lexicon=lexicon, nlp=nlp, children_typed=children_typed,
                                               terminals=terminals,
                                               see_also_refs=res_context_seealsos, link_refs=refs,
                                               context_str=context_str,
                                               context=context)
            t_parse += datetime.now() - t_start
            forest.append(tree_context)
            # logger.debug('leafs: %i' % len(tree_context))
        except Exception as e:
            failed.append((context, e))

    fn = os.path.join(out_path, 'forest-%i' % batch_idx)
    forest.dump(fn)
    lexicon.dump(fn, types_only=True)
    with open(fn + '.failed', 'w', ) as f:
        for uri, e in failed:
            f.write((unicode(uri) + u'\t' + unicode(e) + u'\n').encode('utf8'))

    logger.debug('finished batch %i: %s (failed: %i) t_query: %s t_parse: %s' % (batch_idx, str(datetime.now() - t_start), len(failed), str(t_query), str(t_parse)))


def test_process_all_contexts_parallel(graph,
                                       out_path='/mnt/WIN/ML/data/corpora/DBPEDIANIF_parallel',
                                       steps_save=10000,
                                       n_jobs=4):
    logger.setLevel(logging.DEBUG)

    if not os.path.exists(out_path):
        os.mkdir(out_path)

    # create empty lexicon
    #lexicon = Lexicon(types=[])
    #logger.info('lexicon size: %i' % len(lexicon))

    logger.info('load spacy ...')
    t_start = datetime.now()
    nlp = spacy.load('en_core_web_md')
    logger.info('loaded spacy: %s' % str(datetime.now() - t_start))
    t_start = datetime.now()
    logger.info('create context partitions ... ')
    partitions = partition_all(steps_save, list(graph.subjects(RDF.type, NIF.Context)))
    logger.info('partitions created: %s' % str(datetime.now() - t_start))
    executor = Parallel(n_jobs=n_jobs)
    do = delayed(process_contexts)
    logger.info('start processing ... ')
    tasks = (do(nlp, i, batch, graph, out_path) for i, batch in enumerate(partitions))
    executor(tasks)


if __name__ == '__main__':
    #test_connect_utf8()

    logger.info('set up connection ...')
    Virtuoso = plugin("Virtuoso", Store)
    store = Virtuoso("DSN=VOS;UID=dba;PWD=dba;WideAsUTF16=Y")
    default_graph_uri = "http://dbpedia.org/nif"
    g = Graph(store, identifier=URIRef(default_graph_uri))
    logger.info('connected')

    #test_context_tree(g, context=URIRef(u'http://dbpedia.org/resource/1958_US–UK_Mutual_Defence_Agreement?dbpv=2016-10&nif=context'))
    #test_utf8_context(g)

    process_all_contexts(g)
    #test_process_all_contexts_parallel(g)



