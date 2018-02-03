from __future__ import unicode_literals

import logging
import pprint
from datetime import datetime

import spacy
from rdflib.graph import ConjunctiveGraph as Graph
from rdflib.store import Store
from rdflib.plugin import get as plugin
from rdflib.term import URIRef
from rdflib import Namespace
from rdflib.namespace import RDF, RDFS

from lexicon import Lexicon
from sequence_trees import Forest, tree_from_sorted_parent_triples
from src import preprocessing

"""
prerequisites:
    set up / install:
        virtuoso docker image: 
            see https://joernhees.de/blog/2015/11/23/setting-up-a-linked-data-mirror-from-rdf-dumps-dbpedia-2015-04-freebase-wikidata-linkedgeodata-with-virtuoso-7-2-1-and-docker-optional/
        virtuoso odbc driver (libvirtodbc0) included in libvirtodbc0_7.2_amd64.deb (e.g. from https://github.com/Dockerizing/triplestore-virtuoso7)
            wget https://github.com/Dockerizing/triplestore-virtuoso7/raw/master/libvirtodbc0_7.2_amd64.deb
            sudo apt install /PATH/TO/libvirtodbc0_7.2_amd64.deb
        rdflib:
            pip install rdflib
        virtuoso-python: see https://github.com/maparent/virtuoso-python
            git clone git@github.com:maparent/virtuoso-python.git && cd virtuoso-python
            pip install -r requirements.txt
            python setup.py install
    
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
ITS = Namespace("http://www.w3.org/2005/11/its/rdf#")
ns_dict = {'nif': NIF, 'dbr': DBR, 'its': ITS, 'rdf': RDF, 'rdfs': RDFS}

logger = logging.getLogger('corpus_dbpedia_nif')
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())


def query_see_also_links(graph, initBindings=None):
    q_str = ('select ?context ?linkRefContext where { \
                        ?context a nif:Context . \
                        ?seeAlsoSection a nif:Section . \
                        ?seeAlsoSection nif:superString+ ?context . \
                        ?seeAlsoSection nif:beginIndex ?beginIndex . \
                        ?seeAlsoSection nif:endIndex ?endIndex . \
                        ?link nif:superString+ ?seeAlsoSection . \
                        ?link <http://www.w3.org/2005/11/its/rdf#taIdentRef> ?linkRef . \
                        ?context nif:isString ?contextStr . \
                        FILTER (?endIndex - ?beginIndex > 0) \
                        FILTER (STRLEN(?contextStr) >= ?endIndex) \
                        BIND(SUBSTR(STR(?contextStr), ?beginIndex + 1, ?endIndex - ?beginIndex) AS ?seeAlsoSectionStr) \
                        FILTER (STRSTARTS(STR(?seeAlsoSectionStr), "See also")) \
                        BIND(IRI(CONCAT(STR(?linkRef), "?dbpv=2016-10&nif=context")) AS ?linkRefContext) \
                        } \
                        LIMIT 1000 OFFSET 0 \
                        ')

    res = graph.store.query(q_str, initNs=ns_dict, initBindings=initBindings)
    return res


def query_first_section_structure(graph, initBindings=None):
    q_str = (
        'construct {'
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
        ' VALUES ?p {nif:beginIndex nif:endIndex nif:superString its:taIdentRef rdf:type} .'
        #' ?s nif:superString ?superString .'
        #' ?superString nif:beginIndex ?superOffset .'
        #' ?s its:taIdentRef ?ref .'
        #' ?s ?p2 ?oo2 .'
        
        #' FILTER (?p != nif:referenceContext)'
        '}')
    logger.debug(q_str)
    res = graph.store.query(q_str, initNs=ns_dict, initBindings=initBindings)
    #print(type(res))
    return res


def create_context_tree(nlp, lexicon, children_typed, terminals, context):
    t_start = datetime.now()
    tree_context, terminal_parent_positions, terminal_types = tree_from_sorted_parent_triples(children_typed, lexicon=lexicon, root_id=str(context))
    logger.info('created forest_struct: %s' % str(datetime.now() - t_start))
    t_start = datetime.now()

    terminal_uri_strings, terminal_strings = zip(*[(unicode(uri), context_str[int(begin):int(end)]) for uri, begin, end in terminals])

    def terminal_reader():
        for s in terminal_strings:
            yield s

    def reader_roots():
        for s in terminal_types:
            yield s

    logger.info('parse data ...')
    forest_terminals = lexicon.read_data(reader=terminal_reader, sentence_processor=preprocessing.process_sentence1,
                                         reader_roots=reader_roots,
                                         parser=nlp, batch_size=10000, concat_mode='sequence', inner_concat_mode='tree',
                                         expand_dict=True)
    logger.info('parsed data: %s' % str(datetime.now() - t_start))
    t_start = datetime.now()

    # link terminal roots to (virtual) parents
    for i, root in enumerate(forest_terminals.roots):
        uri_string = terminal_uri_strings[i]
        #parent_uri = parent_uris[uri_string]
        uri_pos = terminal_parent_positions[uri_string]
        forest_terminals.parents[root] = uri_pos - (len(tree_context) + root)

    # append to tree_context
    tree_context.append(forest_terminals)
    logger.info('added terminals: %s' % str(datetime.now() - t_start))

    #children = tree_context.children
    #roots = tree_context.roots
    return tree_context


if __name__ == '__main__':
    logger.info('set up connection ...')
    Virtuoso = plugin("Virtuoso", Store)
    store = Virtuoso("DSN=VOS;UID=dba;PWD=dba;WideAsUTF16=Y")
    default_graph_uri = "http://dbpedia.org/nif"
    g = Graph(store, identifier=URIRef(default_graph_uri))
    logger.info('connected')

    #for i, (nif_context, _, _) in enumerate(g.triples((None, RDF.type, nif.Context))):
    #    if i > 10:
    #        break
    #    print(nif_context)
    t_start = datetime.now()
    #res = query_see_also_links(g)
    #res = query_first_section_structrue(g, "http://dbpedia.org/resource/8th_Canadian_Hussars_(Princess_Louise's)?dbpv=2016-10&nif=context")
    context = URIRef("http://dbpedia.org/resource/Damen_Group?dbpv=2016-10&nif=context")
    res = query_first_section_structure(g, {'context': context})
    logger.info('exec query: %s' % str(datetime.now() - t_start))
    t_start = datetime.now()
    g_structure = Graph()
    for r in res:
        g_structure.add(r)
    #g_structure.addN(res)
    #query_first_section_structrue(g, "http://dbpedia.org/resource/Damen_Group?dbpv=2016-10&nif=context"))
    #q = prepareQuery(
    #    'SELECT ?c WHERE { ?c a nif:Context .} LIMIT 10',
    #    initNs={"rdfs": RDFS, "nif": nif})
    logger.info('new graph: %s' % str(datetime.now() - t_start))
    t_start = datetime.now()
    logger.info('result count: %i' % len(res))
    for i, row in enumerate(res):
        logger.debug(row)

    logger.debug('children (ordered by ?parent_beginIndex DESC(?parent_endIndex) ?child_beginIndex DESC(?child_endIndex)):')
    #for s, p, o in g_structure.triples((None, NIF.subString, None)):
    #    print(s, o)
    #children_typed = g_structure.query('SELECT DISTINCT ?s ?child ?type WHERE {?s a ?type . VALUES ?type {nif:Section nif:Context} . ?s nif:beginIndex ?beginIndex . ?s nif:endIndex ?endIndex . ?child nif:superString ?s . ?child nif:beginIndex ?child_beginIndex . ?child nif:endIndex ?child_endIndex .} ORDER BY ?beginIndex DESC(?endIndex) ?child_beginIndex DESC(?child_endIndex)', initNs=ns_dict)
    children_typed = g_structure.query(
        'SELECT DISTINCT ?child ?type_child ?parent WHERE {?parent a ?type . VALUES ?type {nif:Section nif:Context} . ?child a ?type_child . ?parent nif:beginIndex ?parent_beginIndex . ?parent nif:endIndex ?parent_endIndex . ?child nif:superString ?parent . ?child nif:beginIndex ?child_beginIndex . ?child nif:endIndex ?child_endIndex .} ORDER BY ?parent_beginIndex DESC(?parent_endIndex) ?child_beginIndex DESC(?child_endIndex)',
        initNs=ns_dict)

    for row in children_typed:
        logger.debug(row)

    logger.debug('terminals (ordered by ?beginIndex DESC(?endIndex)):')
    terminals = g_structure.query('SELECT DISTINCT ?terminal ?beginIndex ?endIndex WHERE {?terminal nif:beginIndex ?beginIndex . ?terminal nif:endIndex ?endIndex . ?terminal a ?type . VALUES ?type {nif:Title nif:Paragraph}} ORDER BY ?beginIndex DESC(?endIndex)', initNs=ns_dict)
    for row in terminals:
        #pprint.pprint(row)
        logger.debug(row)

    logger.debug('refs:')
    #for s, p, o in g_structure.triples((None, ITS.taIdentRef, None)):
    #    print(s, o)
    for row in g_structure.query('SELECT DISTINCT ?superString ?targetContext ?superOffset ?beginIndex ?endIndex ?type WHERE {?ref its:taIdentRef ?target . ?ref nif:superString ?superString . ?ref nif:beginIndex ?beginIndex . ?ref nif:endIndex ?endIndex . ?superString nif:beginIndex ?superOffset . ?ref a ?type . BIND(URI(CONCAT(STR(?target), "?dbpv=2016-10&nif=context")) as ?targetContext)}', initNs=ns_dict):
        logger.debug(row)

    logger.debug('str:')
    #for o in g_structure.objects(subject=context, predicate=NIF.isString):
    #    print(o)
    context_str = unicode(g_structure.value(subject=context, predicate=NIF.isString, any=False))
    #context_str = context_str_lit.toPython
    logger.debug(len(context_str))

    logger.info('print result: %s' % str(datetime.now() - t_start))
    t_start = datetime.now()

    # create empty lexicon
    lexicon = Lexicon(types=[])
    logger.debug(len(lexicon))

    logger.info('load spacy ...')
    nlp = spacy.load('en_core_web_md')
    logger.info('loaded spacy: %s' % str(datetime.now() - t_start))
    t_start = datetime.now()

    tree_context = create_context_tree(lexicon=lexicon, nlp=nlp, children_typed=children_typed, terminals=terminals,
                                       context=context)
    logger.info('leafs: %i' % len(tree_context))

    tree_context.visualize('tmp.svg')#, start=0, end=100)
