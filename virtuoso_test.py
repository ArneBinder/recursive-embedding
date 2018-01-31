from rdflib.graph import ConjunctiveGraph as Graph
from rdflib.store import Store
from rdflib.plugin import get as plugin
from rdflib.term import URIRef
from rdflib import Namespace
from rdflib.namespace import RDF

"""
uses https://github.com/maparent/virtuoso-python

requires:
    install: 
        libvirtodbc0 (high-performance database - ODBC libraries (driver)) included in:
            libvirtodbc0_7.2_amd64.deb (https://github.com/Dockerizing/triplestore-virtuoso7)
    
    settings:            
        ~/.odbc.ini (or /etc/odbc.ini) containing:
        "
        [ODBC Data Sources]
        VOS = Virtuoso
        
        [VOS]
        Description = Open Virtuoso
        Driver      = /usr/local/virtuoso-opensource/lib/virtodbcu_r.so
        Address     = localhost:1111
        Locale      = en.UTF-8
        "
        
        and ~/.odbcinst.ini (or /etc/odbcinst.ini) containing:
        "
        [ODBC Drivers]
        Virtuoso = Installed
        
        [Virtuoso]
        Driver = /usr/lib/odbc/virtodbc_r.so
        Setup  = /usr/lib/odbc/virtodbc_r.so
"""

if __name__ == '__main__':
    print('set up connection ...')
    Virtuoso = plugin("Virtuoso", Store)
    store = Virtuoso("DSN=VOS;UID=dba;PWD=dba;WideAsUTF16=Y")
    print('connected')
    default_graph_uri = "http://dbpedia.org/nif"
    g = Graph(store, identifier=URIRef(default_graph_uri))
    nif = Namespace("http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#")

    for i, (nif_context, _, _) in enumerate(g.triples((None, RDF.type, nif.Context))):
        if i > 10:
            break
        print(nif_context)


