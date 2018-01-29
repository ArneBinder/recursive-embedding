## execute via: 
## isql-vt 1111 dba dba VERBOSE=OFF /var/lib/virtuoso-opensource-7/<this-filename>.sql

set blobs on ;

sparql 
define output:format "TSV"
select ?context ?linkRefContext ?seeAlsoSectionStr where {
?context a nif:Context .
FILTER NOT EXISTS { ?context nif:lastSection ?seeAlsoSection . }
?seeAlsoSection a nif:Section .
?seeAlsoSection nif:superString+ ?context .
?seeAlsoSection nif:beginIndex ?beginIndex .
?seeAlsoSection nif:endIndex ?endIndex .
?link nif:superString+ ?seeAlsoSection .
?link <http://www.w3.org/2005/11/its/rdf#taIdentRef> ?linkRef .
?context nif:isString ?contextStr .
FILTER (?endIndex - ?beginIndex > 0)
FILTER (STRLEN(?contextStr) >= ?endIndex)
BIND(SUBSTR(STR(?contextStr), ?beginIndex + 1, ?endIndex - ?beginIndex) AS ?seeAlsoSectionStr)
FILTER (STRSTARTS(STR(?seeAlsoSectionStr), "See also"))
BIND(IRI(CONCAT(STR(?linkRef), "?dbpv=2016-10&nif=context")) AS ?linkRefContext)
} 
LIMIT 10 ;
