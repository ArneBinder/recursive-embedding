#!/bin/sh

# Note: Ensure to enable correct python environment before execution.

## CHANGE THIS!
CORPORA_IN="/mnt/DATA/ML/data/corpora"
CORPORA_OUT="/mnt/DATA/ML/data/corpora_preprocessed/FINAL_SERVER"



## parse SICK
# see src.corpus_sick.parse_rdf for required files
python src/corpus_sick.py PARSE_RDF --in-path "$CORPORA_IN"/SICK --out-path "$CORPORA_OUT"/SICK_RDF --parser spacy
python src/corpus_sick.py PARSE_RDF --in-path "$CORPORA_IN"/SICK --out-path "$CORPORA_OUT"/SICK_RDF --parser corenlp

## parse IMDB Sentiment
# see src.corpus_imdb.parse_rdf for required files
python src/corpus_imdb.py PARSE_RDF --in-path "$CORPORA_IN"/aclImdb --out-path "$CORPORA_OUT"/IMDB_RDF --parser spacy
python src/corpus_imdb.py PARSE_RDF --in-path "$CORPORA_IN"/aclImdb --out-path "$CORPORA_OUT"/IMDB_RDF --parser corenlp

## parse SemEval2010_task8
# see src.corpus_seemval2010task8.parse_rdf for required files
python src/corpus_semeval2010task8.py PARSE_RDF --in-path "$CORPORA_IN"/SemEval2010_task8_all_data --out-path "$CORPORA_OUT"/SEMEVAL2010T8_RDF --parser spacy
python src/corpus_semeval2010task8.py PARSE_RDF --in-path "$CORPORA_IN"/SemEval2010_task8_all_data --out-path "$CORPORA_OUT"/SEMEVAL2010T8_RDF --parser corenlp

## parse TacRED EXCLUDED
## see src.corpus_tacred.parse_rdf for required files
#python src/corpus_tacred.py PARSE_RDF --in-path "$CORPORA_IN"/tacred --out-path "$CORPORA_OUT"/TACRED_RDF
