#!/bin/sh

# Note: Ensure to enable correct python environment before execution.

CORPORA_OUT="/mnt/DATA/ML/data/corpora_out/FINAL"
SPLIT_COUNT="10"

echo "use SPLIT_COUNT=$SPLIT_COUNT"

## SICK
python src/corpus_rdf.py CREATE_INDICES --merged-forest-path "$CORPORA_OUT"/SICK_RDF/spacy_noner_recemb_mc2/forest --split-count "$SPLIT_COUNT" --step-root 2 2>&1
python src/corpus_rdf.py CREATE_INDICES --merged-forest-path "$CORPORA_OUT"/SICK_RDF/corenlp_noner_recemb_mc2/forest --split-count "$SPLIT_COUNT" --step-root 2 2>&1
python src/corpus_rdf.py CREATE_INDICES --merged-forest-path "$CORPORA_OUT"/SICK_RDF/spacy_noner_recemb_edges_mc2/forest --split-count "$SPLIT_COUNT" --step-root 2 2>&1
python src/corpus_rdf.py CREATE_INDICES --merged-forest-path "$CORPORA_OUT"/SICK_RDF/corenlp_noner_recemb_edges_mc2/forest --split-count "$SPLIT_COUNT" --step-root 2 2>&1

## SEMEVAL
python src/corpus_rdf.py CREATE_INDICES --merged-forest-path "$CORPORA_OUT"/SEMEVAL2010T8_RDF/spacy_recemb_ner_mc2/forest --split-count "$SPLIT_COUNT" 2>&1
python src/corpus_rdf.py CREATE_INDICES --merged-forest-path "$CORPORA_OUT"/SEMEVAL2010T8_RDF/corenlp_recemb_ner_mc2/forest --split-count "$SPLIT_COUNT" 2>&1
python src/corpus_rdf.py CREATE_INDICES --merged-forest-path "$CORPORA_OUT"/SEMEVAL2010T8_RDF/spacy_recemb_ner_edges_mc2/forest --split-count "$SPLIT_COUNT" 2>&1
python src/corpus_rdf.py CREATE_INDICES --merged-forest-path "$CORPORA_OUT"/SEMEVAL2010T8_RDF/corenlp_recemb_ner_edges_mc2/forest --split-count "$SPLIT_COUNT" 2>&1
python src/corpus_rdf.py CREATE_INDICES --merged-forest-path "$CORPORA_OUT"/SEMEVAL2010T8_RDF/spacy_recemb_ner_span_mc2/forest --split-count "$SPLIT_COUNT" 2>&1
python src/corpus_rdf.py CREATE_INDICES --merged-forest-path "$CORPORA_OUT"/SEMEVAL2010T8_RDF/corenlp_recemb_ner_span_mc2/forest --split-count "$SPLIT_COUNT" 2>&1

## TACRED
python src/corpus_rdf.py CREATE_INDICES --merged-forest-path "$CORPORA_OUT"/TACRED_RDF/None_recemb_ner_mc2/forest --split-count "$SPLIT_COUNT" 2>&1
python src/corpus_rdf.py CREATE_INDICES --merged-forest-path "$CORPORA_OUT"/TACRED_RDF/None_recemb_ner_edges_mc2/forest --split-count "$SPLIT_COUNT" 2>&1
python src/corpus_rdf.py CREATE_INDICES --merged-forest-path "$CORPORA_OUT"/TACRED_RDF/None_recemb_ner_span_mc2/forest --split-count "$SPLIT_COUNT" 2>&1

## IMDB
python src/corpus_rdf.py CREATE_INDICES --merged-forest-path "$CORPORA_OUT"/IMDB_RDF/spacy_noner_recemb_mc2/forest --split-count "$SPLIT_COUNT" 2>&1
python src/corpus_rdf.py CREATE_INDICES --merged-forest-path "$CORPORA_OUT"/IMDB_RDF/corenlp_noner_recemb_mc2/forest --split-count "$SPLIT_COUNT" 2>&1
python src/corpus_rdf.py CREATE_INDICES --merged-forest-path "$CORPORA_OUT"/IMDB_RDF/spacy_noner_recemb_edges_mc2/forest --split-count "$SPLIT_COUNT" 2>&1
python src/corpus_rdf.py CREATE_INDICES --merged-forest-path "$CORPORA_OUT"/IMDB_RDF/corenlp_noner_recemb_edges_mc2/forest --split-count "$SPLIT_COUNT" 2>&1
