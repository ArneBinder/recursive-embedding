#!/bin/sh

# This script converts datasets in the nif-based format to actual rec-emb data format.
# Note: If the target folder already exists, processing of this one will be skipped!

# Note: Ensure to enable correct python environment before execution.

CORPORA_OUT="/root/corpora_out"
GLOVE_TXT="/root/glove_dir/glove.840B.300d.txt"

cd /root/recursive-embedding

echo "convert SICK corpus..."
python src/corpus_rdf.py CONVERT -i "$CORPORA_OUT"/SICK_RDF/spacy_noner -c sck:vocab#entailment_judgment -g "$GLOVE_TXT" -m 2
python src/corpus_rdf.py CONVERT -i "$CORPORA_OUT"/SICK_RDF/corenlp_noner -c sck:vocab#entailment_judgment -g "$GLOVE_TXT" -m 2
python src/corpus_rdf.py CONVERT -i "$CORPORA_OUT"/SICK_RDF/spacy_noner -c sck:vocab#entailment_judgment -g "$GLOVE_TXT" -m 2 -e
python src/corpus_rdf.py CONVERT -i "$CORPORA_OUT"/SICK_RDF/corenlp_noner -c sck:vocab#entailment_judgment -g "$GLOVE_TXT" -m 2 -e


echo "convert SEMEVAL corpus..."
python src/corpus_rdf.py CONVERT -i "$CORPORA_OUT"/SEMEVAL2010T8_RDF/spacy -c smvl:vocab#relation -g "$GLOVE_TXT" -m 2
python src/corpus_rdf.py CONVERT -i "$CORPORA_OUT"/SEMEVAL2010T8_RDF/corenlp -c smvl:vocab#relation -g "$GLOVE_TXT" -m 2
## link via edges
python src/corpus_rdf.py CONVERT -i "$CORPORA_OUT"/SEMEVAL2010T8_RDF/spacy -c smvl:vocab#relation -g "$GLOVE_TXT" -m 2 -e
python src/corpus_rdf.py CONVERT -i "$CORPORA_OUT"/SEMEVAL2010T8_RDF/corenlp -c smvl:vocab#relation -g "$GLOVE_TXT" -m 2 -e
## link via edges; re-linked entities (LCA)
python src/corpus_rdf.py CONVERT -i "$CORPORA_OUT"/SEMEVAL2010T8_RDF/spacy -c smvl:vocab#relation -g "$GLOVE_TXT" -m 2 -e -l
python src/corpus_rdf.py CONVERT -i "$CORPORA_OUT"/SEMEVAL2010T8_RDF/corenlp -c smvl:vocab#relation -g "$GLOVE_TXT" -m 2 -e -l


## convert TACRED   EXCLUDED
## entity masking; re-linked entities
#python src/corpus_rdf.py CONVERT -i "$CORPORA_OUT"/TACRED_RDF/None -c tac:vocab#relation -g "$GLOVE_TXT" -m 20 -t -l
## entity masking; entity-to-entity spans
#python src/corpus_rdf.py CONVERT -i "$CORPORA_OUT"/TACRED_RDF/None -c tac:vocab#relation -g "$GLOVE_TXT" -m 20 -t -s
## entity masking; re-linked entities; link via deprel
#python src/corpus_rdf.py CONVERT -i "$CORPORA_OUT"/TACRED_RDF/None -c tac:vocab#relation -g "$GLOVE_TXT" -m 20 -t -l -e


echo "convert IMDB corpus (TAKES A WHILE)..."
python src/corpus_rdf.py CONVERT -i "$CORPORA_OUT"/IMDB_RDF/spacy_noner -c imdb:vocab#sentiment -g "$GLOVE_TXT" -m 20
python src/corpus_rdf.py CONVERT -i "$CORPORA_OUT"/IMDB_RDF/corenlp_noner -c imdb:vocab#sentiment -g "$GLOVE_TXT" -m 20
python src/corpus_rdf.py CONVERT -i "$CORPORA_OUT"/IMDB_RDF/spacy_noner -c imdb:vocab#sentiment -g "$GLOVE_TXT" -m 20 -e
python src/corpus_rdf.py CONVERT -i "$CORPORA_OUT"/IMDB_RDF/corenlp_noner -c imdb:vocab#sentiment -g "$GLOVE_TXT" -m 20 -e