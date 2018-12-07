#!/bin/bash

# see https://stanfordnlp.github.io/CoreNLP/other-languages.html 
# and https://github.com/nltk/nltk/wiki/Stanford-CoreNLP-API-in-NLTK

STANFORD_VERSION="stanford-corenlp-full-2018-02-27"
LANGUAGE_MODEL="stanford-english-corenlp-2018-02-27-models.jar"
PORT=9000

# if an argument is given, go to this directory
if [ -n "$1" ]; then
    PORT="$1"
fi

# if an argument is given, go to this directory
if [ -n "$2" ]; then
    cd "$2"
fi

if [ ! -d "$STANFORD_VERSION" ]; then
  echo "create $STANFORD_VERSION"
  if [ ! -f "$STANFORD_VERSION.zip" ]; then
    echo "$STANFORD_VERSION.zip not found. Download it ..."
    wget "http://nlp.stanford.edu/software/$STANFORD_VERSION.zip"
  fi
  unzip "$STANFORD_VERSION.zip"
  rm "$STANFORD_VERSION.zip"
else
  echo "$STANFORD_VERSION" already exists
fi

cd "$STANFORD_VERSION"

if [ ! -f "$LANGUAGE_MODEL" ]; then
  echo "$LANGUAGE_MODEL not found. Download it ..."
  # get english model from https://stanfordnlp.github.io/CoreNLP/history.html#
  wget "http://nlp.stanford.edu/software/$LANGUAGE_MODEL"
fi

# start the server
java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -preload tokenize,ssplit,pos,lemma,ner,parse,depparse -status_port $PORT -port $PORT -timeout 15000 > /dev/null

