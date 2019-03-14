# Preprocess Corpus Data

Preprocess corpus data for the following datasets:
 * SICK: see http://alt.qcri.org/semeval2014/task1/
 * SemEval 2010 Task 8: see http://semeval2.fbk.eu/semeval2.php?location=tasks#T11
 * Large Movie Review Dataset (IMDB): see http://ai.stanford.edu/~amaas/data/sentiment/

NOTE: SICK and SemEval2010T8 are included here ([datasets folder](datasets)). The IMDB dataset has to be downloaded from [http://ai.stanford.edu/~amaas/data/sentiment](http://ai.stanford.edu/~amaas/data/sentiment) and extracted into the same folder as the other datasets.

The preprocessing happens in two steps:
1. parsing: this results in json line file per input file / dir containing records in an intermediate format that is a very much simplified version of [NIF](http://persistence.uni-leipzig.org/nlp2rdf/)
2. conversion: convert into compact, graph based format to train the models with

The latter can be visualized with the [visualization tool](../tools/visualize/README.md)



## HOW TO preprocess

Install:
 * docker
 * docker compose

Clone this repo and switch into this folder:
```bash
git clone https://github.com/ArneBinder/recursive-embedding.git
cd recursive-embedding/docker/preprocessing
```

Rename [`.env.dev`](.env.dev) (or copy) to `.env` and adapt its parameters.

Set execution permission of `scripts`: `chmod +x scripts/*.sh`

Optional:
 * adapt the script files (use Spacy instead of CoreNLP)

To start the parsing process, execute from current folder:

```bash
docker-compose up corpus-parse
```

Afterwards, convert to rec-emb data format:

```bash
docker-compose up corpus-convert
```




