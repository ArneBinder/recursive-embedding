# Preprocess Corpus Data

The **rec-emb data model** is optimized for fast training of rec-emb models. It includes the following:
 * a **lexicon**: it holds id-string mappings, string-hash mappings and embedding vectors
 * a **structure**: a serialized, typed graph that consists of two numpy arrays. One **data**
 array is holding the symbol type for each node. That are string hashes which can be converted
 into lexicon ids. The other is a sparse adjacency matrix encoding edge data.

Currently, the project provides three data sources:
 * [SICK corpus](http://clic.cimec.unitn.it/composes/sick.html)([paper](http://clic.cimec.unitn.it/marco/publications/marelli-etal-sick-lrec2014.pdf)). The SICK corpus
 consists of english sentence pairs extracted from image descriptions that are annotated
 with a *relatedness score* and an *entailment class* (one of `neutral`, `entailment`, or `contradiction`).
 * [SemEval 2010 Task 8 relation extraction corpus](http://semeval2.fbk.eu/semeval2.php?location=tasks#T11)([paper](http://www.aclweb.org/anthology/S10-1006)). It contains
 english sentences where argument pairs are annotated with one out of nine abstract relation types, e.g. `cause-effect` or `message-topic`.
 * [Large Movie Review Dataset (IMDB)](http://ai.stanford.edu/~amaas/data/sentiment)([paper](http://ai.stanford.edu/~amaas/papers/wvSent_acl2011.pdf)). This binary sentiment analysis dataset
 consists of 50,000 english [IMDB](www.imdb.com) movie reviews with clear ratings: negative reviews have a score <= 4 out of 10,
and a positive reviews have a score >= 7 out of 10.


NOTE: SICK and SemEval2010T8 are included here ([datasets folder](datasets)). The IMDB dataset has to be downloaded from [http://ai.stanford.edu/~amaas/data/sentiment](http://ai.stanford.edu/~amaas/data/sentiment) and extracted into the same folder as the other datasets.

The preprocessing happens in two steps:

 1. parsing: This results in one jsonline file per input file (or directory in the case of IMDB) containing records in an intermediate format that is a very much simplified version of [NIF](http://persistence.uni-leipzig.org/nlp2rdf/)
 2. conversion: Convert into compact, graph based format (rec-emb format) to train the models with.

NOTE: Per default, two structural variants are created in the conversion step: (1) word nodes are linked **directly**, or (2) word nodes are linked via dependency **edge** type nodes.

The final rec-emb data can be visualized with the [visualization tool](../tools/visualize/README.md).


## HOW TO preprocess

Install:
 * docker
 * docker compose

1. Clone this repo and switch into this folder:
```bash
git clone https://github.com/ArneBinder/recursive-embedding.git
cd recursive-embedding/docker/preprocessing
```
2. Rename [`.env.dev`](.env.dev) (or copy) to `.env` and adapt its parameters.
3. Set execution permission of `scripts`: `chmod +x scripts/*.sh`
4. Optional: adapt the script files, e.g.
    * use Spacy instead of CoreNLP (parameter `--parser`),
    * restrict output to create only **direct** (default) or **edge** (flag `-e`) linked structure, or
    * adjust the minimal node type count (parameter `-m`): node types (e.g. words) that occur less then this value are replaced with the `UNKNOWN` type.
5. To start the parsing process, execute from current folder:
```bash
docker-compose up corpus-parse
```
6. Afterwards, convert to rec-emb data format:
```bash
docker-compose up corpus-convert
```




