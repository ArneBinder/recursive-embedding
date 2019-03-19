# recursive-embedding (rec-emb)

Train embeddings for hierarchically structured data. rec-emb is a research project.

## Idea

A lot of real world phenomena are structured hierarchically. Creating a semantic model that exploits tree structure seems to be a native need.


### The Embedding Model

The rec-emb embedding model
 * is a Vector Space Model (VSM)
 * is a Compositional Distributional Semantics Model (CDSM)
 * follows the map-reduce paradigm

### The Data Model

The rec-emb data model
 * is a simple serialization format for data that is structured in directed, typed graphs
 * is optimized for fast training of the rec-emb embedding model
 * identifies data by integer ids or hashes -- i.e. is content agnostic
 * links data with directed, unlabeled edges -- i.e. is relation agnostic


**TL;DR**, the `docker` folder provides several starting points:
 * [Preprocessing of datasets](docker/preprocessing)
 * [REST endpoint for corpus visualization](docker/tools/visualize)
 * [Training](docker/train/tensorflow-fold)


## Preprocessing

The **rec-emb data model** includes the following:
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

See the [preprocessing docker setup](docker/preprocessing) for how to parse and convert these datasets into the rec-emb format.


## The rec-emb Embedding Model

Create a single embedding for any tree generated out of the graph structure.
Use `reduce` to combine children and apply `map` along edges. Both functions
*can* depend on the node data (at least one of them *should* depend on it). Together, they form a **Headed Tree Unit (HTU)**.

Implemented `reduce` functions:
 * SUM
 * AVG
 * MAX
 * ATT (attention; default: use node data mapped via a FC as attention weights; split: use a dedicated part of the node data as attention weights; single: use attention weights independent of node data)
 * LSTM
 * GRU
 
Implemented `map` functions:
 * FC (fully connected layer)
 * LSTM (optionally stacked)
 * GRU (optionally stacked)

HTU implementations:
 * default: Aggregate all child embeddings with `reduce` and incorporate the node data via one `map` execution.
 * reverted: Contextualize each child via `map` individually and reduce afterwards.
 * special leaf handling: A substantial amount of nodes are leafs. We propose two HTU variants that take this into account.
    * init state: Use a trainable initialization state instead of the null vector.
    * leaf passing: Omit the `map` step at all. Just execute one FC to allow for different
    dimensionality of (word) embeddings and internal state.

### Similarity Scoring

Given two embeddings, calculate a floating point value that indicates their similarity. Here, one embedding may consist of one or
multiple concatenated tree embeddings, eventually further transformed with one or multiple FCs.
Furthermore, one embedding may be a class label embedding.

Implemented similarity functions:
 * cosine


## Training

See the [training docker setup](docker/train/tensorflow-fold) for how to train and test models for individual tasks.


### TASK: Predict Relatedness Scores (regression)

Predict, how strongly two trees are related.

"Related" can be interpreted in several ways, e.g., in the case of SICK, several annotators scored pairs of sentences intuitively and the resulting scores are averaged.

The [similarity score](#Similarity-Scoring) between two tree embeddings is used as measure for the strength of the relatedness.
In general, one FC is applied to each tree embedding before scoring.

Task instances:
 * Relatedness Prediction on SICK.

### TASK: Multiclass Prediction

Predict, if a tree matches one (or multiple) labels

The [similarity score](#Similarity-Scoring) between one (or multiple concatenated) tree embeddings and
the class label embedding is used as probability that the instance belongs to that class.
In general, one FC is applied to the (concatenated) tree embedding(s).

Task instances:
 * Recognizing Textual Entailment on SICK.
 * Binary Sentiment Analysis on IMDB.
 * Relation Extraction on SemEval2010 Task 8.


## License

Copyright 2018 Arne Binder

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

