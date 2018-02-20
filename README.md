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
 * is a simple serialization format for data that is structured in trees
 * is optimized for fast training of the rec-emb embedding model
 * identifies data by integer ids or hashes -- i.e. is content agnostic
 * links data with directed, unlabeled edges -- i.e. is relation agnostic


**TL;DR**, the `docker` folder provides several starting points:
 * [preprocessing of DBpedia-NIF data](docker/create-corpus/dbpedia-nif/README.md)
 * [preprocessing of SICK corpus data](docker/create-corpus/sick/README.md)
 * [REST endpoint for corpus visualization](docker/tools/visualize/README.md)
 * [tensorflow_fold tf1.3 mkl](docker/tensorflow_fold/tensorflowfold_conda_tf1.3_mkl)


## Preprocessing

The **rec-emb data model** includes the following:
 * a **lexicon**: it holds id-string mappings, id-stringhash mappings and embedding vectors
 * a **forest**: a serialized forest consists of two numpy arrays. One **data** array is holding the data ids or string hashes (can 
 be switched when a lexicon is provided), the other holds **parent** offsets for every data point. Data ids point to 
 embedding vector entries. For training, the parent array can be converted into two arrays, **children** and 
 **children position**. The children array holds amounts of children for every data point followed by the child offsets. 
 The children position array indicates for every data point, where to look in the children array. 

Currently, the project provides two data sources:
 * [SICK corpus](http://clic.cimec.unitn.it/marco/publications/marelli-etal-sick-lrec2014.pdf) data. The SICK corpus consists of sentence pairs extracted from image descriptions that are annotated 
 with a *relatedness score*. See [preprocessing of SICK corpus data](docker/create-corpus/sick/README.md) for a docker 
 image that assists to create a **rec-emb data model** for SICK.
 * [DBpedia-NIF 2016-10](http://wiki.dbpedia.org/downloads-2016-10) data. This corpus builds on [DBpedia](http://wiki.dbpedia.org/), but includes preprocessed [NIF](http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core/nif-core.html) data. It consists of (1) 
 cleaned full text of wikipedia articles, (2) structural information like segmentation into sections, paragraphs, etc., and (3) 
 annotations of links that point to other wikipedia articles. See 
 [preprocessing of DBpedia-NIF data](docker/create-corpus/dbpedia-nif/README.md) for a docker image that assists to 
 create a **rec-emb data model** for DBpedia-NIF.


## The rec-emb Embedding Model

Create a single embedding for any tree in the forest. Use `reduce` to combine children and apply `map` along edges. Both functions *can* depend on the node data (at least one of them *should* depend on it).

Implemented `reduce` functions:
 * summation
 * averaging
 * max
 * attention (default: use node data mapped via a FC as attention weights; split: use a dedicated part of the node data as attention weights; single: use attention weights independent of node data)
 * LSTM
 * GRU
 
Implemented `map` functions:
 * FC (fully connected layer)
 * LSTM
 * GRU

Selected setting:
 * `reduce`: summation
 * `map`: GRU step
 * application order: apply `reduce` before `map` (`reduce > map`)
 
### Scoring

Given two embeddings of trees, calculate a floating point value.

Implemented scoring functions:
 * cosine


## Training

### TASK: Predict Relatedness Scores (regression)

Predict, **how strongly** two trees are related.

"Related" can be interpreted in several ways, e.g., in the case of SICK, several annotators scored pairs of sentences intuitively and the resulting scores are averaged.

The [score](#Scoring) is used as measure for the strength of the relatedness.

### TASK: Predict Matches (binary classification)

Predict, **if** two trees belong together or not. 

"Belonging together" can be interpreted in several ways, e.g., in the case of DBpedia-NIF we selected to match article abstracts against abstracts of other wikipedia articles, that are linked within the section "See also".

The [score](#Scoring) is used as probability for the case that the trees belong together.

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

