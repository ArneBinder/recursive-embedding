# Create a rec-emb data model from SICK relatedness/entailment data

# TODO: describe corpus
See [here](http://clic.cimec.unitn.it/composes/sick.html) and [here](http://alt.qcri.org/semeval2014/task1/index.php?id=data-and-tools).

This docker-compose service converts these reviews into a simple tree serialization optimized for hierarchical neural
network training (see [main readme](../../../README.md)).


The workflow is as follows.

## HOW TO build the data model

Install:
 * docker
 * docker compose

Clone this repository:
```bash
git clone https://github.com/ArneBinder/recursive-embedding.git
cd recursive-embedding
```

Get the SICK dataset: [train data](http://alt.qcri.org/semeval2014/task1/data/uploads/sick_train.zip) and [test data](http://alt.qcri.org/semeval2014/task1/data/uploads/sick_test_annotated.zip). Unzip these files into respective folders `sick_train` and `sick_test_annotated` and move these folders into a common parent directory.

Rename [`docker/create-corpus/sick/.env.dev`](.env.dev) (or copy) to `.env` and configure its parameters, especially set `HOST_CORPORA_IN` where your SICK data is located. The path `HOST_CORPORA_IN` has to point to the parent directory that contains the subdirectories `sick_train` and `sick_test_annotated`. Set `HOST_CORPORA_OUT` to the directory you want to output the corpus files.


To start the (batched) parsing, execute from repository root:

```bash
cd docker/create-corpus/sick && docker-compose up corpus-parse
```

This previous command creates one batch for the train data and one for the test data. To merge the individual lexica into a final one, convert the string hashes to lexicon indices and finally concatenate all data, execute:
```bash
cd docker/create-corpus/sick && docker-compose up corpus-merge
```

Finally, we create (filtered, split and shuffled) index files with:
```bash
cd docker/create-corpus/sick
docker-compose up corpus-indices-test
docker-compose up corpus-indices-train
```
The created files (extension: .idx.[id].npy) contain just the shuffled positions of the roots in the merged corpus.
