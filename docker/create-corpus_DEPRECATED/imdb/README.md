# Create a rec-emb data model from IMDB sentiment data

The [IMDB sentiment dataset](http://ai.stanford.edu/~amaas/data/sentiment/) is for binary sentiment classification and provides a set of 25,000 highly polar movie reviews for training, and 25,000 for testing.
The dataset is served as plain text files containing one review per file. The files are separated into positive and negative reviews (directories `pos` and `neg`).

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

Get the [IMDB dataset](http://ai.stanford.edu/~amaas/data/sentiment/). After unzipping you should have one folder containing subdirectories `train` and `test`.

Rename [`docker/create-corpus/imdb/.env.dev`](.env.dev) (or copy) to `.env` and configure its parameters, especially set `HOST_CORPORA_IN` where your IMDB data is located. The path `HOST_CORPORA_IN` has to point to the directory containing the subdirectories `train` and `test`. Set `HOST_CORPORA_OUT` to the directory you want to output the corpus files.


To start the (batched) parsing, execute from repository root:

```bash
cd docker/create-corpus/imdb && docker-compose up corpus-parse
```

This previous command creates one batch for the train data and one for the test data. To merge the individual lexica into a final one, convert the string hashes to lexicon indices and finally concatenate all data, execute:
```bash
cd docker/create-corpus/imdb && docker-compose up corpus-merge
```

Finally, we create (filtered, split and shuffled) index files with:
```bash
cd docker/create-corpus/imdb
docker-compose up corpus-indices-test
docker-compose up corpus-indices-train
```
The created files (extension: .idx.[id].npy) contain just the shuffled positions of the roots in the merged corpus.
