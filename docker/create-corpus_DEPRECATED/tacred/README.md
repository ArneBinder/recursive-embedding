# Create a rec-emb data model from Tacred relation extraction data

# TODO: describe corpus

This docker-compose service converts these annotated sentences into a simple tree serialization optimized for hierarchical neural
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

Rename [`docker/create-corpus/tacred/.env.dev`](.env.dev) (or copy) to `.env` and configure its parameters, especially set `HOST_CORPORA_IN` where your tacred data is located. The path `HOST_CORPORA_IN` has to point to the parent directory that contains the files `train.jsonl`, `dev.jsonl` and `test.jsonl`. Set `HOST_CORPORA_OUT` to the directory you want to output the corpus files.


To start the (batched) parsing, execute from repository root:

```bash
cd docker/create-corpus/tacred && docker-compose up corpus-parse
```

This previous command creates one batch for the train data and one for the test data. To merge the individual lexica into a final one, convert the string hashes to lexicon indices and finally concatenate all data, execute:
```bash
cd docker/create-corpus/tacred && docker-compose up corpus-merge
```

Finally, we create (filtered, split and shuffled) index files with:
```bash
cd docker/create-corpus/tacred
docker-compose up corpus-indices-test
docker-compose up corpus-indices-train
```
The created files (extension: .idx.[id].npy) contain just the shuffled positions of the roots in the merged corpus.
