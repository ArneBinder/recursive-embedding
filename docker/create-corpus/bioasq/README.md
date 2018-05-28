# Create a rec-emb data model from BioASQ data

TODO: add BioASQ data description

This docker-compose service converts these PubMed records into a simple tree serialization optimized for hierarchical neural
network training (see [main readme](../../../README.md)).

The workflow is as follows. TODO: describe workflow


## HOW TO build the data model

Install:
 * docker
 * docker compose

Clone this repository:
```bash
git clone https://github.com/ArneBinder/recursive-embedding.git
cd recursive-embedding
```

Get the BioASQ dataset. After unzipping you should have one json file like `BIOASQ_DIRECTORY/allMeSH_2018.json`.
Split the file into chunks of e.g. 10000 lines for batch parsing:
```bash
mkdir BIOASQ_DIRECTORY/split && cd BIOASQ_DIRECTORY/split
split -l 10000 ../allMeSH_2018.json
```

Rename [`docker/create-corpus/bioasq/.env.dev`](.env.dev) (or copy) to `.env` and configure its parameters, especially set `HOST_CORPORA_IN` and `BIOASQ_SUBDIR` where your BioASQ data is located (The concatenated path `HOST_CORPORA_IN/BIOASQ_SUBDIR` has to point to the directory containing the split files) and set `HOST_CORPORA_OUT` to the directory you want to output the corpus files (ATTENTION: expected corpus size: ~32 GB (batches) + X GB (final)).

To prepare the data (uniform abstract labels with mappings from https://structuredabstracts.nlm.nih.gov/downloads.shtml), execute from repository root:
```bash
cd docker/create-corpus/bioasq && docker-compose up corpus-bioasq-prepare
```

To start the (batched) parsing, execute from repository root:

```bash
cd docker/create-corpus/bioasq && docker-compose up corpus-bioasq-parse
```
NOTE: The processing can be interrupted any time, restarting continues from the latest position.

This previous command creates batches of processed data. To merge the individual lexica into a final one, convert the string hashes to lexicon indices and finally concatenate all data, execute:
```bash
cd docker/create-corpus/bioasq && docker-compose up corpus-bioasq-merge
```

TODO: implement this
Finally, we create (filtered, splitted and shuffled) index files with:
```bash
cd docker/create-corpus/bioasq && docker-compose up corpus-bioasq-indices
```
The created files (extension: .idx.[id].npy) contain just the shuffled positions of the roots in the merged corpus. But they roots are filtered according to the number of MeSH terms in this article/tree (see parameters MESH_MIN_COUNT and MESH_MAX_COUNT in the .env file for the actual values).
