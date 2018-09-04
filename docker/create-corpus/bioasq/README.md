# Create a rec-emb data model from BioASQ data

The [BioASQ Task 6a](http://bioasq.org/participate/challenges) data contains 13,486,072 annotated articles
 from [PubMed](https://www.ncbi.nlm.nih.gov/pubmed/), where annotated means that [MeSH](https://www.nlm.nih.gov/mesh/)
 terms have been assigned to the articles by the human curators in PubMed.
The dataset is served as a JSON string with the following format:
```json
{"articles": [{"abstractText":"text..", "journal":"journal..", "meshMajor":["mesh1",...,"meshN"],
"pmid":"PMID", "title":"title..", "year":"YYYY"},..., {..}]}
```
The JSON string contains the following fields for each article:
`pmid`: the unique identifier of each article,
`title`: the title of the article,
`abstractText`: the abstract of the article,
`year`: the year the article was published,
`journal`: the journal the article was published, and
`meshMajor`: a list with the major MeSH headings of the article.

This docker-compose service converts these PubMed records into a simple tree serialization optimized for hierarchical neural
network training (see [main readme](../../../README.md)).

*NOTE*: As *rec-emb* focuses on recursive structures, only
[structured abstracts](https://www.nlm.nih.gov/bsd/policy/structured_abstracts.html)(~1/3 of total data) are considered.

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

Finally, we create (filtered, split and shuffled) index files with:
```bash
cd docker/create-corpus/bioasq && docker-compose up corpus-bioasq-indices
```
The created files (extension: .idx.[id].npy) contain just the shuffled positions of the roots in the merged corpus.
