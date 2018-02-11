# Create corpus from SICK corpus data

TODO: add source corpus description
TODO: add target data format description (see [dbpedia-nif](../dbpedia-nif/README.md))

## HOW TO build the corpus

Install:
 * docker
 * docker compose

Clone this repo:
```bash
git clone https://github.com/ArneBinder/recursive-embedding.git
cd recursive-embedding
```

Adapt parameters in [`docker/create-corpus/sick/.env`](docker/create-corpus/sick/.env) file.

To start the processing, execute from project root:

```bash
cd docker/create-corpus/sick && docker-compose up
```



