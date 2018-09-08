# Create a rec-emb data model from the SICK corpus

TODO: add source corpus description

This docker-compose service converts these sentence tuples into a simple tree serialization optimized for hierarchical neural 
network training (see [main readme](../../../README.md)).

## HOW TO build the data model

Install:
 * docker
 * docker compose

Clone this repo:
```bash
git clone https://github.com/ArneBinder/recursive-embedding.git
cd recursive-embedding
```

Rename (or copy) [`docker/create-corpus/sick/.env.dev`](.env.dev) to `.env` and adapt its parameters.

To start the processing, execute from project root:

```bash
cd docker/create-corpus/sick && docker-compose up
```



